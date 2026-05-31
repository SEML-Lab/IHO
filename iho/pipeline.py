import os
from typing import Dict, List, Literal, Optional, Union, cast
import logging
from typeguard import typechecked
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pandas as pd
from judgezoo import Judge
from sentence_transformers import SentenceTransformer

from iho.model_wrapper.ARWrapper import ARWrapper
from iho.model_wrapper.LLaDAWrapper import LLaDAWrapper
from iho.model_wrapper.PolyGuardWrapper import PolyGuardWrapper
from iho.datasets.PreferenceDataset import ThresholdedTopKBottomKPairing, PairingStrategy, PreferenceDataset
from iho.trainer.DPOTrainer import DPOTrainer
from iho.configs.config_helper import PipelineConfig, NamedBehaviourSubsets, SamplingMode, DetectorOverwriteMode
from iho.utils.batch_utils import rebatch_data, flatten_batch_results
from iho.utils.general_utils import generate_wandb_style_name, init_seeds, verify_cuda, save_configs_json, build_attack_dataloader
from iho.utils.ModelManager import ModelManager

logger = logging.getLogger(__name__)

DETECTOR_REFUSAL = "Sorry, I can't assist you with that."

@typechecked
class IHOPipeline:
    def __init__(
        self,
        experiment_path: str,
        run_name: Optional[str],
        configs: PipelineConfig,
        overwrite_existing: bool = False,
    ):
        self.run_name = run_name or generate_wandb_style_name()
        self.run_path = os.path.join(experiment_path, self.run_name)
        self.overwrite_existing = overwrite_existing
        
        self._verify_input(configs)
        init_seeds(configs["general"]['seed'])
        verify_cuda(configs["general"]['device'])

        os.makedirs(os.path.dirname(self.run_path), exist_ok=True)
        
        self.configs = configs
        
        save_configs_json(
            configs=configs,
            run_path=self.run_path,
            logger=logger
        )

        self.model_manager = ModelManager(self)

        self.attacker: Optional[LLaDAWrapper] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.attacked_wrapper: Optional[ARWrapper] = None
        self.detector: Optional[PolyGuardWrapper] = None
        self.judge_model: Optional[Judge] = None
        self._loaded_judge_name: Optional[str] = None
        self.dpo_trainer: Optional[DPOTrainer] = None
        
        self.attack_dataloaders: dict[str, DataLoader] = {}

        behaviour_subsets = configs["attack_dataset"]["behaviour_subsets"]

        for split, subset in behaviour_subsets.items():
            subset = cast(Union[NamedBehaviourSubsets, List[int]], subset)

            self.attack_dataloaders[split] = build_attack_dataloader(
                rows=subset,
                batch_size=configs["batch_sizes"]["attack"],
            )
        
        logger.info("Initialized IHO Pipeline with run name: %s", self.run_path)
    
    def _verify_input(
        self,
        configs: PipelineConfig
    ) -> None:

        lora_config = configs['attack_model'].get('lora_config')
        lora_checkpoint = configs['attack_model'].get('lora_checkpoint')
        
        if lora_config is not None and lora_checkpoint is not None:
            raise ValueError(
                "Use either lora_config or lora_checkpoint, not both."
            )
        if lora_config is None and lora_checkpoint is None:
            raise ValueError(
                "You must either provide a lora_config or lora_checkpoint."
            )

        if configs['general']["embed_attack_prompts"]:
            embedding_model = configs['general']["embedding_model"]
            if not isinstance(embedding_model, str) or not embedding_model.strip():
                raise TypeError("An embedding_model must be provided if embedding requested")
        

    def _get_sample_path(self, cycle_id: Optional[int], mode: SamplingMode) -> str:
        suffix = "_validation" if mode == "validation" else ""
        suffix = "_test" if mode == "testing" else suffix
        if cycle_id is None:
            return os.path.join(self.run_path, f"samples_output{suffix}.parquet")
        else:
            folder = "samples_validation" if  mode == "validation" else "samples"
            return os.path.join(self.run_path, folder, f"cycle_{cycle_id}{suffix}.parquet")

    def _get_checkpoint_path(self, cycle_id: int) -> str:
        return os.path.join(
            self.run_path,
            "checkpoints",
            f"best_cycle_{cycle_id}"
        )
    
    def _get_preference_dataset_path(self, cycle_id: int) -> str:
        return os.path.join(
            self.run_path,
            "dpo_sets",
            f"cycle_{cycle_id}.parquet"
        )
    
    def _sample_exists(self, cycle_id: Optional[int], mode: SamplingMode) -> bool:
        sample_path = self._get_sample_path(cycle_id, mode)
        return os.path.exists(sample_path)
    
    def _checkpoint_exists(self, cycle_id: int) -> bool:
        checkpoint_path = self._get_checkpoint_path(cycle_id)
        return os.path.exists(checkpoint_path)
    
    def _find_last_valid_checkpoint(self, n_cycles: int) -> int:
        for cycle_id in range(n_cycles - 1, -1, -1):
            if self._checkpoint_exists(cycle_id):
                logger.info(f"Found valid checkpoint from cycle {cycle_id}")
                return cycle_id
        
        return -1
        
    def _load_existing_sample(self, cycle_id: Optional[int], mode: SamplingMode) -> pd.DataFrame:
        sample_path = self._get_sample_path(cycle_id, mode)
        logger.info(f"Loading existing sample from {sample_path}")
        return pd.read_parquet(sample_path)
    
    @torch.no_grad()
    def _generate_attack_prompts(
        self,
        batches: List[Dict[str, torch.Tensor]],
        mode: SamplingMode
    ) -> List[Dict]:
        
        if mode != "dpo":
            #If mode==dpo, attacker will be loaded already
            self.model_manager.load_attacker()
        
        assert self.attacker is not None, "Attacker failed to load"
        self.attacker.eval()
        
        all_attack_results = []
        
        for batch in tqdm(batches, total=len(batches), desc="Generating attack prompts"):
            target_texts = batch["target_text"]
            goal_text = batch["goal_text"]
            
            target_sequences = [f"\nAnswer: {t}" for t in target_texts]
            target_ids = self.attacker.encode(target_sequences)
            
            masking_result = self.attacker.mask_tokens(
                token_ids=target_ids,
                masking_mode="attack",
                mask_all=True,
            )
            
            original_prompt_text = self.attacker.tokenizer.batch_decode(
                masking_result.masked_ids,
                skip_special_tokens=False
            )

            inpainted_prompt_ids_full, attack_loglikelihood = self.attacker.predict_masked(
                masked_ids=masking_result.masked_ids,
                steps=self.configs['attack_model']['attacker_step_number'],
                temperature=self.configs['attack_model']['attacker_temperature'],
                remasking=self.configs['attack_model']['remasking'],
                mask_padding=self.configs['attack_model']['mask_padding_tokens'],
            )

            inpainted_prompt_text_full = self.attacker.tokenizer.batch_decode(
                inpainted_prompt_ids_full, skip_special_tokens=False
            )
            
            attacking_prompt_ids = (
                inpainted_prompt_ids_full[masking_result.mask_positions]
                .view(inpainted_prompt_ids_full.size(0), -1)
            )
            
            attacking_prompt_text = self.attacker.tokenizer.batch_decode(
                attacking_prompt_ids, skip_special_tokens=False
            )
            
            all_attack_results.append({
                "jb_index": [int(idx) for idx in batch["jb_index"]],
                "goal_text": goal_text,
                "target_text": target_texts,
                "original_prompt_ids": masking_result.masked_ids.cpu().tolist(),
                "original_prompt_text": original_prompt_text,
                "attacking_prompt_ids": attacking_prompt_ids.cpu().tolist(),
                "attacking_prompt_text": attacking_prompt_text,
                "inpainted_prompt_ids_full": inpainted_prompt_ids_full.cpu().tolist(),
                "inpainted_prompt_text_full": inpainted_prompt_text_full,
                "attack_loglikelihood": attack_loglikelihood.cpu().tolist(),
            })
            
            torch.cuda.empty_cache()

        if mode != "dpo":
        #If mode==dpo, attacker must stay loaded so both IHOPipeline and DPOTrainer reference the same object. Otherwise unload to improve memory usage.
            self.model_manager.unload_attacker()

        return flatten_batch_results(all_attack_results)
    
    @torch.no_grad()
    def _add_embeddings(self, attack_samples: List[Dict], key: str) -> List[Dict]:
        self.model_manager.load_embedding_model()
        assert self.embedding_model is not None, "Embedding model failed to load"

        texts = [s[key] for s in attack_samples]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for s, emb in zip(attack_samples, embeddings):
            s[f"{key}_embedding"] = emb.astype("float32").tolist()

        self.model_manager.unload_embedding_model()

        return attack_samples

    @torch.no_grad()
    def _generate_responses(
        self,
        attack_samples: List[Dict],
        mode: SamplingMode
    ) -> List[Dict]:

        self.model_manager.load_attacked_model()
        assert self.attacked_wrapper is not None, "Attacked model failed to load"

        batches = rebatch_data(
            attack_samples,
            self.configs['batch_sizes']['generate'] // (2 if mode == "dpo" else 1)
        )

        results_with_responses: List[Dict] = []

        for batch_samples in tqdm(batches, total=len(batches), desc="Generating responses"):

            attacking_prompt_text = [
                s["attacking_prompt_text"] for s in batch_samples
            ]

            batch_results = self.attacked_wrapper.generate(
                prompts=attacking_prompt_text,
                use_chat_template=self.configs['attacked_model']['use_chat_template'],
                greedy_sampling=self.configs['attacked_model']['greedy_sampling'],
                max_new_tokens=self.configs['attacked_model']['max_new_tokens'],
                batch_size=self.configs['batch_sizes']['generate'],
                compute_logprobs=self.configs['attacked_model']['compute_logprobs'],
            )

            for sample, result in zip(batch_samples, batch_results):
                results_with_responses.append(sample | result)

            #TODO Benchmark this against doing a single empty_cache() after the entire loop
            torch.cuda.empty_cache()

        self.model_manager.unload_attacked_model()

        return results_with_responses

    @torch.no_grad()
    def _detect_responses(
        self,
        samples_with_responses: List[Dict],
    ) -> List[Dict]:
        
        self.model_manager.load_detector()
        assert self.detector is not None, "Detector model failed to load"

        overwrite_mode: DetectorOverwriteMode = "harmful_content"

        batches = rebatch_data(
            samples_with_responses,
            self.configs["batch_sizes"]["detector"],
        )

        for batch_samples in tqdm(batches, total=len(batches), desc="Detecting responses"):
            prompts   = [s["attacking_prompt_text"]       for s in batch_samples]
            responses = [s["attacked_output"] for s in batch_samples]

            detection_results = self.detector.detect(prompts, responses)

            for sample, det in zip(batch_samples, detection_results):
                sample["harmful_request"]  = det["harmful_request"]
                sample["response_refusal"] = det["response_refusal"]
                sample["harmful_response"] = det["harmful_response"]

                should_overwrite = (
                    det["harmful_request"] or det["harmful_response"]
                    if overwrite_mode == "harmful_content"
                    else det["harmful_response"]
                )

                sample["attacked_output_raw"] = sample["attacked_output"]
                if should_overwrite:
                    sample["attacked_output"] = DETECTOR_REFUSAL

        self.model_manager.unload_detector()
        return samples_with_responses


    @torch.no_grad()
    def _judge_responses(
        self,
        samples_with_responses: List[Dict],
        mode: SamplingMode
    ) -> List[Dict]:
        for judge_mode in self.configs["judge_models_config"].keys():
            self.model_manager.load_judge(judge_mode) #type: ignore
            assert self.judge_model is not None, "Judge model failed to load"

            batches = rebatch_data(
                samples_with_responses,
                self.configs['batch_sizes']['judge'] // (3 if mode == "dpo" else 1)
            )

            for batch_samples in tqdm(batches, total=len(batches), desc=f"Judging responses ({judge_mode})"):
                goal_response_pairs = []
                for sample in batch_samples:
                    goal_response_pairs.append(
                        [
                            {"role": "user", "content": sample["goal_text"]},
                            {"role": "assistant", "content": sample["attacked_output"]},
                        ]
                    )

                scores = self.judge_model(goal_response_pairs)

                for sample, score in zip(batch_samples, scores['p_harmful']):
                    sample[f"judge_score_{judge_mode}"] = score

                # TODO: Benchmark this against doing a single empty_cache() after the entire loop
                torch.cuda.empty_cache()

            self.model_manager.unload_judge()

        return samples_with_responses
        
    
    def sample(
        self,
        mode: SamplingMode = "training",
        save_df: bool = False,
        cycle_id: Optional[int] = None,
    ) -> pd.DataFrame:

        if (
            not self.overwrite_existing 
            and self._sample_exists(cycle_id, mode)
            and mode != "dpo" 
        ):
            logger.info("Sample for cycle %s (mode=%s) already exists, loading from disk", cycle_id, mode)
            return self._load_existing_sample(cycle_id, mode)
        
        if mode not in self.attack_dataloaders or mode not in self.configs['general']['num_sampled_attacks']:
            raise ValueError(f"Invalid mode: {mode}. Must be in {list(self.attack_dataloaders.keys())} AND {list(self.configs['general']['num_sampled_attacks'].keys())}")


        logger.info(f"Generating new sample for cycle {cycle_id} (mode={mode})")
        
        all_batches = []
        min_samples = self.configs['general']['num_sampled_attacks'][mode]
        dataloader = self.attack_dataloaders[mode]
        dataloader_iter = iter(dataloader)
        samples_count = 0
        
        while samples_count < min_samples:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            all_batches.append(batch)
            samples_count += len(batch["jb_index"])


        logger.info("[START] PHASE: ATTACK_GENERATION | MODE: %s | CYCLE: %s", mode, cycle_id)
        logger.info("Generating attack prompts for %d batches (batch_size=%d)", len(all_batches), self.configs['batch_sizes']['attack'])
        attack_samples = self._generate_attack_prompts(all_batches, mode)
        logger.info("[END] PHASE: ATTACK_GENERATION | MODE: %s | CYCLE: %s", mode, cycle_id)

        if self.configs["general"]["embed_attack_prompts"]:
            logger.info("[START] PHASE: EMBEDDING ATTACKS | MODE: %s | CYCLE: %s", mode, cycle_id)
            attack_samples = self._add_embeddings(attack_samples, key="attacking_prompt_text")
            logger.info("[END] PHASE: EMBEDDING ATTACKS | MODE: %s | CYCLE: %s", mode, cycle_id)

        logger.info("[START] PHASE: RESPONSE_GENERATION | MODE: %s | CYCLE: %s", mode, cycle_id)
        logger.info("Generating responses (batch_size=%d)", self.configs['batch_sizes']['generate'])
        samples_with_responses = self._generate_responses(attack_samples, mode)
        logger.info("[END] PHASE: RESPONSE_GENERATION | MODE: %s | CYCLE: %s", mode, cycle_id)

        if self.configs["general"]["embed_attack_prompts"]:
            logger.info("[START] PHASE: EMBEDDING OUTPUTS | MODE: %s | CYCLE: %s", mode, cycle_id)
            attack_samples = self._add_embeddings(samples_with_responses, key="attacked_output")
            logger.info("[END] PHASE: EMBEDDING OUTPUTS | MODE: %s | CYCLE: %s", mode, cycle_id)

        if self.configs["attacked_model"]["model_ids"][0].endswith("-with-detector"):
            logger.info("[START] PHASE: DETECTION | MODE: %s | CYCLE: %s", mode, cycle_id)
            samples_with_responses = self._detect_responses(samples_with_responses)
            logger.info("[END] PHASE: DETECTION | MODE: %s | CYCLE: %s", mode, cycle_id)

        logger.info("[START] PHASE: RESPONSE_JUDGING | MODE: %s | CYCLE: %s", mode, cycle_id)
        logger.info("Judging responses (batch_size=%d)", self.configs['batch_sizes']['judge'])
        final_samples = self._judge_responses(samples_with_responses, mode)
        logger.info("[END] PHASE: RESPONSE_JUDGING | MODE: %s | CYCLE: %s", mode, cycle_id)

        df = pd.DataFrame(final_samples)

        if save_df:
            sample_path = self._get_sample_path(cycle_id, mode)
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            df.to_parquet(sample_path, index=False)
            logger.info(f"Saved sample to {sample_path}")
        
        return df

    def _train_preference(
        self, 
        preference_dataset: PreferenceDataset, 
        cycle_id: int = 0,
        save_checkpoint: bool = True
    ) -> List[float]:
        
        logger.info("[START] PHASE: DPO_TRAINING | DPO TRAINING: ACTIVE | CYCLE: %d", cycle_id)
        
        self.model_manager.load_attacker()
        self.model_manager.load_dpo_trainer()
        assert self.dpo_trainer is not None, "DPO trainer failed to initialize"
        
        if save_checkpoint is False:
            logger.warning("Not saving checkpoints during DPO training. THIS IS NOT RECOMMENDED FOR REAL EXPERIMENTS.")
    
        dataloader = DataLoader(
            preference_dataset,
            batch_size=self.configs['batch_sizes']['dpo'],
            shuffle=True,
        )

        losses = self.dpo_trainer.train(
            dataloader=dataloader,
            n_epochs=self.configs['dpo_training']['dpo_epochs'],
            masking_mode=self.configs['dpo_training']['preference_masking_mode'],
            mask_all=self.configs['dpo_training']['dpo_mask_all'],
            cycle_id=cycle_id,
            run_path=self.run_path,
            checkpoint_every=self.configs["dpo_training"]["checkpoint_every"],
            save_checkpoints=self.configs['dpo_training']['save_checkpoints'],
            patience=self.configs['dpo_training']['patience'],
            sample_fn=self.sample,
            warmup_epochs = self.configs['dpo_training']['warmup_epochs'],
            load_optimizer_state = self.configs['dpo_training']['load_optimizer_state'],
            metric = self.configs['dpo_training']['dpo_metric']
        )
        
        if save_checkpoint:
            checkpoint_path = self._get_checkpoint_path(cycle_id)  # already saved by dpo_trainer
            assert os.path.isdir(checkpoint_path), f"Expected best checkpoint at {checkpoint_path}"
            self.configs['attack_model']['lora_checkpoint'] = checkpoint_path
            logger.info("Using best checkpoint from %s", checkpoint_path)
        
        self.model_manager.unload_dpo_trainer()
        self.model_manager.unload_attacker()

        logger.info("[END] PHASE: DPO_TRAINING | DPO TRAINING: ACTIVE | CYCLE: %d", cycle_id)
        logger.info("Final loss: %s", f"{losses[-1]:.4f}" if losses else "N/A")
        
        return losses
    
    def execute_one_cycle(
        self, 
        pairing_strategy: Optional[PairingStrategy] = None,
        resample_after_training: bool = False,
        cycle_id: int = 0,
        validate: bool = False,
        testing: bool = False
    ) -> None:
        has_sample = self._sample_exists(cycle_id, "training")
        has_checkpoint = self._checkpoint_exists(cycle_id)
        
        if not self.overwrite_existing and has_sample and has_checkpoint:
            logger.info(f"Cycle {cycle_id} already complete, skipping")
            self.configs['attack_model']['lora_checkpoint'] = self._get_checkpoint_path(cycle_id)
            return
        

        self.sample(mode="training", save_df=True, cycle_id=cycle_id)
        
        if validate:
            self.sample(mode="validation", save_df=True, cycle_id=cycle_id)
        
        if testing:
            self.sample(mode="testing", save_df=True, cycle_id=cycle_id)


        if pairing_strategy is None:
            pairing_strategy = ThresholdedTopKBottomKPairing(self.configs['dpo_training']["harmfulness_threshold"])
        
        preference_dataset = PreferenceDataset(
            strategy=pairing_strategy,
            run_path=self.run_path,
            expanding=self.configs['dpo_training']['expanding_dpo_dataset'],
            percent_chosen=self.configs['dpo_training']["percent_chosen"]
        )
        
        logger.info(
            "Generated %d preference pairs for DPO training",
            len(preference_dataset),
        )
        
        preference_path = self._get_preference_dataset_path(cycle_id)
        os.makedirs(os.path.dirname(preference_path), exist_ok=True)
        preference_dataset.to_parquet(preference_path)
        logger.info(f"Saved preference dataset to {preference_path}")
        
        _ = self._train_preference(
            preference_dataset,
            cycle_id=cycle_id,
            save_checkpoint=True
        )

        if resample_after_training:
            self.sample(mode="training" ,save_df=True, cycle_id=cycle_id + 1)
            if validate:
                self.sample(mode="validation", save_df=True, cycle_id=cycle_id + 1)

    def execute_multiple_cycles(
        self,
        n_cycles: int,
        pairing_strategy: Optional[PairingStrategy] = None,
    ) -> None:
        logger.info("Starting execution of %d cycles", n_cycles)
        
        last_checkpoint = self._find_last_valid_checkpoint(n_cycles)
        
        if last_checkpoint >= 0:
            start_cycle = last_checkpoint + 1
            logger.info(f"Resuming from cycle {start_cycle} (found checkpoint from cycle {last_checkpoint})")
            self.configs['attack_model']['lora_checkpoint'] = self._get_checkpoint_path(last_checkpoint)
        else:
            start_cycle = 0
            logger.info("Starting from scratch (no checkpoints found)")
        
        for cycle_id in range(start_cycle, n_cycles):
            last = (cycle_id + 1 == n_cycles)

            if "validation" in self.configs['attack_dataset']['behaviour_subsets'] and self.configs['attack_dataset']['behaviour_subsets']["training"] != self.configs['attack_dataset']['behaviour_subsets']["validation"]:
                validate = True
            else:
                validate = False

            if "testing" in self.configs['attack_dataset']['behaviour_subsets'] and self.configs['attack_dataset']['behaviour_subsets']["training"] != self.configs['attack_dataset']['behaviour_subsets']["testing"]:
                testing = True
            else:
                testing = False

            logger.info("[START] CYCLE: %d/%d", cycle_id + 1, n_cycles)
            self.execute_one_cycle(
                pairing_strategy=pairing_strategy,
                resample_after_training=last,
                cycle_id=cycle_id,
                validate=validate,
                testing=testing
            )
            logger.info("[END] CYCLE: %d/%d", cycle_id + 1, n_cycles)
        
        logger.info("Completed all %d cycles", n_cycles)


DEFAULT_BASE_MODEL = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_CHECKPOINT = "SEML-Lab/IHO-Llama-3-8B-Instruct"


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def load_model(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    device: str = "auto",
    cache_dir: Optional[str] = None,
    dtype: str | torch.dtype = torch.bfloat16,
    attack_size: int = 32,
    **_: object,
) -> LLaDAWrapper:
    return LLaDAWrapper(
        model_name=base_model,
        device=_resolve_device(device),
        cache_dir=cache_dir,
        attacker_input_size=attack_size,
        dtype=_resolve_dtype(dtype),
        lora_checkpoint=checkpoint,
    )


@torch.no_grad()
def generate(
    model: LLaDAWrapper,
    affirmative_response: str | list[str],
    *,
    num_attacks: int = 1,
    batch_size: int = 1,
    return_dict: bool = False,
    attack_size: Optional[int] = None,
    attack_steps: int = 32,
    temperature: float = 0.0,
    remasking: Literal["low_confidence", "random"] = "low_confidence",
    mask_padding: bool = True,
) -> list[str] | list[dict]:
    if isinstance(affirmative_response, str):
        responses = [affirmative_response] * num_attacks
    else:
        responses = [response for response in affirmative_response for _ in range(num_attacks)]

    old_size = model.attacker_input_size
    if attack_size is not None:
        model.attacker_input_size = attack_size

    outputs: list[str] = []
    rows: list[dict] = []
    model.eval()

    try:
        for start in range(0, len(responses), batch_size):
            batch_responses = responses[start : start + batch_size]
            target_ids = model.encode([f"\nAnswer: {response}" for response in batch_responses])
            masking = model.mask_tokens(target_ids, masking_mode="attack", mask_all=True)
            inpainted_ids, loglikelihood = model.predict_masked(
                masking.masked_ids,
                steps=attack_steps,
                temperature=temperature,
                remasking=remasking,
                mask_padding=mask_padding,
            )
            attack_ids = inpainted_ids[masking.mask_positions].view(inpainted_ids.size(0), -1)
            attacks = model.tokenizer.batch_decode(attack_ids, skip_special_tokens=False)
            if return_dict:
                full_texts = model.tokenizer.batch_decode(inpainted_ids, skip_special_tokens=False)
                rows.extend(
                    {
                        "attack": attack,
                        "affirmative_response": response,
                        "attack_ids": attack_id.cpu().tolist(),
                        "inpainted_ids": inpainted_id.cpu().tolist(),
                        "inpainted_text": full_text,
                        "attack_loglikelihood": float(score.item()),
                    }
                    for attack, response, attack_id, inpainted_id, full_text, score in zip(
                        attacks, batch_responses, attack_ids, inpainted_ids, full_texts, loglikelihood
                    )
                )
            else:
                outputs.extend(attacks)
    finally:
        model.attacker_input_size = old_size

    return rows if return_dict else outputs
