import gc
import logging
from typing import List, Optional, Literal, TYPE_CHECKING
from typeguard import typechecked

import torch
if TYPE_CHECKING:
    from iho.IHOPipeline import IHOPipeline
from iho.configs.config_helper import SamplingMode

logger = logging.getLogger(__name__)



@typechecked
def cuda_mem(
    tag: str,
    top_k: int = 10,
    extensive: bool = False,
) -> None:
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9

    logger.info(
        "[%s] allocated=%.2f GB | reserved=%.2f GB",
        tag,
        allocated_gb,
        reserved_gb,
    )

    if not extensive:
        return

    tensors: List[torch.Tensor] = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append(obj)
            elif (
                hasattr(obj, "data")
                and torch.is_tensor(obj.data)
                and obj.data.is_cuda
            ):
                tensors.append(obj.data)
        except Exception:
            continue

    if not tensors:
        print("  No CUDA tensors found.")
        return

    def tensor_size_gb(tensor: torch.Tensor) -> float:
        return tensor.numel() * tensor.element_size() / 1e9

    tensors.sort(key=tensor_size_gb, reverse=True)

    k = min(top_k, len(tensors))
    print(f"  Top {k} largest CUDA tensors:")

    for i, tensor in enumerate(tensors[:k], start=1):
        print(
            f"   #{i:02d} | "
            f"{tensor_size_gb(tensor):6.2f} GB | "
            f"shape={tuple(tensor.shape)} | "
            f"dtype={tensor.dtype} | "
            f"device={tensor.device}"
        )

@typechecked
class ModelManager:
    def __init__(self, pipeline: 'IHOPipeline'):
        self.pipeline = pipeline
    
    def load_attacker(self, checkpoint_path: Optional[str] = None):
        from iho.model_wrapper.LLaDAWrapper import LLaDAWrapper
        
        load_checkpoint = (
            checkpoint_path if checkpoint_path is not None 
            else self.pipeline.configs['attack_model'].get('lora_checkpoint')
        )
        
        if self.pipeline.attacker is None:
            cuda_mem("Loading attacker model")
            self.pipeline.attacker = LLaDAWrapper(
                model_name=self.pipeline.configs['attack_model']['model_id'],
                device=self.pipeline.configs['general']['device'],
                cache_dir=self.pipeline.configs['general']['custom_cache_path'],
                attacker_input_size=self.pipeline.configs['attack_model']['attacker_input_size'],
                dtype=torch.bfloat16,
                lora_config=(
                    self.pipeline.configs['attack_model'].get('lora_config') 
                    if load_checkpoint is None else None
                ),
                lora_checkpoint=load_checkpoint,
                lora_trainable=load_checkpoint is not None,
            )
            cuda_mem("Attacker model loaded")
        elif checkpoint_path is not None:
            cuda_mem(f"Reloading attacker from checkpoint: {checkpoint_path}")
            self.unload_attacker()
            self.pipeline.attacker = LLaDAWrapper(
                model_name=self.pipeline.configs['attack_model']['model_id'],
                device=self.pipeline.configs['general']['device'],
                cache_dir=self.pipeline.configs['general']['custom_cache_path'],
                attacker_input_size=self.pipeline.configs['attack_model']['attacker_input_size'],
                dtype=torch.bfloat16,
                lora_config=None,
                lora_checkpoint=checkpoint_path,
                lora_trainable=True,
            )
            cuda_mem("Attacker model reloaded from checkpoint")
        else:
            logger.warning("Attacker model is already loaded but load_attacker was called again without a checkpoint_path.")
    
    def unload_attacker(self):
        if self.pipeline.attacker is not None:
            cuda_mem("Unloading attacker model")
            del self.pipeline.attacker
            self.pipeline.attacker = None
            torch.cuda.empty_cache()
            cuda_mem("Attacker model unloaded")
    
    def load_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        if not self.pipeline.configs["general"]["embed_attack_prompts"]:
            return

        if self.pipeline.embedding_model is None:
            cuda_mem("Loading embedding model")

            self.pipeline.embedding_model = SentenceTransformer(
                self.pipeline.configs["general"]["embedding_model"],
                device=self.pipeline.configs["general"]["device"],
                cache_folder=self.pipeline.configs["general"]["custom_cache_path"],
            )

            cuda_mem("Embedding model loaded")

    def unload_embedding_model(self):
        if self.pipeline.embedding_model is not None:
            cuda_mem("Unloading embedding model")
            del self.pipeline.embedding_model
            self.pipeline.embedding_model = None
            torch.cuda.empty_cache()
            cuda_mem("Embedding model unloaded")
    


    def load_attacked_model(self):
        from iho.model_wrapper.ARWrapper import ARWrapper, LLMConfig
        
        if self.pipeline.attacked_wrapper is None:
            cuda_mem("Loading attacked model")
            self.pipeline.attacked_wrapper = ARWrapper(
                LLMConfig(
                    model_name=self.pipeline.configs['attacked_model']['model_ids'][0].replace("-with-detector", ""),
                    device=self.pipeline.configs['general']['device'],
                    max_new_tokens=self.pipeline.configs['attacked_model']['max_new_tokens'],
                    dtype=torch.bfloat16
                )
            )
            cuda_mem("Attacked model loaded")
    


    def unload_attacked_model(self):
        cuda_mem("Unloading attacked model")
        wrapper = self.pipeline.attacked_wrapper

        if wrapper is not None:
            del self.pipeline.attacked_wrapper
            self.pipeline.attacked_wrapper = None
            torch.cuda.empty_cache()

        cuda_mem("Attacked model unloaded")
        
    def load_detector(self) -> None:
        from iho.model_wrapper.PolyGuardWrapper import PolyGuardWrapper

        if self.pipeline.detector is not None:
            logger.warning("Detector is already loaded.")
            return

        cuda_mem("Loading detector model")
        if "detector_model" not in self.pipeline.configs:
            raise ValueError("Detector model configuration is missing in configs.")
        
        cfg = self.pipeline.configs["detector_model"]
        self.pipeline.detector = PolyGuardWrapper(
            model_name=cfg["model_id"],
            device=self.pipeline.configs["general"]["device"],
            cache_dir=self.pipeline.configs["general"]["custom_cache_path"],
            max_new_tokens=cfg["max_new_tokens"],
            dtype=torch.bfloat16,
        )
        cuda_mem("Detector model loaded")

    def unload_detector(self) -> None:
        if self.pipeline.detector is None:
            logger.warning("No detector model is currently loaded, unload_detector does nothing.")
            return

        cuda_mem("Unloading detector model")
        del self.pipeline.detector
        self.pipeline.detector = None
        torch.cuda.empty_cache()
        cuda_mem("Detector model unloaded")
    
    def load_judge(self, mode: Literal["training", "validation", "validation_2"]) -> None:
        cuda_mem(f"Loading judge model for mode: {mode}")
        from judgezoo import Judge

        cfg = self.pipeline.configs["judge_models_config"]
        model_name = cfg.get(mode)

        if model_name is None:
            raise ValueError(f"No judge model specified for mode '{mode}' in configs.")

        if self.pipeline.judge_model is not None:
            logger.info("A judge model '%s' is already loaded", self.pipeline._loaded_judge_name)

            if self.pipeline._loaded_judge_name == model_name:
                logger.info(
                    "The requested judge model '%s' is already loaded. Skipping reload.",
                    model_name,
                )
                return

            logger.warning(
                "Requested judge '%s' differs from currently loaded judge '%s'. This might be supported in the future.",
                model_name,
                self.pipeline._loaded_judge_name,
            )
            raise RuntimeError(
                "Cannot load a different judge model without unloading the current one first."
            )

        self.unload_judge()
        self.pipeline.judge_model = Judge.from_name(model_name)
        self.pipeline._loaded_judge_name = model_name
        cuda_mem(f"Loaded judge model for mode: {mode}")

            
    
    def unload_judge(self) -> None:
        from typing import Any
        if self.pipeline.judge_model is None:
            logger.warning("No judge model is currently loaded, so unload_judge does nothing.")
            return

        cuda_mem("Judge model is loaded")

        judge_any: Any = self.pipeline.judge_model
        if hasattr(judge_any, 'classifier') and judge_any.classifier is not None:
            #judge_any.classifier.cpu()
            del judge_any.classifier
            judge_any.classifier = None

        del self.pipeline.judge_model
        self.pipeline.judge_model = None
        self.pipeline._loaded_judge_name = None

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        cuda_mem("Judge model unloaded and GPU memory cleared")
        
    def load_dpo_trainer(self):
        from iho.trainer.DPOTrainer import DPOTrainer
        
        if self.pipeline.dpo_trainer is None:
            if self.pipeline.attacker is None:
                self.load_attacker()
            
            cuda_mem("Initializing DPO trainer")
            self.pipeline.dpo_trainer = DPOTrainer(
                wrapper=self.pipeline.attacker,
                learning_rate=self.pipeline.configs['dpo_training']['learning_rate'],
                beta=self.pipeline.configs['dpo_training']['beta'],
                run_name=self.pipeline.run_name
            )
            cuda_mem("DPO trainer initialized")
    
    def unload_dpo_trainer(self):
        if self.pipeline.dpo_trainer is not None:
            cuda_mem("Unloading DPO trainer")
            del self.pipeline.dpo_trainer
            self.pipeline.dpo_trainer = None
            torch.cuda.empty_cache()
            cuda_mem("DPO trainer unloaded")


            