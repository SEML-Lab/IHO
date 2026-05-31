from typing import List, Optional, Literal, Tuple
from typeguard import typechecked
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel


logger = logging.getLogger(__name__)

@dataclass
class MaskingResult:
    masked_ids: torch.Tensor 
    mask_positions: torch.Tensor
    

@dataclass
class PredictionResult:
    predicted_ids: torch.Tensor
    neg_log_likelihoods: torch.Tensor
    logits: Optional[torch.Tensor] = None

@typechecked
class LLaDAWrapper:
    def __init__(
        self,
        model_name: str = 'GSAI-ML/LLaDA-8B-Base',
        device: str = 'cuda',
        cache_dir: Optional[str] = None,
        mask_token_id: int = 126336,
        attacker_input_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        lora_config: Optional[LoraConfig] = None,
        lora_checkpoint: Optional[str] = None,
        lora_trainable: bool = False
    ):
        self.device = device
        self.mask_token_id = mask_token_id
        self.dtype = dtype
        self.attacker_input_size = attacker_input_size
        
        logger.info(f"Loading model {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            cache_dir=cache_dir
            ).to(device)
        
        self.has_lora = False
        if lora_checkpoint is not None:
            logger.info(f"Loading LoRA checkpoint from {lora_checkpoint}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_checkpoint,
                is_trainable=lora_trainable,
            )
            self.has_lora = True
        elif lora_config is not None:
            logger.info("Initializing LoRA layers based on configuration...")
            self.model = get_peft_model(self.model, lora_config)
            self.has_lora = True
            trainable_params, all_param = self.model.get_nb_trainable_parameters()
            logger.info(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")
        else:
            raise RuntimeError("LLaDAWrapper requires LoRA configuration or checkpoint to be provided.")

        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.tokenizer.mask_token = "<|mdm_mask|>"
        self.padding_token_id = self.tokenizer.pad_token_id
        
        logger.info("Model initalized successfully!")
    
    def train(self) -> 'LLaDAWrapper':
        self.model.train()
        return self
    
    def eval(self) -> 'LLaDAWrapper':
        self.model.eval()
        return self
    
    @contextmanager
    def disable_adapter(self):
        if not self.has_lora:
            yield
            return
        
        self.model.disable_adapter_layers()
        try:
            yield
        finally:
            self.model.enable_adapter_layers()
    
    def save_lora(self, save_path: str) -> None:
        if not self.has_lora:
            raise RuntimeError("Model does not have LoRA adapters to save")
        
        logger.info(f"Saving LoRA adapter to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Adapter saved!")
    
    def merge_lora_into_base(self) -> None:
        if not self.has_lora:
            raise RuntimeError("Model does not have LoRA adapters to merge.")
        
        logger.info("Merging LoRA adapter into base model weights...")
        self.model = self.model.merge_and_unload()
        self.has_lora = False
        logger.info("Merge complete. Model is now a plain base model.")
    
    def save_merged_model(self, save_path: str) -> None:
        if self.has_lora:
            raise RuntimeError("Merge the adapter first before saving.")
        
        logger.info(f"Saving merged model to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Merged model saved!")
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        return encoded['input_ids'].to(self.device)
    
    def mask_tokens(
        self,
        token_ids: torch.Tensor,
        masking_mode: Literal['all', 'prompt', 'attack'],
        mask_all: bool = False
    ) -> MaskingResult:

        B, L = token_ids.shape
        device = token_ids.device

        if masking_mode == "all":
            eligible_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        
        elif masking_mode == "prompt":
            eligible_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
            eligible_mask[:, :self.attacker_input_size] = True

        elif masking_mode == "attack":
            K = self.attacker_input_size
            mask_id = self.mask_token_id

            mask_block = token_ids.new_full((B, K), mask_id)
            token_ids = torch.cat([mask_block, token_ids], dim=1)

            eligible_mask = torch.zeros(token_ids.shape, dtype=torch.bool, device=device)
            eligible_mask[:, :K] = True

        else:
            raise ValueError(f"Unknown masking_mode: {masking_mode}")

        if mask_all:
            mask_positions = eligible_mask.clone()
        else:
            p = torch.rand(B, device=device).view(B, 1) 
            rand = torch.rand((B, L), device=device)
            mask_positions = (rand < p) & eligible_mask

        masked_ids = token_ids.clone()
        masked_ids[mask_positions] = self.mask_token_id

        return MaskingResult(
            masked_ids=masked_ids,
            mask_positions=mask_positions
        )

    def compute_log_likelihood(
        self,
        masked_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        target_ids: torch.Tensor,
        use_base_model: bool = False
    ) -> torch.Tensor:
        context = self.disable_adapter() if use_base_model else torch.no_grad() if not self.model.training else nullcontext()
        
        if use_base_model or not self.model.training:
            with context:
                logits = self.model(masked_ids).logits
        else:
            logits = self.model(masked_ids).logits  
        
        log_probs = F.log_softmax(logits, dim=-1)  
        
        target_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        return (target_log_probs * mask_positions).sum(dim=1)
    
    @torch.no_grad()
    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits

        eps = 1e-20
        #U = torch.rand_like(logits, dtype=torch.bfloat16)
        #gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.bfloat16) + eps) + eps)
        #Done to save memory for now
        return (logits -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.bfloat16) + eps) + eps)) / temperature


    @torch.no_grad()
    def _forward_process_batched(self, batch, fixed_mask, mask_id=126336):

        b, l = batch.shape
        device = batch.device

        target_len = l

        x = torch.randint(1, target_len + 1, (b,), device=device)

        indices = torch.arange(target_len, device=device).unsqueeze(0).expand(b, -1)

        is_mask = indices < x.unsqueeze(1)

        randperm = torch.argsort(torch.rand(b, target_len, device=device), dim=1)
        is_mask = torch.gather(is_mask, 1, randperm)

        is_mask = is_mask & fixed_mask

        noisy_batch = torch.where(is_mask, mask_id, batch)

        mask_ratio = (x / target_len).unsqueeze(1).expand(-1, l)

        return noisy_batch, mask_ratio


    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
            
        return num_transfer_tokens


    @torch.no_grad()
    def predict_masked(
        self,
        masked_ids: torch.Tensor, 
        steps: int = 10,
        temperature: float = 0.0,
        remasking: Literal['low_confidence', 'random'] = 'low_confidence',
        mask_padding: bool = False,
        number_global_remask: int = 8,
        global_remasking: Literal['random', 'low_confidence'] = 'random',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.model.training:
            raise RuntimeError("predict_masked must be run in evaluation mode.")

        device = self.device
        x = masked_ids.to(device)
        batch_size, seq_len = x.shape

        known_mask = (x != self.mask_token_id)
        known_tokens = x.clone()
        global_conf = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        total_loglikelihood = torch.zeros(batch_size, device=device)
        
        special_token_ids = [
            self.tokenizer.bos_token_id, 
            self.tokenizer.eos_token_id, 
            self.tokenizer.pad_token_id
        ]
        special_token_ids = list(set(int(s) for s in special_token_ids if s is not None))

        mask_index_initial = (x == self.mask_token_id)
        num_transfer_tokens = self._get_num_transfer_tokens(
            mask_index_initial, 
            steps
        ) 

        if not isinstance(num_transfer_tokens, torch.Tensor):
            num_transfer_tokens = torch.tensor(num_transfer_tokens, device=device)

        for s in range(steps):
            mask_index = (x == self.mask_token_id) 

            x_l, _ = self._forward_process_batched(x, known_mask, mask_id=self.mask_token_id)

            attention_mask = None
            if mask_padding:
                attention_mask = (x_l != self.padding_token_id).long()

            logits = self.model(x_l, attention_mask).logits 
            logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)

            if len(special_token_ids) > 0:
                logits_with_noise[:, :, special_token_ids] = -float('inf')
            
            x0 = torch.argmax(logits_with_noise, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_logp = torch.gather(
                log_probs, dim=-1, index=x0.unsqueeze(-1)
            ).squeeze(-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)  
                idx = x0.unsqueeze(-1)
                x0_p = torch.gather(p, dim=-1, index=idx).squeeze(-1).to(device)
            elif remasking == 'random':
                x0_p = torch.rand((batch_size, seq_len), device=device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            x0 = torch.where(known_mask, known_tokens, x0)

            neg_inf = torch.tensor(-float('inf'), device=device)
            confidence = torch.where(mask_index, x0_p, neg_inf)
            confidence = torch.where(known_mask, neg_inf, confidence)

            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
            new_mask_index = torch.zeros_like(x, dtype=torch.bool, device=device)

            for b in range(batch_size):
                k = int(num_transfer_tokens[b, s].item())

                if (k == 0 and 
                    s < steps - 1 - number_global_remask and 
                    number_global_remask > 0 and 
                    (s % number_global_remask == 0)):
                    
                    unknown_indices = (~known_mask[b]).nonzero(as_tuple=True)[0]
                    
                    if global_remasking == "random":
                        if len(unknown_indices) >= number_global_remask:
                            rnd = torch.randperm(len(unknown_indices), device=device)[:number_global_remask]
                            random_index = unknown_indices[rnd]
                            new_mask_index[b, random_index] = True
                    elif global_remasking == "low_confidence":
                        if len(unknown_indices) >= number_global_remask:
                            unknown_confidence = global_conf[b][unknown_indices]
                            _, local_indices = unknown_confidence.topk(number_global_remask, largest=False)
                            selected_indices = unknown_indices[local_indices]
                            new_mask_index[b, selected_indices] = True

                k = max(k, 1)
                _, select_index = torch.topk(confidence[b], k=k)
                transfer_index[b, select_index] = True

            x[transfer_index] = x0[transfer_index]
            step_loglik = (chosen_logp * transfer_index).sum(dim=1)
            total_loglikelihood += step_loglik

            global_conf[transfer_index] = confidence[transfer_index].float()
            x[new_mask_index] = self.mask_token_id

        return x, total_loglikelihood