import os
from typing import List, Union, Tuple, Dict, Any, Optional
from typeguard import typechecked
import logging
from dataclasses import dataclass

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iho.utils.ModelManager import cuda_mem

@dataclass
class LLMConfig:
    model_name: str
    device: str = "cuda"
    max_new_tokens: int = 256
    dtype: torch.dtype = torch.bfloat16
    temperature: float = 1.0

@typechecked
class ARWrapper:
    CUSTOM_CACHE_PATH = os.environ.get("IHO_CACHE_DIR") or None

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.device = config.device

        logging.info(
            f"Loading model {self.model_name} on {self.device} "
            f"(dtype={self.config.dtype})"
        )

        if self.model_name == "CAT/local":
            cat_model_path = os.environ.get("IHO_CAT_MODEL_PATH")
            if not cat_model_path:
                raise ValueError(
                    "CAT/local requires IHO_CAT_MODEL_PATH to point to the merged CAT model."
                )
            self.model_name = cat_model_path
            self.tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            self.tokenizer_name = self.model_name
            

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.CUSTOM_CACHE_PATH,
            dtype=self.config.dtype,
        ).to(self.device)  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.CUSTOM_CACHE_PATH,
            padding_side="left",
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "llama" in self.model_name.lower():
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')

        self.model.eval()

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def _compute_sequence_logprobs(
        self,
        sequences: torch.Tensor,  # (B, prompt_len + gen_len)
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass to compute per-sequence log-likelihoods.
        Much cheaper than output_scores + compute_transition_scores:
        - Single forward pass vs num_generated_tokens passes during generation
        - No per-step score tensors stored in memory
        Returns (sequence_logprobs, gen_lengths) both shape (B,).
        """
        with torch.no_grad():
            logits = self.model(input_ids=sequences).logits  # (B, T, vocab)
        cuda_mem("After forward pass for logprobs")
        # Shift: predict token t from context [0..t-1]
        shift_logits = logits[:, :-1].contiguous()    # (B, T-1, vocab)
        del logits
        cuda_mem("After deleting logits")

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        del shift_logits
        cuda_mem("After deleting shifts_logits")

        # Gather log-prob of the actual token at each position
        shift_ids    = sequences[:, 1:].contiguous()  # (B, T-1)
        token_logprobs = log_probs.gather(
            -1, shift_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)
        cuda_mem("After deleting shifts_logits")
        del log_probs

        # Mask: only sum over generated tokens, not prompt or padding
        pad_id = self.tokenizer.eos_token_id
        is_generated = torch.zeros_like(shift_ids, dtype=torch.bool)
        is_generated[:, prompt_len - 1:] = True           # generated region
        is_padding = (shift_ids == pad_id)
        mask = is_generated & ~is_padding

        token_logprobs = token_logprobs * mask
        sequence_logprobs = token_logprobs.sum(dim=-1)    # (B,)
        gen_lengths = mask.sum(dim=-1)                    # (B,)

        return sequence_logprobs, gen_lengths

    @typechecked
    def generate(
        self,
        prompts: Union[str, List[str]],
        use_chat_template: bool = False,
        max_new_tokens: int | None = None,
        system_prompts: Union[str, List[str], None] = None,
        batch_size: int = 32,
        greedy_sampling: bool = False,
        enable_memory_optimization: bool = True,
        compute_logprobs: bool = False
    ) -> List[dict]:

        self.model.eval()
        cuda_mem("Before generation")
        if isinstance(prompts, str):
            prompts = [prompts]
        if system_prompts is not None and isinstance(system_prompts, str):
            system_prompts = [system_prompts]

        if use_chat_template and system_prompts is not None:
            formatted_prompts = [
                [{"role": "system", "content": s}, {"role": "user", "content": p}]
                for p, s in zip(prompts, system_prompts)
            ]
            formatted_prompts = [
                self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in formatted_prompts
            ]
        elif use_chat_template:
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in prompts
            ]
        else:
            formatted_prompts = prompts

        results: List[dict] = []
        max_new_tokens_to_use = max_new_tokens or self.config.max_new_tokens

        for i in range(0, len(formatted_prompts), batch_size):
            batch_prompts = formatted_prompts[i:i + batch_size]

            tokenized = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            )

            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            prompt_len = input_ids.shape[1]

            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens_to_use,
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": not greedy_sampling,
                "return_dict_in_generate": True
            }

            if not greedy_sampling:
                generation_kwargs["top_p"] = 0.9
                generation_kwargs["temperature"] = self.config.temperature

            # static cache is now always safe to use (no output_scores conflict)
            if enable_memory_optimization:
                try:
                    generation_kwargs["cache_implementation"] = "static"
                except (TypeError, AttributeError):
                    logging.warning(
                        "Static cache generation not supported in this transformers version. "
                        "Reserved memory may be substantially higher."
                    )

            with torch.no_grad():
                cuda_mem("Before generation step")
                outputs = self.model.generate(**generation_kwargs)
                cuda_mem("After generation step")

            sequences = outputs.sequences  # (B, prompt_len + gen_len) #type: ignore
            cuda_mem("Here 1")

            del outputs  # free KV cache / any retained tensors immediately
            cuda_mem("Here 2")

            if compute_logprobs:
                sequence_logprobs, gen_lengths = self._compute_sequence_logprobs(
                    sequences, prompt_len
                )
            cuda_mem("After")

            decoded_full = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            decoded_gen = self.tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)

            for j in range(len(decoded_gen)):
                item = {
                    "attacked_output": decoded_gen[j],
                    "output_ids": sequences[j].tolist(),
                    "attacked_output_full": decoded_full[j],
                }

                if compute_logprobs:
                    item.update({
                        "output_loglikelihood": float(sequence_logprobs[j].item()), #type: ignore
                        "output_num_tokens": int(gen_lengths[j].item()), #type: ignore
                    })

                results.append(item)

            if enable_memory_optimization and torch.cuda.is_available():
                del input_ids, attention_mask, sequences
                if compute_logprobs:
                    del sequence_logprobs, gen_lengths
                torch.cuda.empty_cache()

        return results




    @typechecked
    def log_likelihood(
        self,
        texts: Union[str, List[str]],
        outputs: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        enable_memory_optimization: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Computes mean log-likelihood and perplexity per sample.

        If outputs is None:
            Computes unconditional LM likelihood on `texts`.

        If outputs is provided:
            Computes conditional likelihood p(outputs | texts).

        Returns:
            mean_log_likelihoods: List[float]
            perplexities: List[float]
        """
        self.model.eval()

        if isinstance(texts, str):
            texts = [texts]

        if outputs is not None:
            if isinstance(outputs, str):
                outputs = [outputs]
            if len(outputs) != len(texts):
                raise ValueError("outputs must have the same length as texts")

        n = len(texts)

        mean_log_likelihoods: List[float] = []
        perplexities: List[float] = []

        for i in tqdm(range(0, n, batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_outputs = outputs[i:i + batch_size] if outputs is not None else None

            if batch_outputs is None:
                # unconditional LM
                tokenized = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True,
                )

                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)

                with torch.no_grad():
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits

                # shift for LM loss
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_attention_mask = attention_mask[:, 1:]

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=shift_labels.unsqueeze(-1),
                ).squeeze(-1)

                token_log_probs = token_log_probs * shift_attention_mask

            else:
                # conditional likelihood p(outputs | texts)
                combined = [
                    t + o for t, o in zip(batch_texts, batch_outputs)
                ]

                tokenized = self.tokenizer(
                    combined,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True,
                )

                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)

                with torch.no_grad():
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits

                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_attention_mask = attention_mask[:, 1:]

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=shift_labels.unsqueeze(-1),
                ).squeeze(-1)

                # mask out the prompt tokens
                prompt_lens = [
                    len(self.tokenizer(t, add_special_tokens=True)["input_ids"]) - 1
                    for t in batch_texts
                ]
                seq_len = token_log_probs.size(1)
                pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
                start = torch.tensor(prompt_lens, device=self.device).unsqueeze(1)

                mask = (pos >= start) * shift_attention_mask
                token_log_probs = token_log_probs * mask

            # ---- mean per sample ----
            token_counts = (token_log_probs != 0).sum(dim=-1).clamp(min=1)
            mean_ll = token_log_probs.sum(dim=-1) / token_counts
            ppl = torch.exp(-mean_ll)

            mean_log_likelihoods.extend(mean_ll.cpu().tolist())
            perplexities.extend(ppl.cpu().tolist())

            if enable_memory_optimization and torch.cuda.is_available():
                del input_ids, attention_mask, logits
                del shift_logits, shift_labels, shift_attention_mask
                del log_probs, token_log_probs
                torch.cuda.empty_cache()

        return mean_log_likelihoods, perplexities

