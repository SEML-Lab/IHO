import os
from typing import TypedDict, List, Union, Literal, Optional, Dict, Any, overload
from typing_extensions import NotRequired
from peft import LoraConfig
import copy

THE_TRAINING_ONES_SMALL = [1, 3, 7, 17, 25, 29, 47, 58, 77, 81]
THE_TRAINING_ONES_BIG = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 47, 48, 49, 5, 51, 52, 53, 54, 55, 56]
THE_VALIDATION_ONES = [0, 46, 50, 57, 95]
ALL = list(range(100))

THE_TRAINING_ONES_SMALL_STRATIFIED = [0, 19, 20, 39, 40, 55, 62, 78, 84, 91] 
THE_TRAINING_ONES_BIG_STRATIFIED = [0, 2, 5, 6, 8, 9, 10, 14, 15, 16, 17, 19, 20, 21, 22, 25, 27, 28, 31, 34, 35, 36, 37, 39, 40, 42, 43, 45, 46, 49, 50, 54, 55, 56, 57, 59, 61, 62, 64, 65, 66, 68, 71, 74, 76, 77, 78, 79, 81, 82, 83, 84, 88, 89, 91, 92, 93, 97, 98, 99] 
THE_VALIDATION_ONES_STRATIFIED = [3, 7, 12, 13, 23, 24, 30, 32, 44, 47, 51, 53, 60, 69, 72, 75, 86, 87, 90, 95] 
#Please note that the original further splitting non training into Test and Validation was dropped. In the paper we reported the performance of the joint index set when refering to 40 heldout validation behaviours.
THE_TEST_HELD_OUT_STRATIFIED = [1, 4, 11, 18, 26, 29, 33, 38, 41, 48, 52, 58, 63, 67, 70, 73, 80, 85, 94, 96]

ALL_MODELS = ["google/gemma-3-1b-it", "meta-llama/Meta-Llama-3-8B-Instruct", "GraySwanAI/Llama-3-8B-Instruct-RR", "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector", "LLM-LAT/robust-llama3-8b-instruct", "CAT/local", "ContinuousAT/Llama-2-7B-CAT", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-with-detector", "Qwen/Qwen2.5-32B-Instruct"]

JudgeModel = Literal["strong_reject", "harmbench", "llama_guard_3", "llama_guard_4", "jail_judge", "aegis_guard"]
NamedBehaviourSubsets = Literal["ALL", "THE_TRAINING_ONES_SMALL", "THE_TRAINING_ONES_BIG", "THE_VALIDATION_ONES", "THE_UNBROKEN_ONES", "THE_TRAINING_ONES_SMALL_STRATIFIED", "THE_TRAINING_ONES_BIG_STRATIFIED", "THE_VALIDATION_ONES_STRATIFIED", "THE_TEST_HELD_OUT_STRATIFIED"]
SamplingMode = Literal["training", "validation", "testing", "dpo"]
DetectorOverwriteMode = Literal["harmful_content", "harmful_response_only"]


class BehaviourSubsetsConfig(TypedDict):
    training: Union[NamedBehaviourSubsets, List[int]]
    validation: NotRequired[Union[NamedBehaviourSubsets, List[int]]]
    testing: NotRequired[Union[NamedBehaviourSubsets, List[int]]]
    dpo: NotRequired[Union[NamedBehaviourSubsets, List[int]]]

class BehaviourSubsetsConfigOverride(TypedDict, total=False):
    training: Union[NamedBehaviourSubsets, List[int]]
    validation: NotRequired[Union[NamedBehaviourSubsets, List[int]]]
    testing: NotRequired[Union[NamedBehaviourSubsets, List[int]]]
    dpo: NotRequired[Union[NamedBehaviourSubsets, List[int]]]

class AttackDatasetConfig(TypedDict):
    dataset_name: Literal["jbb"]
    behaviour_subsets: BehaviourSubsetsConfig

class AttackDatasetConfigOverride(TypedDict, total=False):
    dataset_name: Literal["jbb"]
    behaviour_subsets: BehaviourSubsetsConfigOverride

class AttackModelConfig(TypedDict):
    model_id: Literal["GSAI-ML/LLaDA-8B-Base"]
    attacker_input_size: int
    attacker_step_number: int
    attacker_temperature: float
    mask_padding_tokens: bool
    remasking: Literal["low_confidence", "random"]
    lora_config: Optional[LoraConfig]
    lora_checkpoint: Optional[str]

class AttackModelConfigOverride(TypedDict, total=False):
    model_id: Literal["GSAI-ML/LLaDA-8B-Base"]
    attacker_input_size: int
    attacker_step_number: int
    attacker_temperature: float
    mask_padding_tokens: bool
    remasking: Literal["low_confidence", "random"]
    lora_config: Optional[Union[LoraConfig, Dict[str, Any]]]
    lora_checkpoint: Optional[str]

class AttackedModelConfig(TypedDict):
    model_ids: List[str]
    max_new_tokens: int
    use_chat_template: bool
    greedy_sampling: bool
    compute_logprobs: bool

class AttackedModelConfigOverride(TypedDict, total=False):
    model_ids: List[str]
    max_new_tokens: int
    use_chat_template: bool
    greedy_sampling: bool
    compute_logprobs: bool

class DetectorModelConfig(TypedDict):
    model_id: str
    max_new_tokens: int

class DetectorModelConfigOverride(TypedDict, total=False):
    model_id: str
    max_new_tokens: int

class JudgeModelsConfig(TypedDict):
    training: JudgeModel
    validation: NotRequired[JudgeModel]
    validation_2: NotRequired[JudgeModel]

class JudgeModelsConfigOverride(TypedDict, total=False):
    training: JudgeModel
    validation: JudgeModel
    validation_2: JudgeModel

class DPOTrainingConfig(TypedDict):
    learning_rate: float
    beta: float
    dpo_epochs: int
    preference_masking_mode: Literal["all", "prompt", "attack"]    
    dpo_mask_all: bool
    checkpoint_every: int
    save_checkpoints: bool
    percent_chosen: float
    harmfulness_threshold: float
    expanding_dpo_dataset: bool
    patience: int
    warmup_epochs: int
    load_optimizer_state: bool
    dpo_metric: Literal["mean", "max", "weighted"]

class DPOTrainingConfigOverride(TypedDict, total=False):
    learning_rate: float
    beta: float
    dpo_epochs: int
    preference_masking_mode: Literal["all", "prompt", "attack"]    
    dpo_mask_all: bool
    checkpoint_every: int
    save_checkpoints: bool
    harmfulness_threshold: float
    percent_chosen: float
    expanding_dpo_dataset: bool
    patience: int
    warmup_epochs: int
    load_optimizer_state: bool
    dpo_metric: Literal["mean", "max", "weighted"]


class GeneralConfig(TypedDict):
    seed: int
    device: str
    custom_cache_path: Optional[str]
    debug: bool
    run_name: Optional[str]
    num_sampled_attacks: Dict[SamplingMode, int]
    num_cycles: int
    gpu_type: str
    embed_attack_prompts: bool
    embedding_model: str
    use_detector: bool

class GeneralConfigOverride(TypedDict, total=False):
    seed: int
    device: str
    custom_cache_path: Optional[str]
    debug: bool
    run_name: Optional[str]
    num_sampled_attacks: Dict[SamplingMode, int]
    num_cycles: int
    gpu_type: str
    embed_attack_prompts: bool
    embedding_model: str
    use_detector: bool


GPUType = Literal["a100", "a100_small", "h100", "h200", "custom"]

class BatchSizeConfig(TypedDict):
    attack: int
    generate: int
    detector: int
    judge: int
    dpo: int

class PipelineConfig(TypedDict):
    attack_dataset: AttackDatasetConfig
    attack_model: AttackModelConfig
    attacked_model: AttackedModelConfig
    detector_model: NotRequired[DetectorModelConfig]
    judge_models_config: JudgeModelsConfig
    dpo_training: DPOTrainingConfig
    general: GeneralConfig
    batch_sizes: BatchSizeConfig

class PipelineConfigOverride(TypedDict, total=False):
    attack_dataset: AttackDatasetConfigOverride
    attack_model: AttackModelConfigOverride
    attacked_model: AttackedModelConfigOverride
    detector_model: DetectorModelConfigOverride
    judge_models_config: JudgeModelsConfigOverride
    dpo_training: DPOTrainingConfigOverride
    general: GeneralConfigOverride
    batch_sizes: BatchSizeConfig

DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
    "attack_dataset": {
        "dataset_name": "jbb",
        "behaviour_subsets": {
            "training": "ALL"
        }
    },
    "attack_model": {
        "model_id": "GSAI-ML/LLaDA-8B-Base",
        "attacker_input_size": 32,
        "mask_padding_tokens": True,
        "remasking": "low_confidence",
        "lora_config": LoraConfig(
            r = 8,
            lora_alpha = 16,
            target_modules = ["q_proj", "v_proj"],
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM"
        ),
        "lora_checkpoint": None,
        "attacker_step_number": 32,
        "attacker_temperature": 0.0,
    },
    "attacked_model": {
        "model_ids": ["GraySwanAI/Llama-3-8B-Instruct-RR"],
        "max_new_tokens": 256,
        "use_chat_template": True,
        "greedy_sampling": False,
        "compute_logprobs": False
    },
    "detector_model": {
        "model_id": "ToxicityPrompts/PolyGuard-Qwen-Smol",
        "max_new_tokens": 64,
    },
    "judge_models_config": {
        "training": "strong_reject",
        "validation": "harmbench"
    },
    "dpo_training": {
        "learning_rate": 1e-5,
        "beta": 0.25,
        "dpo_epochs": 200,
        "preference_masking_mode": "prompt",
        "dpo_mask_all": False,
        "checkpoint_every": 0,
        "save_checkpoints": True,
        "percent_chosen": 0.125,
        "harmfulness_threshold": 0.2,
        "expanding_dpo_dataset": False,
        "patience": 1,
        "warmup_epochs": 0,
        "load_optimizer_state": False,
        "dpo_metric": "mean"
    },
    "general": {
        "seed": 42,
        "device": "cuda",
        "custom_cache_path": os.environ.get("IHO_CACHE_DIR"),
        "debug": False,
        "run_name": None,
        "num_sampled_attacks": {
            "training": 512,
            "validation": 512,
            "testing": 512,
            "dpo": 128
        },
        "num_cycles": 0,
        "gpu_type": "a100", 
        "embed_attack_prompts": True,
        "embedding_model": "all-MiniLM-L6-v2",
        "use_detector": False
    },
    "batch_sizes": { 
        "attack": 0,
        "generate": 0,
        "detector": 0,
        "judge": 0,
        "dpo": 0,
    },
}

DEFAULT_BATCH_SIZES: dict[GPUType, BatchSizeConfig] = {
    "a100": {"attack": 128, "generate": 64, "detector": 64, "judge": 96, "dpo": 16},
    "a100_small": {"attack": 64, "generate": 64, "detector": 64, "judge": 48, "dpo": 16},
    "h100": {"attack": 256, "generate": 192, "detector": 64, "judge": 160, "dpo": 32},
    "h200": {"attack": 512, "generate": 384, "detector": 256, "judge": 352, "dpo": 96},
    "custom": {"attack": 512, "generate": 196, "detector": 256, "judge": 352, "dpo": 96}
}

@overload
def deep_merge(
    base: PipelineConfigOverride,
    override: PipelineConfigOverride,
) -> PipelineConfigOverride: ...


@overload
def deep_merge(
    base: PipelineConfig,
    override: Optional[PipelineConfigOverride],
) -> PipelineConfig: ...

@overload
def deep_merge(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]: ...

def deep_merge(base, override):
    if override is None:
        return copy.deepcopy(base)

    result = copy.deepcopy(base)

    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v

    return result

def load_defaults(gpu: GPUType) -> PipelineConfig:
    cfg = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
    cfg["batch_sizes"] = DEFAULT_BATCH_SIZES[gpu]
    cfg["general"]["gpu_type"] = gpu
    return cfg