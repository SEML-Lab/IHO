import os
import logging
import random
import string
from typeguard import typechecked
from typing import List, Union, Any, Optional
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from iho.datasets.JailbreakBenchDataset import JailbreakBenchDataset
from iho.configs.constants import SEED
from iho.configs.config_helper import PipelineConfig, NamedBehaviourSubsets

logger = logging.getLogger(__name__)


@typechecked
def generate_wandb_style_name() -> str:
    adjectives = [
        "brave", "calm", "eager", "fancy", "gentle", "happy",
        "icy", "jolly", "kind", "lucky", "mighty", "nimble",
        "proud", "quick", "royal", "sharp", "tidy", "witty",
    ]
    nouns = [
        "otter", "falcon", "tiger", "phoenix", "panther",
        "koala", "badger", "lynx", "eagle", "whale",
    ]

    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=4)
    )
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{suffix}"


@typechecked
def init_seeds(seed: int = SEED) -> None:
    logger.info("Initialising seeds with seed=%d", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@typechecked
def verify_cuda(device: str) -> None:
    if device != "cuda":
        logger.warning("Using CPU device: '%s'", device)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    try:
        x = torch.tensor([1.0], device="cuda")
        y = torch.tensor([2.0], device="cuda")
        _ = (x + y).item()
    except Exception as exc:
        raise RuntimeError("CUDA compute test failed.") from exc



@typechecked
def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif hasattr(obj, "to_dict"):
        return to_json_safe(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return to_json_safe(vars(obj))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)

@typechecked
def save_configs_json(
    configs: PipelineConfig,
    run_path: str,
    subdir: str = "configs",
    logger: Optional[logging.Logger] = None,
) -> None:
    configs_dir = os.path.join(run_path, subdir)
    os.makedirs(configs_dir, exist_ok=True)

    for config_name, config in configs.items():
        config_path = os.path.join(configs_dir, f"{config_name}.json")

        with open(config_path, "w") as f:
            json.dump(to_json_safe(config), f, indent=2)

        #if logger is not None:
            #logger.info("Saved %s to %s", config_name, config_path)


@typechecked
def build_attack_dataloader(
    rows: Union[NamedBehaviourSubsets, List[int]],
    batch_size: int,
) -> DataLoader:
    dataset = JailbreakBenchDataset(rows=rows)
    
    use_replacement = (
        rows != "all" and len(dataset) < batch_size
    )
    
    num_training_samples = (
        batch_size if use_replacement else len(dataset)
    )
    
    sampler = RandomSampler(
        dataset,
        replacement=use_replacement,
        num_samples=num_training_samples,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )

@typechecked
def attach_file_logger(logs_dir: Path, level: int = logging.INFO) -> int:
    """
    Attach a new indexed log file handler to the root logger.
    Creates logs/execution_XX.log where XX is auto-incremented based on existing files.
    
    Returns:
        execution_num: The execution number for this log file
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing execution_XX.log files
    existing_logs = sorted(logs_dir.glob("execution_*.log"))
    
    if existing_logs:
        # Extract numbers and get next index
        pattern = re.compile(r"execution_(\d+)\.log")
        indices = []
        for log in existing_logs:
            match = pattern.match(log.name)
            if match:
                indices.append(int(match.group(1)))
        next_index = max(indices) + 1 if indices else 0
    else:
        next_index = 0
    
    # Create new log file path
    new_log_file = logs_dir / f"execution_{next_index:02d}.log"
    
    # Remove existing file handlers from root logger
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
    
    # Create new file handler with mode 'w' (fresh file for this execution)
    file_handler = logging.FileHandler(new_log_file, mode="w")
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    
    return next_index
