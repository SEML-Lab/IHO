import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, List, Tuple, TypedDict
from typeguard import typechecked

import pandas as pd
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


class ScoredSample(TypedDict):
    jb_index: int
    original_prompt_text: str
    inpainted_prompt_text_full: str
    judge_score_training: float
    cycle_id: int


class PreferencePair(TypedDict):
    jb_index: int
    prompt: str
    chosen: str
    rejected: str
    chosen_judge_score: float
    rejected_judge_score: float
    chosen_cycle_id: int     
    rejected_cycle_id: int   


class PairingStrategy(ABC):
    @abstractmethod
    def generate(
        self, 
        *,        
        chosen_pool: List[ScoredSample],
        rejected_pool: List[ScoredSample],
        k: int) -> List[PreferencePair]:
        ...


@typechecked
class ThresholdedTopKBottomKPairing(PairingStrategy):
    def __init__(self, min_score: float = 0.2):
        self.min_score = min_score

    def generate(
        self,
        *,
        chosen_pool: List[ScoredSample],
        rejected_pool: List[ScoredSample],
        k: int,
    ) -> List[PreferencePair]:
        if k <= 0 or not rejected_pool:
            return []

        rejected_sorted = sorted(rejected_pool, key=lambda s: s["judge_score_training"])
        bottom_k = rejected_sorted[:k]

        chosen_candidates = [
            s for s in chosen_pool
            if s["judge_score_training"] >= self.min_score
        ]
        if not chosen_candidates:
            return []

        chosen_sorted = sorted(chosen_candidates, key=lambda s: s["judge_score_training"])
        top_k = chosen_sorted[-k:]

        pair_k = min(len(top_k), len(bottom_k))
        if pair_k == 0:
            return []

        pairs: List[PreferencePair] = []

        for chosen, rejected in zip(
            reversed(top_k[-pair_k:]),
            bottom_k[:pair_k],
        ):
            pairs.append({
                "jb_index": chosen["jb_index"],
                "prompt": chosen["original_prompt_text"],
                "chosen": chosen["inpainted_prompt_text_full"],
                "rejected": rejected["inpainted_prompt_text_full"],
                "chosen_judge_score": chosen["judge_score_training"],
                "rejected_judge_score": rejected["judge_score_training"],
                "chosen_cycle_id": chosen["cycle_id"],
                "rejected_cycle_id": rejected["cycle_id"],
            })

        return pairs


# cycle_0.parquet, cycle_1.parquet, ...
_CYCLE_RE = re.compile(r"cycle_(\d+)\.parquet$")


@typechecked
class PreferenceDataset(TorchDataset[PreferencePair]):
    def __init__(
        self,
        *,
        strategy: PairingStrategy,
        run_path: str,
        expanding: bool = True,
        percent_chosen: float = 0.125
    ):
        if not isinstance(strategy, ThresholdedTopKBottomKPairing):
            raise TypeError(
                "Only ThresholdedTopKBottomKPairing is currently supported."
            )

        self.strategy = strategy
        self.run_path = run_path
        self.expanding = expanding

        self.data = self._build_from_run_path(run_path, strategy, percent_chosen, expanding)

        if not self.data:
            logger.warning("PreferenceDataset initialized with zero preference pairs")
    
    @staticmethod
    def _find_cycles(samples_dir: str) -> List[Tuple[int, str]]:
        cycles: List[Tuple[int, str]] = []
        for fname in os.listdir(samples_dir):
            m = _CYCLE_RE.match(fname)
            if m:
                idx = int(m.group(1))
                cycles.append((idx, os.path.join(samples_dir, fname)))

        if not cycles:
            raise RuntimeError(f"No cycle_*.parquet files found in {samples_dir}")

        return sorted(cycles, key=lambda x: x[0])

    @staticmethod
    def _df_to_samples(df: pd.DataFrame, cycle_id: int) -> List[ScoredSample]:
        samples: List[ScoredSample] = []
        for _, row in df.iterrows():
            samples.append({
                "jb_index": int(row["jb_index"]),
                "original_prompt_text": str(row["original_prompt_text"]),
                "inpainted_prompt_text_full": str(row["inpainted_prompt_text_full"]),
                "judge_score_training": float(row["judge_score_training"]),
                "cycle_id": cycle_id,
            })
        return samples

    @staticmethod
    def _build_from_run_path(
        run_path: str,
        strategy: ThresholdedTopKBottomKPairing,
        percent_chosen: float,
        expanding: bool,

    ) -> List[PreferencePair]:
        samples_dir = os.path.join(run_path, "samples")
        cycles = PreferenceDataset._find_cycles(samples_dir)

        last_idx, last_path = cycles[-1]
        prev_cycles = cycles[:-1]

        current_df = pd.read_parquet(last_path)
        current_samples = PreferenceDataset._df_to_samples(current_df, last_idx)

        if expanding:
            # Original behavior: use all previous cycles + current cycle for chosen pool
            previous_samples: List[ScoredSample] = []
            for idx, path in prev_cycles:
                df = pd.read_parquet(path)
                previous_samples.extend(
                    PreferenceDataset._df_to_samples(df, idx)
                )
            chosen_pool = previous_samples + current_samples
        else:
            # New behavior: only use current cycle for both chosen and rejected pools
            chosen_pool = current_samples

        rejected_pool = current_samples

        grouped_chosen: DefaultDict[Tuple[int, str], List[ScoredSample]] = defaultdict(list)
        grouped_rejected: DefaultDict[Tuple[int, str], List[ScoredSample]] = defaultdict(list)

        for s in chosen_pool:
            key = (s["jb_index"], s["original_prompt_text"])
            grouped_chosen[key].append(s)

        for s in rejected_pool:
            key = (s["jb_index"], s["original_prompt_text"])
            grouped_rejected[key].append(s)



        preference_pairs: List[PreferencePair] = []

        for key in grouped_rejected.keys():
            chosen_samples = grouped_chosen.get(key, [])
            rejected_samples = grouped_rejected[key]
            k = int(len(chosen_samples) * percent_chosen) 

            if not chosen_samples or not rejected_samples:
                continue

            preference_pairs.extend(
                strategy.generate(
                    chosen_pool=chosen_samples,
                    rejected_pool=rejected_samples,
                    k=k
                )
            )

        return preference_pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.data[idx]

    # ---------- Parquet writer ----------
    def to_parquet(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = []
        for pair in self.data:
            rows.append({
                "jb_index": pair["jb_index"],
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "chosen_judge_score": pair["chosen_judge_score"],
                "rejected_judge_score": pair["rejected_judge_score"],
                "chosen_cycle_id": pair["chosen_cycle_id"],
                "rejected_cycle_id": pair["rejected_cycle_id"],
            })
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
