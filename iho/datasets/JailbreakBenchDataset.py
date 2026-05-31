from typing import Union, List, TypedDict
from typeguard import typechecked

from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

from iho.configs.config_helper import THE_TRAINING_ONES_SMALL, THE_TRAINING_ONES_BIG, THE_VALIDATION_ONES, THE_TRAINING_ONES_SMALL_STRATIFIED, THE_TRAINING_ONES_BIG_STRATIFIED, THE_VALIDATION_ONES_STRATIFIED, THE_TEST_HELD_OUT_STRATIFIED, NamedBehaviourSubsets

class JailbreakBenchItem(TypedDict):
    jb_index: int
    goal_text: str
    target_text: str

@typechecked
def rows_to_indices(rows: Union[NamedBehaviourSubsets, List[int]]) -> List[int]:
    if rows == "ALL":
        return list(range(100))
    elif rows == "THE_TRAINING_ONES_SMALL":
        return THE_TRAINING_ONES_SMALL
    elif rows == "THE_TRAINING_ONES_BIG":
        return THE_TRAINING_ONES_BIG
    elif rows == "THE_VALIDATION_ONES":
        return THE_VALIDATION_ONES
    elif rows == "THE_TRAINING_ONES_SMALL_STRATIFIED":
        return THE_TRAINING_ONES_SMALL_STRATIFIED
    elif rows == "THE_TRAINING_ONES_BIG_STRATIFIED":
        return THE_TRAINING_ONES_BIG_STRATIFIED
    elif rows == "THE_VALIDATION_ONES_STRATIFIED":
        return THE_VALIDATION_ONES_STRATIFIED
    elif rows == "THE_TEST_HELD_OUT_STRATIFIED":
        return THE_TEST_HELD_OUT_STRATIFIED
    elif isinstance(rows, list):
        return rows
    else:
        raise ValueError(f"Invalid 'rows' parameter: {rows}")

@typechecked
class JailbreakBenchDataset(TorchDataset):
    def __init__(
        self,
        rows: Union[NamedBehaviourSubsets, List[int]] = "ALL",
    ):
        self.rows = rows  
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        assert isinstance(ds, HFDataset)
        self.full_dataset = ds

        self.index_list = rows_to_indices(rows)

        selected_data = ds.select(self.index_list)
        
        raw_goals = selected_data["Goal"]
        raw_targets = selected_data["Target"]
        
        self.goal_text = raw_goals
        self.target_texts = raw_targets

    def __len__(self) -> int:
        return len(self.index_list)

    def __getitem__(self, idx: int) -> JailbreakBenchItem:
        """
        Returns the original index in jb, the prompt (Goal), and the desired completion (Target).
        """
        original_jb_index = self.index_list[idx] 
        
        return {
            "jb_index": original_jb_index,
            "goal_text": self.goal_text[idx],
            "target_text": self.target_texts[idx],
        }