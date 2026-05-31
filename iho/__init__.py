from iho.pipeline import IHOPipeline, generate, load_model
from iho.configs.config_helper import DEFAULT_PIPELINE_CONFIG, PipelineConfig, deep_merge, load_defaults
from iho.datasets import JailbreakBenchDataset, PreferenceDataset, ThresholdedTopKBottomKPairing

__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "IHOPipeline",
    "JailbreakBenchDataset",
    "PipelineConfig",
    "PreferenceDataset",
    "ThresholdedTopKBottomKPairing",
    "deep_merge",
    "generate",
    "load_defaults",
    "load_model",
    "train",
]


def __getattr__(name: str):
    if name == "train":
        from iho.train import train

        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
