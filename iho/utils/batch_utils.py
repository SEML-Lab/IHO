from typing import List, Dict
from typeguard import typechecked


@typechecked
def rebatch_data(data_list: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    Rebatch data into new batch size for efficient processing.
    """
    return [
        data_list[i : i + batch_size]
        for i in range(0, len(data_list), batch_size)
    ]


@typechecked
def flatten_batch_results(batch_results: List[Dict]) -> List[Dict]:
    """
    Flatten batch results into individual samples.
    """
    if not batch_results:
        return []

    first_batch = batch_results[0]
    batch_sizes = [
        len(v) for v in first_batch.values() if isinstance(v, list)
    ]

    if not batch_sizes:
        return []

    all_samples = []

    for batch in batch_results:
        batch_size = len(
            next(v for v in batch.values() if isinstance(v, list))
        )

        for i in range(batch_size):
            sample = {}
            for key, value in batch.items():
                sample[key] = value[i] if isinstance(value, list) else value
            all_samples.append(sample)

    return all_samples


@typechecked
def batch_samples(samples: List[Dict]) -> Dict:
    """
    Combine individual samples into a batch dictionary.
    """
    if not samples:
        return {}

    batch = {key: [] for key in samples[0].keys()}

    for sample in samples:
        for key, value in sample.items():
            batch[key].append(value)

    return batch
