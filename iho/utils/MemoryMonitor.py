import os
import psutil
import logging
from typing import Dict, Optional, Any, List
from contextlib import contextmanager
import json
from dataclasses import dataclass, asdict
import threading
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    timestamp: float
    cpu_rss_gb: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def __sub__(self, other: 'MemorySnapshot') -> 'MemorySnapshot':
        return MemorySnapshot(
            timestamp=self.timestamp,
            cpu_rss_gb=self.cpu_rss_gb - other.cpu_rss_gb,
            gpu_allocated_gb=self.gpu_allocated_gb - other.gpu_allocated_gb,
            gpu_reserved_gb=self.gpu_reserved_gb - other.gpu_reserved_gb,
        )

@dataclass
class MemoryProfile:
    operation_name: str
    start: MemorySnapshot
    end: MemorySnapshot
    peak: MemorySnapshot
    delta: MemorySnapshot
    total_time_seconds: float
    num_training_samples: int
    metadata: Optional[Dict[str, Any]] = None
    verbose_snapshots: Optional[List[Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "operation_name": self.operation_name,
            "memory": {
                "start": self.start.to_dict(),
                "end": self.end.to_dict(),
                "peak": self.peak.to_dict(),
                "delta": self.delta.to_dict(),
            },
            "timing": {
                "total_time_seconds": self.total_time_seconds,
            }
        }
        
        if self.num_training_samples > 0:
            result["timing"]["num_training_samples"] = self.num_training_samples
            result["timing"]["samples_per_second"] = self.num_training_samples / self.total_time_seconds if self.total_time_seconds > 0 else 0
            result["timing"]["seconds_per_sample"] = self.total_time_seconds / self.num_training_samples if self.num_training_samples > 0 else 0
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        if self.verbose_snapshots:
            result["verbose_snapshots"] = self.verbose_snapshots
            
        return result
    
    def log_summary(self):
        logger.info("=" * 60)
        logger.info("Profile: %s", self.operation_name)
        logger.info("=" * 60)
        logger.info("Memory:")
        logger.info("  Start - CPU RSS: %.2f GB | GPU Alloc: %.2f GB | GPU Reserv: %.2f GB",
                   self.start.cpu_rss_gb, self.start.gpu_allocated_gb, self.start.gpu_reserved_gb)
        logger.info("  End   - CPU RSS: %.2f GB | GPU Alloc: %.2f GB | GPU Reserv: %.2f GB",
                   self.end.cpu_rss_gb, self.end.gpu_allocated_gb, self.end.gpu_reserved_gb)
        logger.info("  Peak  - CPU RSS: %.2f GB | GPU Alloc: %.2f GB | GPU Reserv: %.2f GB",
                   self.peak.cpu_rss_gb, self.peak.gpu_allocated_gb, self.peak.gpu_reserved_gb)
        logger.info("  Delta - CPU RSS: %.2f GB | GPU Alloc: %.2f GB | GPU Reserv: %.2f GB",
                   self.delta.cpu_rss_gb, self.delta.gpu_allocated_gb, self.delta.gpu_reserved_gb)
        logger.info("Timing:")
        logger.info("  Total time: %.2f seconds", self.total_time_seconds)
        if self.num_training_samples > 0:
            logger.info("  Samples: %d", self.num_training_samples)
            logger.info("  Samples/sec: %.2f", self.num_training_samples / self.total_time_seconds if self.total_time_seconds > 0 else 0)
            logger.info("  Sec/sample: %.4f", self.total_time_seconds / self.num_training_samples if self.num_training_samples > 0 else 0)
        if self.metadata:
            logger.info("Metadata:")
            for key, value in self.metadata.items():
                logger.info("  %s: %s", key, value)
        if self.verbose_snapshots:
            logger.info("Verbose snapshots: %d data points collected", len(self.verbose_snapshots))
        logger.info("=" * 60)
    
    def save_to_file(self, filepath: str, execution_num: Optional[int] = None):
        """Save profile to file with optional execution number."""
        filepath_obj = Path(filepath)
        
        # If execution_num provided, insert it into filename
        if execution_num is not None:
            # monitoring.json → monitoring_execution_00.json
            stem = filepath_obj.stem  # "monitoring"
            suffix = filepath_obj.suffix  # ".json"
            new_name = f"{stem}_execution_{execution_num:02d}{suffix}"
            filepath_obj = filepath_obj.parent / new_name
        
        os.makedirs(filepath_obj.parent, exist_ok=True)
        with open(filepath_obj, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Profile saved to %s", filepath_obj)



class MemoryMonitor:
    def __init__(self, poll_interval: float = 0.1, verbose: bool = False):
        self.process = psutil.Process(os.getpid())
        self.cuda_available = torch.cuda.is_available()
        self.poll_interval = poll_interval
        self.verbose = verbose

        self.operation_name: Optional[str] = None
        self.start_snapshot: Optional[MemorySnapshot] = None
        self.peak_snapshot: Optional[MemorySnapshot] = None
        self.start_time: Optional[float] = None
        self.num_training_samples: int = 0
        self.metadata: Optional[Dict[str, Any]] = None
        self.verbose_snapshots: Optional[List[MemorySnapshot]] = None

        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _get_gpu_metadata(self) -> Optional[Dict[str, Any]]:
        if not self.cuda_available:
            return None

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)

        return {
            "device": props.name,
            "index": device_index,
        }


    def get_snapshot(self, relative_time: float = 0.0) -> MemorySnapshot:
        cpu_rss_gb = self.process.memory_info().rss / (1024 ** 3)

        if self.cuda_available:
            gpu_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        else:
            gpu_allocated_gb = 0.0
            gpu_reserved_gb = 0.0

        return MemorySnapshot(
            timestamp=relative_time,
            cpu_rss_gb=cpu_rss_gb,
            gpu_allocated_gb=gpu_allocated_gb,
            gpu_reserved_gb=gpu_reserved_gb,
        )

    def update_peak(self, current: MemorySnapshot):
        if self.peak_snapshot is None:
            self.peak_snapshot = current
            return

        self.peak_snapshot = MemorySnapshot(
            timestamp=current.timestamp,
            cpu_rss_gb=max(self.peak_snapshot.cpu_rss_gb, current.cpu_rss_gb),
            gpu_allocated_gb=max(self.peak_snapshot.gpu_allocated_gb, current.gpu_allocated_gb),
            gpu_reserved_gb=max(self.peak_snapshot.gpu_reserved_gb, current.gpu_reserved_gb),
        )

    def set_num_training_samples(self, num_training_samples: int):
        with self._lock:
            self.num_training_samples = num_training_samples

    def set_metadata(self, metadata: Dict[str, Any]):
        with self._lock:
            if self.metadata is None:
                self.metadata = {}
            self.metadata.update(metadata)

    def _poll_loop(self):
        while self._monitoring and self.start_time is not None:
            relative_time = time.time() - self.start_time
            snapshot = self.get_snapshot(relative_time)
            
            with self._lock:
                self.update_peak(snapshot)
                
                if self.verbose and self.verbose_snapshots is not None:
                    self.verbose_snapshots.append(snapshot)
                    
            time.sleep(self.poll_interval)

    @contextmanager
    def monitor(self, operation_name: str, log_summary: bool = True, save_path: Optional[str] = None, execution_num: Optional[int] = None):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.start_snapshot = self.get_snapshot(0.0)
        self.peak_snapshot = self.start_snapshot
        self.num_training_samples = 0
        self.metadata = self._get_gpu_metadata() if self._get_gpu_metadata() is not None else {}

        
        if self.verbose:
            self.verbose_snapshots = [self.start_snapshot]
        else:
            self.verbose_snapshots = None

        logger.info("Started monitoring: %s", operation_name)

        self._monitoring = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        try:
            yield self
        finally:
            self._monitoring = False
            if self._thread is not None:
                self._thread.join()

            relative_time = time.time() - self.start_time
            end_snapshot = self.get_snapshot(relative_time)
            total_time = relative_time

            with self._lock:
                self.update_peak(end_snapshot)
                num_training_samples = self.num_training_samples
                metadata = self.metadata
                verbose_snapshots = self.verbose_snapshots

            if verbose_snapshots is not None:
                verbose_snapshots.append(end_snapshot)

            delta = MemorySnapshot(
                timestamp=0.0,
                cpu_rss_gb=end_snapshot.cpu_rss_gb - self.start_snapshot.cpu_rss_gb,
                gpu_allocated_gb=end_snapshot.gpu_allocated_gb - self.start_snapshot.gpu_allocated_gb,
                gpu_reserved_gb=end_snapshot.gpu_reserved_gb - self.start_snapshot.gpu_reserved_gb,
            )

            profile = MemoryProfile(
                operation_name=self.operation_name,
                start=self.start_snapshot,
                end=end_snapshot,
                peak=self.peak_snapshot,
                delta=delta,
                total_time_seconds=total_time,
                num_training_samples=num_training_samples,
                metadata=metadata,
                verbose_snapshots=[s.to_dict() for s in verbose_snapshots] if verbose_snapshots else None,
            )

            if log_summary:
                profile.log_summary()

            if save_path:
                profile.save_to_file(save_path, execution_num=execution_num)

            self.operation_name = None
            self.start_snapshot = None
            self.peak_snapshot = None
            self.start_time = None
            self.num_training_samples = 0
            self.metadata = None
            self.verbose_snapshots = None
            self._thread = None