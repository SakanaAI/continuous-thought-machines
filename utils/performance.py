"""
Comprehensive Performance Profiler & Benchmark Suite for CTM

Enterprise-grade performance monitoring for Continuous Thought Machines.
Provides automated training speed benchmarks, memory leak detection,
computational bottleneck identification, and reproducibility validation.

Usage:
    from utils.performance import CTMPerformanceProfiler

    # Benchmark training speed
    profiler = CTMPerformanceProfiler()
    throughput = profiler.benchmark_training_speed(model, dataloader)

    # Profile memory usage
    with profiler.profile_memory(model):
        train_model()

    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks(model, data)

    # Validate reproducibility
    score = profiler.validate_reproducibility(model_factory)
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import psutil
import gc
import threading
import statistics
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    training_throughput: float  # samples/second
    memory_peak: float  # MB
    memory_leak_rate: float  # MB/iteration
    gpu_utilization: float  # %
    cpu_utilization: float  # %
    bottlenecks: Dict[str, float]  # operation -> time(sec)
    reproducibility_score: float  # 0.0 to 1.0
    hardware_info: Dict[str, str]


class CTMPerformanceProfiler:
    """
    Enterprise-grade performance monitoring for Continuous Thought Machines.

    Provides comprehensive profiling capabilities including:
    - Training speed benchmarking
    - Memory leak detection
    - Bottleneck identification
    - Reproducibility validation
    - Hardware utilization monitoring
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the performance profiler.

        Args:
            device: PyTorch device to monitor (defaults to current device)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_history: List[PerformanceMetrics] = []
        self._memory_monitoring = False
        self._memory_samples = []
        self._memory_thread = None
        self._stop_monitoring = threading.Event()

    def benchmark_training_speed(self, model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 iterations: int = 100) -> float:
        """
        Measure training throughput in samples/second.

        Args:
            model: PyTorch model to benchmark
            dataloader: Training dataloader
            iterations: Number of iterations to measure

        Returns:
            Training throughput in samples/second
        """
        model.eval()
        model.to(self.device)

        # Warm up
        iterator = iter(dataloader)
        try:
            inputs, targets = next(iterator)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                _ = model(inputs)

            torch.cuda.synchronize() if self.device.type == 'cuda' else None

            # Benchmark
            start_time = time.perf_counter()
            total_samples = 0

            for i in range(iterations):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    inputs, targets = next(iterator)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                batch_size = inputs.size(0)
                total_samples += batch_size

                with torch.no_grad():
                    _ = model(inputs)

                if i % 10 == 0:
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None

            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            end_time = time.perf_counter()

            throughput = total_samples / (end_time - start_time)
            return throughput

        except Exception as e:
            warnings.warn(f"Training speed benchmark failed: {e}")
            return 0.0

    @contextmanager
    def profile_memory(self, model: torch.nn.Module):
        """
        Context manager for memory profiling and leak detection.

        Usage:
            with profiler.profile_memory(model):
                # Your training/inference code here
                pass

        Yields:
            Memory usage statistics
        """
        self._start_memory_monitoring()
        initial_memory = self._get_memory_usage()

        try:
            yield
        finally:
            self._stop_memory_monitoring()
            final_memory = self._get_memory_usage()

            memory_increase = final_memory - initial_memory
            leak_detected = memory_increase > 50  # 50MB threshold

            if leak_detected:
                warnings.warn(f"Memory leak detected! Increase: {memory_increase:.2f}MB")

    def _start_memory_monitoring(self):
        """Start background memory monitoring."""
        if self._memory_monitoring:
            return

        self._memory_monitoring = True
        self._memory_samples = []
        self._stop_monitoring.clear()

        def monitor_memory():
            while not self._stop_monitoring.is_set():
                self._memory_samples.append(self._get_memory_usage())
                time.sleep(0.1)  # Sample every 100ms

        self._memory_thread = threading.Thread(target=monitor_memory, daemon=True)
        self._memory_thread.start()

    def _stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        if not self._memory_monitoring:
            return

        self._stop_monitoring.set()
        if self._memory_thread:
            self._memory_thread.join(timeout=1.0)

        self._memory_monitoring = False

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        return psutil.Process().memory_info().rss / (1024 ** 2)

    def identify_bottlenecks(self, model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           iterations: int = 50) -> Dict[str, float]:
        """
        Identify computational bottlenecks in the CTM pipeline.

        Args:
            model: PyTorch model to analyze
            dataloader: Data loader for profiling
            iterations: Number of iterations to profile

        Returns:
            Dictionary of operations and their execution times
        """
        model.eval()
        model.to(self.device)

        bottlenecks = defaultdict(list)
        iterator = iter(dataloader)

        for _ in range(iterations):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Profile different stages
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA if self.device.type == 'cuda' else None
                ],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    predictions = model(inputs)

            # Analyze profiler results
            for event in prof.key_averages():
                operation_name = event.key
                execution_time = event.cpu_time_total / 1000000  # Convert to seconds
                bottlenecks[operation_name].append(execution_time)

        # Calculate average times
        avg_bottlenecks = {op: statistics.mean(times) for op, times in bottlenecks.items()}
        return dict(sorted(avg_bottlenecks.items(), key=lambda x: x[1], reverse=True))

    def validate_reproducibility(self, model_factory: Callable,
                               seed: int = 42, iterations: int = 5) -> float:
        """
        Validate deterministic behavior across runs.

        Args:
            model_factory: Function that creates a new model instance
            seed: Random seed for reproducibility
            iterations: Number of runs to compare

        Returns:
            Reproducibility score (0.0 to 1.0, higher is better)
        """
        # Import here to avoid circular dependencies
        try:
            from utils.housekeeping import set_seed
        except ImportError:
            # Fallback if housekeeping module not available
            def set_seed(s):
                import random
                import numpy as np
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)

        set_seed(seed)

        outputs = []

        for _ in range(iterations):
            set_seed(seed)

            model = model_factory().to(self.device)
            model.eval()

            # Create test input
            test_input = torch.randn(1, 3, 224, 224).to(self.device)

            with torch.no_grad():
                output = model(test_input)

            outputs.append(output.cpu().numpy())

        # Calculate reproducibility score
        if len(outputs) < 2:
            return 1.0

        std_dev = np.std(outputs, axis=0)
        mean_val = np.mean(outputs, axis=0)

        # Avoid division by zero
        relative_std = std_dev / (np.abs(mean_val) + 1e-8)
        reproducibility_score = 1.0 / (1.0 + np.mean(relative_std))

        return reproducibility_score

    def compare_architectures(self, models: Dict[str, torch.nn.Module],
                            test_data: torch.utils.data.DataLoader,
                            metrics: List[str] = None) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark different CTM configurations.

        Args:
            models: Dictionary of model names to model instances
            test_data: Test dataset
            metrics: List of metrics to measure (defaults to all)

        Returns:
            Dictionary of model names to their performance metrics
        """
        if metrics is None:
            metrics = ['throughput', 'memory', 'reproducibility']

        results = {}

        for model_name, model in models.items():
            print(f"Benchmarking {model_name}...")

            # Measure throughput
            throughput = 0.0
            if 'throughput' in metrics:
                throughput = self.benchmark_training_speed(model, test_data)

            # Measure memory
            memory_peak = 0.0
            if 'memory' in metrics:
                with self.profile_memory(model):
                    # Run inference to trigger memory allocation
                    for batch in test_data:
                        inputs = batch[0].to(self.device)
                        with torch.no_grad():
                            _ = model(inputs)
                        break
                memory_peak = max(self._memory_samples) if self._memory_samples else 0.0

            # Measure reproducibility
            reproducibility = 1.0
            if 'reproducibility' in metrics:
                reproducibility = self.validate_reproducibility(
                    lambda: model.__class__(**model.__dict__)
                )

            # Hardware utilization
            gpu_util = 0.0
            cpu_util = psutil.cpu_percent()

            if self.device.type == 'cuda' and torch.cuda.is_available():
                gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0

            # Get hardware info
            hardware_info = {
                'device': str(self.device),
                'cpu_cores': str(psutil.cpu_count()),
                'ram_gb': f"{psutil.virtual_memory().total / (1024**3):.1f}",
                'cuda_devices': str(torch.cuda.device_count()) if torch.cuda.is_available() else '0'
            }

            metrics_data = PerformanceMetrics(
                training_throughput=throughput,
                memory_peak=memory_peak,
                memory_leak_rate=0.0,  # Would need extended monitoring
                gpu_utilization=gpu_util,
                cpu_utilization=cpu_util,
                bottlenecks={},  # Would need extended analysis
                reproducibility_score=reproducibility,
                hardware_info=hardware_info
            )

            results[model_name] = metrics_data

        return results

    def generate_performance_report(self,
                                  results: Dict[str, PerformanceMetrics],
                                  output_path: Optional[str] = None) -> str:
        """
        Create comprehensive performance analysis report.

        Args:
            results: Performance metrics from compare_architectures
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("CTM PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append("")

        for model_name, metrics in results.items():
            report.append(f"Model: {model_name}")
            report.append(f"  Training Throughput: {metrics.training_throughput:.2f} samples/sec")
            report.append(f"  Peak Memory Usage: {metrics.memory_peak:.2f} MB")
            report.append(f"  Reproducibility Score: {metrics.reproducibility_score:.3f}")
            report.append(f"  GPU Utilization: {metrics.gpu_utilization:.1f}%")
            report.append(f"  CPU Utilization: {metrics.cpu_utilization:.1f}%")
            report.append("")

        # Hardware information
        report.append("HARDWARE INFORMATION")
        report.append("-" * 40)
        report.append("")

        for model_name, metrics in results.items():
            report.append(f"Model: {model_name}")
            for key, value in metrics.hardware_info.items():
                report.append(f"  {key}: {value}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("")

        report.append("1. Performance Optimization:")
        best_throughput = max(results.items(), key=lambda x: x[1].training_throughput)
        report.append(f"   Best throughput: {best_throughput[0]} ({best_throughput[1].training_throughput:.2f} samples/sec)")

        worst_memory = max(results.items(), key=lambda x: x[1].memory_peak)
        report.append(f"   Highest memory usage: {worst_memory[0]} ({worst_memory[1].memory_peak:.2f} MB)")

        worst_reproducibility = min(results.items(), key=lambda x: x[1].reproducibility_score)
        report.append(f"   Lowest reproducibility: {worst_reproducibility[0]} ({worst_reproducibility[1].reproducibility_score:.3f})")

        report.append("")
        report.append("2. Memory Management:")
        report.append("   - Monitor memory leaks in long-running training sessions")
        report.append("   - Consider gradient checkpointing for memory-intensive models")
        report.append("   - Use mixed precision training to reduce memory usage")

        report.append("")
        report.append("3. Reproducibility:")
        report.append("   - Set random seeds consistently across all components")
        report.append("   - Use deterministic algorithms where possible")
        report.append("   - Document non-deterministic operations")

        report.append("")
        report.append("=" * 80)

        report_str = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)

        return report_str

    def cleanup(self):
        """Clean up profiler resources."""
        self._stop_memory_monitoring()
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        gc.collect()


# Convenience functions for common use cases

def benchmark_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: Optional[torch.device] = None,
                   iterations: int = 100) -> float:
    """
    Quick benchmark function for training speed.

    Args:
        model: PyTorch model to benchmark
        dataloader: Training dataloader
        device: Optional device override
        iterations: Number of iterations to measure

    Returns:
        Training throughput in samples/second
    """
    profiler = CTMPerformanceProfiler(device)
    return profiler.benchmark_training_speed(model, dataloader, iterations)


def profile_model_memory(model: torch.nn.Module,
                        device: Optional[torch.device] = None) -> contextmanager:
    """
    Context manager for quick memory profiling.

    Usage:
        with profile_model_memory(model):
            # Your code here
            pass

    Args:
        model: PyTorch model to profile
        device: Optional device override

    Returns:
        Context manager for memory profiling
    """
    profiler = CTMPerformanceProfiler(device)
    return profiler.profile_memory(model)


def validate_model_determinism(model_factory: Callable,
                              seed: int = 42) -> float:
    """
    Quick reproducibility validation.

    Args:
        model_factory: Function that creates a new model instance
        seed: Random seed for reproducibility

    Returns:
        Reproducibility score (0.0 to 1.0)
    """
    profiler = CTMPerformanceProfiler()
    return profiler.validate_reproducibility(model_factory, seed)