"""Implement various matrix multiplication methods and performance analysis tools."""

import json
import logging
import os
import sys
import threading
import time as time_module
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool, shared_memory
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
# Third-party imports
import numpy as np
import psutil
import pytest
from matplotlib.cm import get_cmap


class SharedMemoryRegistry:
    """Registry for tracking shared memory blocks.

    This class maintains a list of shared memory blocks that need to be cleaned up
    when they are no longer needed.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self.blocks = {}
        self._lock = threading.Lock()

    def register(self, block, name=None):
        """Register a shared memory block for cleanup.

        Parameters
        ----------
        block : shared_memory.SharedMemory
            The shared memory block to register
        name : str, optional
            Name of the shared memory block
        """
        with self._lock:
            if name is None:
                name = str(uuid.uuid4())
            self.blocks[name] = block

    def unregister(self, name):
        """Remove a shared memory block from the registry.

        Parameters
        ----------
        name : str
            Name of the shared memory block to remove
        """
        with self._lock:
            if name in self.blocks:
                self.blocks.pop(name)

    def cleanup(self):
        """Clean up all registered shared memory blocks."""
        with self._lock:
            for name, block in list(self.blocks.items()):
                try:
                    block.close()
                    block.unlink()
                except Exception as e:
                    logging.warning(f"Error cleaning up shared memory block {name}: {e}")
                self.blocks.pop(name)

    def __del__(self):
        """Clean up when the registry is deleted."""
        self.cleanup()


def setup_logging(log_level_str: str = "INFO") -> None:
    """Configure logging with the specified log level.

    Parameters
    ----------
    log_level_str : str
        String representation of the log level (e.g., 'INFO', 'DEBUG')
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = log_levels.get(log_level_str.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler("matrix_multiplication.log")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info(f"Logging initialized at level: {log_level_str}")


def generate_unique_shm_name() -> str:
    """Create a unique name for shared memory block."""
    timestamp = int(time_module.time() * 1000)
    return f"mm_{os.getpid()}_{timestamp}_{uuid.uuid4().hex[:8]}"


def create_shared_array(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    registry: Optional[SharedMemoryRegistry] = None,
) -> Tuple[shared_memory.SharedMemory, np.ndarray, str]:
    """Create a shared memory array.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the array
    dtype : np.dtype
        Data type of the array
    registry : Optional[SharedMemoryRegistry]
        Registry for tracking shared memory blocks

    Returns
    -------
    Tuple[shared_memory.SharedMemory, np.ndarray, str]
        Shared memory block, numpy array view, and block name
    """
    size = int(np.prod(shape))
    itemsize = np.dtype(dtype).itemsize
    nbytes = size * itemsize

    name = f"mm_{os.getpid()}_{int(time_module.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    shared_array[:] = 0

    if registry:
        registry.register(shm, name)

    return shm, shared_array, name


def process_chunk_mp(chunk_data):
    """Process a chunk of matrix multiplication using NumPy operations.

    Parameters
    ----------
    chunk_data : tuple
        Tuple containing (chunk of matrix a, matrix b, start index)

    Returns
    -------
    tuple
        (start index, result chunk)
    """
    a_chunk, b, start = chunk_data
    return start, np.dot(a_chunk, b)


def process_shared_chunk(args):
    """Process a chunk using shared memory multiprocessing.

    Parameters
    ----------
    args : tuple
        Tuple containing (chunk range, shared memory names, shapes, dtype)
    """
    chunk_range, shm_names, shapes, dtype = args
    start, end = chunk_range

    shm_blocks = []
    try:
        shm_a, shared_a = attach_shared_block(shm_names[0], shapes[0], dtype)
        shm_blocks.append(shm_a)

        shm_b, shared_b = attach_shared_block(shm_names[1], shapes[1], dtype)
        shm_blocks.append(shm_b)

        shm_result, shared_result = attach_shared_block(shm_names[2], shapes[2], dtype)
        shm_blocks.append(shm_result)

        shared_result[start:end] = np.dot(shared_a[start:end], shared_b)

    finally:
        for shm in shm_blocks:
            if shm:
                shm.close()


def process_parallel_blocked_chunk(args):
    """Process a block using parallel shared memory multiplication.

    Parameters
    ----------
    args : tuple
        Tuple containing (row range, block size, shared memory names, shapes, dtype)
    """
    i0_range, block_size, shm_names, shapes, dtype = args
    i0_start, i0_end = i0_range

    shm_blocks = []
    try:
        shm_a, shared_a = attach_shared_block(shm_names[0], shapes[0], dtype)
        shm_blocks.append(shm_a)

        shm_b, shared_b = attach_shared_block(shm_names[1], shapes[1], dtype)
        shm_blocks.append(shm_b)

        shm_result, shared_result = attach_shared_block(shm_names[2], shapes[2], dtype)
        shm_blocks.append(shm_result)

        for i0 in range(i0_start, i0_end, block_size):
            i_max = min(i0 + block_size, shapes[0][0])
            for j0 in range(0, shapes[1][1], block_size):
                j_max = min(j0 + block_size, shapes[1][1])
                for k0 in range(0, shapes[0][1], block_size):
                    k_max = min(k0 + block_size, shapes[0][1])

                    block_result = np.dot(
                        shared_a[i0:i_max, k0:k_max], shared_b[k0:k_max, j0:j_max]
                    )
                    shared_result[i0:i_max, j0:j_max] += block_result

    finally:
        for shm in shm_blocks:
            if shm:
                shm.close()


# --- Configuration ---
@dataclass
class Config:
    """Configuration settings for matrix multiplication.

    Attributes
    ----------
    test_sizes : List[int]
        List of matrix sizes to test
    block_size : int
        Size of blocks for blocked multiplication
    verify_size : int
        Size of matrices for verification tests
    worker_counts : List[int]
        List of worker counts to test
    log_level : str
        Logging level (e.g., 'INFO', 'DEBUG')
    output_dir : str
        Directory for output files
    """

    test_sizes: List[int] = field(default_factory=lambda: [100, 250, 500, 1000])
    block_size: int = 32
    verify_size: int = 64
    worker_counts: List[int] = field(default_factory=lambda: [1, 2, 4])
    log_level: str = "INFO"
    output_dir: str = "output"

    @classmethod
    def from_file(cls, filepath: str) -> "Config":
        """Load configuration from JSON file."""
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r") as f:
            return cls(**json.load(f))

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=4)


# --- Helper Functions for Shared Memory ---
def attach_shared_block(
    name: str, shape: Tuple[int, ...], dtype: np.dtype
) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    """Attach to an existing shared memory block.

    Parameters
    ----------
    name : str
        Name of the shared memory block
    shape : tuple of int
        Shape of the array
    dtype : numpy.dtype
        Data type of the array

    Returns
    -------
    tuple
        (SharedMemory object, numpy array view)
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return shm, array
    except Exception as e:
        logging.error(f"Failed to attach to shared memory block {name}: {e}")
        raise


def cleanup_shared_block(
    shm: Optional[shared_memory.SharedMemory],
    registry: Optional[SharedMemoryRegistry] = None,
    name: Optional[str] = None,
) -> None:
    """Clean up a shared memory block.

    Parameters
    ----------
    shm : Optional[shared_memory.SharedMemory]
        SharedMemory object to clean up
    registry : Optional[SharedMemoryRegistry]
        Registry instance for tracking
    name : Optional[str]
        Name of the shared memory block
    """
    if shm:
        try:
            shm.close()
            shm.unlink()
        except Exception as e:
            logging.warning(f"Error during shared memory cleanup: {e}")
        finally:
            if registry and name:
                registry.unregister(name)


# --- Matrix Multiplication Class ---
class MatrixMultiplication:
    """Implement matrix multiplication using various methods.

    This class provides various methods for matrix multiplication,
    including sequential, parallel, and GPU-accelerated implementations.

    Attributes
    ----------
    config : Config
        Configuration settings
    cpu_cores : int
        Number of CPU cores available
    results : dict
        Dictionary to store performance results
    shared_memory_registry : SharedMemoryRegistry
        Registry for tracking shared memory blocks
    """

    def __init__(self, config: Config):
        """Initialize the matrix multiplication instance.

        Parameters
        ----------
        config : Config
            Configuration settings
        """
        self.config = config
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.results = {}
        self.output_dir = Path(config.output_dir)
        self.shared_memory_registry = SharedMemoryRegistry()

        self._setup_output_dir()

        self.methods = {
            "numpy_multiply": self.numpy_multiply,
            "threading_multiply": self.threading_multiply,
            "multiprocessing_multiply": self.multiprocessing_multiply,
            "sequential_blocked_multiply": self.sequential_blocked_multiply,
            "multiprocessing_shared_multiply": self.multiprocessing_shared_multiply,
            "process_pool_executor_multiply": self.process_pool_executor_multiply,
            "thread_pool_executor_multiply": self.thread_pool_executor_multiply,
            "parallel_blocked_shared_multiply": self.parallel_blocked_shared_multiply,
            "sequential_multiply": self.sequential_multiply
        }

    def __del__(self):
        """Clean up resources when the instance is destroyed."""
        if hasattr(self, "shared_memory_registry"):
            self.shared_memory_registry.cleanup()

    def _setup_output_dir(self) -> None:
        """Create and configure the output directory."""
        try:
            self.output_dir.mkdir(exist_ok=True)
            logging.info(f"Output directory set to: {self.output_dir}")
        except Exception as e:
            logging.error(f"Failed to create output directory: {e}")
            raise

    def generate_random_matrices(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two random matrices of specified size."""
        logging.debug(f"Generating {size}x{size} float64 random matrices.")
        return (
            np.random.rand(size, size).astype(np.float64),
            np.random.rand(size, size).astype(np.float64),
        )

    def _measure_performance(
        self, method_func: Callable, a: np.ndarray, b: np.ndarray, workers: int, **kwargs
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """Execute performance measurement for a matrix multiplication method.

        Parameters
        ----------
        method_func : Callable
            Function that implements the multiplication method
        a : np.ndarray
            First input matrix
        b : np.ndarray
            Second input matrix
        workers : int
            Number of workers to use
        **kwargs
            Additional arguments passed to the method

        Returns
        -------
        Tuple[Optional[np.ndarray], float, float]
            Tuple containing (result matrix, execution time, CPU usage)
        """
        result = None
        exec_time = float("inf")
        cpu_usage = 0.0

        try:
            start_cpu = psutil.cpu_times_percent()
            start_time = time_module.perf_counter()

            result = method_func(a, b, workers, **kwargs) if kwargs else method_func(a, b, workers)

            end_time = time_module.perf_counter()
            end_cpu = psutil.cpu_times_percent()

            exec_time = end_time - start_time
            # Calculate total CPU usage across all cores
            cpu_usage = sum(
                (end_cpu.user - start_cpu.user,
                 end_cpu.system - start_cpu.system)
            )

        except Exception as e:
            logging.error(f"Error in matrix multiplication: {str(e)}")

        return result, exec_time, cpu_usage

    def numpy_multiply(self, a: np.ndarray, b: np.ndarray, _: int = 1) -> np.ndarray:
        """NumPy matrix multiplication (baseline implementation).

        Parameters
        ----------
        a : np.ndarray
            First input matrix
        b : np.ndarray
            Second input matrix
        _ : int
            Unused worker count parameter (for API consistency)

        Returns
        -------
        np.ndarray
            Result of matrix multiplication
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")
        return np.dot(a, b)

    def threading_multiply(
        self, a: np.ndarray, b: np.ndarray, num_threads: Optional[int] = None
    ) -> np.ndarray:
        """Threading-based matrix multiplication using NumPy operations for each chunk.

        Note: Due to Python's GIL, threading may not provide significant speedup for CPU-bound operations.
        However, NumPy operations can partially release the GIL during computation.
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        num_threads = num_threads or self.cpu_cores
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)

        # Calculate optimal chunk size based on matrix size and number of threads
        chunk_size = max(1, min(256, a.shape[0] // num_threads))
        chunks = [
            (i, min(i + chunk_size, a.shape[0]))
            for i in range(0, a.shape[0], chunk_size)
        ]

        def process_chunk(start: int, end: int) -> None:
            """Process a chunk using NumPy's optimized matrix multiplication."""
            result[start:end] = np.dot(a[start:end], b)

        threads = []
        for start, end in chunks:
            thread = threading.Thread(target=process_chunk, args=(start, end))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return result

    def multiprocessing_multiply(
        self, a: np.ndarray, b: np.ndarray, num_processes: Optional[int] = None
    ) -> np.ndarray:
        """Multiprocessing-based matrix multiplication using NumPy operations.

        This implementation:
        1. Uses NumPy's optimized dot product for each chunk
        2. Minimizes data transfer by sending only necessary chunks
        3. Optimizes chunk size for better cache utilization
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        num_processes = num_processes or self.cpu_cores
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)

        # Calculate optimal chunk size based on matrix size and cache considerations
        chunk_size = max(1, min(256, a.shape[0] // num_processes))
        chunks = [
            (i, min(i + chunk_size, a.shape[0]))
            for i in range(0, a.shape[0], chunk_size)
        ]

        # Prepare data for processes - only send necessary chunks
        chunk_data = [(a[start:end], b, start) for start, end in chunks]

        # Create and start processes
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_chunk_mp, chunk_data)

        # Collect results in correct order
        for start, chunk_result in results:
            result[start : start + chunk_result.shape[0]] = chunk_result

        return result

    def process_pool_executor_multiply(
        self, a: np.ndarray, b: np.ndarray, num_workers: Optional[int] = None
    ) -> np.ndarray:
        """Matrix multiplication using ProcessPoolExecutor.

        This implementation:
        1. Uses concurrent.futures.ProcessPoolExecutor for parallel processing
        2. Reuses the process_chunk_mp function for consistency
        3. Optimizes chunk distribution for better performance
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        num_workers = num_workers or self.cpu_cores
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)

        # Calculate optimal chunk size based on matrix size and cache considerations
        chunk_size = max(1, min(256, a.shape[0] // num_workers))
        chunks = [
            (i, min(i + chunk_size, a.shape[0]))
            for i in range(0, a.shape[0], chunk_size)
        ]

        # Prepare data for processes - only send necessary chunks
        chunk_data = [(a[start:end], b, start) for start, end in chunks]

        # Use ProcessPoolExecutor to distribute work
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process chunks in parallel and collect results
            results = list(executor.map(process_chunk_mp, chunk_data))

            # Assemble results in correct order
            for start, chunk_result in results:
                result[start : start + chunk_result.shape[0]] = chunk_result

        return result

    def thread_pool_executor_multiply(
        self, a: np.ndarray, b: np.ndarray, num_workers: Optional[int] = None
    ) -> np.ndarray:
        """Matrix multiplication using ThreadPoolExecutor.

        This implementation:
        1. Uses concurrent.futures.ThreadPoolExecutor for parallel processing
        2. Optimizes chunk size for better cache utilization
        3. Takes advantage of NumPy's GIL-releasing operations
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        num_workers = num_workers or self.cpu_cores
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)

        # Calculate optimal chunk size based on matrix size and number of workers
        chunk_size = max(1, min(256, a.shape[0] // num_workers))
        chunks = [
            (i, min(i + chunk_size, a.shape[0]))
            for i in range(0, a.shape[0], chunk_size)
        ]

        def process_chunk(chunk_range: Tuple[int, int]) -> Tuple[int, np.ndarray]:
            """Process a chunk of matrix multiplication and return the result."""
            start, end = chunk_range
            chunk_result = np.dot(a[start:end], b)
            return start, chunk_result

        try:
            # Use ThreadPoolExecutor to distribute work
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Process chunks in parallel and collect results
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

                # Assemble results in correct order
                for future in futures:
                    start, chunk_result = future.result()
                    result[start : start + chunk_result.shape[0]] = chunk_result

            return result
        except Exception as e:
            logging.error(f"Error in thread_pool_executor_multiply: {e}")
            raise

    def _calculate_optimal_block_size(self, matrix_size: int) -> int:
        """Calculate optimal block size based on matrix size and CPU cache."""
        l1_cache_size = 32 * 1024  # 32KB typical L1 cache
        double_size = 8  # bytes per double
        max_elements = l1_cache_size // (3 * double_size)
        block_size = int(np.sqrt(max_elements))

        # Ensure block size divides matrix size evenly if possible
        if matrix_size % block_size == 0:
            return block_size

        # Find largest divisor less than or equal to block_size
        for size in range(block_size, 0, -1):
            if matrix_size % size == 0:
                return size
        return block_size

    def sequential_blocked_multiply(
        self, a: np.ndarray, b: np.ndarray, block_size: Optional[int] = None
    ) -> np.ndarray:
        """Sequential blocked matrix multiplication with optimized block size."""
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        block_size = block_size or self._calculate_optimal_block_size(a.shape[0])
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)

        # Process blocks
        for i in range(0, a.shape[0], block_size):
            for j in range(0, b.shape[1], block_size):
                for k in range(0, a.shape[1], block_size):
                    i_end = min(i + block_size, a.shape[0])
                    j_end = min(j + block_size, b.shape[1])
                    k_end = min(k + block_size, a.shape[1])

                    block_result = np.dot(a[i:i_end, k:k_end], b[k:k_end, j:j_end])
                    result[i:i_end, j:j_end] += block_result

        return result

    def multiprocessing_shared_multiply(
        self, a: np.ndarray, b: np.ndarray, num_processes: Optional[int] = None
    ) -> np.ndarray:
        """Matrix multiplication using shared memory multiprocessing.

        This implementation:
        1. Uses shared memory to minimize data copying between processes
        2. Directly writes results to shared memory
        3. Optimizes chunk size for better cache utilization
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        num_processes = num_processes or self.cpu_cores
        dtype = np.float64

        # Create shared memory blocks
        shared_blocks = []
        try:
            # Create and initialize shared memory for matrices
            shm_a, shared_a, _ = create_shared_array(
                a.shape, dtype, self.shared_memory_registry
            )
            shared_blocks.append(shm_a)
            shared_a[:] = a

            shm_b, shared_b, _ = create_shared_array(
                b.shape, dtype, self.shared_memory_registry
            )
            shared_blocks.append(shm_b)
            shared_b[:] = b

            shm_result, shared_result, _ = create_shared_array(
                (a.shape[0], b.shape[1]), dtype, self.shared_memory_registry
            )
            shared_blocks.append(shm_result)

            # Calculate optimal chunk size
            chunk_size = max(1, min(256, a.shape[0] // num_processes))
            chunks = [
                (i, min(i + chunk_size, a.shape[0]))
                for i in range(0, a.shape[0], chunk_size)
            ]

            # Prepare arguments for worker processes
            shm_names = [shm_a.name, shm_b.name, shm_result.name]
            shapes = [a.shape, b.shape, (a.shape[0], b.shape[1])]
            worker_args = [(chunk, shm_names, shapes, dtype) for chunk in chunks]

            # Process chunks in parallel
            with Pool(processes=num_processes) as pool:
                pool.map(process_shared_chunk, worker_args)

            # Copy result from shared memory
            result = np.array(shared_result)
            return result

        finally:
            # Clean up all shared memory blocks
            for shm in shared_blocks:
                cleanup_shared_block(shm, self.shared_memory_registry, shm.name)

    def parallel_blocked_shared_multiply(
        self,
        a: np.ndarray,
        b: np.ndarray,
        num_processes: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> np.ndarray:
        """Multiply matrices using parallel blocked algorithm with shared memory.

        This implementation:
        1. Uses shared memory to minimize data copying between processes
        2. Applies cache-friendly blocked multiplication algorithm
        3. Parallelizes across row blocks to avoid synchronization issues
        4. Optimizes block size for cache utilization

        Parameters
        ----------
        a : np.ndarray
            First input matrix
        b : np.ndarray
            Second input matrix
        num_processes : Optional[int]
            Number of processes to use
        block_size : Optional[int]
            Size of blocks for blocked multiplication

        Returns
        -------
        np.ndarray
            Result of matrix multiplication
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        num_processes = num_processes or self.cpu_cores
        block_size = block_size or self._calculate_optimal_block_size(a.shape[0])
        dtype = np.float64

        shared_blocks = []
        try:
            # Create and initialize shared memory for matrices
            shm_a, shared_a, _ = create_shared_array(
                a.shape, dtype, self.shared_memory_registry
            )
            shared_blocks.append(shm_a)
            shared_a[:] = a

            shm_b, shared_b, _ = create_shared_array(
                b.shape, dtype, self.shared_memory_registry
            )
            shared_blocks.append(shm_b)
            shared_b[:] = b

            shm_result, shared_result, _ = create_shared_array(
                (a.shape[0], b.shape[1]), dtype, self.shared_memory_registry
            )
            shared_blocks.append(shm_result)
            shared_result[:] = 0  # Initialize result to zeros

            # Calculate row block ranges for each process
            num_row_blocks = (a.shape[0] + block_size - 1) // block_size
            blocks_per_process = max(1, num_row_blocks // num_processes)
            row_ranges = []

            for i in range(0, a.shape[0], blocks_per_process * block_size):
                end = min(i + blocks_per_process * block_size, a.shape[0])
                if end > i:  # Avoid empty ranges
                    row_ranges.append((i, end))

            # Prepare arguments for worker processes
            shm_names = [shm_a.name, shm_b.name, shm_result.name]
            shapes = [a.shape, b.shape, (a.shape[0], b.shape[1])]
            worker_args = [
                (row_range, block_size, shm_names, shapes, dtype)
                for row_range in row_ranges
            ]

            # Process blocks in parallel
            with Pool(processes=min(num_processes, len(row_ranges))) as pool:
                pool.map(process_parallel_blocked_chunk, worker_args)

            # Copy result from shared memory
            result = np.array(shared_result)
            return result

        finally:
            # Clean up all shared memory blocks
            for shm in shared_blocks:
                cleanup_shared_block(shm, self.shared_memory_registry, shm.name)

    def evaluate_performance(
        self, matrix_sizes: Optional[List[int]] = None, num_runs: int = 1
    ) -> Dict:
        """Evaluate performance of all matrix multiplication methods.

        Parameters
        ----------
        matrix_sizes : Optional[List[int]]
            List of matrix sizes to test (default: from config)
        num_runs : int
            Number of runs for each size (default: 1)

        Returns
        -------
        Dict
            Nested dictionary containing performance results for all methods
        """
        matrix_sizes = matrix_sizes or self.config.test_sizes
        results = {
            "sizes": matrix_sizes,
            "methods": {},
            "workers": self.config.worker_counts + [1],
            "cpu_usage": {},
        }

        # Pre-generate matrices for consistent comparison
        matrices = {
            size: self.generate_random_matrices(size) for size in matrix_sizes
        }

        # Test NumPy baseline
        results["methods"]["NumPy (Baseline)"] = {}
        for size in matrix_sizes:
            a, b = matrices[size]
            results["methods"]["NumPy (Baseline)"][size] = {"single": 0.0}
            for _ in range(num_runs):
                _, exec_time, cpu_usage = self._measure_performance(
                    self.numpy_multiply, a, b, 1
                )
                results["methods"]["NumPy (Baseline)"][size]["single"] += exec_time
                results["cpu_usage"][f"numpy_{size}"] = cpu_usage
            results["methods"]["NumPy (Baseline)"][size]["single"] /= num_runs

        # Test parallel methods
        parallel_methods = {
            "Threading": self.threading_multiply,
            "Multiprocessing": self.multiprocessing_multiply,
            "ProcessPool": self.process_pool_executor_multiply,
            "ThreadPool": self.thread_pool_executor_multiply,
            "SharedMemory": self.multiprocessing_shared_multiply,
        }

        for method_name, method_func in parallel_methods.items():
            results["methods"][method_name] = {}
            for size in matrix_sizes:
                results["methods"][method_name][size] = {}
                a, b = matrices[size]
                for workers in self.config.worker_counts:
                    total_time = 0.0
                    for _ in range(num_runs):
                        _, exec_time, cpu_usage = self._measure_performance(
                            method_func, a, b, workers
                        )
                        total_time += exec_time
                        results["cpu_usage"][f"{method_name}_{size}_{workers}"] = cpu_usage
                    results["methods"][method_name][size][str(workers)] = total_time / num_runs

        # Test blocked methods
        blocked_methods = {
            "Sequential Blocked": self.sequential_blocked_multiply,
            "Parallel Blocked": self.parallel_blocked_shared_multiply,
        }

        for method_name, method_func in blocked_methods.items():
            results["methods"][method_name] = {}
            for size in matrix_sizes:
                results["methods"][method_name][size] = {}
                a, b = matrices[size]
                block_size = self._calculate_optimal_block_size(size)
                
                if method_name == "Sequential Blocked":
                    total_time = 0.0
                    for _ in range(num_runs):
                        _, exec_time, cpu_usage = self._measure_performance(
                            method_func, a, b, block_size
                        )
                        total_time += exec_time
                        results["cpu_usage"][f"{method_name}_{size}"] = cpu_usage
                    results["methods"][method_name][size]["single"] = total_time / num_runs
                else:
                    for workers in self.config.worker_counts:
                        total_time = 0.0
                        for _ in range(num_runs):
                            _, exec_time, cpu_usage = self._measure_performance(
                                method_func, a, b, workers, block_size=block_size
                            )
                            total_time += exec_time
                            results["cpu_usage"][f"{method_name}_{size}_{workers}"] = cpu_usage
                        results["methods"][method_name][size][str(workers)] = total_time / num_runs

        return results

    def plot_performance(self, results: Dict) -> None:
        """Create performance comparison graphs for matrix multiplication methods.
        
        Parameters
        ----------
        results : Dict
            Dictionary containing performance results from evaluate_performance
        """
        if not results or "methods" not in results:
            logging.warning("No results to plot")
            return
        
        sizes = results["sizes"]
        methods = results["methods"]
        workers = results["workers"]
        
        # Set up the plot style
        plt.style.use("seaborn-v0_8")  # Use the updated style name
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Plot execution time vs matrix size for each method and worker count
        for method_name, size_data in methods.items():
            for worker_count in workers:
                key = str(worker_count) if worker_count > 1 else "single"
                times = []
                valid_sizes = []
                
                for size in sizes:
                    if (
                        size in size_data
                        and key in size_data
                        and size_data[size] > 0
                    ):
                        times.append(size_data[size][key])
                        valid_sizes.append(size)
                
                if times:
                    label = (
                        f"{method_name} ({key} "
                        f"worker{'s' if key != 'single' else ''})"
                    )
                    ax1.plot(valid_sizes, times, marker="o", label=label)
        
        ax1.set_xlabel("Matrix Size (N×N)")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_title("Matrix Multiplication Performance Comparison")
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.set_yscale("log")
        
        # Plot speedup relative to NumPy baseline
        baseline_method = "NumPy (Baseline)"
        if baseline_method in methods:
            baseline_data = methods[baseline_method]
            for method_name, size_data in methods.items():
                if method_name == baseline_method:
                    continue
                
                for worker_count in workers:
                    key = str(worker_count) if worker_count > 1 else "single"
                    speedups = []
                    valid_sizes = []
                    
                    for size in sizes:
                        if (
                            size in size_data
                            and key in size_data
                            and size in baseline_data
                            and "single" in baseline_data[size]
                            and baseline_data[size]["single"] > 0
                        ):
                            speedup = baseline_data[size]["single"] / size_data[size][key]
                            speedups.append(speedup)
                            valid_sizes.append(size)
                    
                    if speedups:
                        label = (
                            f"{method_name} ({key} "
                            f"worker{'s' if key != 'single' else ''})"
                        )
                        ax2.plot(valid_sizes, speedups, marker="o", label=label)
        
        ax2.set_xlabel("Matrix Size (N×N)")
        ax2.set_ylabel("Speedup (relative to NumPy)")
        ax2.set_title("Speedup Comparison")
        ax2.grid(True)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.axhline(y=1, color="k", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(
            self.output_dir, f"performance_comparison_{timestamp}.png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        logging.info(f"Performance plots saved to: {plot_path}")
        
        # Save results to JSON
        results_path = os.path.join(
            self.output_dir, f"performance_results_{timestamp}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Performance results saved to: {results_path}")

    def sequential_multiply(self, a: np.ndarray, b: np.ndarray, _: int = 1) -> np.ndarray:
        """Basic sequential matrix multiplication for illustration.

        This is a naive implementation using nested loops for educational purposes.
        For real applications, use numpy_multiply instead.

        Parameters
        ----------
        a : np.ndarray
            First input matrix
        b : np.ndarray
            Second input matrix
        _ : int
            Unused worker count parameter (for API consistency)

        Returns
        -------
        np.ndarray
            Result of matrix multiplication
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: {a.shape} and {b.shape}")

        result = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                for k in range(a.shape[1]):
                    result[i, j] += a[i, k] * b[k, j]
        return result


# --- Unit Tests ---
@pytest.fixture
def matrix_multiplication():
    """Fixture for matrix multiplication tests."""
    return MatrixMultiplication(Config())


def test_sequential_multiply(matrix_multiplication):
    """Test sequential matrix multiplication."""
    size = 4
    a, b = matrix_multiplication.generate_random_matrices(size)
    result = matrix_multiplication.sequential_multiply(a, b)
    assert result.shape == (size, size)
    assert np.allclose(result, np.matmul(a, b))


def test_numpy_multiply(matrix_multiplication):
    """Test NumPy matrix multiplication."""
    size = 4
    a, b = matrix_multiplication.generate_random_matrices(size)
    result = matrix_multiplication.numpy_multiply(a, b)
    assert result.shape == (size, size)
    assert np.allclose(result, np.matmul(a, b))


# ... [Add more tests as needed] ...


# --- Main Execution ---
def main():
    """Main function to run matrix multiplication benchmarks."""
    try:
        config = Config.from_file("config.json")
        mm = MatrixMultiplication(config)

        results = mm.evaluate_performance(matrix_sizes=config.test_sizes, num_runs=3)
        if results:
            logging.info("Performance evaluation completed successfully.")
            mm.plot_performance(results)

            with open(Path(config.output_dir) / "results.json", "w") as f:
                json.dump(results, f, indent=4)
            logging.info(f"Results saved to {config.output_dir}/results.json")
        else:
            logging.error("Performance evaluation failed.")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        return 1
    return 0


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    sys.exit(main())
