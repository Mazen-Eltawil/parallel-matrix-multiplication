"""Test matrix multiplication implementations."""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from matrix_multiplication import Config, MatrixMultiplication

# Test configuration
TEST_CONFIG = Config(
    test_sizes=[64],
    block_size=32,
    verify_size=64,
    worker_counts=[1, 2],
    log_level="INFO",
    output_dir="test_output"
)

@pytest.fixture
def matrix_mult():
    """Create a MatrixMultiplication instance for testing."""
    return MatrixMultiplication(TEST_CONFIG)

@pytest.fixture
def test_matrices():
    """Generate test matrices."""
    size = TEST_CONFIG.verify_size
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    expected = np.dot(a, b)
    return a, b, expected

def test_numpy_multiply(matrix_mult, test_matrices):
    """Test NumPy multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.numpy_multiply(a, b, 1)
    np.testing.assert_array_almost_equal(result, expected)

def test_sequential_multiply(matrix_mult, test_matrices):
    """Test sequential multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.sequential_multiply(a, b, 1)
    np.testing.assert_array_almost_equal(result, expected)

def test_threading_multiply(matrix_mult, test_matrices):
    """Test threaded multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.threading_multiply(a, b, 2)
    np.testing.assert_array_almost_equal(result, expected)

def test_multiprocessing_multiply(matrix_mult, test_matrices):
    """Test multiprocessing multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.multiprocessing_multiply(a, b, 2)
    np.testing.assert_array_almost_equal(result, expected)

def test_sequential_blocked_multiply(matrix_mult, test_matrices):
    """Test sequential blocked multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.sequential_blocked_multiply(a, b, 1)
    np.testing.assert_array_almost_equal(result, expected)

def test_multiprocessing_shared_multiply(matrix_mult, test_matrices):
    """Test multiprocessing with shared memory multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.multiprocessing_shared_multiply(a, b, 2)
    np.testing.assert_array_almost_equal(result, expected)

def test_process_pool_executor_multiply(matrix_mult, test_matrices):
    """Test process pool executor multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.process_pool_executor_multiply(a, b, 2)
    np.testing.assert_array_almost_equal(result, expected)

def test_thread_pool_executor_multiply(matrix_mult, test_matrices):
    """Test thread pool executor multiplication."""
    a, b, expected = test_matrices
    result = matrix_mult.thread_pool_executor_multiply(a, b, 2)
    np.testing.assert_array_almost_equal(result, expected)

def test_parallel_blocked_shared_multiply(matrix_mult, test_matrices):
    """Test parallel blocked multiplication with shared memory."""
    a, b, expected = test_matrices
    result = matrix_mult.parallel_blocked_shared_multiply(a, b, 2)
    np.testing.assert_array_almost_equal(result, expected)

def test_invalid_matrix_dimensions(matrix_mult):
    """Test handling of invalid matrix dimensions."""
    a = np.random.rand(3, 4)
    b = np.random.rand(5, 6)
    with pytest.raises(ValueError):
        matrix_mult.numpy_multiply(a, b, 1)

def test_performance_evaluation(matrix_mult):
    """Test performance evaluation functionality."""
    results = matrix_mult.evaluate_performance(num_runs=1)
    assert isinstance(results, dict)
    assert "methods" in results
    assert "cpu_usage" in results

def test_plot_performance(matrix_mult, tmp_path):
    """Test performance plotting functionality."""
    results = matrix_mult.evaluate_performance(num_runs=1)
    matrix_mult.plot_performance(results)
    plot_files = list(Path(matrix_mult.config.output_dir).glob("performance_comparison_*.png"))
    assert len(plot_files) > 0

def test_cleanup(matrix_mult):
    """Test cleanup of shared memory."""
    matrix_mult.shared_memory_registry.cleanup()
    assert len(matrix_mult.shared_memory_registry.blocks) == 0

if __name__ == "__main__":
    pytest.main(["-v"])