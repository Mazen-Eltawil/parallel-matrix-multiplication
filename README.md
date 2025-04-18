# Parallel Matrix Multiplication System

A comprehensive implementation of matrix multiplication using various parallel programming techniques in Python. This project demonstrates and compares different approaches to matrix multiplication, from basic sequential implementation to advanced parallel processing methods.

## Features

* Multiple implementation methods:
  * Sequential (Basic implementation for educational purposes)
  * NumPy (Baseline for performance comparison)
  * Threading (Using Python's threading module)
  * Multiprocessing (Using process pools)
  * Process Pool Executor (Using concurrent.futures)
  * Thread Pool Executor (Using concurrent.futures)
  * Shared Memory Multiprocessing (Using shared memory for efficiency)
  * Sequential Blocked (Cache-aware implementation)
  * Parallel Blocked Shared (Combining blocking with parallel processing)

* Advanced Features:
  * Performance analysis and visualization
  * Comprehensive error handling and logging
  * Configurable parameters via JSON
  * Cache-aware blocked multiplication
  * Interactive GUI with real-time visualization
  * Automated performance testing
  * Detailed performance reports

## Requirements

* Python 3.8+
* Required packages (install via requirements.txt):
  * NumPy
  * Matplotlib
  * psutil
  * pytest
  * tkinter (usually comes with Python)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mazen-Eltawil/parallel-matrix-multiplication.git
cd parallel-matrix-multiplication
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. GUI Application
Run the graphical interface:
```bash
python matrix_multiplication_gui.py
```

### 2. Performance Analysis
Run the benchmark suite:
```bash
python matrix_multiplication.py
```

### 3. Run Tests
Execute the test suite:
```bash
pytest test_matrix_multiplication.py -v
```

### 4. Configuration
Edit `config.json` to customize:
```json
{
    "test_sizes": [100, 250, 500, 1000],
    "block_size": 32,
    "verify_size": 64,
    "worker_counts": [1, 2, 4],
    "log_level": "INFO",
    "output_dir": "output"
}
```

## Project Structure

```
parallel-matrix-multiplication/
├── matrix_multiplication.py      # Core implementation
├── matrix_multiplication_gui.py  # GUI interface
├── test_matrix_multiplication.py # Test suite
├── generate_complete_report.py   # Report generation
├── config.json                  # Configuration file
├── requirements.txt             # Main dependencies
├── requirements-dev.txt         # Development dependencies
├── run_checks.bat              # Quality checks script
├── .flake8                     # Linter configuration
├── example_outputs/            # Example results
│   ├── performance_comparison_*.png
│   └── performance_results_*.json
└── output/                     # Results directory
```

## Performance Analysis

The system measures and compares:
* Execution time for different matrix sizes
* CPU utilization
* Speedup ratios relative to NumPy baseline
* Memory usage patterns
* Cache efficiency (for blocked methods)

### Example Performance Results

Performance metrics are saved in two formats:
1. Visual plots (`performance_comparison_*.png`)
2. Detailed JSON data (`performance_results_*.json`)

The analysis includes:
* Execution time comparison across methods
* Speedup relative to NumPy baseline
* CPU usage statistics
* Scaling with different worker counts

## GUI Features

The graphical interface provides:
* Matrix size and method selection
* Real-time visualization of:
  * Input matrices
  * Result matrix
  * Computation progress
* Performance metrics display
* Error handling and user feedback

## Error Handling and Logging

The system includes comprehensive error handling:
* Input validation
* Resource management
* Exception handling
* Detailed logging to both file and console

## Development

### Running Quality Checks
```bash
# On Windows:
run_checks.bat
# On Unix/MacOS:
pytest test_matrix_multiplication.py -v
flake8 .
```

### Adding New Methods
To add a new multiplication method:
1. Add the method to `MatrixMultiplication` class
2. Update the `methods` dictionary in `__init__`
3. Add corresponding tests
4. Update the GUI if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* NumPy team for the efficient array operations
* Python multiprocessing library developers
* The open-source community for testing and feedback 
