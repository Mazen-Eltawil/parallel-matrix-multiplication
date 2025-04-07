# Parallel Matrix Multiplication System

A comprehensive implementation of matrix multiplication using various parallel programming techniques in Python.

## Features

- Multiple implementation methods:
  - Sequential
  - NumPy (baseline)
  - Threading
  - Multiprocessing Pool
  - Process Pool Executor
  - Thread Pool Executor
  - Multiprocessing Shared Memory
  - Sequential Blocked
  - Parallel Blocked Shared
  - GPU (CuPy) support
- Performance analysis and visualization
- Comprehensive error handling and logging
- Unit tests
- Configuration management
- Cache-aware blocked multiplication
- Interactive GUI with matrix visualization

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- psutil
- pytest (for testing)
- CuPy (optional, for GPU support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mazen-Eltawil/parallel-matrix-multiplication.git
cd parallel-matrix-multiplication
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the GUI application:
```bash
python matrix_multiplication_gui.py
```

2. Run the performance analysis:
```bash
python matrix_multiplication.py
```

3. Run tests:
```bash
pytest test_matrix_multiplication.py -v
```

4. Configure the program by editing `config.json`:
```json
{
    "test_sizes": [100, 250, 500, 1000],
    "block_size": 32,
    "verify_size": 64,
    "worker_counts": [1, 2, 4],
    "gpu_enabled": true,
    "log_level": "INFO",
    "output_dir": "output"
}
```

## Output

The program generates:
- Performance plots in the `output` directory
- Log file: `matrix_multiplication.log`
- Configuration file: `config.json`

## Performance Analysis

The program measures:
- Execution time
- CPU utilization
- Speedup ratios
- Memory usage

## GUI Features

The GUI application provides:
- Configuration controls for matrix sizes and methods
- Real-time visualization of matrix multiplication
- Progress tracking and logging
- Performance results display
- Animation speed control

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.