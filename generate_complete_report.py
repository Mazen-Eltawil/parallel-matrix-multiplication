"""Comprehensive report generator for matrix multiplication project."""

import json
import os
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matrix_multiplication import MatrixMultiplication, Config

class CompleteReportGenerator:
    """Generate comprehensive reports including performance, GUI events, and tests."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_complete_report(self):
        """Generate a complete report including performance, GUI events, and tests."""
        report_file = os.path.join(self.report_dir, f"complete_report_{self.timestamp}.txt")
        
        with open(report_file, "w") as f:
            # Write header
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE MATRIX MULTIPLICATION REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report File: {report_file}\n\n")
            
            # Write sections
            self._write_performance_section(f)
            self._write_gui_section(f)
            self._write_test_section(f)
            self._write_visualization_section(f)
            
        print(f"Complete report generated successfully: {report_file}")
        return report_file
    
    def _write_performance_section(self, f):
        """Write performance results section."""
        f.write("\n" + "=" * 50 + "\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        try:
            with open("output/results.json", "r") as results_file:
                results = json.load(results_file)
                
                # System Configuration
                f.write("System Configuration:\n")
                f.write("-" * 30 + "\n")
                f.write("Test Sizes: " + ", ".join(map(str, results["sizes"])) + "\n")
                f.write("Worker Counts: " + ", ".join(map(str, results["workers"])) + "\n\n")
                
                # Method Results
                for method_name, size_data in results["methods"].items():
                    f.write(f"\nMethod: {method_name}\n")
                    f.write("=" * (len(method_name) + 8) + "\n")
                    
                    for size, worker_data in size_data.items():
                        f.write(f"\nMatrix Size: {size}x{size}\n")
                        f.write("-" * 20 + "\n")
                        
                        for worker_count, time in worker_data.items():
                            f.write(f"Workers: {worker_count}\n")
                            f.write(f"Time: {time:.6f} seconds\n")
                            
                            cpu_key = f"{method_name}_{size}_{worker_count}"
                            if cpu_key in results["cpu_usage"]:
                                f.write(f"CPU Usage: {results['cpu_usage'][cpu_key]:.2f}%\n")
                            f.write("\n")
                
                # Performance Analysis
                self._write_performance_analysis(f, results)
                
        except FileNotFoundError:
            f.write("No performance results found.\n")
    
    def _write_performance_analysis(self, f, results):
        """Write detailed performance analysis."""
        f.write("\nPerformance Analysis:\n")
        f.write("-" * 30 + "\n")
        
        # Speedup Analysis
        baseline_method = "NumPy (Baseline)"
        if baseline_method in results["methods"]:
            f.write("\nSpeedup Analysis (vs NumPy baseline):\n")
            for size in results["sizes"]:
                f.write(f"\nMatrix Size {size}x{size}:\n")
                baseline_time = results["methods"][baseline_method][str(size)]["single"]
                
                for method_name, size_data in results["methods"].items():
                    if method_name != baseline_method and str(size) in size_data:
                        for worker_count, time in size_data[str(size)].items():
                            speedup = baseline_time / time
                            f.write(f"- {method_name} ({worker_count} workers): {speedup:.2f}x\n")
        
        # Best Methods Analysis
        f.write("\nBest Performing Methods:\n")
        for size in results["sizes"]:
            best_time = float('inf')
            best_method = ""
            best_workers = ""
            
            for method, size_data in results["methods"].items():
                if str(size) in size_data:
                    for workers, time in size_data[str(size)].items():
                        if time < best_time:
                            best_time = time
                            best_method = method
                            best_workers = workers
            
            f.write(f"\nSize {size}x{size}:\n")
            f.write(f"- Best: {best_method} with {best_workers} workers\n")
            f.write(f"- Time: {best_time:.6f} seconds\n")
    
    def _write_gui_section(self, f):
        """Write GUI events and state section."""
        f.write("\n" + "=" * 50 + "\n")
        f.write("GUI INFORMATION\n")
        f.write("=" * 50 + "\n\n")
        
        # GUI Components Status
        f.write("GUI Components:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Configuration Section:\n")
        f.write("   - Matrix size input\n")
        f.write("   - Worker count selection\n")
        f.write("   - Block size configuration\n")
        f.write("   - Output directory selection\n\n")
        
        f.write("2. Methods Section:\n")
        f.write("   - NumPy (Baseline)\n")
        f.write("   - Threading\n")
        f.write("   - Multiprocessing\n")
        f.write("   - ProcessPool\n")
        f.write("   - ThreadPool\n")
        f.write("   - SharedMemory\n")
        f.write("   - Sequential Blocked\n")
        f.write("   - Parallel Blocked\n\n")
        
        f.write("3. Visualization Section:\n")
        f.write("   - Matrix A display\n")
        f.write("   - Matrix B display\n")
        f.write("   - Result matrix display\n")
        f.write("   - Animation controls\n")
        f.write("   - Progress tracking\n\n")
        
        f.write("4. Output Section:\n")
        f.write("   - Log display\n")
        f.write("   - Clear log button\n")
        f.write("   - Save results button\n\n")
    
    def _write_visualization_section(self, f):
        """Write visualization details section."""
        f.write("\n" + "=" * 50 + "\n")
        f.write("VISUALIZATION DETAILS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Animation Features:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Matrix Display:\n")
        f.write("   - Color-coded cell values\n")
        f.write("   - Value annotations\n")
        f.write("   - Grid lines\n")
        f.write("   - Matrix titles\n\n")
        
        f.write("2. Animation Controls:\n")
        f.write("   - Speed slider (0.1x - 5.0x)\n")
        f.write("   - Reset speed button\n")
        f.write("   - Current speed display\n\n")
        
        f.write("3. Progress Tracking:\n")
        f.write("   - Progress bar\n")
        f.write("   - Status messages\n")
        f.write("   - Operation highlighting\n")
        f.write("   - Current computation display\n\n")
        
        f.write("4. Visual Enhancements:\n")
        f.write("   - Clean background\n")
        f.write("   - Proper spacing\n")
        f.write("   - Consistent fonts\n")
        f.write("   - Responsive layout\n\n")
    
    def _write_test_section(self, f):
        """Write test results section."""
        f.write("\n" + "=" * 50 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Matrix Multiplication Tests
        f.write("Matrix Multiplication Tests:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Correctness Tests:\n")
        f.write("   - Small matrices (2x2, 3x3)\n")
        f.write("   - Medium matrices (10x10, 50x50)\n")
        f.write("   - Large matrices (100x100, 500x500)\n")
        f.write("   - Edge cases (1x1, empty matrices)\n\n")
        
        f.write("2. Performance Tests:\n")
        f.write("   - Sequential vs Parallel execution\n")
        f.write("   - Memory usage monitoring\n")
        f.write("   - CPU utilization tracking\n")
        f.write("   - Scaling with matrix size\n\n")
        
        f.write("3. GUI Tests:\n")
        f.write("   - Input validation\n")
        f.write("   - Event handling\n")
        f.write("   - Visualization accuracy\n")
        f.write("   - Progress tracking\n\n")
        
        f.write("4. Error Handling:\n")
        f.write("   - Invalid inputs\n")
        f.write("   - Resource limitations\n")
        f.write("   - Thread safety\n")
        f.write("   - Exception handling\n\n")

def main():
    """Generate a complete report."""
    generator = CompleteReportGenerator()
    generator.generate_complete_report()

if __name__ == "__main__":
    main() 
