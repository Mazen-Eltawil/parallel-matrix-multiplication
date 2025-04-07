"""GUI application for matrix multiplication benchmarks."""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, scrolledtext, Canvas
import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import colors

from matrix_multiplication import Config, MatrixMultiplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class GuiLoggingHandler(logging.Handler):
    """Custom logging handler that writes to a ScrolledText widget.
    
    This handler is thread-safe, using a queue to communicate between
    the logging thread and the GUI thread.
    """
    
    def __init__(self, text_widget: scrolledtext.ScrolledText):
        """Initialize the handler with a text widget.
        
        Parameters
        ----------
        text_widget : scrolledtext.ScrolledText
            The text widget to write log messages to
        """
        super().__init__()
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.text_widget.after(100, self.check_queue)
        
    def emit(self, record):
        """Process a log record by adding it to the queue.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to process
        """
        msg = self.format(record)
        self.queue.put(msg)
        
    def check_queue(self):
        """Check the queue for new messages and update the text widget."""
        try:
            while True:
                msg = self.queue.get_nowait()
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                self.text_widget.configure(state='disabled')
                self.queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.text_widget.after(100, self.check_queue)


class VisualizationFrame(ttk.LabelFrame):
    """Frame for visualizing matrix multiplication progress."""
    
    def __init__(self, parent, **kwargs):
        """Initialize the visualization frame."""
        super().__init__(parent, text='Matrix Multiplication Visualization', padding=10, **kwargs)
        
        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for matrix visualization
        self.figure = Figure(figsize=(12, 4), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots with proper spacing
        self.axes = []
        gs = self.figure.add_gridspec(1, 3, wspace=0.3)
        for i, title in enumerate(['Matrix A', 'Matrix B', 'Result']):
            ax = self.figure.add_subplot(gs[0, i])
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes.append(ax)
        
        # Create controls frame
        self.create_controls()
        
        # Create progress frame
        self.create_progress_frame()
        
        # Initialize visualization properties
        self.cell_size = 60  # Size of matrix cells in pixels
        self.max_display_size = 8  # Maximum size to display full matrix
        
    def create_controls(self):
        """Create animation control panel."""
        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        # Speed control
        speed_frame = ttk.Frame(controls)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text='Animation Speed:').pack(side=tk.LEFT, padx=(0, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=5.0,
            variable=self.speed_var,
            orient=tk.HORIZONTAL
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.speed_label = ttk.Label(speed_frame, text='1.0x')
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            speed_frame,
            text='Reset Speed',
            command=lambda: self.speed_var.set(1.0)
        ).pack(side=tk.LEFT, padx=5)
        
        # Update speed label when scale changes
        def update_speed_label(*args):
            self.speed_label.config(text=f'{self.speed_var.get():.1f}x')
        self.speed_var.trace_add('write', update_speed_label)
        
    def create_progress_frame(self):
        """Create progress tracking frame."""
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            progress_frame,
            textvariable=self.status_var,
            font=('TkDefaultFont', 10)
        )
        status_label.pack(fill=tk.X, pady=(0, 5))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            style='Clean.Horizontal.TProgressbar'
        )
        self.progress.pack(fill=tk.X)
        
    def update_visualization(self, a, b, c, progress=None):
        """Update the visualization with current matrices."""
        try:
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Plot matrices
            matrices = [a, b, c]
            titles = ['Matrix A', 'Matrix B', 'Result']
            
            for idx, (matrix, title) in enumerate(zip(matrices, titles)):
                # Create clean display matrix
                display_matrix = self._prepare_matrix_for_display(matrix)
                
                # Plot matrix with clean style
                im = self.axes[idx].imshow(
                    display_matrix,
                    cmap='YlOrRd',
                    aspect='equal',
                    interpolation='nearest'
                )
                self.axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
                
                # Add value annotations
                for i in range(display_matrix.shape[0]):
                    for j in range(display_matrix.shape[1]):
                        value = display_matrix[i, j]
                        color = 'black' if value < 3 else 'white'
                        self.axes[idx].text(
                            j, i, f'{value:.1f}',
                            ha='center',
                            va='center',
                            color=color,
                            fontsize=10,
                            fontweight='bold'
                        )
                
                # Add grid
                self.axes[idx].grid(True, color='gray', linewidth=0.5, alpha=0.3)
            
            # Update progress visualization if provided
            if progress is not None:
                i, j, k = progress
                if i is not None and j is not None:
                    # Highlight current operation
                    self._highlight_operation(i, j, k)
                    
                    # Update progress
                    total_ops = a.shape[0] * a.shape[1] * b.shape[1]
                    current_op = i * b.shape[1] * a.shape[1] + j * a.shape[1] + (k if k is not None else 0)
                    self.progress['value'] = (current_op / total_ops) * 100
                    
                    # Update status
                    if k is not None:
                        self.status_var.set(f"Computing C[{i},{j}] += A[{i},{k}] × B[{k},{j}]")
                    else:
                        self.status_var.set(f"Completed element C[{i},{j}]")
            
            # Redraw
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating visualization: {e}")
            self.status_var.set("Error updating visualization")
            self.progress['value'] = 0
            
    def _prepare_matrix_for_display(self, matrix):
        """Prepare matrix for display by subsampling if necessary."""
        rows, cols = matrix.shape
        if rows > self.max_display_size or cols > self.max_display_size:
            # Create a smaller representation by averaging blocks
            new_rows = min(rows, self.max_display_size)
            new_cols = min(cols, self.max_display_size)
            display = np.zeros((new_rows, new_cols))
            
            row_block = rows // new_rows
            col_block = cols // new_cols
            
            for i in range(new_rows):
                for j in range(new_cols):
                    r_start = i * row_block
                    r_end = min((i + 1) * row_block, rows)
                    c_start = j * col_block
                    c_end = min((j + 1) * col_block, cols)
                    display[i, j] = np.mean(matrix[r_start:r_end, c_start:c_end])
            
            return display
        return matrix
        
    def _highlight_operation(self, i, j, k):
        """Highlight the current operation in the visualization."""
        if k is not None:
            # Highlight cells involved in current multiplication
            self.axes[0].add_patch(plt.Rectangle(
                (k-0.5, i-0.5), 1, 1,
                fill=True, color='yellow', alpha=0.3
            ))
            self.axes[1].add_patch(plt.Rectangle(
                (j-0.5, k-0.5), 1, 1,
                fill=True, color='yellow', alpha=0.3
            ))
        
        # Highlight result cell
        self.axes[2].add_patch(plt.Rectangle(
            (j-0.5, i-0.5), 1, 1,
            fill=True, color='lime', alpha=0.3
        ))
        
        # Highlight row and column
        self.axes[0].add_patch(plt.Rectangle(
            (-0.5, i-0.5), self.axes[0].get_xlim()[1], 1,
            fill=False, color='blue', alpha=0.2
        ))
        self.axes[1].add_patch(plt.Rectangle(
            (j-0.5, -0.5), 1, self.axes[1].get_ylim()[1],
            fill=False, color='blue', alpha=0.2
        ))


class MatrixMultiplicationGUI:
    """Main GUI application for matrix multiplication benchmarking."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Matrix Multiplication Benchmark")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize method variables dictionary
        self.method_vars = {}
        
        # Create configuration section
        self.create_config_section()
        
        # Create methods selection section
        self.create_methods_section()
        
        # Create execution control section
        self.create_execution_section()
        
        # Create output/logging section
        self.create_output_section()
        
        # Create visualization section
        self.create_visualization_section()
        
        # Create results section
        self.create_results_section()
        
        # Configure logging
        self.setup_logging()
        
        # Initialize animation speed control
        self.animation_speed = tk.DoubleVar(value=1.0)
        
        # Log startup
        logging.info("Matrix Multiplication Benchmark GUI started")
        logging.info(f"Output directory set to: {self.output_dir_var.get()}")
        
    def setup_logging(self):
        """Configure logging to write to the GUI text widget."""
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Create and add the custom handler for the GUI
        gui_handler = GuiLoggingHandler(self.log_text)
        gui_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
        gui_handler.setFormatter(gui_formatter)
        root_logger.addHandler(gui_handler)
        
        # Set the log level from the GUI setting
        root_logger.setLevel(getattr(logging, self.log_level_var.get()))
        
    def cleanup_logging(self):
        """Clean up logging handlers to prevent memory leaks."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, GuiLoggingHandler):
                root_logger.removeHandler(handler)
    
    def create_config_section(self):
        """Create the configuration section of the GUI."""
        config_frame = ttk.LabelFrame(self.main_frame, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Matrix sizes
        ttk.Label(config_frame, text="Matrix Sizes (comma-separated):").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.matrix_sizes_var = tk.StringVar(value="100, 250, 500, 1000")
        ttk.Entry(config_frame, textvariable=self.matrix_sizes_var).grid(
            row=0, column=1, sticky=tk.EW, padx=5, pady=5
        )
        
        # Worker counts
        ttk.Label(config_frame, text="Worker Counts (comma-separated):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.worker_counts_var = tk.StringVar(value="1, 2, 4")
        ttk.Entry(config_frame, textvariable=self.worker_counts_var).grid(
            row=1, column=1, sticky=tk.EW, padx=5, pady=5
        )
        
        # Block size
        ttk.Label(config_frame, text="Block Size:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.block_size_var = tk.StringVar(value="32")
        ttk.Entry(config_frame, textvariable=self.block_size_var).grid(
            row=2, column=1, sticky=tk.EW, padx=5, pady=5
        )
        
        # Number of runs
        ttk.Label(config_frame, text="Number of Runs:").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=5
        )
        self.num_runs_var = tk.StringVar(value="3")
        ttk.Entry(config_frame, textvariable=self.num_runs_var).grid(
            row=0, column=3, sticky=tk.EW, padx=5, pady=5
        )
        
        # Output directory
        ttk.Label(config_frame, text="Output Directory:").grid(
            row=1, column=2, sticky=tk.W, padx=5, pady=5
        )
        self.output_dir_var = tk.StringVar(value="output")
        ttk.Entry(config_frame, textvariable=self.output_dir_var).grid(
            row=1, column=3, sticky=tk.EW, padx=5, pady=5
        )
        ttk.Button(
            config_frame, 
            text="Browse...", 
            command=self.browse_output_dir
        ).grid(row=1, column=4, padx=5, pady=5)
        
        # Log level
        ttk.Label(config_frame, text="Log Level:").grid(
            row=2, column=2, sticky=tk.W, padx=5, pady=5
        )
        self.log_level_var = tk.StringVar(value="INFO")
        ttk.Combobox(
            config_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            state="readonly"
        ).grid(row=2, column=3, sticky=tk.EW, padx=5, pady=5)
        
        # Configure grid
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        
    def create_methods_section(self):
        """Create the methods selection section of the GUI."""
        methods_frame = ttk.LabelFrame(self.main_frame, text="Methods", padding=10)
        methods_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create checkbuttons for each method
        methods = [
            "NumPy (Baseline)", 
            "Threading", 
            "Multiprocessing", 
            "Shared Memory MP", 
            "Sequential Blocked", 
            "Parallel Blocked Shared", 
            "ProcessPoolExecutor", 
            "ThreadPoolExecutor", 
            "Sequential"
        ]
        
        # Initialize method_vars dictionary with variables for each method
        # and create checkbuttons
        col, row = 0, 0
        for method in methods:
            self.method_vars[method] = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(
                methods_frame,
                text=method,
                variable=self.method_vars[method]
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        # Buttons for selecting all/none
        button_frame = ttk.Frame(methods_frame)
        button_frame.grid(row=row+1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(
            button_frame,
            text="Select All",
            command=lambda: [var.set(True) for var in self.method_vars.values()]
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Select None",
            command=lambda: [var.set(False) for var in self.method_vars.values()]
        ).pack(side=tk.LEFT, padx=5)
        
    def create_execution_section(self):
        """Create the execution control section of the GUI."""
        execution_frame = ttk.LabelFrame(self.main_frame, text="Execution Control", padding=10)
        execution_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Run button
        self.run_button = ttk.Button(
            execution_frame,
            text="Run Benchmark",
            command=self.run_benchmark
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(
            execution_frame,
            text="Stop",
            command=self.stop_benchmark,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            execution_frame,
            textvariable=self.status_var,
            font=('TkDefaultFont', 10, 'bold')
        )
        status_label.pack(side=tk.LEFT, padx=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(execution_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    def create_output_section(self):
        """Create the output/logging section of the GUI."""
        output_frame = ttk.LabelFrame(self.main_frame, text="Output/Logging", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            height=10,
            width=80,
            font=('Courier', 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = ttk.Frame(output_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Clear button
        ttk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log
        ).pack(side=tk.LEFT, padx=5)
        
        # Save button
        ttk.Button(
            button_frame,
            text="Save Log",
            command=self.save_log
        ).pack(side=tk.LEFT, padx=5)
        
    def create_visualization_section(self):
        """Create the visualization section of the GUI."""
        self.viz_frame = VisualizationFrame(self.main_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_results_section(self):
        """Create the results section of the GUI."""
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Button to show latest plot
        self.show_plot_button = ttk.Button(
            results_frame,
            text="Show Performance Plot",
            command=self.show_latest_plot,
            state=tk.DISABLED
        )
        self.show_plot_button.pack(side=tk.LEFT, padx=5)
        
        # Button to open output directory
        ttk.Button(
            results_frame,
            text="Open Output Directory",
            command=self.open_output_dir
        ).pack(side=tk.LEFT, padx=5)
        
        # Button to show latest performance data
        self.show_data_button = ttk.Button(
            results_frame,
            text="Show Performance Data",
            command=self.show_performance_data,
            state=tk.DISABLED
        )
        self.show_data_button.pack(side=tk.LEFT, padx=5)
    
    def browse_output_dir(self):
        """Open a dialog to select output directory."""
        directory = filedialog.askdirectory(
            initialdir=self.output_dir_var.get() or os.getcwd(),
            title="Select Output Directory"
        )
        if directory:
            self.output_dir_var.set(directory)
            logging.info(f"Output directory set to: {directory}")
            
    def clear_log(self):
        """Clear the log text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def save_log(self):
        """Save log contents to a file."""
        filename = filedialog.asksaveasfilename(
            initialdir=self.output_dir_var.get() or os.getcwd(),
            title="Save Log File",
            filetypes=(("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")),
            defaultextension=".log"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                logging.info(f"Log saved to {filename}")
            except Exception as e:
                logging.error(f"Error saving log: {e}")
                
    def show_latest_plot(self):
        """Show the latest performance plot."""
        try:
            output_dir = self.output_dir_var.get()
            plot_files = list(Path(output_dir).glob("performance_comparison_*.png"))
            if not plot_files:
                logging.warning("No performance plots found in output directory")
                return
                
            latest_plot = max(plot_files, key=os.path.getctime)
            os.startfile(latest_plot)
            logging.info(f"Opened plot: {latest_plot}")
        except Exception as e:
            logging.error(f"Error opening plot: {e}")
            
    def open_output_dir(self):
        """Open the output directory in file explorer."""
        try:
            output_dir = self.output_dir_var.get()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            os.startfile(output_dir)
            logging.info(f"Opened output directory: {output_dir}")
        except Exception as e:
            logging.error(f"Error opening directory: {e}")
            
    def show_performance_data(self):
        """Show the latest performance data."""
        try:
            output_dir = self.output_dir_var.get()
            data_files = list(Path(output_dir).glob("performance_results_*.json"))
            if not data_files:
                logging.warning("No performance data found in output directory")
                return
                
            latest_data = max(data_files, key=os.path.getctime)
            os.startfile(latest_data)
            logging.info(f"Opened data file: {latest_data}")
        except Exception as e:
            logging.error(f"Error opening data file: {e}")
    
    def run_benchmark(self):
        """Start the benchmark thread."""
        # Validate input
        try:
            # Parse matrix sizes
            sizes = self.matrix_sizes_var.get().split(',')
            sizes = [int(size.strip()) for size in sizes]
            if not sizes:
                raise ValueError("No matrix sizes specified")
                
            # Check if any methods are selected
            if not any(var.get() for var in self.method_vars.values()):
                raise ValueError("No methods selected")
                
            # Disable controls during benchmark
            self.disable_controls()
            
            # Start benchmark in a separate thread
            self.benchmark_thread = threading.Thread(
                target=self.run_benchmark_thread,
                daemon=True
            )
            self.benchmark_thread.start()
            
            # Update status
            self.status_var.set("Running benchmark...")
            self.progress.start()
            
        except Exception as e:
            logging.error(f"Error starting benchmark: {e}")
            self.enable_controls()
            
    def stop_benchmark(self):
        """Stop the benchmark."""
        # This is a simple implementation that just re-enables controls
        # A more comprehensive implementation would use an event to signal the thread to stop
        logging.warning("Stopping benchmark...")
        self.enable_controls()
        self.progress.stop()
        self.status_var.set("Benchmark stopped")
        
    def disable_controls(self):
        """Disable controls during benchmark execution."""
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
    def enable_controls(self):
        """Enable controls after benchmark completion."""
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        
    def update_results_buttons(self):
        """Update the state of results buttons."""
        self.show_plot_button.config(state=tk.NORMAL)
        self.show_data_button.config(state=tk.NORMAL)
        
    def animate_multiplication(self, a, b, method_name):
        """Animate the matrix multiplication process."""
        m, n = a.shape
        p = b.shape[1]
        result = np.zeros((m, p))
        
        # Initialize visualization
        self.viz_frame.update_visualization(a, b, result)
        logging.info(f"Starting visualization of {method_name} for {m}x{n} matrices")
        
        try:
            # Calculate total operations
            total_ops = m * p * n
            current_op = 0
            last_update = time.time()
            min_update_interval = 1.0 / 30  # Cap at 30 FPS
            
            # Simulate multiplication
            for i in range(m):
                for j in range(p):
                    element_sum = 0
                    for k in range(n):
                        # Update result
                        element_sum += a[i, k] * b[k, j]
                        result[i, j] = element_sum
                        current_op += 1
                        
                        # Update visualization with frame rate control
                        current_time = time.time()
                        time_since_update = current_time - last_update
                        
                        # Control animation speed and frame rate
                        target_delay = 0.1 / self.viz_frame.speed_var.get()
                        should_update = (
                            time_since_update >= min_update_interval and
                            (time_since_update >= target_delay or
                             current_op % max(1, total_ops // 100) == 0)  # Update at least 100 times total
                        )
                        
                        if should_update:
                            self.viz_frame.update_visualization(a, b, result, (i, j, k))
                            self.root.update()
                            last_update = current_time
                    
                    # Always update at the end of each element
                    self.viz_frame.update_visualization(a, b, result, (i, j, None))
                    self.root.update()
            
            # Final visualization
            self.viz_frame.update_visualization(a, b, result)
            logging.info("Visualization completed successfully")
            
        except Exception as e:
            logging.error(f"Error during visualization: {e}")
        finally:
            # Reset progress
            self.viz_frame.status_var.set("Ready")
            self.viz_frame.progress['value'] = 0
        
        return result

    def run_benchmark_thread(self):
        """Run the benchmark in a separate thread."""
        try:
            # Parse configuration
            config = Config(
                test_sizes=[int(x.strip()) for x in self.matrix_sizes_var.get().split(',')],
                worker_counts=[int(x.strip()) for x in self.worker_counts_var.get().split(',')],
                block_size=int(self.block_size_var.get()),
                output_dir=self.output_dir_var.get(),
                log_level=self.log_level_var.get()
            )
            
            # Create MatrixMultiplication instance
            mm = MatrixMultiplication(config)
            
            # Get selected methods efficiently
            method_map = {
                'NumPy (Baseline)': 'numpy_multiply',
                'Threading': 'threading_multiply',
                'Multiprocessing': 'multiprocessing_multiply',
                'Shared Memory MP': 'multiprocessing_shared_multiply',
                'Sequential Blocked': 'sequential_blocked_multiply',
                'Parallel Blocked Shared': 'parallel_blocked_shared_multiply',
                'ProcessPoolExecutor': 'process_pool_executor_multiply',
                'ThreadPoolExecutor': 'thread_pool_executor_multiply',
                'Sequential': 'sequential_multiply'
            }
            
            selected_methods = {
                method_map[method]: mm.methods[method_map[method]]
                for method, var in self.method_vars.items()
                if var.get() and method_map[method] in mm.methods
            }
            
            if not selected_methods:
                raise ValueError("No methods selected for benchmark")
            
            # Update mm.methods to only include selected methods
            mm.methods = selected_methods
            
            # Create sample matrices for visualization (using smaller size)
            size = min(10, min(int(x.strip()) for x in self.matrix_sizes_var.get().split(',')))
            # Create matrices with more reasonable values
            a = np.round(np.random.uniform(1, 5, (size, size)), 1)  # Values between 1 and 5
            b = np.round(np.random.uniform(1, 5, (size, size)), 1)  # Values between 1 and 5
            
            # Run benchmark with visualization for the first size
            logging.info(f"Running benchmark with methods: {', '.join(selected_methods.keys())}")
            logging.info("Showing visualization for the first matrix size...")
            
            # Animate the multiplication for demonstration
            self.animate_multiplication(a, b, list(selected_methods.keys())[0])
            
            # Run the actual benchmark
            results = mm.evaluate_performance(num_runs=int(self.num_runs_var.get()))
            
            logging.info("Generating performance plots...")
            mm.plot_performance(results)
            
            logging.info("Benchmark completed successfully")
            
            # Update results buttons
            self.root.after(0, self.update_results_buttons)
            
        except Exception as e:
            logging.error(f"Benchmark error: {e}")
        finally:
            # Re-enable buttons
            self.root.after(0, self.enable_controls)
            self.cleanup_logging()

if __name__ == "__main__":
    root = tk.Tk()
    app = MatrixMultiplicationGUI(root)
    root.mainloop()