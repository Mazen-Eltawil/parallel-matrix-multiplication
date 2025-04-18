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
    """Main GUI class for matrix multiplication benchmarks."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI."""
        self.root = root
        self.root.title('Parallel Matrix Multiplication Benchmark')
        self.root.geometry('1000x800')  # Increased size for visualization
        
        # Initialize variables
        self.benchmark_thread = None
        self.logging_handler = None
        self.method_vars = {}
        self.animation_speed = tk.DoubleVar(value=1.0)  # Animation speed multiplier
        
        # Create main sections
        self.create_config_section()
        self.create_methods_section()
        self.create_execution_section()
        self.create_visualization_section()  # New visualization section
        self.create_output_section()
        self.create_results_section()
        
        # Load default config
        self.load_default_config()
        
    def create_config_section(self):
        """Create the configuration section of the GUI."""
        config_frame = ttk.LabelFrame(self.root, text='Configuration', padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Matrix sizes
        ttk.Label(config_frame, text='Matrix Sizes:').grid(row=0, column=0, sticky=tk.W)
        self.matrix_sizes_var = tk.StringVar(value='100, 250, 500')
        size_entry = ttk.Entry(config_frame, textvariable=self.matrix_sizes_var)
        size_entry.grid(row=0, column=1, sticky=tk.EW)
        self.create_tooltip(size_entry, "Comma-separated list of matrix sizes to test (e.g., 100, 250, 500)")
        
        # Worker counts
        ttk.Label(config_frame, text='Worker Counts:').grid(row=1, column=0, sticky=tk.W)
        self.worker_counts_var = tk.StringVar(value='1, 2, 4')
        worker_entry = ttk.Entry(config_frame, textvariable=self.worker_counts_var)
        worker_entry.grid(row=1, column=1, sticky=tk.EW)
        self.create_tooltip(worker_entry, "Comma-separated list of worker counts (e.g., 1, 2, 4)")
        
        # Block size
        ttk.Label(config_frame, text='Block Size:').grid(row=2, column=0, sticky=tk.W)
        self.block_size_var = tk.StringVar(value='32')
        block_entry = ttk.Entry(config_frame, textvariable=self.block_size_var)
        block_entry.grid(row=2, column=1, sticky=tk.EW)
        self.create_tooltip(block_entry, "Size of blocks for blocked multiplication methods (e.g., 32)")
        
        # Number of runs
        ttk.Label(config_frame, text='Number of Runs:').grid(row=3, column=0, sticky=tk.W)
        self.num_runs_var = tk.StringVar(value='3')
        runs_entry = ttk.Entry(config_frame, textvariable=self.num_runs_var)
        runs_entry.grid(row=3, column=1, sticky=tk.EW)
        self.create_tooltip(runs_entry, "Number of times to run each test for averaging (e.g., 3)")
        
        # Output directory
        ttk.Label(config_frame, text='Output Directory:').grid(row=4, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value='output')
        dir_entry = ttk.Entry(config_frame, textvariable=self.output_dir_var)
        dir_entry.grid(row=4, column=1, sticky=tk.EW)
        browse_btn = ttk.Button(config_frame, text='Browse...', command=self.browse_output_dir)
        browse_btn.grid(row=4, column=2)
        self.create_tooltip(dir_entry, "Directory where results and plots will be saved")
        
        # Log level
        ttk.Label(config_frame, text='Log Level:').grid(row=5, column=0, sticky=tk.W)
        self.log_level_var = tk.StringVar(value='INFO')
        log_combo = ttk.Combobox(
            config_frame,
            textvariable=self.log_level_var,
            values=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            state='readonly'
        )
        log_combo.grid(row=5, column=1, sticky=tk.EW)
        self.create_tooltip(log_combo, "Logging verbosity level")
        
        # Configure grid
        config_frame.columnconfigure(1, weight=1)
        
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget.
        
        Parameters
        ----------
        widget : tk.Widget
            The widget to add a tooltip to
        text : str
            The tooltip text
        """
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())
            
        widget.bind('<Enter>', show_tooltip)
        
    def create_methods_section(self):
        """Create the methods selection section of the GUI."""
        methods_frame = ttk.LabelFrame(self.root, text='Methods Selection', padding=10)
        methods_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Available methods
        methods = [
            'NumPy (Baseline)',
            'Threading',
            'Multiprocessing',
            'Shared Memory MP',
            'Sequential Blocked',
            'Parallel Blocked Shared',
            'ProcessPoolExecutor',
            'ThreadPoolExecutor',
            'Sequential'
        ]
        
        # Create checkbuttons
        for i, method in enumerate(methods):
            var = tk.BooleanVar(value=True)
            self.method_vars[method] = var
            ttk.Checkbutton(
                methods_frame,
                text=method,
                variable=var
            ).grid(row=i // 3, column=i % 3, sticky=tk.W, padx=5)
            
    def create_execution_section(self):
        """Create the execution control section of the GUI."""
        exec_frame = ttk.LabelFrame(self.root, text='Execution Control', padding=10)
        exec_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        self.load_config_btn = ttk.Button(
            exec_frame,
            text='Load Config',
            command=self.load_config
        )
        self.load_config_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_config_btn = ttk.Button(
            exec_frame,
            text='Save Config',
            command=self.save_config
        )
        self.save_config_btn.pack(side=tk.LEFT, padx=5)
        
        self.run_btn = ttk.Button(
            exec_frame,
            text='Run Benchmark',
            command=self.run_benchmark
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
    def create_visualization_section(self):
        """Create the visualization section of the GUI."""
        # Create visualization frame
        self.viz_frame = VisualizationFrame(self.root)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_output_section(self):
        """Create the output/logging section of the GUI."""
        output_frame = ttk.LabelFrame(self.root, text='Output / Logging', padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrolled text widget for logging
        self.log_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')
        
    def create_results_section(self):
        """Create the results display section of the GUI."""
        results_frame = ttk.LabelFrame(self.root, text='Results', padding=10)
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Placeholder for results buttons
        self.results_buttons_frame = ttk.Frame(results_frame)
        self.results_buttons_frame.pack(fill=tk.X)
        
    def browse_output_dir(self):
        """Open a directory browser dialog for selecting the output directory."""
        directory = filedialog.askdirectory(
            initialdir=self.output_dir_var.get(),
            title='Select Output Directory'
        )
        if directory:
            self.output_dir_var.set(directory)
            
    def clear_log(self):
        """Clear the log text widget."""
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')
        
    def load_default_config(self):
        """Load default configuration from config.json if it exists."""
        try:
            with open('config.json', 'r') as f:
                config_data = json.load(f)
                # Remove gpu_enabled if present
                config_data.pop('gpu_enabled', None)
                config = Config(**config_data)
                
            self.matrix_sizes_var.set(', '.join(map(str, config.test_sizes)))
            self.worker_counts_var.set(', '.join(map(str, config.worker_counts)))
            self.block_size_var.set(str(config.block_size))
            self.output_dir_var.set(config.output_dir)
            self.log_level_var.set(config.log_level)
        except Exception as e:
            logging.warning(f"Could not load default config: {e}")
            
    def load_config(self):
        """Load configuration from a JSON file."""
        filename = filedialog.askopenfilename(
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            title='Load Configuration'
        )
        if filename:
            try:
                config = Config.from_file(filename)
                self.matrix_sizes_var.set(', '.join(map(str, config.test_sizes)))
                self.worker_counts_var.set(', '.join(map(str, config.worker_counts)))
                self.block_size_var.set(str(config.block_size))
                self.output_dir_var.set(config.output_dir)
                self.log_level_var.set(config.log_level)
            except Exception as e:
                logging.error(f"Error loading config: {e}")
                
    def save_config(self):
        """Save current configuration to a JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            title='Save Configuration'
        )
        if filename:
            try:
                config = {
                    'test_sizes': [int(x.strip()) for x in self.matrix_sizes_var.get().split(',')],
                    'worker_counts': [int(x.strip()) for x in self.worker_counts_var.get().split(',')],
                    'block_size': int(self.block_size_var.get()),
                    'output_dir': self.output_dir_var.get(),
                    'log_level': self.log_level_var.get()
                }
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
            except Exception as e:
                logging.error(f"Error saving config: {e}")
                
    def setup_logging(self):
        """Set up logging to the GUI's text widget."""
        self.logging_handler = GuiLoggingHandler(self.log_text)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logging_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(self.logging_handler)
        root_logger.setLevel(getattr(logging, self.log_level_var.get()))
        
    def cleanup_logging(self):
        """Clean up logging handler."""
        if self.logging_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.logging_handler)
            self.logging_handler = None
            
    def update_results_buttons(self):
        """Update the results buttons based on available files."""
        # Clear existing buttons
        for widget in self.results_buttons_frame.winfo_children():
            widget.destroy()
            
        # Look for plot files
        output_dir = Path(self.output_dir_var.get())
        if output_dir.exists():
            plot_files = list(output_dir.glob('performance_comparison_*.png'))
            for plot_file in plot_files:
                ttk.Button(
                    self.results_buttons_frame,
                    text=f'View {plot_file.name}',
                    command=lambda f=plot_file: os.startfile(f)
                ).pack(side=tk.LEFT, padx=5)
                
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
            
    def enable_controls(self):
        """Enable control buttons."""
        self.run_btn.configure(state='normal')
        self.load_config_btn.configure(state='normal')
        self.save_config_btn.configure(state='normal')
        
    def disable_controls(self):
        """Disable control buttons."""
        self.run_btn.configure(state='disabled')
        self.load_config_btn.configure(state='disabled')
        self.save_config_btn.configure(state='disabled')
        
    def validate_inputs(self):
        """Validate user inputs before running benchmark."""
        try:
            # Check matrix sizes
            sizes = [int(x.strip()) for x in self.matrix_sizes_var.get().split(',')]
            if not sizes or any(size <= 0 for size in sizes):
                raise ValueError("Matrix sizes must be positive integers")
            
            # Check worker counts
            workers = [int(x.strip()) for x in self.worker_counts_var.get().split(',')]
            if not workers or any(count <= 0 for count in workers):
                raise ValueError("Worker counts must be positive integers")
            
            # Check block size
            block_size = int(self.block_size_var.get())
            if block_size <= 0:
                raise ValueError("Block size must be a positive integer")
            
            # Check number of runs
            num_runs = int(self.num_runs_var.get())
            if num_runs <= 0:
                raise ValueError("Number of runs must be a positive integer")
            
            # Check if any method is selected
            if not any(var.get() for var in self.method_vars.values()):
                raise ValueError("At least one method must be selected")
            
            return True
            
        except ValueError as e:
            logging.error(f"Validation error: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during validation: {e}")
            return False

    def run_benchmark(self):
        """Start the benchmark in a separate thread."""
        if not self.validate_inputs():
            return
            
        # Disable buttons during benchmark
        self.run_btn.configure(state='disabled')
        
        # Clear log
        self.clear_log()
        
        # Setup logging
        self.setup_logging()
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_dir_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start benchmark thread
        self.benchmark_thread = threading.Thread(target=self.run_benchmark_thread)
        self.benchmark_thread.start()


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = MatrixMultiplicationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main() 
