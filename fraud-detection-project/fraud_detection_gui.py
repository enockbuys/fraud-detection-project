import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sqlite3
import os
import sys
import subprocess
import io
import pandas as pd
from datetime import datetime
from pathlib import Path
import queue
from src.common.classification_summary import classification_summary
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = SCRIPT_DIR
COMMON_DIR = SRC_DIR / "common"
PIPELINE1_DIR = SRC_DIR / "pipeline1"
PIPELINE2_DIR = SRC_DIR / "pipeline2"
PIPELINE3_DIR = SRC_DIR / "pipeline3"

# Add paths to sys.path
for path in [SRC_DIR, COMMON_DIR, PIPELINE1_DIR, PIPELINE2_DIR, PIPELINE3_DIR]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Source directory: {SRC_DIR}")

# Function to install missing packages
def install_missing_package(package_name):
    #Install a missing package using pip"
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully!")
        return True
    except Exception as e:
        print(f"Failed to install {package_name}: {e}")
        return False


# Import backend modules
BACKEND_AVAILABLE = False
train_gan_p1 = None
main_pipeline1 = None
train_gan_tf = None
main_pipeline2 = None
train_gan_ctgan = None
main_pipeline3 = None
compare_pipelines = None
predict = None

try:
    from pipeline1.train_gan import main as train_gan_p1_main
    from pipeline1.main_pipeline1 import main as main_pipeline1_main

    train_gan_p1 = type('obj', (), {'main': train_gan_p1_main})
    main_pipeline1 = type('obj', (), {'main': main_pipeline1_main})
    print("Pipeline 1 modules imported successfully")

    from pipeline2.train_gan_tf import main as train_gan_tf_main
    from pipeline2.main_pipeline2 import main as main_pipeline2_main

    train_gan_tf = type('obj', (), {'main': train_gan_tf_main})
    main_pipeline2 = type('obj', (), {'main': main_pipeline2_main})
    print("Pipeline 2 modules imported successfully")

    from pipeline3.train_gan_ctgan import main as train_gan_ctgan_main
    from pipeline3.main_pipeline3 import main as main_pipeline3_main

    train_gan_ctgan = type('obj', (), {'main': train_gan_ctgan_main})
    main_pipeline3 = type('obj', (), {'main': main_pipeline3_main})
    print("Pipeline 3 modules imported successfully")

    try:
        import compare_pipelines
        print("Comparison module imported successfully")
    except ImportError as e:
        if "tabulate" in str(e):
            print("tabulate package missing, attempting to install...")
            if install_missing_package("tabulate"):
                import compare_pipelines
                print("Comparison module imported successfully")
            else:
                compare_pipelines = None
        else:
            raise

    import predict
    print("Prediction module imported successfully")

    BACKEND_AVAILABLE = True
    print("All backend modules loaded successfully")
except Exception as e:
    print(f"Error loading backend modules: {e}")
    BACKEND_AVAILABLE = False


class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Login")
        self.root.geometry("300x250")
        self.root.resizable(False, False)

        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Username:").pack(pady=5)
        self.username = ttk.Entry(frame)
        self.username.pack(pady=5)

        ttk.Label(frame, text="Password:").pack(pady=5)
        self.password = ttk.Entry(frame, show="*")
        self.password.pack(pady=5)

        ttk.Button(frame, text="Login", command=self.login).pack(pady=10)

    def login(self):
        if self.username.get() == "admin" and self.password.get() == "password":
            self.root.destroy()
            MainWindow().run()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials")

    def run(self):
        self.root.mainloop()

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fraud Detection System")
        self.root.geometry("900x600")
        self.root.minsize(800, 500)

        # Database path
        self.db_path = PROJECT_ROOT / "src" / "fraud_detection.db"

        # Running processes and flags
        self.running_threads = {}
        self.stop_flags = {'gan': False, 'pipeline': False, 'comparison': False}
        self.processes = {'gan': None, 'pipeline': None}

        # Queues for output
        self.gan_queue = queue.Queue()
        self.pipeline_queue = queue.Queue()
        self.comparison_queue = queue.Queue()
        self.prediction_queue = queue.Queue()

        self.setup_ui()
        self.start_output_polling()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # GAN Training tab
        self.init_gan_training_tab()

        # Pipeline Training tab
        self.init_pipeline_training_tab()

        # Comparison tab
        self.init_comparison_tab()

        # Prediction tab
        self.init_prediction_tab()

        # Database Viewer tab
        self.init_database_viewer_tab()

        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Logout", command=self.logout)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="System Info", command=self.show_system_info)
        help_menu.add_command(label="About", command=self.show_about)

    def init_gan_training_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="GAN Training")

        # Pipeline selection
        select_frame = ttk.Frame(tab)
        select_frame.pack(fill=tk.X, pady=10)

        ttk.Label(select_frame, text="Select Pipeline:").pack(side=tk.LEFT, padx=5)
        self.gan_pipeline_var = tk.StringVar(value="Pipeline 1")
        pipelines = ["Pipeline 1", "Pipeline 2", "Pipeline 3"]
        self.gan_pipeline_combo = ttk.Combobox(select_frame, textvariable=self.gan_pipeline_var, values=pipelines, state="readonly")
        self.gan_pipeline_combo.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)

        self.train_gan_btn = ttk.Button(button_frame, text="Train GAN", command=lambda: self.start_gan_training(self.gan_pipeline_var.get()))
        self.train_gan_btn.pack(side=tk.LEFT, padx=5)

        self.stop_gan_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_gan_training, state=tk.DISABLED)
        self.stop_gan_btn.pack(side=tk.LEFT, padx=5)

        # Output
        self.gan_output = scrolledtext.ScrolledText(tab, font=('Courier', 10), height=20)
        self.gan_output.pack(fill=tk.BOTH, expand=True, pady=10)
        self.gan_output.tag_configure("error", foreground="red")

    def init_pipeline_training_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Pipeline Training")

        # Pipeline selection
        select_frame = ttk.Frame(tab)
        select_frame.pack(fill=tk.X, pady=10)

        ttk.Label(select_frame, text="Select Pipeline:").pack(side=tk.LEFT, padx=5)
        self.pipeline_var = tk.StringVar(value="Pipeline 1")
        pipelines = ["Pipeline 1", "Pipeline 2", "Pipeline 3"]
        self.pipeline_combo = ttk.Combobox(select_frame, textvariable=self.pipeline_var, values=pipelines, state="readonly")
        self.pipeline_combo.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)

        self.train_pipeline_btn = ttk.Button(button_frame, text="Train Pipeline", command=lambda: self.start_pipeline_training(self.pipeline_var.get()))
        self.train_pipeline_btn.pack(side=tk.LEFT, padx=5)

        self.stop_pipeline_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_pipeline_training, state=tk.DISABLED)
        self.stop_pipeline_btn.pack(side=tk.LEFT, padx=5)

        # Output
        self.pipeline_output = scrolledtext.ScrolledText(tab, font=('Courier', 10), height=20)
        self.pipeline_output.pack(fill=tk.BOTH, expand=True, pady=10)
        self.pipeline_output.tag_configure("error", foreground="red")

    def init_comparison_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Model Comparison")

        # Buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)

        self.compare_btn = ttk.Button(button_frame, text="Run Comparison", command=self.run_comparison)
        self.compare_btn.pack(side=tk.LEFT, padx=5)

        self.stop_compare_btn = ttk.Button(button_frame, text="Stop Comparison", command=self.stop_comparison, state=tk.DISABLED)
        self.stop_compare_btn.pack(side=tk.LEFT, padx=5)

        # Output
        self.comparison_output = scrolledtext.ScrolledText(tab, font=('Courier', 10), height=20)
        self.comparison_output.pack(fill=tk.BOTH, expand=True, pady=10)

    def init_prediction_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Prediction")

        # Model list
        self.model_tree = ttk.Treeview(tab, columns=("Pipeline", "Name", "Size (KB)", "Modified"), show="headings")
        self.model_tree.heading("Pipeline", text="Pipeline")
        self.model_tree.heading("Name", text="Name")
        self.model_tree.heading("Size (KB)", text="Size (KB)")
        self.model_tree.heading("Modified", text="Modified")
        self.model_tree.pack(fill=tk.BOTH, expand=True, pady=10)

        # Buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Refresh Models", command=self.refresh_models).pack(side=tk.LEFT, padx=5)
        self.predict_btn = ttk.Button(button_frame, text="Run Prediction", command=self.run_prediction, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        # Manual prediction section
        manual_frame = ttk.Frame(tab)
        manual_frame.pack(fill=tk.X, pady=10)

        ttk.Label(manual_frame, text="Enter Record (comma-separated, 30 values: Class, V1-V28, Amount):").pack(side=tk.LEFT, padx=5)
        self.manual_record_entry = tk.Text(manual_frame, height=3, width=50)
        self.manual_record_entry.pack(side=tk.LEFT, padx=5)

        self.manual_predict_btn = ttk.Button(manual_frame, text="Predict Single Record", command=self.run_manual_prediction, state=tk.DISABLED)
        self.manual_predict_btn.pack(side=tk.LEFT, padx=5)

        # Output
        self.prediction_output = scrolledtext.ScrolledText(tab, font=('Courier', 10), height=10)
        self.prediction_output.pack(fill=tk.BOTH, expand=True, pady=10)

        self.model_tree.bind("<<TreeviewSelect>>", self.on_model_select)
        self.refresh_models()

    def init_database_viewer_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Database Viewer")

        # Table selection
        select_frame = ttk.Frame(tab)
        select_frame.pack(fill=tk.X, pady=10)

        ttk.Label(select_frame, text="Select Table:").pack(side=tk.LEFT, padx=5)
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(select_frame, textvariable=self.table_var, state="readonly")
        self.table_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(select_frame, text="Refresh Tables", command=self.refresh_tables).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="View Data", command=self.view_table_data).pack(side=tk.LEFT, padx=5)

        # Data display
        self.data_tree = ttk.Treeview(tab, show="headings")
        self.data_tree.pack(fill=tk.BOTH, expand=True, pady=10)

        self.refresh_tables()

    def refresh_tables(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            self.table_combo['values'] = tables
            if tables:
                self.table_var.set(tables[0])
            conn.close()
        except Exception as e:
            messagebox.showerror("Database Error", str(e))

    def view_table_data(self):
        table = self.table_var.get()
        if not table:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            conn.close()

            self.data_tree.delete(*self.data_tree.get_children())
            self.data_tree["columns"] = list(df.columns)
            for col in df.columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)

            for row in df.itertuples(index=False):
                self.data_tree.insert("", "end", values=row)

        except Exception as e:
            messagebox.showerror("Database Error", str(e))

    def start_output_polling(self):
        self.poll_gan_output()
        self.poll_pipeline_output()
        self.poll_comparison_output()
        self.poll_prediction_output()

    def poll_gan_output(self):
        try:
            while not self.gan_queue.empty():
                line = self.gan_queue.get_nowait()
                if "ERROR" in line.upper():
                    self.gan_output.insert(tk.END, line + "\n", "error")
                else:
                    self.gan_output.insert(tk.END, line + "\n")
                self.gan_output.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_gan_output)

    def poll_pipeline_output(self):
        try:
            while not self.pipeline_queue.empty():
                line = self.pipeline_queue.get_nowait()
                if "ERROR" in line.upper():
                    self.pipeline_output.insert(tk.END, line + "\n", "error")
                else:
                    self.pipeline_output.insert(tk.END, line + "\n")
                self.pipeline_output.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_pipeline_output)

    def poll_comparison_output(self):
        try:
            while not self.comparison_queue.empty():
                line = self.comparison_queue.get_nowait()
                self.comparison_output.insert(tk.END, line + "\n")
                self.comparison_output.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_comparison_output)

    def poll_prediction_output(self):
        try:
            while not self.prediction_queue.empty():
                line = self.prediction_queue.get_nowait()
                self.prediction_output.insert(tk.END, line + "\n")
                self.prediction_output.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_prediction_output)

    def get_script_path(self, pipeline, is_gan=True):
        if pipeline == "Pipeline 1":
            return PIPELINE1_DIR / ('train_gan.py' if is_gan else 'main_pipeline1.py')
        elif pipeline == "Pipeline 2":
            return PIPELINE2_DIR / ('train_gan_tf.py' if is_gan else 'main_pipeline2.py')
        elif pipeline == "Pipeline 3":
            return PIPELINE3_DIR / ('train_gan_ctgan.py' if is_gan else 'main_pipeline3.py')
        return None

    def start_gan_training(self, pipeline):
        script_path = self.get_script_path(pipeline, is_gan=True)
        if not script_path or not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return

        self.gan_output.delete(1.0, tk.END)
        self.train_gan_btn.config(state=tk.DISABLED)
        self.stop_gan_btn.config(state=tk.NORMAL)

        def run_training():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=False  # Binary mode for both
                )
                self.processes['gan'] = process

                # Read stdout in binary and decode
                def read_stdout():
                    for byte_line in iter(lambda: process.stdout.readline(4096), b''):
                        try:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            if line:
                                self.gan_queue.put(line)
                        except UnicodeDecodeError:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            self.gan_queue.put(f"WARNING: Non-decodable output in stdout: {line}")
                    process.stdout.close()

                # Read stderr in binary and decode
                def read_stderr():
                    for byte_line in iter(lambda: process.stderr.readline(4096), b''):
                        try:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            if line:
                                self.gan_queue.put(f"ERROR: {line}")
                        except UnicodeDecodeError:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            self.gan_queue.put(f"WARNING: Non-decodable output in stderr: {line}")
                    process.stderr.close()

                threading.Thread(target=read_stdout, daemon=True).start()
                threading.Thread(target=read_stderr, daemon=True).start()

                process.wait()
            except Exception as e:
                self.gan_queue.put(f"ERROR: {str(e)}")
            finally:
                self.root.after(0, lambda: self.train_gan_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_gan_btn.config(state=tk.DISABLED))
                self.processes['gan'] = None

        threading.Thread(target=run_training, daemon=True).start()

    def stop_gan_training(self):
        if self.processes['gan']:
            self.processes['gan'].terminate()
            self.processes['gan'].wait()
            self.gan_queue.put("Training stopped by user.")
            self.train_gan_btn.config(state=tk.NORMAL)
            self.stop_gan_btn.config(state=tk.DISABLED)
            self.processes['gan'] = None

    def start_pipeline_training(self, pipeline):
        script_path = self.get_script_path(pipeline, is_gan=False)
        if not script_path or not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return

        self.pipeline_output.delete(1.0, tk.END)
        self.train_pipeline_btn.config(state=tk.DISABLED)
        self.stop_pipeline_btn.config(state=tk.NORMAL)

        def run_training():
            try:
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=False  # Binary mode for both
                )
                self.processes['pipeline'] = process

                # Read stdout in binary and decode
                def read_stdout():
                    for byte_line in iter(lambda: process.stdout.readline(4096), b''):
                        try:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            if line:
                                self.pipeline_queue.put(line)
                        except UnicodeDecodeError:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            self.pipeline_queue.put(f"WARNING: Non-decodable output in stdout: {line}")
                    process.stdout.close()

                # Read stderr in binary and decode
                def read_stderr():
                    for byte_line in iter(lambda: process.stderr.readline(4096), b''):
                        try:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            if line:
                                self.pipeline_queue.put(f"ERROR: {line}")
                        except UnicodeDecodeError:
                            line = byte_line.decode('utf-8', errors='replace').strip()
                            self.pipeline_queue.put(f"WARNING: Non-decodable output in stderr: {line}")
                    process.stderr.close()

                threading.Thread(target=read_stdout, daemon=True).start()
                threading.Thread(target=read_stderr, daemon=True).start()

                process.wait()
            except Exception as e:
                self.pipeline_queue.put(f"ERROR: {str(e)}")
            finally:
                self.root.after(0, lambda: self.train_pipeline_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_pipeline_btn.config(state=tk.DISABLED))
                self.processes['pipeline'] = None

        threading.Thread(target=run_training, daemon=True).start()

    def stop_pipeline_training(self):
        if self.processes['pipeline']:
            self.processes['pipeline'].terminate()
            self.processes['pipeline'].wait()
            self.pipeline_queue.put("Training stopped by user.")
            self.train_pipeline_btn.config(state=tk.NORMAL)
            self.stop_pipeline_btn.config(state=tk.DISABLED)
            self.processes['pipeline'] = None

    def run_comparison(self):
        if not compare_pipelines:
            messagebox.showerror("Error", "Comparison module not available")
            return

        self.comparison_output.delete(1.0, tk.END)
        self.compare_btn.config(state=tk.DISABLED)
        self.stop_compare_btn.config(state=tk.NORMAL)
        self.stop_flags['comparison'] = False

        def run_comp():
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

                results = compare_pipelines.compare_pipelines_table()
                if results:
                    compare_pipelines.generate_executive_summary(results, Path(compare_pipelines.__file__).parent.parent / "results" / "comparison")

                # Get captured output
                output = sys.stdout.getvalue()
                error = sys.stderr.getvalue()

                for line in output.splitlines():
                    self.comparison_queue.put(line)
                for line in error.splitlines():
                    self.comparison_queue.put(f"ERROR: {line}")

            except Exception as e:
                self.comparison_queue.put(f"ERROR: {str(e)}")
                self.root.after(0, lambda: messagebox.showerror("Comparison Error", str(e)))
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                self.root.after(0, lambda: self.compare_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_compare_btn.config(state=tk.DISABLED))

        threading.Thread(target=run_comp, daemon=True).start()

    def stop_comparison(self):
        self.stop_flags['comparison'] = True
        self.comparison_queue.put("Comparison stopped by user.")
        self.compare_btn.config(state=tk.NORMAL)
        self.stop_compare_btn.config(state=tk.DISABLED)

    def refresh_models(self):
        self.model_tree.delete(*self.model_tree.get_children())
        model_files = predict.scan_model_files()
        print(f"Number of model files found: {len(model_files)}")  # Debug print
        for model in model_files:
            self.model_tree.insert("", "end", values=(
                model['pipeline'],
                model['name'],
                f"{model['size_kb']:.1f}",
                model['modified'].strftime("%Y-%m-%d %H:%M:%S")
            ))
        self.predict_btn.config(state=tk.DISABLED)  # Reset to disabled until selection

    def on_model_select(self, event):
        selected = self.model_tree.selection()
        print(f"Selected items: {selected}")  # Debug print
        if selected:
            self.predict_btn.config(state=tk.NORMAL)
            self.manual_predict_btn.config(state=tk.NORMAL)  # Enable Predict Single Record button
        else:
            self.predict_btn.config(state=tk.DISABLED)
            self.manual_predict_btn.config(state=tk.DISABLED)  # Disable if no selection

    def run_prediction(self):
        selected = self.model_tree.selection()
        if not selected:
            return

        item = self.model_tree.item(selected[0])
        values = item['values']
        model_info = {
            'pipeline': values[0],
            'name': values[1],
            'path': os.path.join(PROJECT_ROOT, "src", "results", values[0], values[1])
        }

        self.prediction_output.delete(1.0, tk.END)
        self.predict_btn.config(state=tk.DISABLED)

        def run_pred():
            try:
                model = predict.load_model(model_info['path'])
                X_predict, y_predict = predict.load_predict_data()
                predictions, probabilities = predict.predict_with_model(model, X_predict)
                metrics = classification_summary(y_predict, predictions, probabilities)

                output = f"PREDICTION RESULTS\n"
                output += f"Model: {model_info['pipeline']}/{model_info['name']}\n"
                output += f"Samples: {len(X_predict)}\n"
                output += f"Accuracy: {metrics['accuracy']:.4f}\n"
                output += f"Precision: {metrics['precision']:.4f}\n"
                output += f"Recall: {metrics['recall']:.4f}\n"
                output += f"F1-Score: {metrics['f1_score']:.4f}\n"

                if 'roc_auc' in metrics:
                    output += f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
                if 'pr_auc' in metrics:
                    output += f"PR-AUC: {metrics['pr_auc']:.4f}\n"

                output += f"\nConfusion Matrix:\n"
                output += f"True Positives: {metrics['true_positives']}\n"
                output += f"False Positives: {metrics['false_positives']}\n"
                output += f"False Negatives: {metrics['false_negatives']}\n"
                output += f"True Negatives: {metrics['true_negatives']}\n"

                self.prediction_queue.put(output)

                output_dir = os.path.dirname(model_info['path'])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")

                results_df = pd.DataFrame({
                    'actual': y_predict,
                    'predicted': predictions
                })

                if probabilities is not None:
                    if probabilities.ndim > 1:
                        results_df['probability_0'] = probabilities[:, 0]
                        results_df['probability_1'] = probabilities[:, 1]
                    else:
                        results_df['probability'] = probabilities

                results_df.to_csv(output_file, index=False)
                self.prediction_queue.put(f"\nPredictions saved to: {output_file}")

            except FileNotFoundError as e:
                self.prediction_queue.put(f"ERROR: {str(e)}")
            except Exception as e:
                self.prediction_queue.put(f"ERROR: {str(e)}")
            finally:
                self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

        threading.Thread(target=run_pred, daemon=True).start()

    def run_manual_prediction(self):
        record_str = self.manual_record_entry.get("1.0", tk.END).strip()
        if not record_str:
            messagebox.showerror("Error", "Please enter a record.")
            return

        selected = self.model_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a model.")
            return

        item = self.model_tree.item(selected[0])
        values = item['values']
        model_path = os.path.join(PROJECT_ROOT, "src", "results", values[0], values[1])
        pipeline = values[0]

        self.prediction_output.delete(1.0, tk.END)
        self.manual_predict_btn.config(state=tk.DISABLED)

        def manual_transform(record_str):
            try:
                record = [float(x.strip()) for x in record_str.split(',')]
                if len(record) != 30:
                    raise ValueError("Invalid record format. Expected 30 values: Class, V1-V28, Amount.")

                # Extract features as per Preprocessor
                Class = record[0]  # Ignore for prediction, but included for consistency
                V = record[1:29]   # V1 to V28
                Amount = record[29]

                Hour = 0.0  # Assume Time=0 for manual input, as in Preprocessor
                log_amount = np.log1p(max(Amount, 0))

                X = np.array(V + [Hour, log_amount], dtype=np.float32)
                return X
            except Exception as e:
                raise ValueError(f"Error transforming record: {str(e)}")

        def run_manual_pred():
            try:
                X = manual_transform(record_str)
                model = predict.load_model(model_path)

                # Handle Pipeline 1's 1D input requirement
                if pipeline == "Pipeline 1":
                    prediction = model.predict(X)  # No reshape for 1D models
                else:
                    prediction = model.predict(X.reshape(1, -1))[0]

                if hasattr(model, 'predict_proba'):
                    if pipeline == "Pipeline 1":
                        proba = model.predict_proba(X)
                    else:
                        proba = model.predict_proba(X.reshape(1, -1))
                    if proba.ndim == 2:
                        probability = proba[0, 1] if pipeline != "Pipeline 1" else proba[1]
                    else:
                        probability = proba[0]
                else:
                    probability = None

                output = f"MANUAL PREDICTION RESULTS\n"
                output += f"Model: {values[0]}/{values[1]}\n"
                output += f"Prediction: {'Fraud' if prediction == 1 else 'Legit'}\n"
                if probability is not None:
                    output += f"Probability of Fraud: {probability:.4f}\n"

                self.prediction_queue.put(output)

            except Exception as e:
                self.prediction_queue.put(f"ERROR: {str(e)}")
            finally:
                self.root.after(0, lambda: self.manual_predict_btn.config(state=tk.NORMAL))

        threading.Thread(target=run_manual_pred, daemon=True).start()

    def show_system_info(self):
        info = f"""System Diagnostic Information

Script Location: {SCRIPT_DIR}
Project Root: {PROJECT_ROOT}
Source Directory: {SRC_DIR}
Source Exists: {SRC_DIR.exists()}

Database Path: {self.db_path}
Database Exists: {self.db_path.exists()}

Backend Available: {BACKEND_AVAILABLE}

Modules Loaded:
  train_gan_p1: {train_gan_p1 is not None}
  main_pipeline1: {main_pipeline1 is not None}
  train_gan_tf: {train_gan_tf is not None}
  main_pipeline2: {main_pipeline2 is not None}
  train_gan_ctgan: {train_gan_ctgan is not None}
  main_pipeline3: {main_pipeline3 is not None}
  compare_pipelines: {compare_pipelines is not None}
  predict: {predict is not None}

Active Processes:
  GAN Training: {'Running' if self.processes['gan'] and self.processes['gan'].poll() is None else 'Idle'}
  Pipeline Training: {'Running' if self.processes['pipeline'] and self.processes['pipeline'].poll() is None else 'Idle'}
"""
        messagebox.showinfo("System Information", info)

    def show_about(self):
        about_text = """Fraud Detection System v2.0

Features:
 Real-time training output
 Process termination control
 Three pipeline implementations
 GAN-based data augmentation
 Comprehensive model comparison
 Interactive prediction interface

Developed for Credit Card Fraud Detection
Uses: Random Forest, XGBoost, GANs, CTGAN"""
        messagebox.showinfo("About", about_text)

    def logout(self):
        # Check for running processes
        active_processes = []
        if self.processes['gan'] and self.processes['gan'].poll() is None:
            active_processes.append("GAN Training")
        if self.processes['pipeline'] and self.processes['pipeline'].poll() is None:
            active_processes.append("Pipeline Training")

        if active_processes:
            msg = f"The following processes are still running:\n"
            msg += "\n".join(f"  - {p}" for p in active_processes)
            msg += "\n\nAre you sure you want to logout? This will terminate all processes."

            if not messagebox.askyesno("Active Processes", msg):
                return
        elif not messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            return

        # Terminate processes
        for key in ['gan', 'pipeline']:
            if self.processes[key] and self.processes[key].poll() is None:
                self.processes[key].terminate()
                self.processes[key].wait()

        self.root.destroy()
        LoginWindow().run()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.logout)
        self.root.mainloop()

if __name__ == "__main__":
    login = LoginWindow()
    login.run()