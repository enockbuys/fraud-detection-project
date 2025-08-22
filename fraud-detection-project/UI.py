import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import time
import numpy as np
from preprocessor import Preprocessor
from my_random_forest import MyRandomForest
from classification_summary import classification_summary
from db_manager import DBManager
from conditional_gan import ConditionalGAN


class FraudDetectionGUI:
    def __init__(self):

        self.window = tk.Tk()
        self.window.title("Fraud Detection System")
        self.window.geometry("700x700")
        self.window.configure(bg="#f0f0f0")


        self.file_path = ""
        self.features_train = None
        self.labels_train = None
        self.features_validation = None
        self.labels_validation = None
        self.features_test = None
        self.labels_test = None
        self.is_training = False
        self.db_manager = None

        self.create_header()
        self.create_file_section()
        self.create_training_section()
        self.create_results_section()

    def create_header(self):
        header_frame = tk.Frame(self.window, bg="#2c3e50", height=80)
        header_frame.pack(fill="x", padx=10, pady=10)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="Fraud Detection System",
                               font=("Arial", 20, "bold"),
                               bg="#2c3e50", fg="white")
        title_label.pack(pady=20)

    def create_file_section(self):

        file_frame = tk.LabelFrame(self.window, text="Step 1: Load Your Data",
                                   font=("Arial", 12, "bold"),
                                   bg="#f0f0f0", padx=20, pady=20)
        file_frame.pack(fill="x", padx=20, pady=10)

        self.file_button = tk.Button(file_frame, text="Choose CSV File",
                                     font=("Arial", 11),
                                     bg="#3498db", fg="white",
                                     width=15, height=2,
                                     command=self.select_file)
        self.file_button.pack(pady=10)

        self.file_info_label = tk.Label(file_frame, text="No file selected",
                                        font=("Arial", 10),
                                        bg="#f0f0f0", fg="#7f8c8d")
        self.file_info_label.pack()

        self.data_info_frame = tk.Frame(file_frame, bg="#f0f0f0")

        self.stats_label = tk.Label(self.data_info_frame, text="",
                                    font=("Arial", 10),
                                    bg="#f0f0f0", fg="#2c3e50")
        self.stats_label.pack(pady=10)

    def create_training_section(self):
        training_frame = tk.LabelFrame(self.window, text="Step 2: Train Model",
                                       font=("Arial", 12, "bold"),
                                       bg="#f0f0f0", padx=20, pady=20)
        training_frame.pack(fill="x", padx=20, pady=10)

        model_frame = tk.Frame(training_frame, bg="#f0f0f0")
        model_frame.pack(pady=10)

        tk.Label(model_frame, text="Model Type:", font=("Arial", 10),
                 bg="#f0f0f0").pack(side="left")

        self.model_var = tk.StringVar(value="Random Forest")
        model_options = ["Random Forest", "Random Forest + CGAN"]

        self.model_menu = tk.OptionMenu(model_frame, self.model_var, *model_options)
        self.model_menu.config(font=("Arial", 10), width=20)
        self.model_menu.pack(side="left", padx=10)

        params_frame = tk.Frame(training_frame, bg="#f0f0f0")
        params_frame.pack(pady=10)

        tk.Label(params_frame, text="Trees:", font=("Arial", 10),
                 bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.trees_entry = tk.Entry(params_frame, width=8, font=("Arial", 10))
        self.trees_entry.insert(0, "10")
        self.trees_entry.grid(row=0, column=1, padx=5)

        tk.Label(params_frame, text="Max Depth:", font=("Arial", 10),
                 bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.depth_entry = tk.Entry(params_frame, width=8, font=("Arial", 10))
        self.depth_entry.insert(0, "10")
        self.depth_entry.grid(row=0, column=3, padx=5)

        cgan_frame = tk.Frame(training_frame, bg="#f0f0f0")
        cgan_frame.pack(pady=10)

        tk.Label(cgan_frame, text="CGAN Epochs:", font=("Arial", 10),
                 bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.epochs_entry = tk.Entry(cgan_frame, width=8, font=("Arial", 10))
        self.epochs_entry.insert(0, "500")
        self.epochs_entry.grid(row=0, column=1, padx=5)

        tk.Label(cgan_frame, text="Batch Size:", font=("Arial", 10),
                 bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.batch_entry = tk.Entry(cgan_frame, width=8, font=("Arial", 10))
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=0, column=3, padx=5)

        self.train_button = tk.Button(training_frame, text="Start Training",
                                      font=("Arial", 12, "bold"),
                                      bg="#27ae60", fg="white",
                                      width=20, height=2,
                                      command=self.start_training)
        self.train_button.pack(pady=15)

        self.progress_label = tk.Label(training_frame, text="",
                                       font=("Arial", 10),
                                       bg="#f0f0f0", fg="#e74c3c")
        self.progress_label.pack()

    def create_results_section(self):
        results_frame = tk.LabelFrame(self.window, text="Step 3: View Results",
                                      font=("Arial", 12, "bold"),
                                      bg="#f0f0f0", padx=20, pady=20)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=100, width=60,
                                                      font=("Courier", 9),
                                                      bg="white", fg="#2c3e50",
                                                      relief="sunken", bd=2)
        self.results_text.pack(pady=10, fill="both", expand=True)

        self.results_text.insert("1.0", "Results will appear here after training...\n\n" +
                                 "Random Forest: Shows validation and test results\n" +
                                 "Random Forest + CGAN: Shows results for multiple augmentation percentages (5%, 10%, 15%)\n" +
                                 "Each will show detailed metrics for both validation and test sets\n\n" +
                                 "Metrics explained:\n" +
                                 "• Accuracy: How often the model is correct\n" +
                                 "• Precision: Of predicted frauds, how many are real\n" +
                                 "• Recall: Of real frauds, how many are caught\n" +
                                 "• F1 Score: Balance between precision and recall")
        self.results_text.config(state="disabled")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            self.file_path = file_path
            filename = file_path.split("/")[-1]
            self.file_info_label.config(text=f"Selected: {filename}", fg="#27ae60")

            self.process_file()

    def process_file(self):
        try:
            self.file_info_label.config(text="Processing file", fg="#f39c12")
            self.window.update()

            preprocessor = Preprocessor(self.file_path)
            (self.features_train, self.labels_train,
             self.features_validation, self.labels_validation,
             self.features_test, self.labels_test) = preprocessor.clean_and_split()

            total_samples = len(self.labels_train) + len(self.labels_validation) + len(self.labels_test)
            fraud_cases = np.sum(self.labels_train) + np.sum(self.labels_validation) + np.sum(self.labels_test)
            features_count = self.features_train.shape[1]

            stats_text = f"Data loaded successfully!\n"
            stats_text += f"Total samples: {total_samples}\n"
            stats_text += f"Features: {features_count}\n"
            stats_text += f"Training samples: {len(self.labels_train)}\n"
            stats_text += f"Validation samples: {len(self.labels_validation)}\n"
            stats_text += f"Test samples: {len(self.labels_test)}\n"
            stats_text += f"Total fraud cases: {fraud_cases}\n"
            stats_text += f"Total normal cases: {total_samples - fraud_cases}"

            self.stats_label.config(text=stats_text)
            self.data_info_frame.pack(pady=10)

            self.file_info_label.config(text=f"Ready to train!", fg="#27ae60")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")
            self.file_info_label.config(text="Error loading file", fg="#e74c3c")

    def start_training(self):
        if self.features_train is None:
            messagebox.showwarning("Warning", "Please select and load a CSV file first!")
            return

        if self.is_training:
            messagebox.showinfo("Info", "Training is already in progress!")
            return

        self.is_training = True
        self.train_button.config(text="Training...", state="disabled")

        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()

    def run_training(self):
        try:
            num_trees = int(self.trees_entry.get())
            max_depth = int(self.depth_entry.get())
            model_type = self.model_var.get()

            self.results_text.config(state="normal")
            self.results_text.delete("1.0", "end")
            self.results_text.config(state="disabled")

            if model_type == "Random Forest":
                self.train_random_forest(num_trees, max_depth)
            else:
                self.train_cgan_random_forest(num_trees, max_depth)

        except Exception as e:
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")
        finally:
            self.is_training = False
            self.train_button.config(text="Start Training", state="normal")
            self.progress_label.config(text="")

    def train_random_forest(self, num_trees, max_depth):
        self.progress_label.config(text="Training Random Forest...")
        self.window.update()

        rf_model = MyRandomForest(number_of_trees=num_trees,
                                  max_depth=max_depth,
                                  min_samples_split=2)
        rf_model.fit(self.features_train, self.labels_train)

        predictions_validation = rf_model.predict(self.features_validation)
        predictions_test = rf_model.predict(self.features_test)

        self.display_random_forest_results(predictions_validation, predictions_test)

    def train_cgan_random_forest(self, num_trees, max_depth):
        epochs = int(self.epochs_entry.get())
        batch_size = int(self.batch_entry.get())

        self.progress_label.config(text="Training CGAN on fraud samples...")
        self.window.update()

        X_fraud = self.features_train[self.labels_train == 1]
        y_fraud = self.labels_train[self.labels_train == 1]

        def normalize(data):
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normalized = (data - mins) / ranges
            return normalized * 2 - 1

        def denormalize(data, original):
            mins = original.min(axis=0)
            maxs = original.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            denorm = (data + 1) / 2
            return denorm * ranges + mins

        X_fraud_normalized = normalize(X_fraud)

        cgan = ConditionalGAN(output_dim=self.features_train.shape[1])
        cgan.train(X_fraud_normalized, y_fraud.reshape(-1, 1), epochs=epochs, batch_size=batch_size)

        results = {}
        synthetic_percentages = [0.05, 0.10, 0.15]

        for i, percentage in enumerate(synthetic_percentages):
            self.progress_label.config(text=f"Testing {percentage * 100}% synthetic data ({i + 1}/3)...")
            self.window.update()

            n_synthetic = int(len(X_fraud) * percentage)
            X_synthetic = cgan.generate_samples(n_synthetic)
            X_synthetic = denormalize(X_synthetic, X_fraud)
            y_synthetic = np.ones(n_synthetic, dtype=np.int64)

            X_combined = np.vstack([self.features_train, X_synthetic])
            y_combined = np.concatenate([self.labels_train, y_synthetic])

            rf_model = MyRandomForest(number_of_trees=num_trees,
                                      max_depth=max_depth,
                                      min_samples_split=2)
            rf_model.fit(X_combined, y_combined)

            predictions_validation = rf_model.predict(self.features_validation)
            predictions_test = rf_model.predict(self.features_test)

            results[percentage] = {
                'val_predictions': predictions_validation,
                'test_predictions': predictions_test
            }

        self.display_cgan_results(results, synthetic_percentages)

    def display_random_forest_results(self, val_predictions, test_predictions):
        results_text = "RANDOM FOREST RESULTS\n"
        results_text += "=" * 80 + "\n\n"

        results_text += "VALIDATION SET RESULTS:\n"
        results_text += "-" * 40 + "\n"
        val_metrics = self.calculate_metrics(self.labels_validation, val_predictions)
        results_text += self.format_metrics(val_metrics)
        results_text += "\n"

        results_text += "TEST SET RESULTS:\n"
        results_text += "-" * 40 + "\n"
        test_metrics = self.calculate_metrics(self.labels_test, test_predictions)
        results_text += self.format_metrics(test_metrics)
        results_text += "\n"

        results_text += "PERFORMANCE SUMMARY:\n"
        results_text += "-" * 40 + "\n"
        results_text += f"Validation Accuracy: {val_metrics['accuracy']:.1%}\n"
        results_text += f"Test Accuracy: {test_metrics['accuracy']:.1%}\n"
        results_text += f"Validation F1 Score: {val_metrics['f1_score']:.3f}\n"
        results_text += f"Test F1 Score: {test_metrics['f1_score']:.3f}\n"

        self.results_text.config(state="normal")
        self.results_text.insert("1.0", results_text)
        self.results_text.config(state="disabled")

    def display_cgan_results(self, results, percentages):
        results_text = "RANDOM FOREST + CGAN RESULTS\n"
        results_text += "=" * 80 + "\n\n"

        for percentage in percentages:
            val_predictions = results[percentage]['val_predictions']
            test_predictions = results[percentage]['test_predictions']

            results_text += f"SYNTHETIC DATA: {percentage * 100}% AUGMENTATION\n"
            results_text += "=" * 60 + "\n\n"

            results_text += "VALIDATION SET RESULTS:\n"
            results_text += "-" * 40 + "\n"
            val_metrics = self.calculate_metrics(self.labels_validation, val_predictions)
            results_text += self.format_metrics(val_metrics)
            results_text += "\n"

            results_text += "TEST SET RESULTS:\n"
            results_text += "-" * 40 + "\n"
            test_metrics = self.calculate_metrics(self.labels_test, test_predictions)
            results_text += self.format_metrics(test_metrics)
            results_text += "\n"

            results_text += f"SUMMARY FOR {percentage * 100}% AUGMENTATION:\n"
            results_text += "-" * 40 + "\n"
            results_text += f"Validation Accuracy: {val_metrics['accuracy']:.1%}\n"
            results_text += f"Test Accuracy: {test_metrics['accuracy']:.1%}\n"
            results_text += f"Validation F1 Score: {val_metrics['f1_score']:.3f}\n"
            results_text += f"Test F1 Score: {test_metrics['f1_score']:.3f}\n\n"
            results_text += "=" * 60 + "\n\n"

        results_text += "COMPARISON ACROSS ALL AUGMENTATION LEVELS:\n"
        results_text += "=" * 60 + "\n"
        results_text += f"{'Augmentation':<15} {'Val Acc':<10} {'Test Acc':<10} {'Val F1':<10} {'Test F1':<10}\n"
        results_text += "-" * 60 + "\n"

        for percentage in percentages:
            val_predictions = results[percentage]['val_predictions']
            test_predictions = results[percentage]['test_predictions']
            val_metrics = self.calculate_metrics(self.labels_validation, val_predictions)
            test_metrics = self.calculate_metrics(self.labels_test, test_predictions)

            results_text += f"{percentage * 100}%{'':<12} {val_metrics['accuracy']:.1%}{'':<5} {test_metrics['accuracy']:.1%}{'':<5} {val_metrics['f1_score']:.3f}{'':<5} {test_metrics['f1_score']:.3f}\n"

        self.results_text.config(state="normal")
        self.results_text.insert("1.0", results_text)
        self.results_text.config(state="disabled")

    def calculate_metrics(self, true_labels, pred_labels):
        tp = np.sum((true_labels == 1) & (pred_labels == 1))
        tn = np.sum((true_labels == 0) & (pred_labels == 0))
        fp = np.sum((true_labels == 0) & (pred_labels == 1))
        fn = np.sum((true_labels == 1) & (pred_labels == 0))

        accuracy = (tp + tn) / len(true_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'accuracy': accuracy,'precision': precision,'recall': recall,'f1_score': f1_score,'tp': tp,'tn': tn,'fp': fp,'fn': fn}

    def format_metrics(self, metrics):
        text = f"Accuracy:  {metrics['accuracy']:.1%} (Overall correctness)\n"
        text += f"Precision: {metrics['precision']:.1%} (Fraud predictions that were correct)\n"
        text += f"Recall:    {metrics['recall']:.1%} (Actual frauds we caught)\n"
        text += f"F1 Score:  {metrics['f1_score']:.3f} (Balance score)\n\n"

        text += f"Confusion Matrix:\n"
        text += f"True Positives (Caught fraud):{metrics['tp']}\n"
        text += f"True Negatives (Correct normal):{metrics['tn']}\n"
        text += f"False Positives (False alarms):{metrics['fp']}\n"
        text += f"False Negatives (Missed fraud): {metrics['fn']}\n"
        return text

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = FraudDetectionGUI()
    app.run()