import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from src.common.classification_summary import classification_summary
def setup_import_paths():
    #Setup import paths for all pipeline modules
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, 'src')

    paths_to_add = [
        src_path,
        os.path.join(src_path, 'pipeline1'),
        os.path.join(src_path, 'pipeline2'),
        os.path.join(src_path, 'pipeline3'),
        os.path.join(src_path, 'common')
    ]

    for path in paths_to_add:
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)

def load_model(model_path):
    #Load model with import path management
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        # Add pipeline directories to path for custom modules
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.join(project_root, 'src')

        pipeline_dirs = [
            os.path.join(src_path, 'pipeline1'),
            os.path.join(src_path, 'pipeline2'),
            os.path.join(src_path, 'pipeline3'),
            os.path.join(src_path, 'common')
        ]
        for pipeline_dir in pipeline_dirs:
            if pipeline_dir not in sys.path and os.path.exists(pipeline_dir):
                sys.path.insert(0, pipeline_dir)

        with open(model_path, 'rb') as f:
            return pickle.load(f)
def scan_model_files():
    #Scan for available model files in results directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "src", "results")

    model_files = []

    if not os.path.exists(results_dir):
        return model_files

    for pipeline in ["pipeline1", "pipeline2", "pipeline3"]:
        pipeline_dir = os.path.join(results_dir, pipeline)
        if not os.path.exists(pipeline_dir):
            continue

        for file in os.listdir(pipeline_dir):
            if file.endswith('.pkl') or file.endswith('.keras'):
                file_path = os.path.join(pipeline_dir, file)
                file_size = os.path.getsize(file_path) / 1024
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                model_files.append({
                    'pipeline': pipeline,
                    'name': file,
                    'path': file_path,
                    'size_kb': file_size,
                    'modified': mod_time
                })
    return sorted(model_files, key=lambda x: (x['pipeline'], x['name']))
def display_model_selection(model_files):
    #Display available models in a formatted table
    print("  FRAUD DETECTION PREDICTION SYSTEM")

    if not model_files:
        print("No model files found in results directory")
        return None

    print("\nAvailable Model Files")
    print("#    Pipeline     Model Name                               Size (KB)    Modified")

    for i, model in enumerate(model_files, 1):
        print(f"{i:<4} {model['pipeline']:<12} {model['name']:<38} {model['size_kb']:<10.1f}  {model['modified'].strftime('%Y-%m-%d %H:%M')}")

    return model_files

def load_predict_data():
    #Load the prediction dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predict_path = os.path.join(project_root, "src", "data", "processed", "predict_set.npz")
    data = np.load(predict_path)
    return data['X_predict'], data['y_predict']

def predict_with_model(model, X_data):
    #Make predictions with the loaded model
    if hasattr(model, 'predict_proba'):
        predictions = model.predict(X_data)
        probabilities = model.predict_proba(X_data)
        if probabilities.ndim > 1 and probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
        return predictions, probabilities
    elif hasattr(model, 'predict'):
        predictions = model.predict(X_data)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_data)
            if probabilities.ndim > 1 and probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]
        else:
            probabilities = None
        return predictions, probabilities
    else:
        raise Exception("Model doesn't have predict method")

def main():
    #Main prediction function
    setup_import_paths()
    try:
        model_files = scan_model_files()
        displayed_files = display_model_selection(model_files)

        if not displayed_files:
            return

        while True:
            choice = input("\nSelect a model:\n  [Enter model #] - Use specific model\n  [Enter 'q'] - Quit\n\nYour choice: ").strip()

            if choice.lower() == 'q':
                print("Goodbye!")
                return
            try:
                model_index = int(choice) - 1
                if 0 <= model_index < len(displayed_files):
                    selected_model = displayed_files[model_index]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(displayed_files)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")

        print(f"\nLoading model: {selected_model['pipeline']}/{selected_model['name']}")
        model = load_model(selected_model['path'])

        X_predict, y_predict = load_predict_data()
        print(f"Loaded {len(X_predict)} samples for prediction")

        predictions, probabilities = predict_with_model(model, X_predict)
        metrics = classification_summary(y_predict, predictions, probabilities)

        print("PREDICTION RESULTS")
        print(f"Model: {selected_model['pipeline']}/{selected_model['name']}")
        print(f"Samples: {len(X_predict)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"PR-AUC: {metrics['pr_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"True Negatives: {metrics['true_negatives']}")

        output_dir = os.path.dirname(selected_model['path'])
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
        print(f"\nPredictions saved to: {output_file}")

    except KeyboardInterrupt:
        print("\nPrediction cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()