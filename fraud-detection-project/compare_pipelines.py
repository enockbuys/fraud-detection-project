import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from tabulate import tabulate

def compare_pipelines_table(db_path=None, output_dir=None):
    #Compare performance of all three pipelines using enhanced database information.
    # Set default paths if not provided
    if db_path is None:
        project_dir = Path(__file__).parent.parent
        db_path = project_dir / "src" / "fraud_detection.db"

    if output_dir is None:
        project_dir = Path(__file__).parent.parent
        output_dir = project_dir / "results" / "comparison"

    os.makedirs(output_dir, exist_ok=True)

    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

    # Read enhanced pipeline results
    try:
        query = """
        SELECT 
            run_id,
            pipeline_name,
            model_type,
            model_params,
            synthetic_percent,
            training_time,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1_score,
            val_roc_auc,
            val_pr_auc,
            val_true_positives,
            val_false_positives,
            val_false_negatives,
            val_true_negatives,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1_score,
            test_roc_auc,
            test_pr_auc,
            test_true_positives,
            test_false_positives,
            test_false_negatives,
            test_true_negatives,
            run_timestamp
        FROM pipeline_results
        ORDER BY pipeline_name, synthetic_percent, run_timestamp DESC
        """

        df = pd.read_sql_query(query, conn)
        print(f"Loaded {len(df)} pipeline results from database")

        # Read GAN training data
        gan_query = """
        SELECT 
            pipeline_name,
            gan_type,
            epochs,
            batch_size,
            latent_dim,
            learning_rate,
            final_g_loss,
            final_d_loss,
            synthetic_quality,
            mean_diff,
            std_diff,
            kl_div,
            training_time
        FROM gan_training
        ORDER BY pipeline_name, run_timestamp DESC
        """
        gan_df = pd.read_sql_query(gan_query, conn)
        print(f"Loaded {len(gan_df)} GAN training records from database")

    except Exception as e:
        print(f"Error reading database: {e}")
        conn.close()
        return None

    conn.close()

    if df.empty:
        print("No pipeline results found in the database")
        return None

    # Parse model parameters from JSON
    def parse_model_params(params_str):
        try:
            return json.loads(params_str)
        except:
            return {"raw_params": params_str}

    df['model_params_parsed'] = df['model_params'].apply(parse_model_params)

    # Generate comprehensive comparison tables
    comparison_results = generate_comprehensive_tables(df, gan_df, output_dir)

    return comparison_results


def generate_comprehensive_tables(df, gan_df, output_dir):
    #Generate comprehensive comparison tables using enhanced database information.
    results = {}

    print("ENHANCED FRAUD DETECTION PIPELINES COMPARISON")
    print()

    # Table 1: Best Models by Pipeline (Composite Score)
    print("BEST MODELS BY COMPOSITE SCORE (F1 + AUC)")

    best_models_data = []
    for pipeline in df['pipeline_name'].unique():
        pipeline_data = df[df['pipeline_name'] == pipeline]

        # Calculate composite score (F1 + AUC)
        pipeline_data = pipeline_data.copy()
        pipeline_data['composite_score'] = (
                pipeline_data['test_f1_score'] * 0.5 +
                pipeline_data['test_roc_auc'] * 0.3 +
                pipeline_data['test_pr_auc'] * 0.2
        )

        best_idx = pipeline_data['composite_score'].idxmax()
        best_model = pipeline_data.loc[best_idx]

        # Extract key parameters
        params = best_model['model_params_parsed']
        key_params = extract_key_parameters(params, best_model['model_type'])

        best_models_data.append({
            'Pipeline': pipeline,
            'Best_Model': best_model['model_type'],
            'Synthetic_%': best_model['synthetic_percent'],
            'Test_F1': best_model['test_f1_score'],
            'Test_ROC_AUC': best_model['test_roc_auc'],
            'Test_PR_AUC': best_model['test_pr_auc'],
            'Composite_Score': best_model['composite_score'],
            'Key_Parameters': key_params,
            'Train_Time': f"{best_model['training_time']:.1f}s"
        })

    best_models_df = pd.DataFrame(best_models_data)
    print(tabulate(best_models_df, headers='keys', tablefmt='grid', floatfmt=".4f", showindex=False))
    results['best_models'] = best_models_df

    print()

    # Table 2: GAN Performance Comparison
    if not gan_df.empty:
        print("GAN TRAINING PERFORMANCE COMPARISON")

        gan_perf_data = []
        for _, gan_row in gan_df.iterrows():
            gan_perf_data.append({
                'Pipeline': gan_row['pipeline_name'],
                'GAN_Type': gan_row['gan_type'],
                'Synthetic_Quality': gan_row['synthetic_quality'],
                'Mean_Diff': gan_row['mean_diff'],
                'KL_Div': gan_row['kl_div'],
                'Training_Time': f"{gan_row['training_time']:.1f}s",
                'Epochs': gan_row['epochs'],
                'Batch_Size': gan_row['batch_size']
            })

        gan_perf_df = pd.DataFrame(gan_perf_data)
        print(tabulate(gan_perf_df, headers='keys', tablefmt='grid', floatfmt=".4f", showindex=False))
        results['gan_performance'] = gan_perf_df

        print()

    # Table 3: Synthetic Data Impact Analysis
    print("SYNTHETIC DATA IMPACT ANALYSIS")

    impact_data = []
    for pipeline in df['pipeline_name'].unique():
        pipeline_data = df[df['pipeline_name'] == pipeline]

        # Baseline (0% synthetic)
        baseline_data = pipeline_data[pipeline_data['synthetic_percent'] == 0.0]
        # Augmented (best synthetic percentage)
        augmented_data = pipeline_data[pipeline_data['synthetic_percent'] > 0.0]

        if not baseline_data.empty and not augmented_data.empty:
            baseline_f1 = baseline_data['test_f1_score'].max()
            augmented_f1 = augmented_data['test_f1_score'].max()
            best_synth_perc = augmented_data.loc[augmented_data['test_f1_score'].idxmax(), 'synthetic_percent']

            improvement = ((augmented_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 > 0 else 0

            impact_data.append({
                'Pipeline': pipeline,
                'Baseline_F1': baseline_f1,
                'Best_Augmented_F1': augmented_f1,
                'Improvement_%': improvement,
                'Optimal_Synthetic_%': best_synth_perc * 100,
                'Effectiveness': 'High' if improvement > 5 else 'Medium' if improvement > 2 else 'Low'
            })

    impact_df = pd.DataFrame(impact_data)
    if not impact_df.empty:
        print(tabulate(impact_df, headers='keys', tablefmt='grid', floatfmt=".4f", showindex=False))
        results['synthetic_impact'] = impact_df
    else:
        print("No synthetic data impact data available")
    print()

    # Table 4: Model Efficiency Analysis
    print("MODEL EFFICIENCY ANALYSIS")

    efficiency_data = []
    for pipeline in df['pipeline_name'].unique():
        pipeline_data = df[df['pipeline_name'] == pipeline]

        avg_training_time = pipeline_data['training_time'].mean()
        avg_f1_score = pipeline_data['test_f1_score'].mean()
        efficiency_score = avg_f1_score / avg_training_time if avg_training_time > 0 else 0

        efficiency_data.append({
            'Pipeline': pipeline,
            'Avg_Train_Time(s)': avg_training_time,
            'Avg_Test_F1': avg_f1_score,
            'Efficiency_Score': efficiency_score,
            'Models_Tested': len(pipeline_data)
        })

    efficiency_df = pd.DataFrame(efficiency_data)
    print(tabulate(efficiency_df, headers='keys', tablefmt='grid', floatfmt=".4f", showindex=False))
    results['efficiency_analysis'] = efficiency_df

    print()

    # Table 5: Overall Recommendations
    print("OVERALL RECOMMENDATIONS AND INSIGHTS")

    # Find best overall pipeline
    if 'best_models' in results:
        best_overall = results['best_models'].loc[results['best_models']['Composite_Score'].idxmax()]

        print(f"   BEST OVERALL PERFORMER: {best_overall['Pipeline']}")
        print(f"   Model: {best_overall['Best_Model']}")
        print(f"   Synthetic Data: {best_overall['Synthetic_%'] * 100:.1f}%")
        print(f"   Test F1: {best_overall['Test_F1']:.4f}")
        print(f"   ROC-AUC: {best_overall['Test_ROC_AUC']:.4f}")
        print(f"   Key Parameters: {best_overall['Key_Parameters']}")
        print()

    # Synthetic data recommendations
    if 'synthetic_impact' in results and not results['synthetic_impact'].empty:
        best_impact = results['synthetic_impact'].loc[results['synthetic_impact']['Improvement_%'].idxmax()]
        print(f"   BEST SYNTHETIC DATA IMPACT: {best_impact['Pipeline']}")
        print(f"   Improvement: +{best_impact['Improvement_%']:.1f}%")
        print(f"   Optimal Percentage: {best_impact['Optimal_Synthetic_%']:.1f}%")
        print()

    # Efficiency recommendations
    if 'efficiency_analysis' in results:
        best_efficiency = results['efficiency_analysis'].loc[
            results['efficiency_analysis']['Efficiency_Score'].idxmax()]
        print(f"   MOST EFFICIENT PIPELINE: {best_efficiency['Pipeline']}")
        print(f"   Efficiency Score: {best_efficiency['Efficiency_Score']:.4f}")
        print(f"   Avg Training Time: {best_efficiency['Avg_Train_Time(s)']:.1f}s")
        print()

    # Save all tables to CSV files
    for table_name, table_data in results.items():
        csv_path = output_dir / f"{table_name}.csv"
        table_data.to_csv(csv_path, index=False)

    print(f"  All comparison tables saved to: {output_dir}")

    return results

def extract_key_parameters(params, model_type):
    #Extract the most important parameters based on model type.
    if isinstance(params, dict):
        if 'random_forest' in model_type.lower() or 'rf' in model_type.lower():
            key_params = []
            for key in ['n_estimators', 'n_trees', 'max_depth', 'min_samples_split', 'max_features']:
                if key in params:
                    key_params.append(f"{key}:{params[key]}")
            return ', '.join(key_params) if key_params else str(params)

        elif 'xgb' in model_type.lower():
            key_params = []
            for key in ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight']:
                if key in params:
                    key_params.append(f"{key}:{params[key]}")
            return ', '.join(key_params) if key_params else str(params)

        else:
            # Return first 3 parameters for other models
            return ', '.join([f"{k}:{v}" for k, v in list(params.items())[:3]])

    return str(params)[:50]  # Truncate if too long


def generate_executive_summary(results, output_dir):
    #Generate a concise executive summary table.
    if 'best_models' not in results:
        return

    print("EXECUTIVE SUMMARY")

    summary_data = []
    for _, row in results['best_models'].iterrows():
        summary_data.append({
            'Rank': len(summary_data) + 1,
            'Pipeline': row['Pipeline'],
            'Best_Model': row['Best_Model'],
            'Test_F1': row['Test_F1'],
            'ROC_AUC': row['Test_ROC_AUC'],
            'Synthetic_%': f"{row['Synthetic_%'] * 100:.1f}%",
            'Training_Time': row['Train_Time']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test_F1', ascending=False)

    print(tabulate(summary_df, headers='keys', tablefmt='grid', floatfmt=".4f", showindex=False))

    # Save executive summary
    summary_df.to_csv(output_dir / "executive_summary.csv", index=False)
    print(f"\n  Executive summary saved to: {output_dir / 'executive_summary.csv'}")


if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate package...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        from tabulate import tabulate

    results = compare_pipelines_table()

    if results:
        generate_executive_summary(results, Path(__file__).parent.parent / "results" / "comparison")
        print("\nPipeline comparison completed successfully!")
    else:
        print("Pipeline comparison failed!")