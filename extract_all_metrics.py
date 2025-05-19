import json
import os
import pandas as pd
import torch
from tabulate import tabulate  # Make sure to have tabulate installed (pip install tabulate)

def extract_metrics_from_model(model_path):
    """Extract metrics from a saved model checkpoint"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return {}
    
    try:
        checkpoint = torch.load(model_path)
        # Try to get all metrics if they exist
        if 'all_metrics' in checkpoint:
            return checkpoint['all_metrics']
        else:
            # Otherwise construct from individual metrics
            metrics = {
                'Accuracy': checkpoint.get('accuracy', 0),
                'F1': checkpoint.get('f1', 0),
                'InferenceLatencyMS': checkpoint.get('inference_time_ms', 0),
            }
            return metrics
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return {}

def load_json_metrics(json_path):
    """Load metrics from a JSON file"""
    if not os.path.exists(json_path):
        print(f"Metrics file {json_path} not found.")
        return []
    
    try:
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics {json_path}: {e}")
        return []

def extract_all_metrics():
    """Extract and combine metrics from all available sources"""
    all_metrics = []
    
    # Check for JSON metrics files
    json_files = [
        'complete_metrics.json',
        'evaluation_metrics.json',
        'model_metrics.json'
    ]
    
    for json_file in json_files:
        metrics = load_json_metrics(json_file)
        if isinstance(metrics, list):
            all_metrics.extend(metrics)
        elif isinstance(metrics, dict):
            all_metrics.append(metrics)
    
    # Check for model files
    model_files = [
        'optimal_protein_model.pt'
    ]
    
    for model_file in model_files:
        metrics = extract_metrics_from_model(model_file)
        if metrics:
            # Add model name if it doesn't exist
            if 'Model' not in metrics:
                metrics['Model'] = os.path.splitext(model_file)[0]
            all_metrics.append(metrics)
    
    # Check for CSV metrics files
    csv_files = [
        'model_metrics.csv',
        'simplified_model_metrics.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    all_metrics.append(row.to_dict())
            except Exception as e:
                print(f"Error loading CSV {csv_file}: {e}")
    
    # Remove duplicates based on model name
    unique_metrics = {}
    for metrics in all_metrics:
        if 'Model' in metrics:
            model_name = metrics['Model']
            unique_metrics[model_name] = metrics
    
    return list(unique_metrics.values())

def format_metrics_table(metrics_list):
    """Format metrics into a readable table"""
    if not metrics_list:
        return "No metrics found."
    
    # Define the columns we want
    columns = [
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 
        'Parameters', 'ModelSizeMB', 'PeakMemoryMB',
        'InferenceLatencyMS', 'TrainingTimeTotal'
    ]
    
    # Prepare data for tabulation
    table_data = []
    for metrics in metrics_list:
        row = []
        for col in columns:
            if col in metrics:
                # Format numeric values
                value = metrics[col]
                if isinstance(value, float) and col != 'Parameters':
                    value = round(value, 3)
                elif col == 'Parameters' and isinstance(value, (int, float)):
                    value = f"{int(value):,}"
                row.append(value)
            else:
                row.append('N/A')
        table_data.append(row)
    
    # Sort by accuracy if available
    try:
        table_data.sort(key=lambda x: float(x[columns.index('Accuracy')]) if x[columns.index('Accuracy')] != 'N/A' else 0, reverse=True)
    except:
        pass
    
    # Create table
    table = tabulate(table_data, headers=columns, tablefmt="grid")
    return table

def save_formatted_metrics(metrics_list, output_file="all_model_metrics.json"):
    """Save all metrics to a formatted JSON file"""
    # Make sure all numeric values are rounded to 3 decimal places
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, float) and key != 'Parameters':
                metrics[key] = round(value, 3)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    
    print(f"Saved all metrics to {output_file}")

if __name__ == "__main__":
    print("Extracting and summarizing all model metrics...")
    all_metrics = extract_all_metrics()
    
    if all_metrics:
        # Print table summary
        table = format_metrics_table(all_metrics)
        print("\nModel Performance Summary:")
        print(table)
        
        # Save to JSON
        save_formatted_metrics(all_metrics)
        
        # Save to CSV for easier analysis
        try:
            df = pd.DataFrame(all_metrics)
            df.to_csv("all_model_metrics.csv", index=False)
            print("Metrics also saved to all_model_metrics.csv")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    else:
        print("No metrics found.")
