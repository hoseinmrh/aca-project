import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metrics(csv_file='model_metrics_complete.csv'):
    """
    Create visualizations for model performance metrics
    """
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Run optimized_main.py first.")
        return
    
    # Load metrics data
    df = pd.read_csv(csv_file)
    
    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 12))
    
    # 1. Plot accuracy metrics
    plt.subplot(2, 2, 1)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    df_plot = df[['Model'] + metrics_to_plot].melt(
        id_vars=['Model'], 
        value_vars=metrics_to_plot,
        var_name='Metric', 
        value_name='Value'
    )
    
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_plot)
    plt.title('Classification Performance Metrics')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    
    # 2. Plot memory usage and model size
    plt.subplot(2, 2, 2)
    memory_metrics = ['ModelSizeMB', 'PeakMemoryMB']
    df_memory = df[['Model'] + memory_metrics].melt(
        id_vars=['Model'], 
        value_vars=memory_metrics,
        var_name='Metric', 
        value_name='MB'
    )
    
    sns.barplot(x='Model', y='MB', hue='Metric', data=df_memory)
    plt.title('Memory Usage')
    plt.xticks(rotation=45)
    plt.ylabel('Memory (MB)')
    
    # 3. Plot inference latency
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='InferenceLatency', data=df, color='green')
    plt.title('Inference Latency per Sample')
    plt.xticks(rotation=45)
    plt.ylabel('Time (seconds)')
    
    # 4. Plot training time per epoch
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='TrainTimePerEpochSec', data=df, color='orange')
    plt.title('Training Time per Epoch')
    plt.xticks(rotation=45)
    plt.ylabel('Time (seconds)')
    
    # Layout adjustments
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    print("Visualization saved as model_performance_comparison.png")
    
    # Create a radar chart for comparing models
    plt.figure(figsize=(10, 8))
    
    # Normalize metrics for radar chart
    metrics_for_radar = ['Accuracy', 'F1', 'Parameters', 'PeakMemoryMB', 
                        'InferenceLatency', 'TrainTimePerEpochSec']
    
    # Invert metrics where lower is better
    df_radar = df[['Model'] + metrics_for_radar].copy()
    for col in ['Parameters', 'PeakMemoryMB', 'InferenceLatency', 'TrainTimePerEpochSec']:
        df_radar[col] = 1 - (df_radar[col] - df_radar[col].min()) / (df_radar[col].max() - df_radar[col].min())
    
    # Normalize accuracy and F1 where higher is better
    for col in ['Accuracy', 'F1']:
        df_radar[col] = (df_radar[col] - df_radar[col].min()) / (df_radar[col].max() - df_radar[col].min())
    
    # Create radar chart
    from math import pi
    
    # Number of variables
    categories = metrics_for_radar
    N = len(categories)
    
    # Create angle for each variable
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model in enumerate(df_radar['Model']):
        values = df_radar.loc[i, metrics_for_radar].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison', size=15)
    plt.tight_layout()
    plt.savefig('model_radar_comparison.png', dpi=300)
    print("Radar chart saved as model_radar_comparison.png")

if __name__ == "__main__":
    plot_metrics()
