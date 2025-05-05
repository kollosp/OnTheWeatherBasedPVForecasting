import pandas as pd
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init
init()

# Read the CSV file with multi-level columns
df = pd.read_csv("cm/concat.csv", header=[0,1])

# Extract unique model names
df['model_name'] = df['Name']['Param'].apply(lambda x: x.split('_')[0])
unique_models = df['model_name'].unique()
print("\nUnique model names:")
print("==================")
for model in sorted(unique_models):
    print(model)

def color_metric(value, min_val, max_val):
    # Normalize value between 0 and 1 (inverted - lower is better)
    norm = 1 - ((value - min_val) / (max_val - min_val) if max_val != min_val else 0)
    # Convert to color (red->yellow->green)
    if norm <= 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (norm * 2))
    else:
        # Yellow to Green
        r = int(255 * ((1 - norm) * 2))
        g = 255
    b = 0
    return f"\033[38;2;{r};{g};{b}m{value:.4f}\033[0m"

def print_model_results(df):
    # Get global min/max for color scaling
    mae_means = df['MAE']['Mean'].values
    mse_means = df['MSE']['Mean'].values
    mae_min, mae_max = mae_means.min(), mae_means.max()
    mse_min, mse_max = mse_means.min(), mse_means.max()
    
    # Get base MAE for sorting
    base_metrics = {}
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        base_row = model_df[model_df['Name']['Param'].str.contains('_base')]
        if not base_row.empty:
            base_metrics[model] = base_row['MAE']['Mean'].values[0]
    
    # Sort models by base MAE
    sorted_models = sorted(base_metrics.keys(), key=lambda x: base_metrics[x])
    
    # Print tables in sorted order
    for model in sorted_models:
        model_df = df[df['model_name'] == model]
        results = []
        
        # Create table rows
        for _, row in model_df.iterrows():
            full_name = row['Name']['Param'].replace(',', ', ')  # Add space after commas
            mae_mean = row['MAE']['Mean']
            mse_mean = row['MSE']['Mean']
            
            colored_mae = f"{color_metric(mae_mean, mae_min, mae_max)}±{row['MAE']['Std']:.4f}"
            colored_mse = f"{color_metric(mse_mean, mse_min, mse_max)}±{row['MSE']['Std']:.4f}"
            
            results.append([full_name, colored_mae, colored_mse])
        
        # Print table for this model
        print(f"\n{model} Performance Metrics (Base MAE: {base_metrics[model]:.4f}):")
        print(tabulate(results, 
                      headers=["Model Configuration", "MAE (μ±σ)", "MSE (μ±σ)"],
                      tablefmt='grid',
                      maxcolwidths=[None, 20, 20],
                      disable_numparse=True))

print_model_results(df)