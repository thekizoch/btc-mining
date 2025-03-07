"""
Script to analyze ASIC price trends for the 19-25 TH/s category over the last 3 years
Created: 2024-03-07
Changes:
- Fixed model selection logic
- Added numerical stability improvements
- Fixed date handling for March 7, 2025 reference
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import json
from typing import Callable, Tuple, Dict

def load_and_prepare_data(filepath: str = 'asic_price_data.csv', reference_date: str = '2025-03-07') -> pd.DataFrame:
    """Load and prepare the ASIC price data for fitting."""
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime with UTC timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Set reference date and calculate 3 years ago
    ref_date = pd.to_datetime(reference_date, utc=True)
    three_years_ago = ref_date - timedelta(days=3*365)
    
    # Filter data
    df = df[df['timestamp'] <= ref_date]  # Only use data up to reference date
    df = df[df['timestamp'] >= three_years_ago]
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    earliest_date = df['timestamp'].min()
    latest_date = df['timestamp'].max()
    
    print(f"\nData range in dataset (last 3 years from {reference_date}):")
    print(f"Earliest date: {earliest_date.strftime('%Y-%m-%d')}")
    print(f"Latest date: {latest_date.strftime('%Y-%m-%d')}")
    
    # Print some data validation info
    price_column = 'Price/TH (19-25 TH/s)'
    print(f"\nData points for {price_column}: {df[price_column].notna().sum()}")
    print(f"Missing values: {df[price_column].isna().sum()}")
    
    return df

def linear_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear function: y = ax + b"""
    return a * x + b

def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: y = ax^b"""
    # Add small epsilon to prevent divide by zero
    x = np.maximum(x, 1e-10)
    return a * np.power(x, b)

def exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential function: y = a*exp(bx)"""
    return a * np.exp(b * x)

def logarithmic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logarithmic function: y = a*log(x) + b"""
    # Add small epsilon to prevent divide by zero
    x = np.maximum(x, 1e-10)
    return a * np.log(x) + b

def fit_model(func: Callable, x: np.ndarray, y: np.ndarray, p0: Tuple[float, float] = None) -> Tuple[np.ndarray, float, float]:
    """Fit a model and return parameters and metrics."""
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=2000)
        y_pred = func(x, *popt)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return popt, r2, rmse
    except Exception as e:
        print(f"Warning: Failed to fit model: {str(e)}")
        return None, 0, float('inf')

def plot_fits(df: pd.DataFrame, fits: Dict[str, Tuple], price_column: str) -> None:
    """Plot the data and fitted curves."""
    plt.figure(figsize=(12, 8))
    
    # Filter out any NaN values
    df_clean = df.dropna(subset=[price_column])
    
    # Convert timestamps to days since start for fitting
    days_since_start = (df_clean['timestamp'] - df_clean['timestamp'].min()).dt.total_seconds() / (24*60*60)
    plt.scatter(df_clean['timestamp'], df_clean[price_column], alpha=0.5, label='Data')
    
    x_smooth = np.linspace(days_since_start.min(), days_since_start.max(), 1000)
    dates_smooth = pd.date_range(start=df_clean['timestamp'].min(), end=df_clean['timestamp'].max(), periods=1000)
    
    for name, (func, params, r2, rmse) in fits.items():
        if params is not None:
            y_smooth = func(x_smooth, *params)
            plt.plot(dates_smooth, y_smooth, label=f'{name} (R² = {r2:.3f}, RMSE = {rmse:.1f})')
    
    plt.xlabel('Date')
    plt.ylabel('Price per TH/s (USD)')
    plt.title('ASIC Price Trends (19-25 TH/s): Model Fits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('asic_price_fits.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_best_model(fits: Dict[str, Tuple], df: pd.DataFrame, price_column: str) -> None:
    """Save the best model parameters to JSON."""
    # Find valid fits (exclude None and infinite RMSE)
    valid_fits = {}
    for name, (func, params, r2, rmse) in fits.items():
        if params is not None and not np.isinf(rmse) and rmse >= 0:
            valid_fits[name] = (func, params, float(r2), float(rmse))  # Convert to Python float
    
    if not valid_fits:
        print("\nNo valid models found!")
        return
    
    # Find best model (highest R² with reasonable RMSE)
    best_model = max(valid_fits.items(), key=lambda x: x[1][2])  # x[1][2] is R²
    model_name, (func, params, r2, rmse) = best_model
    
    # Clean data for date range
    df_clean = df.dropna(subset=[price_column])
    start_date = df_clean['timestamp'].min()
    end_date = df_clean['timestamp'].max()
    
    # Prepare model data
    model_data = {
        "model_name": model_name,
        "parameters": params.tolist(),
        "metrics": {
            "r2": float(r2),
            "rmse": float(rmse)
        },
        "date_range": {
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d')
        },
        "function_type": {
            "exponential": "y = a*exp(bx)",
            "linear": "y = ax + b",
            "power_law": "y = ax^b",
            "logarithmic": "y = a*log(x) + b"
        }.get(model_name.lower(), "unknown"),
        "notes": "x is days since start_date, y is price per TH/s in USD"
    }
    
    # Save to JSON
    with open('asic_price_model.json', 'w') as f:
        json.dump(model_data, f, indent=4)
    
    print(f"\nBest model ({model_name}) parameters saved to asic_price_model.json")
    print(f"Model equation: {model_data['function_type']}")
    print(f"Parameters: a = {params[0]:.6f}, b = {params[1]:.6f}")
    print(f"R² = {r2:.3f}, RMSE = {rmse:.3f}")

def main():
    # Load and prepare data
    df = load_and_prepare_data(reference_date='2025-03-07')
    price_column = 'Price/TH (19-25 TH/s)'
    
    # Clean data by removing NaN values
    df_clean = df.dropna(subset=[price_column])
    
    if len(df_clean) == 0:
        print("Error: No valid data points found after cleaning!")
        return
        
    # Convert timestamps to days since start for fitting
    days_since_start = (df_clean['timestamp'] - df_clean['timestamp'].min()).dt.total_seconds() / (24*60*60)
    x = days_since_start.values
    y = df_clean[price_column].values
    
    # Dictionary to store all fits
    fits = {}
    
    # Try different models
    models = {
        'Linear': (linear_func, None),
        'Power Law': (power_law, (1, 0.5)),
        'Exponential': (exponential, (1, 0.0001)),
        'Logarithmic': (logarithmic, (1, 0))
    }
    
    print(f"\nAnalyzing ASIC price trends for {price_column}\n")
    print(f"{'Model':<15} {'R²':>10} {'RMSE':>10}")
    print("-" * 35)
    
    for name, (func, p0) in models.items():
        params, r2, rmse = fit_model(func, x, y, p0)
        fits[name] = (func, params, r2, rmse)
        print(f"{name:<15} {r2:>10.3f} {rmse:>10.1f}")
    
    # Plot the results
    plot_fits(df_clean, fits, price_column)
    print("\nPlot saved as 'asic_price_fits.png'")

    # Save best model parameters
    save_best_model(fits, df_clean, price_column)

    # Print additional statistics
    print("\nPrice Statistics (USD/TH):")
    print(f"Current Price: {df_clean[price_column].iloc[-1]:.2f}")
    print(f"Average Price: {df_clean[price_column].mean():.2f}")
    print(f"Min Price: {df_clean[price_column].min():.2f}")
    print(f"Max Price: {df_clean[price_column].max():.2f}")

if __name__ == "__main__":
    main() 