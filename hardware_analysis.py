"""
Hardware Analysis for Bitcoin Mining ASICs

This script analyzes the progression of ASIC hardware metrics over a 20-year period:
- Projects price per TH/s using exponential model
- Projects efficiency (J/TH) improvements
- Calculates what hardware specs you could get for ~$8,375 at 4-year intervals
- Shows the evolution of key metrics over time

Created: 2024
Last updated: Fixed price model calculations to properly account for start date
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Constants
INVESTMENT_AMOUNT = 8375  # USD
START_DATA_HASHRATE = 335  # TH/s
CURRENT_PRICE_PER_TH = INVESTMENT_AMOUNT / START_DATA_HASHRATE  # Should be ~$25/TH

ANALYSIS_YEARS = 20
INTERVAL_YEARS = 4

# Load ASIC price model
with open('asic_price_model.json', 'r') as f:
    price_model = json.load(f)

# Model start date from JSON
MODEL_START_DATE = datetime.strptime(price_model['date_range']['start_date'], '%Y-%m-%d')
START_DATE = datetime.today()

# Calculate days since model start date
DAYS_SINCE_MODEL_START = (START_DATE - MODEL_START_DATE).days

# Extract model parameters
A = price_model['parameters'][0]  # Initial price/TH
B = price_model['parameters'][1]  # Decay rate

# Validate our starting point matches reality
calculated_current_price = A * np.exp(B * DAYS_SINCE_MODEL_START)
print("\nModel Validation:")
print("=" * 80)
print(f"Actual current price per TH: ${CURRENT_PRICE_PER_TH:.2f}")
print(f"Model calculated price per TH: ${calculated_current_price:.2f}")
print(f"Days since model start: {DAYS_SINCE_MODEL_START}")
print("-" * 80)

# Efficiency model parameters (based on historical improvements)
INITIAL_JTH = 30  # J/TH for current gen
EFFICIENCY_IMPROVEMENT_RATE = -0.001  # Slightly slower than price improvements

def calculate_price_per_th(days_from_model_start):
    """Calculate price per TH/s using the exponential model."""
    return A * np.exp(B * days_from_model_start)

def calculate_efficiency(days_from_start):
    """Calculate J/TH using an exponential improvement model."""
    return INITIAL_JTH * np.exp(EFFICIENCY_IMPROVEMENT_RATE * days_from_start)

# Calculate metrics at each interval
intervals = range(0, ANALYSIS_YEARS + 1, INTERVAL_YEARS)
dates = [START_DATE + timedelta(days=365*years) for years in intervals]
days_from_model_start = [DAYS_SINCE_MODEL_START + (years * 365) for years in intervals]

# Calculate metrics
price_per_th = [calculate_price_per_th(days) for days in days_from_model_start]
efficiency = [calculate_efficiency(years * 365) for years in intervals]  # Efficiency from today
hashrate_per_investment = [INVESTMENT_AMOUNT / price for price in price_per_th]

# Print analysis
print("\nBitcoin ASIC Hardware Analysis (20-year projection)")
print("=" * 80)
print(f"{'Year':<10} {'Date':<12} {'$/TH':<10} {'J/TH':<10} {'TH/$8375':<12}")
print("-" * 80)

for i, year in enumerate(intervals):
    date_str = dates[i].strftime('%Y-%m-%d')
    print(f"{year:<10} {date_str:<12} {price_per_th[i]:,.2f} {efficiency[i]:,.2f} {hashrate_per_investment[i]:,.2f}")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Price per TH plot
ax1.plot(dates, price_per_th, 'b-', linewidth=2, label='Price per TH/s')
ax1.set_title('ASIC Price per TH/s Over Time', pad=20, fontsize=12)
ax1.set_ylabel('USD per TH/s', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)

# Efficiency plot
ax2.plot(dates, efficiency, 'g-', linewidth=2, label='Efficiency (J/TH)')
ax2.set_title('ASIC Efficiency Over Time', pad=20, fontsize=12)
ax2.set_ylabel('Joules per TH/s', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)

# Format x-axis dates
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Date', fontsize=10)

plt.tight_layout()
plt.savefig('hardware_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional analysis
print("\nKey Insights:")
print("=" * 80)
print(f"Initial hashrate purchasable (2024): {hashrate_per_investment[0]:,.2f} TH/s")
print(f"Final hashrate purchasable (2044): {hashrate_per_investment[-1]:,.2f} TH/s")
print(f"Efficiency improvement: {(1 - efficiency[-1]/efficiency[0])*100:.1f}%")
print(f"Price per TH reduction: {(1 - price_per_th[-1]/price_per_th[0])*100:.1f}%") 