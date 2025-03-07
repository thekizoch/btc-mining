"""
Script to analyze Bitcoin investment strategies over a 3-year period.

- Calculates portfolio value for buying BTC vs. mining BTC
- Compares different network hashrate growth scenarios
- Plots portfolio value and operating margins
"""

import numpy as np
import matplotlib.pyplot as plt

# File updated: Changed analysis period to 3 years and added halving dates information
# Last halving: April 2024 (Block reward: 6.25 -> 3.125 BTC)
# Next halving: ~April 2028 (outside our 3-year analysis window)

# Constants
MONTHS = 36  # 3 years
INITIAL_INVESTMENT = 8375  # USD (cost of miner)
BTC_PRICE_INITIAL = 90000  # USD/BTC
MONTHLY_INVESTMENT = 300  # USD (hosting cost)
BTC_GROWTH_RATE = 0.01  # 1% monthly growth
HASHRATE_INITIAL = 335 / 1_000_000  # EH/s (335 TH/s)
HASHRATE_DEGRADATION = 0.001  # 0.1% monthly decrease
NETWORK_HASHRATE = 800  # EH/s
NETWORK_HASHRATE_GROWTH_SCENARIOS = {
    'Aggressive': 0.02,    # 2% monthly (25% yearly)
    'Moderate': 0.01,      # 1% monthly (12.7% yearly)
    'Conservative': 0.005   # 0.5% monthly (6.2% yearly)
}
BLOCKS_PER_DAY = 144  # Bitcoin blocks/day
DAYS_PER_MONTH = 365 / 12  # Simplified

# Block reward (fixed at 3.125 BTC after April 2024 halving)
def get_block_reward(month):
    return 3.125  # Fixed reward for our analysis period (April 2024 - April 2027)

# Track BTC price over time
btc_price = BTC_PRICE_INITIAL
btc_prices = [btc_price]
for month in range(1, MONTHS):
    btc_price *= (1 + BTC_GROWTH_RATE)
    btc_prices.append(btc_price)

# Calculate cumulative monthly investment
cumulative_investment = [INITIAL_INVESTMENT]
for month in range(1, MONTHS):
    cumulative_investment.append(cumulative_investment[-1] + MONTHLY_INVESTMENT)

# Scenario 1: Buying Bitcoin
btc_owned = INITIAL_INVESTMENT / BTC_PRICE_INITIAL  # Initial BTC purchase
btc_price = BTC_PRICE_INITIAL
buying_portfolio_values = [INITIAL_INVESTMENT]  # Start with initial investment

for month in range(1, MONTHS):
    # BTC price grows
    btc_price *= (1 + BTC_GROWTH_RATE)
    # Buy more BTC monthly
    monthly_btc_bought = MONTHLY_INVESTMENT / btc_price
    btc_owned += monthly_btc_bought
    # Calculate portfolio value
    portfolio_value = btc_owned * btc_price
    buying_portfolio_values.append(portfolio_value)

# Scenario 2: Mining Bitcoin with different network growth rates
mining_scenarios = {}
mining_margins = {}  # Store monthly operating margins for each scenario
for scenario, growth_rate in NETWORK_HASHRATE_GROWTH_SCENARIOS.items():
    hashrate = HASHRATE_INITIAL
    network_hashrate = NETWORK_HASHRATE
    mined_btc_total = 0
    portfolio_values = [0]  # Start at 0
    monthly_margins = [0]  # Start margins at 0
    btc_price = BTC_PRICE_INITIAL
    
    for month in range(1, MONTHS):
        # Calculate monthly mining reward
        block_reward = get_block_reward(month)
        # All hashrates are in EH/s, so ratios remain the same
        miner_fraction = hashrate / network_hashrate
        mined_btc = miner_fraction * block_reward * BLOCKS_PER_DAY * DAYS_PER_MONTH
        
        # Calculate monthly revenue and margin
        monthly_revenue = mined_btc * btc_price
        monthly_margin = monthly_revenue - MONTHLY_INVESTMENT
        monthly_margins.append(monthly_margin)
        
        # Accumulate mined BTC
        mined_btc_total += mined_btc
        # BTC price grows
        btc_price *= (1 + BTC_GROWTH_RATE)
        # Network hashrate grows at scenario rate
        network_hashrate *= (1 + growth_rate)
        # Portfolio value is just the value of mined BTC
        portfolio_value = mined_btc_total * btc_price
        portfolio_values.append(portfolio_value)
        
        # Miner degrades
        hashrate *= (1 - HASHRATE_DEGRADATION)
    
    mining_scenarios[scenario] = portfolio_values
    mining_margins[scenario] = monthly_margins

# Calculate total costs for mining
total_mining_costs = INITIAL_INVESTMENT + (MONTHLY_INVESTMENT * MONTHS)

# Plotting
months = np.arange(MONTHS)

# Create figure with two subplots, smaller size
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 10.8))
fig.suptitle('Bitcoin Investment Comparison Over 3 Years\n(April 2024 - April 2027)', fontsize=14, y=0.95)

# Plot 1: Buying BTC Portfolio Value with BTC Price
ax1.plot(months, buying_portfolio_values, color='blue', label='Portfolio Value')
ax1.plot(months, cumulative_investment, color='gray', linestyle=':', label='Cumulative Investment')
ax1.axhline(y=INITIAL_INVESTMENT, color='gray', linestyle='--', label='Initial Investment')
ax1.set_title('Portfolio Value - Buying BTC Strategy')
ax1.set_xlabel('Months')
ax1.set_ylabel('Portfolio Value (USD)')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.legend(loc='upper left')

# Add BTC price to first plot (right axis)
ax1_btc = ax1.twinx()
ax1_btc.plot(months, btc_prices, color='gold', label='BTC Price', linestyle=':')
ax1_btc.set_ylabel('BTC Price (USD)', color='gold')
ax1_btc.tick_params(axis='y', labelcolor='gold')
ax1_btc.set_yscale('log')
ax1_btc.legend(loc='upper right')

# Plot 2: Mining BTC Portfolio Value and Operating Margins
colors = {'Aggressive': 'red', 'Moderate': 'orange', 'Conservative': 'green'}
for scenario, values in mining_scenarios.items():
    growth_pct = NETWORK_HASHRATE_GROWTH_SCENARIOS[scenario] * 100
    ax2.plot(months, values, color=colors[scenario], 
             label=f'{scenario} ({growth_pct:.1f}%/mo)')  # Added monthly percentage

# Add operating margins (right axis)
ax2_margin = ax2.twinx()
for scenario, margins in mining_margins.items():
    ax2_margin.plot(months, margins, color=colors[scenario], linestyle=':', alpha=0.5,
                   label='_nolegend_')  # Hide from legend
    
ax2.set_title('Mining Strategy: Portfolio Value and Monthly Operating Margin\n(Dotted Lines: Operating Margins)')
ax2.set_xlabel('Months')
ax2.set_ylabel('Portfolio Value (USD)')
ax2_margin.set_ylabel('Monthly Operating Margin (USD)')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Add horizontal line at 0 margin
ax2_margin.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Simplified legend
ax2.legend(loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save figure instead of showing it
plt.savefig('contract_analysis.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Print final values in a table-like format
def print_separator(length=80):
    print("=" * length)

print_separator()
print(f"{'BITCOIN INVESTMENT ANALYSIS - 3 YEAR PROJECTION (2024-2027)':^80}")
print_separator()
print(f"Final BTC Price: ${btc_prices[-1]:>20,.2f}")
print(f"Total Investment: ${(INITIAL_INVESTMENT + (MONTHLY_INVESTMENT * MONTHS)):>20,.2f}")
print_separator()

print(f"{'STRATEGY COMPARISON':^80}")
print_separator()
print(f"{'Buying Strategy:':<40}")
print(f"  {'Initial Investment:':<30}${INITIAL_INVESTMENT:>15,.2f}")
print(f"  {'Monthly Investment:':<30}${MONTHLY_INVESTMENT:>15,.2f}")
print(f"  {'Final Portfolio Value:':<30}${buying_portfolio_values[-1]:>15,.2f}")
print_separator()

print(f"{'Mining Strategy:':<40}")
print(f"  {'Initial Investment:':<30}${INITIAL_INVESTMENT:>15,.2f}")
print(f"  {'Monthly Hosting Cost:':<30}${MONTHLY_INVESTMENT:>15,.2f}")
print()
print(f"{'Final Portfolio Values by Network Growth Rate:':<40}")
for scenario, values in mining_scenarios.items():
    growth_pct = NETWORK_HASHRATE_GROWTH_SCENARIOS[scenario] * 100
    final_margin = mining_margins[scenario][-1]
    print(f"  {f'{scenario} Network Growth ({growth_pct:.1f}% monthly):':<45}${values[-1]:>15,.2f}")
    print(f"  {f'Final Monthly Operating Margin:':<45}${final_margin:>15,.2f}")
print_separator()
