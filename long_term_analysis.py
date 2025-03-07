"""
Long-term Bitcoin Mining Analysis (20 years)
- Compares two strategies: Buy & Hold vs Mining
- Both strategies reinvest $8,375 every 4 years
- Mining strategy includes hardware reinvestment and ongoing costs
- Buy strategy uses reinvestment amount to purchase more BTC
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import math

# Load ASIC price model
with open('asic_price_model.json', 'r') as f:
    asic_model = json.load(f)

# Constants
YEARS = 20
MONTHS = YEARS * 12
REINVESTMENT_PERIOD = 48  # months (4 years)
INITIAL_INVESTMENT = 8375  # USD (cost of miner)
BTC_PRICE_INITIAL = 90000  # USD/BTC
MONTHLY_INVESTMENT = 300  # USD (hosting cost)
BTC_GROWTH_RATE = 0.01  # 1% monthly growth
HASHRATE_INITIAL = 335 / 1_000_000  # EH/s (335 TH/s)
HASHRATE_DEGRADATION = 0.001  # 0.1% monthly decrease
NETWORK_HASHRATE = 800  # EH/s
NETWORK_HASHRATE_GROWTH_SCENARIOS = {
    'Aggressive': 0.02,    # 2% monthly
    'Moderate': 0.01,      # 1% monthly
    'Conservative': 0.005   # 0.5% monthly
}
BLOCKS_PER_DAY = 144
DAYS_PER_MONTH = 365 / 12

def get_block_reward(month):
    """Calculate block reward considering halvings every 4 years"""
    halvings = month // 48  # Approximate halving every 48 months
    return 3.125 / (2 ** halvings)

def calculate_hash_power(investment_date):
    """Calculate hash power purchasable with INITIAL_INVESTMENT using ASIC price model"""
    model_start = datetime.strptime(asic_model['date_range']['start_date'], '%Y-%m-%d')
    days_since_start = (investment_date - model_start).days
    
    # Use exponential model: y = a*exp(bx)
    a, b = asic_model['parameters']
    price_per_th = a * math.exp(b * days_since_start)
    
    # Calculate TH/s purchasable with INITIAL_INVESTMENT
    th_purchased = INITIAL_INVESTMENT / price_per_th
    return th_purchased / 1_000_000  # Convert to EH/s

# Initialize arrays for tracking
months = np.arange(MONTHS)
btc_prices = [BTC_PRICE_INITIAL]
cumulative_investment = [INITIAL_INVESTMENT]

# Calculate BTC price progression
btc_price = BTC_PRICE_INITIAL
for month in range(1, MONTHS):
    btc_price *= (1 + BTC_GROWTH_RATE)
    btc_prices.append(btc_price)
    cumulative_investment.append(cumulative_investment[-1] + MONTHLY_INVESTMENT)

# Scenario 1: Buying Bitcoin
btc_owned = INITIAL_INVESTMENT / BTC_PRICE_INITIAL
buying_portfolio_values = [INITIAL_INVESTMENT]

for month in range(1, MONTHS):
    btc_price = btc_prices[month]
    
    # Regular monthly investment
    monthly_btc_bought = MONTHLY_INVESTMENT / btc_price
    btc_owned += monthly_btc_bought
    
    # Every 4 years, invest INITIAL_INVESTMENT in BTC
    if month % REINVESTMENT_PERIOD == 0:
        reinvestment_btc = INITIAL_INVESTMENT / btc_price
        btc_owned += reinvestment_btc
        cumulative_investment[month] += INITIAL_INVESTMENT
    
    portfolio_value = btc_owned * btc_price
    buying_portfolio_values.append(portfolio_value)

# Scenario 2: Mining with reinvestment
mining_scenarios = {}
mining_margins = {}

for scenario, growth_rate in NETWORK_HASHRATE_GROWTH_SCENARIOS.items():
    network_hashrate = NETWORK_HASHRATE
    mined_btc_total = 0
    portfolio_values = [0]
    monthly_margins = [0]
    
    # Track hash power from each investment
    hash_cohorts = []
    investment_dates = []
    
    # Initial investment
    start_date = datetime.now()
    initial_hash = calculate_hash_power(start_date)
    hash_cohorts.append(initial_hash)
    investment_dates.append(start_date)
    
    btc_price = BTC_PRICE_INITIAL
    
    for month in range(1, MONTHS):
        # Reinvest every REINVESTMENT_PERIOD months
        if month % REINVESTMENT_PERIOD == 0:
            investment_date = start_date + timedelta(days=month*30)
            new_hash = calculate_hash_power(investment_date)
            hash_cohorts.append(new_hash)
            investment_dates.append(investment_date)
        
        # Calculate current total hash power considering degradation
        current_hash = 0
        for i, (hash_power, inv_date) in enumerate(zip(hash_cohorts, investment_dates)):
            months_since_purchase = month - (inv_date - start_date).days / 30
            if months_since_purchase >= 0:
                degraded_hash = hash_power * (1 - HASHRATE_DEGRADATION) ** months_since_purchase
                current_hash += degraded_hash
        
        # Calculate mining rewards
        block_reward = get_block_reward(month)
        miner_fraction = current_hash / network_hashrate
        mined_btc = miner_fraction * block_reward * BLOCKS_PER_DAY * DAYS_PER_MONTH
        
        # Calculate revenue and margins
        btc_price = btc_prices[month]
        monthly_revenue = mined_btc * btc_price
        monthly_margin = monthly_revenue - MONTHLY_INVESTMENT
        monthly_margins.append(monthly_margin)
        
        # Accumulate mined BTC
        mined_btc_total += mined_btc
        portfolio_value = mined_btc_total * btc_price
        portfolio_values.append(portfolio_value)
        
        # Network hashrate grows
        network_hashrate *= (1 + growth_rate)
    
    mining_scenarios[scenario] = portfolio_values
    mining_margins[scenario] = monthly_margins

# Plotting
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle(f'Bitcoin Investment Comparison Over {YEARS} Years\nWith 4-Year Reinvestment Cycles', fontsize=14, y=0.95)

# Plot 1: Buy & Hold with BTC Price
ax1.plot(months/12, buying_portfolio_values, color='blue', label='Portfolio Value')
ax1.set_title('Buy & Hold Strategy')
ax1.set_xlabel('Years')
ax1.set_ylabel('Portfolio Value (USD)')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.legend(loc='upper left')

# Add BTC price to first plot (right axis)
ax1_btc = ax1.twinx()
ax1_btc.plot(months/12, btc_prices, color='gold', label='BTC Price', linestyle=':')
ax1_btc.set_ylabel('BTC Price (USD)', color='gold')
ax1_btc.tick_params(axis='y', labelcolor='gold')
ax1_btc.set_yscale('log')
ax1_btc.legend(loc='upper right')

# Plot 2: Mining Strategy with Operating Margins
colors = {'Aggressive': 'red', 'Moderate': 'orange', 'Conservative': 'green'}
for scenario, values in mining_scenarios.items():
    growth_pct = NETWORK_HASHRATE_GROWTH_SCENARIOS[scenario] * 100
    ax2.plot(months/12, values, color=colors[scenario], 
             label=f'Portfolio ({scenario}: {growth_pct:.1f}%/mo)')

ax2.set_title('Mining Strategy')
ax2.set_xlabel('Years')
ax2.set_ylabel('Portfolio Value (USD)')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')
ax2.legend(loc='upper left')

# Add operating margins to second plot (right axis)
ax2_margin = ax2.twinx()
for scenario, margins in mining_margins.items():
    ax2_margin.plot(months/12, margins, color=colors[scenario], linestyle=':', alpha=0.5,
                   label=f'Margin ({scenario})')
ax2_margin.set_ylabel('Monthly Operating Margin (USD)')
ax2_margin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('long_term_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
def print_separator(length=80):
    print("=" * length)

print_separator()
print(f"{'BITCOIN LONG-TERM INVESTMENT ANALYSIS - {YEARS} YEAR PROJECTION':^80}")
print_separator()
print(f"Initial Investment: ${INITIAL_INVESTMENT:,.2f}")
print(f"Monthly Operating Cost: ${MONTHLY_INVESTMENT:,.2f}")
print(f"Reinvestment Cycle: {REINVESTMENT_PERIOD//12} years")
print(f"Final BTC Price: ${btc_prices[-1]:,.2f}")
print_separator()

print(f"{'Buy & Hold Strategy':^80}")
print(f"Final Portfolio Value: ${buying_portfolio_values[-1]:,.2f}")
print(f"Total Investment: ${cumulative_investment[-1]:,.2f}")
print(f"Total BTC Holdings: {btc_owned:.8f}")
print_separator()

print(f"{'Mining Strategy Results':^80}")
for scenario, values in mining_scenarios.items():
    print(f"\n{scenario} Network Growth Scenario:")
    print(f"Final Portfolio Value: ${values[-1]:,.2f}")
    print(f"Final Monthly Operating Margin: ${mining_margins[scenario][-1]:,.2f}")
print_separator()
