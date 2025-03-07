"""
Script to fetch ASIC price index data from hashrateindex.com
Created: 2024-03-07
Changes:
- Updated to use correct API endpoint
- Added parsing for different ASIC price tiers (under19, 19to25, 25to38, etc.)
"""

import requests
import pandas as pd
import json
from datetime import datetime
from typing import Dict

def fetch_asic_price_data(currency: str = "USD", span: str = "ALL") -> pd.DataFrame:
    """
    Fetch ASIC price index data from hashrateindex.com
    Args:
        currency: Currency for prices (default: USD)
        span: Time span for data (default: ALL)
    """
    base_url = "https://data.hashrateindex.com/hi-api/hashrateindex/asic/price-index"
    
    # Headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://data.hashrateindex.com/',
        'Origin': 'https://data.hashrateindex.com'
    }
    
    # Build URL with parameters
    url = f"{base_url}?currency={currency}&span={span}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Rename columns for clarity
        column_descriptions = {
            'under19': 'Price/TH (Under 19 TH/s)',
            '19to25': 'Price/TH (19-25 TH/s)',
            '25to38': 'Price/TH (25-38 TH/s)',
            '38to68': 'Price/TH (38-68 TH/s)',
            'above68': 'Price/TH (Above 68 TH/s)',
            'close': 'BTC Price (USD)'
        }
        df = df.rename(columns=column_descriptions)
        
        # Save to CSV
        output_file = 'asic_price_data.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        
        # Display first few rows and data info
        print("\nFirst few rows of the data:")
        print(df.head())
        print("\nDataset information:")
        print(df.info())
        
        # Print some basic statistics
        print("\nBasic statistics for each price tier:")
        print(df.describe())
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    df = fetch_asic_price_data() 