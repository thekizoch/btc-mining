"""
Script to fetch Bitcoin mining hardware efficiency data from IEA and save to CSV.
Created: 2024-03-07
Changes:
- Updated to extract embedded JSON data from HTML
- Added parsing of chart data for ASIC, FPGA, GPU, and CPU categories
"""

import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
import re
from typing import Dict, List
import html

def clean_json_string(s: str) -> str:
    """Clean the JSON string by removing escaped quotes and fixing formatting."""
    # Remove escaped quotes and convert HTML entities
    s = html.unescape(s)
    # Remove any remaining escaped quotes
    s = s.replace('\\"', '"')
    return s

def extract_chart_data(html_content: str) -> Dict:
    """Extract the chart data from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the div with the chart data
    chart_div = soup.find('div', {'class': 'm-chart-block', 'data-behavior': 'chart'})
    if not chart_div:
        raise ValueError("Could not find chart div in HTML")
    
    # Get the chart options JSON
    chart_options = chart_div.get('data-chart-chartoptions', '{}')
    chart_options = clean_json_string(chart_options)
    
    # Parse the JSON
    data = json.loads(chart_options)
    return data

def process_series_data(series_data: List[Dict]) -> pd.DataFrame:
    """Convert series data to a DataFrame."""
    rows = []
    for series in series_data:
        category = series['name']
        for point in series['data']:
            rows.append({
                'category': category,
                'device_name': point['name'],
                'hashrate_mhs': point['x'],  # MH/s
                'efficiency_mhj': point['y']  # MH/J
            })
    return pd.DataFrame(rows)

def fetch_mining_efficiency_data():
    # IEA chart page URL
    url = "https://www.iea.org/data-and-statistics/charts/efficiency-of-bitcoin-mining-hardware"
    
    # Headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Extract and parse the data
        chart_data = extract_chart_data(response.text)
        
        # Process the series data
        df = process_series_data(chart_data['series'])
        
        # Save to CSV
        output_file = 'bitcoin_mining_efficiency.csv'
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        
        # Display first few rows
        print("\nFirst few rows of the data:")
        print(df.head())
        
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
    fetch_mining_efficiency_data()