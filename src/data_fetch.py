import json
import sys
import os
import requests
import datetime
import time
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)

API_URL = "https://api.jquants.com"

# Load refresh token from environment variable
refreshtoken = os.getenv('JQUANTS_REFRESH_TOKEN')
if not refreshtoken:
    print("Error: JQUANTS_REFRESH_TOKEN environment variable not set!")
    print("Please create a .env file with your refresh token.")
    sys.exit(1)

# Create data directory if it doesn't exist
data_dir = Path("/app/data")
price_data_dir = Path("/app/data/price_data")
data_dir.mkdir(exist_ok=True)

# idToken取得
res = requests.post(f"{API_URL}/v1/token/auth_refresh?refreshtoken={refreshtoken}")
if res.status_code == 200:
    id_token = res.json()['idToken']
    headers = {'Authorization': 'Bearer {}'.format(id_token)}
    print("idTokenの取得に成功しました。")
else:
    print(res.json()["message"])
    sys.exit(1)

date = "20250501"  # You can make this configurable too

params = {}
params["date"] = date

# Get listed companies info
print(f"Fetching listed companies info for date: {date}")
res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=headers)

if res.status_code == 200:
    d = res.json()
    data = d["info"]
    while "pagination_key" in d:
        params["pagination_key"] = d["pagination_key"]
        res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=headers)
        d = res.json()
        data += d["info"]
    
    list_df = pd.DataFrame(data)
    print(f"Found {len(list_df)} companies")
    
    # Save to data directory
    list_csv_path = data_dir / "stock_list.csv"
    list_df.to_csv(list_csv_path, index=False)
    print(f"Stock list saved to: {list_csv_path}")
else:
    print("Error fetching listed companies:")
    print(res.json())
    sys.exit(1)

# Get daily quotes for each company
print("\nFetching daily quotes for each company...")
unique_codes = list_df["Code"].unique()

# Testing mode - only process first 5 companies
if False:  # Change to False when ready for full run
    unique_codes = unique_codes[:5]
    print(f"Testing mode: Processing only {len(unique_codes)} companies")

for i, code in enumerate(unique_codes, 1):
    time.sleep(0.3)
    print(f"Processing {code} ({i}/{len(unique_codes)})")
    
    params = {"code": code}
    
    res = requests.get(f"{API_URL}/v1/prices/daily_quotes", params=params, headers=headers)
    
    if res.status_code == 200:
        d = res.json()
        data = d["daily_quotes"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(f"{API_URL}/v1/prices/daily_quotes", params=params, headers=headers)
            d = res.json()
            data += d["daily_quotes"]
        
        df = pd.DataFrame(data)
        
        # Save to data directory with proper filename
        csv_path = data_dir / f"{code}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} records to: {csv_path}")
    else:
        print(f"  Error fetching data for {code}:")
        print(f"  {res.json()}")

print("\nData fetching completed!")