#!/usr/bin/env python3
"""
Example script demonstrating the included libraries
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime

def main():
    print("🐍 Python Development Environment Ready!")
    print("=" * 50)
    
    # Test numpy
    print("\n📊 Testing NumPy:")
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Array: {arr}")
    print(f"Mean: {np.mean(arr)}")
    
    # Test pandas
    print("\n📈 Testing Pandas:")
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Tokyo']
    })
    print(df)
    
    # Test requests
    print("\n🌐 Testing Requests:")
    try:
        response = requests.get('https://httpbin.org/json')
        if response.status_code == 200:
            print("✅ HTTP request successful!")
            print(f"Response keys: {list(response.json().keys())}")
        else:
            print(f"❌ HTTP request failed with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print(f"\n🕐 Current time: {datetime.now()}")
    print("\n🚀 Environment is ready for development!")

if __name__ == "__main__":
    main()