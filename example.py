import json
import sys
import requests

from IPython.display import display
import pandas as pd

pd.set_option("display.max_columns", None)

API_URL = "https://api.jquants.com"

refreshtoken = "XXXXXXXXXX"

# idToken取得
res = requests.post(f"{API_URL}/v1/token/auth_refresh?refreshtoken={refreshtoken}")
if res.status_code == 200:
    id_token = res.json()['idToken']
    headers = {'Authorization': 'Bearer {}'.format(id_token)}
    display("idTokenの取得に成功しました。")
else:
    display(res.json()["message"])


date = "20250501"#@param {type:"string"}

params = {}
params["date"] = date

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
  display(list_df)
  list_df.to_csv("stock_list.csv")
else:
  print(res.json())


params = {}
for code in unique(list_df["Code"]):
    params["code"] = code

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
    display(df)
    df.to_csv(f"${code}.csv")
    else:
    print(res.json())