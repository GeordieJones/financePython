import requests
import pandas as pd
import json



API_KEY = '9PYBJ0V552GDDNLF'
symbol = 'AAPL'
function = 'INCOME_STATEMENT'

url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={API_KEY}"

r = requests.get(url)
data = r.json()
#print(json.dumps(data, indent=2))

df = pd.DataFrame(data['quarterlyReports'])

# Display the DataFrame
print(df.columns)
