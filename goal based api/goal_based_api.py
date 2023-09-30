import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
from datetime import date
from sklearn import preprocessing 
from sklearn.preprocessing import MinMaxScaler
import random
import math
import json
from flask import Flask, jsonify, request
import sqlite3
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

portfolio_analysis = {}
ex_rate = 81
years = 7
z = None
user = 'Merken'

df = pd.read_csv(r'final_data7.csv')
cry = pd.read_csv(r'crypto_final1.csv')

def norm(arr):
  return MinMaxScaler().fit_transform(np.array(arr).reshape(-1,1))


features = ['NAME OF COMPANY', 'SYMBOL','marketCap','sector','volitility','std','price', 'Sscore', 'beta', 'cagr','market_cagr', 'net_cagr','returns','sharp_ratio','spearman','pearson', 'kendall','period']
featuresc = ['Sscore', 'Symbol','Market Cap', 'Name','cagr', 'currency', 'marketCap', 'period', 'returns', 'sharp_ratio','std', 'volitility','price']
le = preprocessing.LabelEncoder()
new = df[features].copy().dropna(axis=0, subset=['Sscore'])
new['avgcorr'] = new[['spearman','pearson', 'kendall']].sum(axis='columns')/3
new['intsector'] = le.fit_transform(new['sector'])
new['spscore'] = norm(new['Sscore']*new['period']/new['volitility'])
new['Sscore'] = norm(new['Sscore'])
new['vscore'] = norm(new['Sscore']/new['returns'])
new = new.query(f'period >= {years}')
portfo = None
cry = cry[featuresc]
cry['spscore'] = cry['Sscore']*cry['period']/cry['volitility']
cry['price'] *= ex_rate
xl = 'final_data7.csv'

from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('my_cagr_model.h5')
print(joblib.__version__)
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

df = pd.read_csv(xl)

def minmax_normalize_dict(input_dict):
    if not input_dict:
        return {}

    min_value = min(input_dict.values())
    max_value = max(input_dict.values())

    # Check for division by zero
    if max_value == min_value:
        return {key: 1 for key in input_dict}

    return {key: (value - min_value) / (max_value - min_value) for key, value in input_dict.items()}


def preprocess_new_data(new_data):
    global scaler,pca
    # Extract stock symbols before preprocessing
    symbols = new_data['SYMBOL'].tolist()  # Replace 'SYMBOL' with the actual column name for stock symbols

    # Drop the specified columns
    columns_to_drop = ['market_cagr', 'returns']
    df_cleaned = new_data.drop(columns=columns_to_drop, errors='ignore')

    # Separate numerical and categorical columns
    numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    # Fill missing values for numerical columns with their mean
    for col in numerical_cols:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    # Fill missing values for categorical columns with their mode
    for col in categorical_cols:
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    # Identify and replace infinite values
    infinite_cols = [col for col in numerical_cols if np.isinf(df_cleaned[col]).any()]
    df_cleaned[infinite_cols] = df_cleaned[infinite_cols].replace([np.inf, -np.inf], np.nan)
    for col in infinite_cols:
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)

    # Scale numerical columns
    df_cleaned[numerical_cols] = scaler.transform(df_cleaned[numerical_cols])

    # One-hot encoding for categorical variables
    df_cleaned = pd.get_dummies(df_cleaned)

    # Apply PCA
    X_final = pca.transform(df_cleaned.drop(columns=['cagr'], errors='ignore'))

    return X_final, symbols


new_data_preprocessed, symbols = preprocess_new_data(df)  # Implement this function based on your original preprocessing steps

# Make predictions
predictions = loaded_model.predict(new_data_preprocessed)
predictions_list = predictions.flatten().tolist()

# Create a dictionary mapping stock symbols to predicted CAGR values
symbol_to_prediction = dict(zip(symbols, predictions_list))


def portfolio(row):
  global portfo
  if '.NS' in row['Symbol']:
    row['quantity'] = round(portfo[row['Symbol']]/row['price'])
    row['purchase'] = row['quantity']*row['price']
  else:
    row['purchase'] = portfo[row['Symbol']]
    row['quantity'] = portfo[row['Symbol']]/row['price']
  return row


def final_portfolio(port,amt,risk):
  global portfo,symbol_to_prediction
  data = build(**port)
  tickers = bucket(r = data['r'],risk=risk)
  start = "2015-01-01"
  df = yf.download(tickers, start=start)
  data = df['Adj Close']
  log_returns = np.log(data/data.shift())
  a = new[new['SYMBOL'].isin(tickers)][['SYMBOL','cagr','price','period','std','marketCap']].rename({'SYMBOL': 'Symbol'}, axis='columns')
  b = cry[cry['Symbol'].isin(tickers)][['Symbol','cagr','price','period','std','marketCap']]
  ab = pd.concat([a,b])
  mc = dict(zip(ab['Symbol'],(ab['period']**3)*ab['marketCap']**2))
  for m in mc.keys():
    if '.NS' not in m:
      mc[m] = mc[m]*0.3

  n = 500*len(tickers)
  weights = np.zeros((n, len(tickers)))
  exp_rtns = np.zeros(n)
  exp_vols = np.zeros(n)
  sharpe_ratios = np.zeros(n)
  ai_preds = minmax_normalize_dict(symbol_to_prediction)

  # markcap = []

  markcap = [mc[tick] * ai_preds[tick] if tick in ai_preds.keys() else mc[tick]*0.2 for tick in tickers]


  for i in range(n):
      weight = np.random.random(len(tickers))
      weight /= weight.sum()
      weights[i] = weight

      exp_rtns[i] = np.sum(log_returns.mean()*weight)*252
      exp_vols[i] = np.sqrt(np.dot(weight.T, np.dot(log_returns.cov()*252, weight)))

      sharpe_ratios[i] = (exp_rtns[i] / (exp_vols[i])**2)*sum(weight*(markcap))

  index = np.where((norm(exp_vols).reshape(-1) <= 0.6) & (norm(exp_rtns).reshape(-1) >= 0) & (norm(exp_rtns).reshape(-1) <= 1))[0]
  # import matplotlib.pyplot as plt
  # fig, ax = plt.subplots()
  # ax.scatter(exp_vols, exp_rtns, c=sharpe_ratios)
  # ax.scatter(exp_vols[np.where(sharpe_ratios == sharpe_ratios[index].max())], exp_rtns[np.where(sharpe_ratios == sharpe_ratios[index].max())], c='r')
  # ax.set_xlabel('Expected Volatility')
  # ax.set_ylabel('Expected Return')
  print(index)
  investments = weights[sharpe_ratios[index].argmax()]*amt
  portfo = dict(zip(tickers,investments))
  abc = ab.apply(portfolio,axis=1).drop_duplicates(subset='Symbol')
  abc['cagr'] *= (1 - abc['std']*10)
  abc['exp_cagr'] = abc['cagr'] *(abc['purchase']/abc['purchase'].sum()) *100
  abc['exp_returns'] = abc['cagr'] * abc['purchase']
  abc['percentage'] = (abc['purchase']/abc['purchase'].sum())*100
  return abc

# @app.route('/infostocks', methods=['POST','GET'])
def info_on():
  data = request.json
  msft = yf.Ticker(data['ticker'] + '.NS')
  info = msft.info
  return info

def build(pv=None, fv=None, r=None,n=None):

  if fv == None:
    fv = pv * (((1 + (r/100.0)) ** n))
  elif r == None:
    r = (((fv/pv)**(1/n))-1)*100
  elif n == None:
    n = (np.log(fv/pv))/np.log(1+r/100)
  return ({
      'pv':pv,'fv':fv
      ,'r':r,'n':n})

def bucket(n=15,cryp_split=0.3,r=12,risk=0):
  r /= 100
  rmin = r*(risk/100)
  ncrypto = math.floor(n*cryp_split)
  nstocks = n - ncrypto - 5
  # print(nstocks)
  df = new.query(f'cagr >= {rmin}')
  df1 = cry.query(f'cagr >= {rmin}').drop_duplicates()
  numeric_columns = df.select_dtypes(include=['int', 'float'])
#   print(df.groupby('sector')[numeric_columns.columns].mean())
  inedx = list(df.groupby('sector')[numeric_columns.columns].mean().sort_values(by=['spscore'],ascending=False).index)
  sectors, cryp = inedx[:5] , []
  # sectors.append('RELIANCE.NS')
  bucket = ['RELIANCE.NS']

  for x in range(nstocks):
    sectors.append(random.choice(inedx))

  for x in range(ncrypto):
    bucket.append(random.choice(list(cry.sort_values(by=['spscore'],ascending=False)['Symbol'].head(ncrypto*3))))

  for sec in sectors:
    #  print(df.loc[df['sector'] == sec].sort_values(by=['spscore'],ascending=False))
     candidates = list(df.loc[df['sector'] == sec].sort_values(by=['spscore'],ascending=False)['SYMBOL'].head(5))

    #  print(candidates)
     bucket.append(random.choice(candidates))

  return list(set(bucket))

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

@app.route('/get_portfolio', methods=['POST','GET'])  ##api endppoint
def get_data():
    data=request.json
    global abc,portfolio_analysis
    global z
    # Retrieve the data from the request
    # print('executed')
    # data = request.json
    print(data)
    
    obj = {}
    obj["pv"] = data["pv"]
    obj["fv"] = data["fv"]
    obj["r"] = data["r"]
    obj["n"] = data["n"]

    print(obj)
    

    data1=[obj,obj["pv"],data["risk"]]                      # amount value same as present value

    port, amt, risk = build(**data1[0]), data1[1], data1[2]
    print(port)

    # Process the data or perform any required calculations
    abc = final_portfolio(port, amt,risk)
    abc.drop(['period','std'], axis=1,inplace=True)

    portfolio_analysis = {
        'crypto_percentage': abc[abc['Symbol'].str.contains('USD', regex=True, na=False)]['percentage'].sum(),
        'stocks_percentage': abc[abc['Symbol'].str.contains('.NS', regex=True, na=False)]['percentage'].sum(),
        'stocks_expected_return': abc[abc['Symbol'].str.contains('.NS', regex=True, na=False)]['exp_cagr'].sum(),
        'crypto_expected_return': abc[abc['Symbol'].str.contains('USD', regex=True, na=False)]['exp_cagr'].sum(),
        'total_expected_return': abc['exp_cagr'].sum(),
        'total_purchase': abc['purchase'].sum()
    }

    new_port = build(fv=port['fv'],r=portfolio_analysis['total_expected_return'],pv=portfolio_analysis['total_purchase']
                     ,n=None)
    portfolio_analysis['future_value'] = new_port['fv']
    portfolio_analysis['years'] = new_port['n']
    print(abc)
    print(portfolio_analysis)
    z = abc[['Symbol','quantity']]
    # insert_purchase_order(z)
    # Return the processed data as a JSON response
    stocks = abc[abc['Symbol'].str.contains('.NS', regex=True, na=False)].to_json(orient = 'records')
    crypto = abc[abc['Symbol'].str.contains('USD', regex=True, na=False)].to_json(orient = 'records')
    print(crypto)
    return jsonify(abc.to_json(orient = 'records'), portfolio_analysis,stocks,crypto)




if __name__ == '__main__':
    while True:
        try:
            app.run(host='0.0.0.0',debug=True,port=8000)
        except Exception as e:
            print("Error occurred:", e)
            print("Try different prompt or try something different...")

# @app.route('/table', methods=['POST','GET'])
# def get_abc():
#    global abc,portfolio_analysis
#    stocks = abc[abc['Symbol'].str.contains('.NS', regex=True, na=False)].to_json(orient = 'records')
#    crypto = abc[abc['Symbol'].str.contains('USD', regex=True, na=False)].to_json(orient = 'records')
#   #  for x in [stocks,crypto]:
#   #       analysis = {
#   #       'crypto_percentage': abc[abc['Symbol'].str.contains('USD', regex=True, na=False)]['percentage'].sum(),
#   #       'stocks_percentage': abc[abc['Symbol'].str.contains('.NS', regex=True, na=False)]['percentage'].sum(),
#   #       'stocks_expected_return': abc[abc['Symbol'].str.contains('.NS', regex=True, na=False)]['exp_cagr'].sum(),
#   #       'crypto_expected_return': abc[abc['Symbol'].str.contains('USD', regex=True, na=False)]['exp_cagr'].sum(),
#   #       'total_expected_return': abc['exp_cagr'].sum(),
#   #       'total_purchase': abc['purchase'].sum()
#   #   }
#    return [abc.to_json(orient = 'records'), portfolio_analysis,stocks,crypto]




