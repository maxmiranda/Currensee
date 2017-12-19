from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import quandl
import math
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


df = quandl.get("BCHARTS/BITFINEXUSD")
df.to_csv('eth.csv')
df = df[['Open','High','Low','Close','Volume (BTC)']]
df.to_csv('bitcoin.csv')

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
forecast_col = 'Close'
df.fillna(-99999, inplace = True)


daysback = 0.01
forecast_out = int(math.ceil(daysback * len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
#X_lately = X[-forecast_out:]
#df.dropna(inplace=True)
X = preprocessing.scale(X)

#y = np.array(df['label'])
y = np.array(df['label'])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train)
print(y_train)
clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

#print(accuracy)
