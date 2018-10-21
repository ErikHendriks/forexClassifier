import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go

from plotly import tools
from featureFunctionsOrig import *


"""
## Load up data and create moving average
"""
df = pd.read_csv('DATA/EURUSD1.csv')
#df.columns = [['date','open','high','low','close','volume']]
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
df['Symbol'] = 'EURUSD'
df = df.drop_duplicates(keep=False)
df = df.iloc[:200]

ma30 = df.close.rolling(center=False,window=30).mean()
ma50 = df.close.rolling(center=False,window=50).mean()
ma100 = df.close.rolling(center=False,window=100).mean()


"""
## Get function data
"""
## Heiken Ashi Candles
#HAresults = holder.heikenashi(df,[1])
#HA = HAresults.candles[1]


## Detrend Data Function
#detrended = holder.detrend(df,method='difference')

## Fourier and Sine
#f = holder.fourier(df,[10,15],method='difference')
#s = holder.sine(df,[10,15],method='difference')

## Wiliams Accumulated Distribution Line
#wadl = holder.wadl(df,[15])
#line = wadl.wadl[15]
 
## Resample Data
#resampled = holder.OHLCresample(df,'15H')
#resampled.index = resampled.index.droplevel(0)

## Momentum indicator
#momentum = holder.momentum(df,[10])
#results = momentum.close[10]

## Stochastic Oscillator
#stochastic = holder.stochastic(df,[14,15])
#results = stochastic.close[14]

## Williams %R
#williams = holder.williams(df,[15])
#results = williams.close[15]

## Price Rate Of Change
#proc = holder.proc(df,[30])
#results = proc.proc[30]

## Accumulation Distribution Oscillator
#AD = holder.adosc(df,[30])
#results = AD.AD[30]

## Moving Average Convergence Divergence
#macd = holder.macd(df,[15,30])
#results = macd.signal

## CCI (Commodity Channel Index)
#cci = holder.cci(df,[30])
#results = cci.cci[30]

## Bollinger Bands
#bollinger = holder.bollinger(df,[20],2)
#results = bollinger.bands[20]

## Price Averages
#avs = holder.paverage(df,[20])
#results = avs.avs[20]

## Slope Function
#s = holder.slopes(df,[20])
#results = s.slope[20]

"""
## Plot stuff
"""
trace0 = go.Ohlc(x=df.index.to_pydatetime(),open=df.open,high=df.high,low=df.low,close=df.close,name='EUR/USD')
#trace1 = go.Scatter(x=df.index,y=ma30,name='ma30')
#trace11 = go.Scatter(x=df.index,y=ma50,name='ma50')
#trace111 = go.Scatter(x=df.index,y=ma100,name='ma100')

#trace2 = go.Scatter(x=df.index,y=detrended,name='detrending')
#trace2 = go.Scatter(x=line.index,y=line.close.values,name='wadl')
#trace2 = go.Scatter(x=results.index,y=results.close.values,name='momentum indicator')
#trace2 = go.Scatter(x=results.index,y=results.K.values,name='stochastic oscillator')
#trace2 = go.Scatter(x=results.index,y=results.R.values,name='williams %R')
#trace2 = go.Scatter(x=results.index,y=results.close.values,name='proc')
#trace2 = go.Scatter(x=results.index,y=results.AD.values,name='adosc')
#trace2 = go.Scatter(x=results.index,y=results.SL.values,name='macd')
#trace2 = go.Scatter(x=results.index,y=results.close.values,name='cci')
#trace2 = go.Scatter(x=results.index,y=results.upper.values,name='bollinger bands')
#trace2 = go.Scatter(x=results.index,y=results.close.values,name='price average')
#trace2 = go.Scatter(x=results.index,y=results.high.values,name='slope')

#trace3 = go.Candlestick(x=HA.index,open=HA.open.values,high=HA.high.values,low=HA.low.values,close=HA.close.values,name='heiken ashi')
#trace3 = go.Ohlc(x=resampled.index.to_pydatetime(),open=resampled.open.values,high=resampled.high.values,low=resampled.low.values,close=resampled.close.values,name='resampled')

#data = [trace0,trace1]
fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)

fig.append_trace(trace0,1,1)
#fig.append_trace(trace1,1,1)
#fig.append_trace(trace11,1,1)
#fig.append_trace(trace111,1,1)

fig.append_trace(trace2,2,1)

#fig.append_trace(trace3,3,1)

#py.offline.plot(fig,filename='test.html')


