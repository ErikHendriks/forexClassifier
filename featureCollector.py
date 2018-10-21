import pandas as pd
import numpy as np
import time
import plotly as py
import plotly.graph_objs as go

from plotly import tools
from featureFunctionsOrig import *

totalStart = time.time()

"""
## Load CSV Data
"""
data = pd.read_csv('DATA/EURUSD1.csv')
#data.columns = [['date','open','high','low','close','volume']]
data.date = pd.to_datetime(data.date)
data = data.set_index(data.date)
data = data[['date','open','high','low','close','volume']]
data['Symbol'] = 'EURUSD'
data = data.iloc[:200]
prices = data.drop_duplicates(keep=False)
#print(prices)


"""
## Create lists of each period required by the functions
"""
momentumKey = [3,4,5,8,9,10]
stochasticKey = [3,4,5,8,9,10]
williamsKey = [6,7,8,9,10]
procKey = [12,13,14,15]
wadlKey = [15]
adoscKey = [2,3,4,5]
macdKey = [15,30]
cciKey = [15]
bollingerKey = [15]
heikenashiKey = [15]
paverageKey = [2]
slopeKey = [3,4,5,10,20,30]
fourierKey = [10,20,30]
sineKey = [5,6]

keylist = [momentumKey,stochasticKey,williamsKey,procKey,
            wadlKey,adoscKey,macdKey,cciKey,bollingerKey,
            heikenashiKey,paverageKey,slopeKey,fourierKey,
            sineKey]


"""
## Calculate all the features
"""
momentumDict = holder.momentum(prices,momentumKey)
print('1')
stochasticDict = holder.stochastic(prices,stochasticKey)
print('2')
williamsDict = holder.williams(prices,williamsKey)
print('3')
procDict = holder.proc(prices,procKey)
print('4')
wadlDict = holder.wadl(prices,wadlKey)
print('5')
adoscDict = holder.adosc(prices,adoscKey)
print('6')
macdDict = holder.macd(prices,macdKey)
print('7')
cciDict = holder.cci(prices,cciKey)
print('8')
bollingerDict = holder.bollinger(prices,bollingerKey,2)
print('9')
HKA = holder.OHLCresample(prices,'15H')
heikenashiDict = holder.heikenashi(HKA,heikenashiKey)
print('10')
paverageDict = holder.paverage(prices,paverageKey)
print('11')
slopeDict = holder.slopes(prices,slopeKey)
print('12')
fourierDict = holder.fourier(prices,fourierKey,method='difference')
print('13')
sineDict = holder.sine(prices,sineKey)
print('14')

"""
## Create list of dictionaries
"""
dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close,
            procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line,
            cciDict.cci,bollingerDict.bands,heikenashiDict.candles,
            paverageDict.avs,slopeDict.slope,fourierDict.coeffs,
            sineDict.coeffs]

"""
## List of 'base' column names
"""
columnFeature = ['momentum','stoch','will','proc','wadl','adosc','macd','cci',
                  'bollinger','heikenashi','paverage','slope','fourier','sine']

"""
## populate the masterframe
"""
masterFrame = pd.DataFrame(index=prices.index)

for i in range(0,len(dictlist)):

    if columnFeature[i] == 'macd':

        columnID = columnFeature[i] + str(keylist[6][0]) + str(keylist[6][1])
        masterFrame[columnID] = dictlist[i]

    else:

        for j in keylist[i]:

            for k in list(dictlist[i][j]):

                columnID = columnFeature[i] + str(j) + k[0]
                masterFrame[columnID] = dictlist[i][j][k]


threshold = round(0.7*len(masterFrame))
masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]
print('15')


"""
## Heiken Ashi is resampled, remove empty data
"""
masterFrame.heikenashi15open = masterFrame.heikenashi15open.fillna(method='bfill')
masterFrame.heikenashi15high = masterFrame.heikenashi15high.fillna(method='bfill')
masterFrame.heikenashi15low = masterFrame.heikenashi15low.fillna(method='bfill')
masterFrame.heikenashi15close = masterFrame.heikenashi15close.fillna(method='bfill')
print('16')


"""
## Drop culomns that have 30% or more NAN data
"""
masterFrameCleaned = masterFrame.copy()
masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
masterFrameCleaned.to_csv('DATA/masterFrame.csv')

totalTime = time.time()-totalStart
print('Feature calculation finished in:', totalTime, 'seconds')


HAresults = holder.heikenashi(HKA,heikenashiKey)
HA = HAresults.candles[15]
trace3 = go.Candlestick(x=HA.index,open=HA.open.values,high=HA.high.values,low=HA.low.values,close=HA.close.values,name='heiken ashi')
fig = tools.make_subplots(rows=1,cols=1,shared_xaxes=True)
fig.append_trace(trace3,1,1)
py.offline.plot(fig,filename='test.html')
















