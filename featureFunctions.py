import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
from mpl_finance import _candlestick
from matplotlib.dates import date2num
from datetime import datetime


class holder:
    1
    def heikenashi(prices,periods):
        """
        Heiken Ashi Candles

        prices: dataframe of OHLC and volume data
        periods: periods for which to create candles
        return: heiken ashi candles

                  open+high+low+close
        HAclose = -------------------
                           4

                 HAopen,prev+HAclose,prev

        HAopen = ------------------------
                            2

        HAhigh = max(high,HAopen,HAclose)


        HAlow = min(low,HAopen,HAclose)

        """
        results = holder()
        dct = {}

        HAclose = prices[['open','high','low','close']].sum(axis=1)/4
        HAopen = HAclose.copy()
        HAopen.iloc[0] = HAclose.iloc[0]
        HAhigh = HAclose.copy()
        HAlow = HAclose.copy()

        for i in range(1,len(prices)):
            HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
            HAhigh.iloc[i] = np.array([prices.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
            HAlow.iloc[i] = np.array([prices.low.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).min()

        df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
        df.columns = [['open','high','low','close']]
        #df.index = df.MultiIndex.droplevel(0)
        dct[periods[0]] = df
        results.candles = dct

        return results


    def detrend(prices,method='linear'):
        """
        Detrend Data

        prices: dataframe of OHLC currency data
        method: method by which to detrend, 'linear' or 'difference'
        return: the detrended price series
        """
        if method == 'difference':

            detrended = prices.close[1:]-prices.close[:-1].values

        elif method == 'linear':

            x = np.arange(0,len(prices))
            y = prices.close.values

            model = LinearRegression()
            model.fit(x.reshape(-1,1),y.reshape(-1,1))
            trend = model.predict(x.reshape(-1,1))
            trend = trend.reshape((len(prices),))
            detrended = prices.close - trend

        else:

            print('Wrong input for detrending: Options are linear or difference')

        return detrended


    def fseries(x,a0,a1,b1,w):
        """
        Fourier fit of detrended data

        x: hours/days
        a0: first fourier series coefficient
        a1: second fourier series coefficient
        b1: third fourier series coefficient
        w: fourier series frequency

        return: value of fourier function

        F = a0 + a1 cos(wx) + b1 sin(wx)
        """

        f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)

        return f


    def sseries(x,a0,b1,w):
        """
        Sine fit of detrended data

        x: hours/days
        a0: first sine series coefficient
        b1: second sine series coefficient
        w: fourier sine frequency

        return: value sine function

        F = a0 + b1 sin(wx)
        """

        f = a0 + b1 * np.sin(w * x)

        return f


    def fourier(prices,periods,method='difference'):
        """
        prices: OHLC dataframe
        periods: list to compute coefficients
        method: method to detrend data
        return: dict of dataframes containing coefficients for given periods
        """

        results = holder()
        dct = {}

        plot = False

        detrended = holder.detrend(prices,method)

        for i in range(0,len(periods)):

            coeffs = []

            for j in range(periods[i],len(prices)-periods[i]):

                x = np.arange(0,periods[i])
                y = detrended.iloc[j-periods[i]:j]

                with warnings.catch_warnings():
                    warnings.simplefilter('error',OptimizeWarning)

                    try:

                        res = scipy.optimize.curve_fit(holder.fseries,x,y)

                    except (RuntimeError,OptimizeWarning):

                        res = np.empty((1,4))
                        res[0,:] = np.NAN

                if plot == True:

                    xt = np.linspace(0,periods[i],100)
                    yt = holder.fseries(xt,res[0][0],res[0][1],res[0][2],res[0][3])

                    plt.plot(x,y)
                    plt.plot(xt,yt,'r')
                    plt.show()

                coeffs = np.append(coeffs,res[0],axis=0)

            warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)

            coeffs = np.array(coeffs).reshape(((len(coeffs)//4,4)))

            df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]])
            df.columns = [['a0','a1','b1','w']]
            df = df.fillna(method='bfill')

            dct[periods[i]] = df

        results.coeffs = dct

        return results


    def sine(prices,periods,method='difference'):
        """
        Sine

        prices: OHLC dataframe
        periods: list to compute coefficients
        method: method to detrend data
        return: dict of dataframes containing coefficients for given periods

        """

        results = holder()
        dct = {}

        plot = False

        detrended = holder.detrend(prices,method)

        for i in range(0,len(periods)):

            coeffs = []

            for j in range(periods[i],len(prices)-periods[i]):

                x = np.arange(0,periods[i])
                y = detrended.iloc[j-periods[i]:j]

                with warnings.catch_warnings():
                    warnings.simplefilter('error',OptimizeWarning)

                    try:

                        res = scipy.optimize.curve_fit(holder.sseries,x,y)

                    except (RuntimeError,OptimizeWarning):

                        res = np.empty((1,3))
                        res[0,:] = np.NAN

                if plot == True:

                    xt = np.linspace(0,periods[i],100)
                    yt = holder.sseries(xt,res[0][0],res[0][1],res[0][2])

                    plt.plot(x,y)
                    plt.plot(xt,yt,'r')
                    plt.show()

                coeffs = np.append(coeffs,res[0],axis=0)

            warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)

            coeffs = np.array(coeffs).reshape(((len(coeffs)//3,3)))

            df = pd.DataFrame(coeffs,index=prices.iloc[periods[i]:-periods[i]])
            df.columns = [['a0','b1','w']]
            df = df.fillna(method='bfill')

            dct[periods[i]] = df

        results.coeffs = dct

        return results



    def wadl(prices,periods):
        """
        Williams Accumulation Distrubution Line

        prices: dataframe of OHLC prices
        periods: list of periods to calculate the function
        return: wadl for each period

        True Range High & Low
        TRH = max(current high,previous close)
        TRL = min(current low,previous colse

        Compare Current Close(CC) to Previous Close(PC) to Price Move(PM)
        CC > PC
            PM = CC - TRL
        CC < PC
            PM = CC - TRH
        CC == PC
            PM = 0

        Calculate Accumulation Distribution(AD)
        AD = PM * volume

        Calculate Total Accumulation Distribution(WAD)
        WAD = AD(current) + AD(previous)
        """

        results = holder()
        dct = {}

        for i in range(0,len(periods)):

            WAD = []

            for j in range(periods[i],len(prices)-periods[i]):

                TRH = np.array([prices.high.iloc[j],prices.close.iloc[j-1]]).max()
                TRL = np.array([prices.low.iloc[j],prices.close.iloc[j-1]]).min()

                if prices.close.iloc[j] > prices.close.iloc[j-1]:

                    PM = prices.close.iloc[j] - TRL

                elif prices.close.iloc[j] < prices.close.iloc[j-1]:

                    PM = prices.close.iloc[j] - TRH

                elif prices.close.iloc[j] == prices.close.iloc[j-1]:

                    PM = 0

                else:

                    print('Unknown Error')


                AD = PM * prices.volume.iloc[j]
                WAD = np.append(WAD,AD)

            WAD = WAD.cumsum()
            WAD = pd.DataFrame(WAD,index=prices.iloc[periods[i]:-periods[i]].index)
            WAD.columns = [['close']]

            dct[periods[i]] = WAD

        results.wadl = dct

        return results


    def OHLCresample(DataFrame,TimeFrame,column='ask'):
        """
        DataFrame: dataframe to resample
        TimeFrame: timeframe to resample
        column: column to resample (bid or ask) default=ask
        return: resampled OHLC data for given timeframe
        """

        #grouped = DataFrame.groupby('date')

        if np.any(DataFrame.columns == 'Ask'):

            if column == 'ask':

                ask = DataFrame['Ask'].resample(TimeFram).ohlc()
                askVol = DataFrame['AskVol'].resample(TimeFrame).count()
                resampled = pd.DataFrame(ask)
                resampled['AskVol'] = askVol

            elif column == 'bid':

                bid = DataFrame['Bid'].resample(TimeFrame).ohlc()
                bidVol = DataFrame['BidVol'].resample(TimeFrame).ohlc()
                resampled = pd.DataFrame(bid)
                resampled['BidVol'] = bidVol

            else:

                raise ValueError('Column must be a string. Either ask or bid')


        elif np.any(DataFrame.columns == 'close'):

            open = DataFrame['open'].resample(TimeFrame).ohlc()
            close = DataFrame['close'].resample(TimeFrame).ohlc()
            high = DataFrame['high'].resample(TimeFrame).ohlc()
            low = DataFrame['low'].resample(TimeFrame).ohlc()
            volume  = DataFrame['volume'].resample(TimeFrame).count()

            resampled = pd.DataFrame(open)
            resampled['high'] = high
            resampled['low'] = low
            resampled['close'] = close
            resampled['volume'] = volume

        resampled = resampled.dropna()

        return resampled


    def momentum(prices,periods):
        """
        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: momentum indicator
        """

        results = holder()
        open = {}
        close = {}

        for i in range(0,len(periods)):

            open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:] - prices.open.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)
            close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)

            open[periods[i]].columns = [['open']]
            close[periods[i]].columns = [['close']]


        results.open = open
        results.close = close

        return results

    def stochastic(prices,periods):
        """
        Stochastic Oscillator

        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: stochastic oscillator function values
        """

        results = holder()
        close = {}

        for i in range(0,len(periods)):

            Ks = []

            for j in range(periods[i],len(prices)-periods[i]):

                C = prices.close.iloc[j+1]
                H = prices.high.iloc[j-periods[i]:j].max()
                L = prices.low.iloc[j-periods[i]:j].min()

                if H == L:

                    K = 0

                else:

                    K = 100*(C-L)/(H-L)

                Ks = np.append(Ks,K)

            df = pd.DataFrame(Ks,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            df.columns = [['K']]
            df['D'] = df.K.rolling(3).mean()
            df = df.dropna()

            close[periods[i]] = df


        results.close = close

        return results


    def williams(prices,periods):
        """
        Williams %R

        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: wiliams oscillator function values
        """

        results = holder()
        close = {}

        for i in range(0,len(periods)):

            Rs = []

            for j in range(periods[i],len(prices)-periods[i]):

                C = prices.close.iloc[j+1]
                H = prices.high.iloc[j-periods[i]:j].max()
                L = prices.low.iloc[j-periods[i]:j].min()

                if H == L:

                    R = 0

                else:

                    R = -100*(H-C)/(H-L)

                Rs = np.append(Rs,R)

            df = pd.DataFrame(Rs,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            df.columns = [['R']]
            df = df.dropna()

            close[periods[i]] = df


        results.close = close

        return results


    def proc(prices,periods):
        """
        Price Rate Of Change

        prices: dataframe of OHLC data
        periods: list of periods to calculate function value
        return: PROC values for indicated periods
        """

        results = holder()
        proc = {}

        for i in range(0,len(periods)):

            proc[periods[i]] = pd.DataFrame((prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values)/prices.close.iloc[:-periods[i]].values)
            proc[periods[i]].columns = [['close']]

        results.proc = proc

        return results


    def adosc(prices,periods):
        """
        Accumulation Distribution Oscillator

        prices: dataframe of OHLC data
        periods: list of periods to calculate indicator
        return: indicator values for indicated periods
        """

        results = holder()
        accdist = {}

        for i in range(0,len(periods)):

            AD = []

            for j in range(periods[i],len(prices)-periods[i]):

                C = prices.close.iloc[j+1]
                H = prices.high.iloc[j-periods[i]:j].max()
                L = prices.low.iloc[j-periods[i]:j].min()
                V = prices.volume.iloc[j+1]

                if H == L:

                    CLV = 0

                else:

                    CLV = ((C-L)-(H-C))/(H-L)

                AD = np.append(AD,CLV*V)

            AD = AD.cumsum()
            AD = pd.DataFrame(AD,index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            AD.columns = [['AD']]

            accdist[periods[i]] = AD

        results.AD = accdist

        return results


    def macd(prices,periods):
        """
        Moving Average Convergence Divergence

        prices: dataframe of OHLC data
        periods: 1x2 array of EMA values
        return: MACD for given periods
        """

        results = holder()

        EMA1 = prices.close.ewm(span=periods[0]).mean()
        EMA2 = prices.close.ewm(span=periods[1]).mean()

        MACD = pd.DataFrame(EMA1-EMA2)
        MACD.columns = [['SL']]

        SigMACD = MACD.rolling(3).mean()
        SigMACD.columns = [['SL']]

        results.line = MACD
        results.signal = SigMACD

        return results



    def cci(prices,periods):
        """
        CCI (Commodity Channel Index)

        prices: dataframe of OHLC data
        periods: list of periods to compute indicator
        return: MACD for given periods
        """

        results = holder()
        CCI = {}

        for i in range(0,len(periods)):

            MA = prices.close.rolling(periods[i]).mean()
            std = prices.close.rolling(periods[i]).std()

            D = (prices.close-MA)/std

            CCI[periods[i]] = pd.DataFrame((prices.close-MA)/(0.015*D))
            CCI[periods[i]].columns = [['close']]


        results.cci = CCI

        return results


    def bollinger(prices,periods,deviations):
        """
        Bollinger Bands

        prices: dataframe of OHLC data
        periods: list of periods to compute bollinger bands
        deviations: deviations to use to calculate bands(upper & lower)
        return: bollinger bands
        """

        results = holder()
        boll = {}

        for i in range(0,len(periods)):

            mid = prices.close.rolling(periods[i]).mean()
            std = prices.close.rolling(periods[i]).std()

            upper = mid+deviations*std
            lower = mid-deviations*std

            df = pd.concat((upper,mid,lower),axis=1)
            df.columns = [['upper','mid','lower']]

            boll[periods[i]] = df

        results.bands = boll

        return results


    def paverage(prices,periods):
        """
        Price Average

        prices: dataframe of OHLC data
        periods: list of periods to compute indicator values
        return: averages over the given periods
        """

        results = holder()
        avs = {}

        for i in range(0,len(periods)):

            avs[periods[i]] = pd.DataFrame(prices[['open','high','low','close']].rolling(periods[i]).mean())


        results.avs = avs

        return results


    def slopes(prices,periods):
        """
        Slope Function

        prices: dataframe of OHLC data
        periods: list of periods to compute indicator values
        return: slopes over the given periods
        """

        results = holder()
        slope = {}

        for i in range(0,len(periods)):

            ms = []

            for j in range(periods[i],len(prices)-periods[i]):

                y = prices.high.iloc[j-periods[i]:j].values
                x = np.arange(0,len(y))

                res = stats.linregress(x,y=y)
                m = res.slope

                ms = np.append(ms,m)


            ms = pd.DataFrame(ms,index=prices.iloc[periods[i]:-periods[i]].index)
            ms.columns = [['high']]

            slope[periods[i]] = ms

        results.slope = slope

        return results


