from iqoptionapi.stable_api import IQ_Option
import time
import logging
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from matplotlib.animation import FuncAnimation
import numpy as np
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
login = IQ_Option('patradanai_n@hotmail.com', 'letminjojo')
login.set_max_reconnect(-1)
endTime = time.time()
dataCandleSum = []
dataClose = []
dataOpen = []
dataHigh = []
dataLow = []
countMt = 0
countWin = 0
countLose = 0
maxMt = 0

fig = plt.figure()
ax1 = plt.subplot2grid((1, 1), (0, 0))
# while True:
#     # Reconnect
#     if login.check_connect() == False:
#         print("try reconnect")
#         login.connect()
#     time.sleep(1)

# for i in range(2):
#     dataCandle = login.get_candles('EURUSD', 60, 1000, endTime)
#     dataCandleSum = dataCandleSum + dataCandle
#     endTime = int(dataCandle[0]['from']) - 1

# for data in dataCandleSum:
#     dataClose.append(data['close'])
#     dataOpen.append(data['open'])
#     dataHigh.append(data['max'])
#     dataLow.append(data['min'])

# for i in range(len(dataOpen)-3):

#     if countMt > maxMt:
#         maxMt = countMt

#     if dataHigh[i] > dataHigh[i+1] and dataLow[i] > dataLow[i+1]:
#         if dataOpen[i+2] > dataClose[i+2]:
#             countWin += 1
#             countMt = 0
#         else:
#             countLose += 1
#             countMt += 1
#     elif dataHigh[i] < dataHigh[i+1] and dataLow[i] < dataLow[i+1]:
#         if dataOpen[i+2] < dataClose[i+2]:
#             countWin += 1
#             countMt = 0
#         else:
#             countLose += 1
#             countMt += 1

# print('win = ', countWin)
# print('loss = ', countLose)
# print('mt = ', countMt)


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


def movingAverage(values, window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas  # as a numpy array


def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow


def matplotRealTime(i):
    ohlc = []
    endTime = time.time()
    dataCandle = login.get_candles('EURUSD', 60, 100, endTime)
    dataIndex = 1

    if len(dataCandle) > 0:
        for data in dataCandle:
            candleData = dataIndex, data['open'], data['max'], data['min'], data['close'], data['volume']
            ohlc.append(candleData)
            dataIndex += 1

    ax1.clear()
    ax1.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("IQ OPTION")
    candlestick_ohlc(ax1, ohlc, width=0.4,
                     colorup='#77d879', colordown='#db3f3f')


animation = FuncAnimation(fig, matplotRealTime, interval=1000)

plt.show()
# goal = 'EURUSD'
# size = 60
# maxdict = 10
# print("Start Steam...")
# login.start_candles_stream(goal, size, maxdict)


# print('print candles')
# cc = login.get_realtime_candles(goal, size)
# for k in cc:
#     print(goal, "size", k, cc[k])

# print("Stop candle")

# login.stop_candles_stream(goal, size)
