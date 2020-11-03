#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:06:38 2020

Published to GitHub as alpha version on November 3, 2020

goldilocks_filter.py

accepts a ticker or list of tickers
reads raw csv files into Pandas DataFrame
computes daily gain / loss percent for all days
forms pdf by yearcolumn
computes pseudo-tradelist for each year
computes safe-f and CAR25 for each year
writes disk file with ticker, safe-f, CAR25 values


@author: howard bandy
October 6, 2020

Open source
License:  MIT

"""

import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import statistics


def make_one_equity_sequence(
    trades,
    fraction,
    number_days_in_forecast,
    number_trades_in_forecast,
    initial_capital         ):

    """
    Given a set of trades, draw a random sequence of trades
    and form an equity sequence.
    
    Parameters:
    trades:           the set of trades to be analyzed
    fraction:         the proportion of the trading account
                      to be used for each trade.
    number_days_in_forecast:    Length in days of forecast.                  
    number_trades_in_forecast:  Length in trades of the equity sequence.
    initial_capital:  Starting value of the trading account.
    
    Returns:  
    Two scalars:
    equity:  The equity at the end of the sequence in dollars.
    max_drawdown:  The maximum drawdown experienced in the sequence
            as a proportion of highest equity marked to market
            after each trade.
    """

    #  initialize sequence

    equity = initial_capital
    max_equity = equity
    drawdown = 0.0
    max_drawdown = 0.0

    #  form the equity curve to display, if desired
    daily_equity = np.zeros(number_days_in_forecast)

    #  form sequence

    for i in range(number_trades_in_forecast):
        trade_index = random.randint(0, len(trades) - 1)
        trade = trades[trade_index]
        trade_dollars = equity * fraction * trade
        equity = equity + trade_dollars
        daily_equity[i] = equity
        max_equity = max(equity, max_equity)
        drawdown = (max_equity - equity) / max_equity
        max_drawdown = max(drawdown, max_drawdown)
    #  if necessary, fill remaining days    
    for i in range(number_trades_in_forecast,number_days_in_forecast):
        daily_equity[i] = equity
        
#    plt.plot(daily_equity)
#    plt.show()
    
    return (equity, max_drawdown)

def analyze_distribution_of_drawdown(
    trades,
    fraction,
    number_days_in_forecast,
    number_trades_in_forecast,
    initial_capital,
    tail_percentile,
    number_equity_in_CDF   ):

    """
    Returns:
    tail_risk:  The maximum drawdown at the tail_percentile
                    of the distribution using the 
                    current value of the position size.
    """
    equity_list = []
    max_dd_list = []
    sorted_max_dd = []

    for i in range(number_equity_in_CDF):
        equity, max_drawdown = make_one_equity_sequence(
                                trades, 
                                fraction, 
                                number_days_in_forecast,
                                number_trades_in_forecast,
                                initial_capital)
        equity_list.append(equity)
        max_dd_list.append(max_drawdown)
#        print (f'equity:  {equity:0.3f}  max_dd:  {max_drawdown:0.3f}')
    sorted_max_dd = np.sort(max_dd_list)
#    plt.plot(sorted_max_dd)
#    plt.show()

    tail_risk = np.percentile(sorted_max_dd, 100 - tail_percentile)

    return tail_risk

def form_distribution_of_equity(
    trades,
    fraction,
    number_days_in_forecast,
    number_trades_in_forecast,
    initial_capital,
    number_equity_in_CDF       ):
    
#    plt.hist(trades,bins=50)
#    plt.show()
    equity_list = []
    max_dd_list = []
    sorted_equity = []

    for i in range(number_equity_in_CDF):
        equity, max_drawdown = make_one_equity_sequence(
                                trades, 
                                fraction, 
                                number_days_in_forecast,
                                number_trades_in_forecast,
                                initial_capital)
        equity_list.append(equity)
        max_dd_list.append(max_drawdown)

    sorted_equity = np.sort(equity_list)
#    plt.plot(sorted_equity)
#    plt.show()

    return sorted_equity

#@numba.jit(target='cuda')
def risk_normalization(
        trades, 
        number_days_in_forecast, 
        number_of_trades_in_forecast,
        initial_capital, 
        tail_percentile, 
        drawdown_tolerance, 
        number_equity_in_CDF  
        ):

    #  Set number of repetitions of calculation to get stddev of mean
    number_repetitions = 10
    safe_fs = []
    TWR25s = []
    CAR25s = []
    
    #  Set accuracy condition
    desired_accuracy = 0.003
    
    for rep in range(number_repetitions):
    
        #  Fraction is initially set to use all available funds
        #  It will be adjusted in response to the risk of drawdown.
        #  The final value of fraction is safe-f
        
        fraction = 1.0
        done = False
        while not done:
#            print(f"fraction this pass:  {fraction:0.3f}")
            tail_risk = analyze_distribution_of_drawdown(
                                                 trades, 
                                                 fraction,
                                                 number_days_in_forecast,
                                                 number_trades_in_forecast,
                                                 initial_capital,
                                                 tail_percentile,
                                                 number_equity_in_CDF)
        
#            print(f"tail_risk this pass: {tail_risk:0.3f}")
            if abs(tail_risk - drawdown_tolerance) < desired_accuracy:
                done = True
            else:
#                print (f'fraction:  {fraction:0.3f}  drawdown_tolerance: {drawdown_tolerance:0.3f}  tail_risk: {tail_risk:0.3f}')
                fraction = fraction * drawdown_tolerance / tail_risk
#            print (f'fraction this pass: {fraction:0.3f}')    
        
#        print(f'final value: safe_f: {fraction:0.3f}')
        
        #  Compute CAR25
        #  fraction == safe_f
        #  Compute CDF of equity
        #  TWR25 is 25th percentile
        #  CAR25 is 25th percentile
        
        CDF_equity = form_distribution_of_equity(trades, 
                                                 fraction,
                                                 number_days_in_forecast,
                                                 number_trades_in_forecast,
                                                 initial_capital,
                                                 number_equity_in_CDF)
        TWR25 = np.percentile(CDF_equity, 25)
        # print(f'terminal wealth: {TWR25:9.0f}')
        
        CAR25 = 100.0 * (math.exp((252.0 / number_days_in_forecast) * 
                                 math.log(TWR25/initial_capital)) - 1.0)
        
        # print(f'Compound Annual Return: {CAR25:0.3f}%')
    
        safe_fs.append(fraction)
        TWR25s.append(TWR25)
        CAR25s.append(CAR25)
    
    #  end of rep loop
       
        
    # print(safe_fs)
    # print(TWR25s)
    # print(CAR25s)
    
    print (f'mean and standard deviation are based on {number_repetitions}'
           ' calculations')    
    safe_f_mean = statistics.mean(safe_fs)
    print (f'safe_f_mean:   {safe_f_mean:0.3f}')
    if number_repetitions > 1:
        safe_f_stdev = statistics.stdev(safe_fs)
        print (f'safe_f_stdev:  {safe_f_stdev:0.3f}')
    else:
        safe_f_stdev = 1.0
        print ('standard deviation calculation is not meaningful')
    
    TWR25_mean = statistics.mean(TWR25s)
    print (f'TWR25_mean:   {TWR25_mean:0.0f}')
    if number_repetitions > 1:
        TWR25_stdev = statistics.stdev(TWR25s)
        print (f'TWR25_stdev:  {TWR25_stdev:0.3f}')
    else:
        TWR25_stdev = 1.0
        print ('standard deviation calculation is not meaningful')
    
    CAR25_mean = statistics.mean(CAR25s)
    print (f'CAR25_mean:   {CAR25_mean:0.3f}%')
    if number_repetitions > 1:
        CAR25_stdev = statistics.stdev(CAR25s)
        print (f'CAR25_stdev:  {CAR25_stdev:0.3f}%')
    else:
        CAR25_stdev = 1.0
        print ('standard deviation calculation is not meaningful')
    
    return (safe_f_mean, CAR25_mean)




################   Main program starts

filepath='/FinancialData/FromNorgate/US_Equities/'
ticker = 'XLV.csv'
print (f'Using stock: {ticker}')

df = pd.read_csv(filepath+ticker,index_col='Date',parse_dates=(True))

# print (df.head())
# print (df.tail())
# print(df.info())
df.columns = ['symbol', 'o', 'h', 'l', 'c', 'v']
# print (df.head())
# print (df.tail())
# print(df.info())

first_date = df.index.min()
#print (f'first_date: {first_date}')
last_date = df.index.max()  
#print (f'last_date: {last_date}')
today_date = dt.datetime.now()
#print (f'today_date: {today_date}')

first_year = '2017'
final_year = '2020'
print ('isolating period to test: ', first_year, final_year)
#subset = df.loc['2017':'2020'].copy()
subset = df.loc[first_year:final_year].copy()
#print (subset.head())
#print (subset.tail())

#  working with 'subset' DataFrame
#  assume target is close to next close
#  signals are beLong or beFlat
#  trader expects to be 70% correct (accuracy)

subset['gain_ahead'] = (subset['c'].shift(-1) - subset['c']) / subset['c']
subset.dropna(inplace=True)
print(f'\n\nnumber of entries in historical data for this period: {len(subset)}')
#print (subset.tail())

acc_list = []
safef_list = []
CAR25_list = []

for acc_int in range(80,81):
    accuracy = 0.01*acc_int
    acc_list.append(accuracy)
#    accuracy = 0.65
    print (f'\nanticipated accuracy: {accuracy:0.3f}')
    
    #  sort gain_ahead into gains (price rises) and losses (price drops)
    gains = []
    losses = []
    for chg in subset['gain_ahead']:
        if chg > 0:
            gains.append(chg)
        else:
            losses.append(chg)    
    print (f'number gains: {len(gains)}  number losses: {len(losses)}')
    
    #  pick 4 years -- 1008 elements to create best_estimate
    #  pick 2 years -- 500 days
    length_in_sample = 500
    trades = np.zeros(length_in_sample)
    #random.seed = 13331
    for i in range(length_in_sample):
        #  pick a trade
        rn1 = random.uniform(0.0,1.0)
        if (rn1 < accuracy):
            # winner
            rn2 = random.randint(0,len(gains)-1)
            trade = gains[rn2]
            trades[i] = trade
        else:
            # loser    
            rn2 = random.randint(0,len(losses)-1)
            trade = losses[rn2]
            trades[i] = trade
    
    # print ('trades: ')
    # for i in range(20):
    #     print (f'{trades[i]:0.4f}') 
    # if np.isnan(trades).any():
    #     print ('trades has nan values')
    
    number_days_in_forecast = 252  #  1 year    504   # 2 years
    number_trades_in_forecast = number_days_in_forecast
    initial_capital = 100000
    tail_percentile = 5
    drawdown_tolerance = 0.10
    number_equity_in_CDF=1000
    
    print (f'number of trades in trades "best_estimate": {len(trades)}')          
    print (f'number_days_in_forecast: {number_days_in_forecast}')
    print (f'number_trades_in_forecast: {number_trades_in_forecast}')
    print (f'initial_capital: {initial_capital}')
    print (f'tail_percentile: {tail_percentile}')
    print (f'drawdown_tolerance: {drawdown_tolerance:0.2f}')
    print (f'number_equity_in_CDF: {number_equity_in_CDF} ')
    
    
    safe_f, CAR25 = risk_normalization(
                        trades,
                        number_trades_in_forecast,
                        number_days_in_forecast,
                        initial_capital,
                        tail_percentile,
                        drawdown_tolerance,
                        number_equity_in_CDF
                      )

    safef_list.append(safe_f)
    CAR25_list.append(CAR25)

print (acc_list, CAR25_list)

plt.scatter(acc_list,CAR25_list)
plt.show()

plt.scatter(acc_list,safef_list)
plt.show()

####  end   ####