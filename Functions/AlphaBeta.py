#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:19:54 2024

@author: arnold
"""
from matplotlib import pyplot as plt
import numpy as np

def AlphaBeta(strategy, market, plot=False):
    x, y = market, strategy
    
    coefficients = np.polyfit(x, y, deg=1)
    
    beta = coefficients[0]
    beta = np.round(beta, decimals=3)
    
    alpha = coefficients[1]
    alpha = np.round(alpha, decimals=3)
    
    if plot:
        y_pred = np.polyval(coefficients, x)
        
        plt.scatter(x, y, label='Original Data')
        plt.plot(x, y_pred, color='red', label='Linear Regression')
        text = f'Alpha: {alpha:.2f}\nBeta: {beta:.2f}'
        plt.text(0.2, 0.8, text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('Market Return')  # Changed label
        plt.ylabel('Strategy Return')  # Changed label
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return alpha, beta

def expanded_risk_metrics(strategy_returns, market_returns, rf_rate=0, annualize_para=np.sqrt(252)):
    # Sharpe Ratio
    excess_returns = strategy_returns - rf_rate
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Treynor Ratio
    alpha, beta = AlphaBeta(strategy_returns, market_returns)
    treynor = np.mean(excess_returns) / beta
    
    # Sortino Ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    sortino = np.mean(excess_returns) / np.std(downside_returns)
    
    return {
        'sharpe': annualize_para*sharpe,
        'treynor': annualize_para**2*treynor,
        'sortino': sortino
    }

def downside_metrics(strategy_returns, market_returns):
    
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    
    # Max Drawdown
    max_drawdown = np.min(drawdowns)
    
    # Max Drawdown Duration
    drawdown_periods = []
    current_duration = 0
    peak = True
    
    for i in range(len(cum_returns)):
        if peak:
            if drawdowns[i] < 0:
                peak = False
                current_duration = 1
        else:
            if drawdowns[i] == 0:
                peak = True
                drawdown_periods.append(current_duration)
                current_duration = 0
            else:
                current_duration += 1
                
    # If still in drawdown at the end, add the current duration
    if current_duration > 0:
        drawdown_periods.append(current_duration)
        
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    
    # Value at Risk (VaR)
    var_95 = np.percentile(strategy_returns, 5)
    
    # Conditional VaR (CVaR) / Expected Shortfall
    cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
    
    # Downside Beta
    down_market_returns = market_returns[market_returns < 0]
    down_strategy_returns = strategy_returns[market_returns < 0]
    down_beta = np.cov(down_strategy_returns, down_market_returns)[0,1] / np.var(down_market_returns)
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'down_beta': down_beta
    }