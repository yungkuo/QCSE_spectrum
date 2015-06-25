# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:36:59 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt

def x(lamp, c1, c2, c3, x):
    c1d = np.diff(c1/lamp)
    c2d = np.diff(c2/lamp)
    c3d = np.diff(c3/lamp)
    
    def movingaverage(interval, window_size):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'same')
            
    window_size=5
    c1ds = movingaverage(c1d, window_size)
    c2ds = movingaverage(c2d, window_size)
    c3ds = movingaverage(c3d, window_size)
    slope_max = np.zeros(5)
    slope_max[0] = int(*np.where(c1ds == c1ds.max()))
    slope_max[1] = int(*np.where(c1ds == c1ds.min()))
    slope_max[2] = int(*np.where(c2ds == c2ds.max()))
    slope_max[3] = int(*np.where(c3ds == c3ds.max()))
    slope_max[4] = int(*np.where(c3ds == c3ds.min()))
    
    lambda_fit = np.polyfit(slope_max, np.array([580,620,700,550,630]), 6)
    p = np.polyval(lambda_fit, x)
    
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(np.arange(0,512,1), c1/lamp*100, label='600/40')
    ax[0].plot(np.arange(0,512,1), c2/lamp*100, label='700LP')
    ax[0].plot(np.arange(0,512,1), c3/lamp*100, label='590/80')
    ax[1].plot(np.arange(0.5,511.5,1), c1d, label='600/40d', marker='o', markersize=1)
    ax[1].plot(np.arange(0.5,511.5,1), c2d, label='700LPd',  marker='o', markersize=1)
    ax[1].plot(np.arange(0.5,511.5,1), c3d, label='590/80d',  marker='o', markersize=1)
    ax[1].plot(np.arange(0.5,511.5,1), c1ds, label='600/40ds', linewidth=2)
    ax[1].plot(np.arange(0.5,511.5,1), c2ds, label='700LPds', linewidth=2)
    ax[1].plot(np.arange(0.5,511.5,1), c3ds, label='590/80ds', linewidth=2)
    ax[2].plot(np.arange(0,512,1), p)
    ax[2].scatter(slope_max, [580,620,700,550,630])
    ax[0].set_ylabel('%T')
    ax[1].set_ylabel('dT')
    ax[2].set_ylabel('Wavelength (nm)')
    ax[2].set_ylim(400, 800)    
    ax[2].set_xlabel('Pixel')
    ax[0].set_xlim(0,len(x))
    ax[0].legend(loc=2, frameon='None', fontsize=10)
    ax[1].legend(loc=2, frameon='None', fontsize=10)
    
    return p