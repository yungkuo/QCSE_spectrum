# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:36:59 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def polynomial(p, y):
    return p[0]+p[1]*y+p[2]*y**2

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def x_lambda(lamp, c1, c2, c3, x):
    window_size=20
    c1T = c1/lamp*100
    c2T = c2/lamp*100
    c3T = c3/lamp*100
    c1d = np.diff(c1/lamp)
    c2d = np.diff(c2/lamp)
    c3d = np.diff(c3/lamp)
    c1ds = movingaverage(c1d, window_size)
    c2ds = movingaverage(c2d, window_size)
    c3ds = movingaverage(c3d, window_size)
    slope_max = np.zeros((5,2))
    slope_max[0,0] = int(*np.where(c1ds == np.max(c1ds[200:280])))
    slope_max[1,0] = int(*np.where(c1ds == np.min(c1ds[280:350])))
    slope_max[2,0] = int(*np.where(c2ds == np.max(c2ds[300:380])))
    slope_max[3,0] = int(*np.where(c3ds == np.max(c3ds[200:280])))
    slope_max[4,0] = int(*np.where(c3ds == np.min(c3ds[280:350])))
    slope_max[:,1] = [580,620,700,550,630]
    print slope_max
    y = slope_max[:, 0]
    dx = np.delete(x-0.5,len(x)-1)
    def constraint_1st_der(p):
        return p[1]+2*p[2]*y

    def objective(p):
        return ((polynomial(p, y)-slope_max[:, 1])**2).sum()

    cons = (dict(type='ineq', fun=constraint_1st_der))
    res = opt.minimize(objective, x0=np.array([0., 0., 0.]), method='SLSQP', constraints=cons)
    if res.success:
        pars = res.x
        p = polynomial(pars, x)

        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(x, c1T, label='600/40')
        ax[0].plot(x, c2T, label='700LP')
        ax[0].plot(x, c3T, label='590/80')
        ax[1].plot(dx, c1d, label='600/40d', marker='o', markersize=1)
        ax[1].plot(dx, c2d, label='700LPd',  marker='o', markersize=1)
        ax[1].plot(dx, c3d, label='590/80d',  marker='o', markersize=1)
        ax[1].plot(dx, c1ds, label='600/40ds', linewidth=2)
        ax[1].plot(dx, c2ds, label='700LPds', linewidth=2)
        ax[1].plot(dx, c3ds, label='590/80ds', linewidth=2)
        ax[2].plot(x, p, '-')
        ax[2].plot(y, slope_max[:, 1], 'ro')
        ax[0].set_ylabel('%T')
        ax[1].set_ylabel('dT')
        ax[0].set_ylim(np.min(c3T[slope_max[:, 0].min()-50:slope_max[:, 0].max()+50]), np.max(c3T[slope_max[:, 0].min()-50: slope_max[:, 0].max()+50]))
        ax[1].set_ylim(np.min(c3d[slope_max[:, 0].min()-50:slope_max[:, 0].max()+50]), np.max(c3d[slope_max[:, 0].min()-50: slope_max[:, 0].max()+50]))
        ax[2].set_ylabel('Wavelength (nm)')
        ax[2].set_xlim(slope_max[:, 0].min()-50, slope_max[:, 0].max()+50)
        ax[2].set_ylim(400, 800)
        ax[2].set_xlabel('Pixel')
        #ax[0].set_xlim(0,len(x))
        ax[0].legend(loc=2, frameon='None', fontsize=10)
        ax[1].legend(loc=2, frameon='None', fontsize=10)
    else:
        print 'Failed'
    return p, fig