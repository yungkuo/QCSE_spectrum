# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:36:59 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import libtiff

def polynomial(p, y):
    return p[0]+p[1]*y+p[2]*y**2

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def x_lambda(lamp, c1, c2, c3, plot, x):
    # lamp, c1, c2, c3 are filepath of the calibration tif figures
    # if plot == 1: plot the lamp, c1, c2, c3 figures; else: no plot
    # x: pixel coordinates
    pixel_mean = [200,512]
    pixel_bg = [0,150]
    print 'pixels taken to calculate transmittance intensity:{}'.format(pixel_mean)
    print 'pixels taken to calculate background:{}'.format(pixel_bg)

    mov = libtiff.TiffFile(c1)
    c1 = mov.get_tiff_array()
    mov = libtiff.TiffFile(c2)
    c2 = mov.get_tiff_array()
    mov = libtiff.TiffFile(c3)
    c3 = mov.get_tiff_array()
    mov = libtiff.TiffFile(lamp)
    lamp = mov.get_tiff_array()

    if plot == 1:
        fig, ax = plt.subplots(1,4, figsize=(8,2))
        ax[0].imshow(np.array(lamp[0,:,:]))
        ax[1].imshow(np.array(c1[0,:,:]))
        ax[2].imshow(np.array(c2[0,:,:]))
        ax[3].imshow(np.array(c3[0,:,:]))
    c1 = np.mean(c1[0,pixel_mean[0]:pixel_mean[1],:],dtype='d', axis=0)-np.mean(c1[0,pixel_bg[0]:pixel_bg[1],:],dtype='d', axis=0)
    c2 = np.mean(c2[0,pixel_mean[0]:pixel_mean[1],:],dtype='d', axis=0)-np.mean(c2[0,pixel_bg[0]:pixel_bg[1],:],dtype='d', axis=0)
    c3 = np.mean(c3[0,pixel_mean[0]:pixel_mean[1],:],dtype='d', axis=0)-np.mean(c3[0,pixel_bg[0]:pixel_bg[1],:],dtype='d', axis=0)
    lamp = np.mean(lamp[0,pixel_mean[0]:pixel_mean[1],:],dtype='d', axis=0)-np.mean(lamp[0,pixel_bg[0]:pixel_bg[1],:],dtype='d', axis=0)

    window_size = 20
    c1T = movingaverage(c1/lamp*100, window_size)
    c2T = movingaverage(c2/lamp*100, window_size)
    c3T = movingaverage(c3/lamp*100, window_size)
    c1d = np.diff(c1T)
    c2d = np.diff(c2T)
    c3d = np.diff(c3T)
    c1ds = movingaverage(c1d, window_size)
    c2ds = movingaverage(c2d, window_size)
    c3ds = movingaverage(c3d, window_size)
    slope_max = np.zeros((6,2))
    slope_max[0,0] = int(*np.where(c1ds == np.max(c1ds[150:250])))
    slope_max[1,0] = int(*np.where(c1ds == np.min(c1ds[150:250])))
    slope_max[2,0] = int(*np.where(c2ds == np.max(c2ds[50:170])))
    slope_max[3,0] = int(*np.where(c2ds == np.min(c2ds[50:170])))
    slope_max[4,0] = int(*np.where(c3ds == np.max(c3ds[50:170])))
    slope_max[5,0] = int(*np.where(c3ds == np.min(c3ds[50:170])))
    slope_max[:,1] = [530,500,630,550,620,580]
    print slope_max
    y = slope_max[:, 0]
    dx = np.delete(x-0.5,len(x)-1)

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(x, c1T, label='510/20')
    ax[0].plot(x, c2T, label='590/80')
    ax[0].plot(x, c3T, label='600/40')
    #ax[1].plot(dx, c1d, label='510/20d', marker='o', markersize=1)
    #ax[1].plot(dx, c2d, label='590/80d',  marker='o', markersize=1)
    #ax[1].plot(dx, c3d, label='600/40d',  marker='o', markersize=1)
    ax[1].plot(dx, c1ds, label='510/20ds', linewidth=1)
    ax[1].plot(dx, c2ds, label='590/80ds', linewidth=1)
    ax[1].plot(dx, c3ds, label='600/40ds', linewidth=1)
    ax[0].set_ylabel('%T')
    ax[1].set_ylabel('dT')
    ax[1].set_xlim(slope_max[:, 0].min()-50, slope_max[:, 0].max()+50)
    ax[0].set_ylim(-1,100)
    ax[0].legend(loc=2, frameon=False, fontsize=10)
    ax[1].legend(loc=2, frameon=False, fontsize=10)
    for i in range(6):
        ax[0].axvline(x=slope_max[i, 0], color='0.7')
        ax[1].axvline(x=slope_max[i, 0], color='0.7')
        ax[2].axvline(x=slope_max[i, 0], color='0.7')
    ax[2].plot(y, slope_max[:, 1], 'ro')
    ax[2].set_ylabel('Wavelength (nm)')
    def constraint_1st_der(p):
        return p[1]+2*p[2]*y
    def constraint_2nd_der(p):
        return 2*p[2]
    def objective(p):
        return ((polynomial(p, y)-slope_max[:, 1])**2).sum()

    cons = (dict(type='ineq', fun=constraint_1st_der))
    res = opt.minimize(objective, x0=np.array([-1.0, -1.0, -1.0]), method='SLSQP', constraints=cons)
    if res.success:
        pars = res.x
        p = polynomial(pars, x)
        ax[2].plot(x, p, '-')
        ax[2].set_ylim(400, 800)
        ax[2].set_xlabel('Pixel')
        #ax[0].set_xlim(0,len(x))
    else:
        print 'Failed'
    return p, fig