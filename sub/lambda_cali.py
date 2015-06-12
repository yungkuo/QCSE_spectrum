# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 20:06:34 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt


filePath='E:/QCSE data/'

def lambda_cali(bulb, cali1,cali2,cali3):
    print 'calibrate wavelength with optical filters'

    bulb = np.tile(bulb,(3,1,1)).reshape(3,512,512)
    im = np.append(np.append(cali1,cali2),cali3).reshape(3,512,512)
    immean = np.mean(im, axis=1)
    bulbmean = np.mean(bulb[0,:,:], axis=0)
    maxf0 = float(np.mean(np.where(immean[0,:] == immean[0,:].max())))
    maxf1 = float(np.mean(np.where(immean[1,:] == immean[1,:].max())))
    maxf2 = float(np.mean(np.where(immean[2,:] == immean[2,:].max())))
    normalf0 = bulbmean*immean[0,maxf0]/bulbmean[maxf0]
    normalf1 = bulbmean*immean[1,maxf1]/bulbmean[maxf1]
    normalf2 = bulbmean*immean[2,maxf2]/bulbmean[maxf2]
    immean_cr = immean/([normalf0,normalf1,normalf2])
    
    imdiff = np.diff(immean_cr)
    x = np.arange(0.,512, 1)
    xdiff = np.arange(0.5,511.5,1)
    
    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
        
    window_size=10

    imdiffsmth = np.zeros((3,511))
    for i in range(3):        
        imdiffsmth[i,:] = movingaverage(imdiff[i,:],window_size)

    slope_max = np.zeros(4)
    slope_max[0] = int(*np.where(imdiffsmth[0,:] == imdiffsmth[0,350:420].max()))
    slope_max[1] = int(*np.where(imdiffsmth[0,:] == imdiffsmth[0,350:420].min()))
    slope_max[2] = int(*np.where(imdiffsmth[1,:] == imdiffsmth[1,420:460].max()))
    slope_max[3] = int(*np.where(imdiffsmth[2,:] == imdiffsmth[2,200:500].max()))

    lambda_fit = np.polyfit(slope_max, np.array([530,570,600,700]), 3)
    p = np.polyval(lambda_fit, x)



    fig, axarr = plt.subplots(8,1, sharex=True)
    axarr[0].imshow(bulb[0,:,:])
    axarr[1].imshow(cali1)
    axarr[2].imshow(cali2)
    axarr[3].imshow(cali3)
    axarr[4].plot(x, immean_cr[0,:])
    axarr[5].plot(x, immean_cr[1,:])
    axarr[6].plot(x, immean_cr[2,:])
    for tl in axarr[4].get_yticklabels():
        tl.set_color('b')
    for tl in axarr[5].get_yticklabels():
        tl.set_color('b')
    for tl in axarr[6].get_yticklabels():
        tl.set_color('b')
        axarr3 = axarr[4].twinx()
        axarr4 = axarr[5].twinx()
        axarr5 = axarr[6].twinx()
        axarr3.plot(xdiff, imdiff[0,:],'y')
        axarr4.plot(xdiff, imdiff[1,:],'y')
        axarr5.plot(xdiff, imdiff[2,:],'y')
        axarr3.plot(xdiff, imdiffsmth[0,:],'m')
        axarr4.plot(xdiff, imdiffsmth[1,:],'m')
        axarr5.plot(xdiff, imdiffsmth[2,:],'m')
    for tl in axarr3.get_yticklabels():
        tl.set_color('m')
    for tl in axarr4.get_yticklabels():
        tl.set_color('m')
    for tl in axarr5.get_yticklabels():
        tl.set_color('m')
        axarr[7].plot(slope_max, [530,570,600,700], 'ro')
        axarr[7].plot(x, p)
        axarr[7].set_ylim([500,780])
        axarr[7].set_xlim([0,len(np.mean(cali1, axis=0))])
        fig.canvas.draw()
        axarr[0].annotate('Light bulb',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')
        axarr[7].annotate('Wavelength calibration',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='k')
        axarr[1].annotate('550/40BP',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')
        axarr[2].annotate('600LP',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')
        axarr[3].annotate('700LP',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')  
  
        axarr[7].axvline(x=slope_max[0],ymin=-2,ymax=12,c="red",linewidth=1,zorder=1, clip_on=False)
        axarr[7].axvline(x=slope_max[1],ymin=-2,ymax=12,c="red",linewidth=1, zorder=1,clip_on=False)
        axarr[7].axvline(x=slope_max[2],ymin=-2,ymax=12,c="red",linewidth=1,zorder=100, clip_on=False)
        axarr[7].axvline(x=slope_max[3],ymin=-2,ymax=12,c="red",linewidth=1, zorder=1,clip_on=False)
    return p