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
        
    window_size=5

    imdiffsmth = np.zeros((3,511))
    for i in range(3):        
        imdiffsmth[i,:] = movingaverage(imdiff[i,:],window_size)

    slope_max = np.zeros(4)
    slope_max[0] = int(*np.where(imdiffsmth[0,:] == imdiffsmth[0,0:300].max()))
    slope_max[1] = int(*np.where(imdiffsmth[0,:] == imdiffsmth[0,0:300].min()))
    slope_max[2] = int(*np.where(imdiffsmth[1,:] == imdiffsmth[1,200:300].max()))
    slope_max[3] = int(*np.where(imdiffsmth[2,:] == imdiffsmth[2,300:400].max()))

    lambda_fit = np.polyfit(slope_max, np.array([530,570,600,700]), 3)
    p = np.polyval(lambda_fit, x)



    fig, axarr = plt.subplots(4,2, sharex=True)
    axarr[0,0].imshow(bulb[0,:,:])
    axarr[0,1].imshow(cali1)
    axarr[2,0].imshow(cali2)
    axarr[2,1].imshow(cali3)
    axarr[1,1].plot(x, immean_cr[0,:])
    axarr[3,0].plot(x, immean_cr[1,:])
    axarr[3,1].plot(x, immean_cr[2,:])
    for tl in axarr[1,1].get_yticklabels():
        tl.set_color('b')
    for tl in axarr[3,0].get_yticklabels():
        tl.set_color('b')
    for tl in axarr[3,1].get_yticklabels():
        tl.set_color('b')
        axarr3 = axarr[1,1].twinx()
        axarr4 = axarr[3,0].twinx()
        axarr5 = axarr[3,1].twinx()
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
        axarr[1,0].plot(slope_max, [530,570,600,700], 'ro')
        axarr[1,0].plot(x, p)
        axarr[1,0].set_ylim([500,780])
        axarr[1,0].set_xlim([0,len(np.mean(cali1, axis=0))])
        fig.canvas.draw()
        axarr[0,0].annotate('Light bulb',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')
        axarr[1,0].annotate('Wavelength calibration',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='k')
        axarr[0,1].annotate('550/40BP',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='b')
        axarr[2,0].annotate('600LP',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')
        axarr[2,1].annotate('700LP',xy=(0,0), xytext=(0.05,0.8), xycoords='axes fraction', color='w')  
    return p