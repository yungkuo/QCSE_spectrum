# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 20:06:34 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt




def lambda_cali(refimg):
    print 'calibrate wavelength with refimg'
    #fig = plt.imshow(refimg)
    immean = np.mean(refimg, axis=0)
    fig, (ax, ax2) = plt.subplots(2,1,sharex=True)
    ax.plot(immean, 'b')
    ax.set_xlim([0,len(immean)])
    pts = np.array(plt.ginput(3))  

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    imsmooth = movingaverage(immean,5)
    ax.plot(imsmooth, 'r-') 
    local_max = []
    for i in range(3):
        local_max = np.append(local_max, np.where(imsmooth == imsmooth[pts[i,0]-20:pts[i,0]+21:1].max()))
    lambda_fit = np.polyfit(local_max, np.array([532,594,645]), 2)
    x = np.arange(0,len(immean),1)
    p = np.polyval(lambda_fit, x)
    ax2.plot(x, p)
    ax2.plot(local_max, [532,594,645], 'go')
    ax2.set_xlim([0,len(immean)])
    fig.canvas.draw()
    ax2.set_title('wavelength calibration')
    
    return p