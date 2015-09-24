# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:07:49 2015

@author: yungkuo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

filePath = '/Users/yungkuo/Documents/Data/ND/092315/'
x = np.load(filePath+'42V_4Hza_x_lambda.npy')
def listdir_nohidden(path):
    for f in os.listdir(path):
        if f.endswith('specs.npy'):
            yield f
fig, ax = plt.subplots()
for count, file in enumerate(listdir_nohidden(filePath)):
    current_file = os.path.join(filePath, file)
    I1 = np.load(current_file)
    V = file.split('V')[0]
    Hz = file.split('_')[1].split('Hz')[0]
    ax.plot(x, I1[0], '-', label='{}Von'.format(V)+'({}Hz)'.format(Hz))
    ax.plot(x, I1[1], 'o', label='{}Voff'.format(V)+'({}Hz)'.format(Hz))
    ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('Fluorescence Intensity (a.u.)')
