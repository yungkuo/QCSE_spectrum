# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:07:49 2015

@author: yungkuo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

filePath = '/Users/yungkuo/Documents/Data/ND/092415/'
x = np.load(filePath+'67V_4Hzx_lambda.npy')
def listdir_nohidden(path):
    for f in os.listdir(path):
        if f.endswith('2_specs.npy'):
            yield f
V = []
Fon = []
Foff = []
fig, ax = plt.subplots()
for count, file in enumerate(listdir_nohidden(filePath)):
    current_file = os.path.join(filePath, file)
    I1 = np.load(current_file)
    F1 = np.sum(I1[0])
    F2 = np.sum(I1[1])
    V1 = file.split('V')[0]
    Hz = file.split('_')[1].split('Hz')[0]
    ax.plot(x, I1[0], '-', label='{}Von'.format(V1)+'({}Hz)'.format(Hz))
    ax.plot(x, I1[1], 'o', markersize=4, label='{}Voff'.format(V1)+'({}Hz)'.format(Hz))
    ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('Fluorescence Intensity (a.u.)')
    V = np.append(V,V1)
    Fon = np.append(Fon, F1)
    Foff = np.append(Foff, F2)
fig, ax = plt.subplots()
ax.scatter(V, Fon, label='Von')
ax.scatter(V, Foff, label='Voff')
ax.set_xlabel('voltage(V)')
ax.set_ylabel('integrated fluorescence intensity')