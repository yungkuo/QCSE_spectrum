# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:37:32 2015

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt

filePath = '/Users/yung/Documents/Data/QCSE/061515 CdSeCdS rod/'
fileName = '_result.txt'
fname = '/Users/yung/Documents/Data/QCSE/061515 CdSeCdS rod/_result.txt'
f = np.loadtxt(fname)
print f
S = filePath.split('/')
S = S[len(S)-2]
print S
fig, ax = plt.subplots()
ax.hist(f, bins=len(f)/1, color='b', histtype='stepfilled',alpha=0.5, label=S+'(n={})'.format(len(f)))
ax.set_title('QCSE $\Delta$$\lambda$ statistical histogram')
ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
ax.set_xlabel('$\Delta$$\lambda$ (nm)')
ax.set_ylabel('Counts')

filePath = '/Users/yung/Documents/Data/QCSE/061115 CdSe(Te)@CdS/'
fileName = '_result.txt'
fname = '/Users/yung/Documents/Data/QCSE/061115 CdSe(Te)@CdS/_result.txt'
f = np.loadtxt(fname)
print f
S = filePath.split('/')
S = S[len(S)-2]
print S
ax.hist(f, bins=len(f)/1, color='r', histtype='stepfilled',alpha=0.5, label=S+'(n={})'.format(len(f)))
ax.set_title('QCSE $\Delta$$\lambda$ statistical histogram')
ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
fig.canvas.draw()
