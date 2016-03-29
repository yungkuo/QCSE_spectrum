# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff
import lmfit
"""
Import files
"""
#filePath='E:/NPLs spectrum/150522/'
filePath = '/Users/yungkuo/Google Drive/032816 ONT QD wide field/raw data/with 0.731MOhm resistor 20x/'
fileName = '-1.2V-13.2V'
"""
Import movie; Define parameters
"""
datapath = filePath+fileName
tiffimg = tff.TiffFile(datapath+'.tif')
data = tiffimg.asarray().shape
frame = data[0]
movie = tiffimg.asarray()
frame_start = 2
dt = 1/8
movie[0:frame_start,:,:] = np.zeros((data[1], data[2]))
x_lambda = np.arange(0,data[2],1)
savefig = 0
"""
Extract spectrum
"""
movie_mean = np.mean(movie, axis=0)
spec = np.mean(movie, axis=1)
Von_spec = np.array([spec[i,:] for i in range(frame_start, frame) if i%2==1])
Voff_spec = np.array([spec[i,:] for i in range(frame_start, frame) if i%2==0])
Von_spec_mean = np.mean(Von_spec, axis=0)
Voff_spec_mean = np.mean(Voff_spec, axis=0)
Von_peaks = np.sum(Von_spec*x_lambda, axis=1)/np.sum(Von_spec, axis=1)
Voff_peaks = np.sum(Voff_spec*x_lambda, axis=1)/np.sum(Voff_spec, axis=1)

fig2, ax = plt.subplots(6,1, figsize=(8,18), sharex=True)
ax[0] = plt.subplot2grid((5,1), (0,0), rowspan=3)
ax[1] = plt.subplot2grid((5,1), (3,0), rowspan=1, sharex=ax[0])
ax[2] = plt.subplot2grid((5,1), (4,0), rowspan=1, sharex=ax[0])
ax[0].imshow(np.log(movie_mean), cmap='gray', interpolation='None')
ax[1].plot(Von_spec_mean, 'r')
ax[1].plot(Voff_spec_mean, 'b')
ax[0].set_title('Mean image')
ax[0].set_xlim(0,512)
ax[1].set_ylabel('Intensity')
ax[2].set_xlabel('Pixels')
ax[2].plot(x_lambda, Von_spec_mean-Voff_spec_mean)
ax[2].set_title('Von-Voff')
ax[2].set_ylabel('Von-Voff')
plt.subplots_adjust(hspace = 0.5)
fig2.canvas.draw()

fig3, ax = plt.subplots()
counts, bins, patches = ax.hist(Voff_peaks, bins=30, histtype='stepfilled',color='b', alpha=0.5, label='Voff')
counts, bins, patches = ax.hist(Von_peaks, bins=bins, histtype='stepfilled',color='r', alpha=0.5, label='Von')
ax.set_xticks(bins)
ax.set_xticklabels(np.around(bins,4), rotation=45)
ax.set_title('Peak "center of mass"')
ax.set_ylabel('Counts')
ax.set_xlabel('Wavelength (nm)')
fig3.tight_layout()

#%%
"""
Extract Von and Voff spectra and fit Gaussian
"""
def gauss(x, A, mu, sigma, b):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+b
fig4, ax = plt.subplots()
x = x_lambda
gmod = lmfit.Model(gauss)
params1 = gmod.make_params()
params1['A'].set(value=np.max(Von_spec_mean), min=0)
params1['mu'].set(value=float(np.where(Von_spec_mean == Von_spec_mean.max())[0]), min=0, max=512)
params1['sigma'].set(value=10, max=100)
params1['b'].set(value=100, min=0)
params2 = gmod.make_params()
params2['A'].set(value=np.max(Voff_spec_mean), min=0)
params2['mu'].set(value=float(np.where(Voff_spec_mean == Voff_spec_mean.max())[0]), min=0, max=512)
params2['sigma'].set(value=10, max=100)
params2['b'].set(value=100, min=0)
weights1 = np.ones(len(x))
weights1[params1['mu'].value-55:params1['mu'].value+55] = 1
weights2 = np.ones(len(x))
weights2[params2['mu'].value-55:params2['mu'].value+55] = 1
result1 = gmod.fit(Von_spec_mean, x=x, weights=weights1 , **params1)
result2 = gmod.fit(Voff_spec_mean, x=x, weights=weights2 , **params2)
dL_mean = result1.best_values['mu']-result2.best_values['mu']

ax.plot(x, Von_spec_mean, 'r.', label='Von Data')
ax.plot(x, Voff_spec_mean, 'b.', label='Voff Data')
ax.plot(x, result1.best_fit, '-', label='Von ({} nm)'.format(round(result1.best_values['mu'],3)), color='r')
ax.plot(x, result2.best_fit, '-', label='Voff ({} nm)'.format(round(result2.best_values['mu'],3)), color='b')
ax.annotate('$\Delta$$\lambda$ = {} nm'.format(round(dL_mean,3)), xy=(1,1), xytext=(0.02,0.9), xycoords='axes fraction', fontsize=12)
ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity')
plt.subplots_adjust(hspace = 0.5)
#%%
Von_fit = []
Voff_fit = []
dL = []
for i in range((frame-frame_start)/2):
    gmod = lmfit.Model(gauss)
    params1 = gmod.make_params()
    params1['A'].set(value=np.max(Von_spec[i,:]), min=0)
    params1['mu'].set(value=float(np.where(Von_spec[i,:] == Von_spec[i,:].max())[0][0]), min=0, max=512)
    params1['sigma'].set(value=10, max=100)
    params1['b'].set(value=100)
    params2 = gmod.make_params()
    params2['A'].set(value=np.max(Voff_spec[i,:]), min=0)
    params2['mu'].set(value=float(np.where(Voff_spec[i,:] == Voff_spec[i,:].max())[0][0]), min=0, max=512)
    params2['sigma'].set(value=10, max=100)
    params2['b'].set(value=100)
    result1 = gmod.fit(Von_spec[i,:] , x=x, **params1)
    result2 = gmod.fit(Voff_spec[i,:], x=x, **params2)
    Von_fit = np.append(Von_fit, result1.best_values['mu'])
    Voff_fit = np.append(Voff_fit, result2.best_values['mu'])
    dL = np.append(dL, result1.best_values['mu']-result2.best_values['mu'])

fig5, ax = plt.subplots()
for n in range((frame-frame_start)/2):
    Vonn, bins, patches = ax.hist(Von_fit, bins=40, histtype='stepfilled', alpha=0.5, label='Von', color='r')
    Voffn, bins, patches = ax.hist(Voff_fit, bins=bins, histtype='stepfilled', alpha=0.5, label='Voff', color='b')
    ax.set_xticks(bins)
    ax.set_xticklabels(np.around(bins,4), rotation=45)

#%%
if savefig ==1:
    fig2.savefig(filePath+'results/'+fileName+'.fig2_MeanImg.png', format='png', bbox_inches = 'tight')
    fig3.savefig(filePath+'results/'+fileName+'.fig3_PCMhist.png', format='png', bbox_inches = 'tight')
    fig4.savefig(filePath+'results/'+fileName+'.fig4_Gfit.png', format='png', bbox_inches = 'tight')
    fig5.savefig(filePath+'results/'+fileName+'.fig5_Gfithist.png', format='png', bbox_inches = 'tight')
'''
print dL
f = open(filePath+'_result.txt','a')
f.write('{},'.format(fileName)+'{}\n'.format(dL)) # python will convert \n to os.linesep
f.close()
'''
