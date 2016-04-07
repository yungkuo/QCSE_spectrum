# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
#import libtiff
import tifffile as tff
import matplotlib.animation as animation
from sub import point1, lambda_cali_v2
#from scipy.ndimage.filters import gaussian_filter1d
#from scipy.optimize import curve_fit
import lmfit
import pandas as pd
"""
Import files
"""
#filePath='E:/NPLs spectrum/150522/'
filePath = '/Users/yungkuo/Google Drive/040416 869B Zn coated/'
fileName = '015'
c1 = '510.20.tif'
c2 = '590.80.tif'
c3 = '600.40.tif'
lamp = 'lamp.tif'
"""
Control panel
"""
wavelength_range = (500,650)
framerate = 8       # in unit of Hz
frame_start = 2
frame_stop = 0    # if =0, frame_stop = last frame
scan_w = 3          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l = 25         # extract scan_l*2 pixels in length = spectral width
plot_cali = 0
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
mean_fig = 1        # 1 = display mean movie image, 2 = display mean(log) image, else = display differencial image
findparticle_nstd = 1 # box mean > image[boundary].mean + findparticle_nstd * image[boundary].std is considered a particle
iterations_to_find_threshold = 0
nstd = 1.5            # set blinking threshold to mean(blink off)+std(blink off)*nstd
fit_gauss = 0      #1: fit Gaussian to averaged spectrum, else: fit polynomial, degree = 9
savefig = 0         # 1 = Yes, save figures, else = No, don't save
#%%
"""
Import movie; Define parameters
"""
datapath = filePath+'raw data/'
tiffimg = tff.TiffFile(datapath+fileName+'.tif')
data = tiffimg.asarray().shape
frame = data[0]
movie = tiffimg.asarray()
dt = 1/framerate
movie[0:frame_start,:,:] = np.zeros((data[1], data[2]))
if frame_stop != 0:
    movie = movie[0:frame_stop,:,:]
    frame = frame_stop
x = np.arange(0,data[2],1)
x_frame = np.arange(0,frame,1)
#%%
"""
Calibrating wavelength
"""
c1 = datapath + c1
c2 = datapath + c2
c3 = datapath + c3
lamp = datapath + lamp
x_lambda, fig1 = lambda_cali_v2.x_lambda(lamp, c1, c2, c3, plot_cali, x)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
boundary = np.zeros(2)
boundary[0] = (find_nearest(x_lambda[0:300], wavelength_range[0]))
boundary[1] = (find_nearest(x_lambda[0:300], wavelength_range[1]))
#%%
"""
Play movie
"""
if playmovie == 1:
    fig=plt.figure()
    ims = []
    for i in range(frame):
        im=plt.imshow(np.log(movie[i,:,:]),vmin=np.log(movie.min()),vmax=np.log(movie.max()),cmap='hot')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True, repeat_delay=1000)
    plt.title('movie')
#%%
"""
Define points (QDs) of interest and background regions
"""
def findparticle(image, boundary):
    pts = []
    for i in np.arange(scan_w,data[1]-scan_w,1, dtype='int'):
        for j in np.arange(boundary[1],boundary[0],1, dtype='int'):
            if image[i,j] == np.max(image[(i-scan_w):(i+scan_w),(j-scan_l):(j+scan_l)]):
                #if (np.sum(image[(i-1):(i+2),(j-1):(j+2)])-image[i,j])/8 > (np.sum(image[(i-2):(i+3),(j-2):(j+3)])-np.sum(image[(i-1):(i+2),(j-1):(j+2)]))/16:
                if np.mean(image[(i-scan_w):(i+scan_w),(j-scan_l):(j+scan_l)]) > np.mean(image[:,boundary[1]:boundary[0]])+findparticle_nstd*np.std(image[:,boundary[1]:boundary[0]]):
                    pt = [j,i]
                    pts = np.append(pts, pt)
    return np.reshape(pts,[len(pts)/2,2])

fig2, ax = plt.subplots(figsize=(10,10))
fig2.tight_layout()
if mean_fig == 1:
    image = np.mean(movie, axis=0)
    ax.imshow(image, cmap='gray', interpolation='None')
    ax.set_title('Mean image')
elif mean_fig == 2:
    image = np.mean(np.log(movie[frame_start:,:,:]), axis=0)
    ax.imshow(image, cmap='gray', interpolation='None')
    ax.set_title('Mean(log) image')
else:
    image = np.mean(np.diff(movie, axis=0), axis=0)
    ax.imshow(image, cmap='gray', interpolation='None')
    ax.set_title('Differential image')

pts = findparticle(image, boundary)
bg_indx= np.ones(data[1])
for i in range(len(pts[:,1])):
    bg_indx[int(pts[i,1])-scan_w : int(pts[i,1])+scan_w] = 0
bg_mask = np.tile(bg_indx,(data[2],1)).T

for i in range(len(pts[:,0])):
    ax.plot(pts[i,0],pts[i,1], 'r+')
#ax.imshow(image*bg_mask, alpha=0.1, cmap='afmhot')
ax.set_xlim(0, data[2])
ax.set_ylim(data[1], 0)
ax.plot((pts[:,0]-scan_l, pts[:,0]-scan_l, pts[:,0]+scan_l, pts[:,0]+scan_l,pts[:,0]-scan_l),
        (pts[:,1]+scan_w, pts[:,1]-scan_w, pts[:,1]-scan_w, pts[:,1]+scan_w,pts[:,1]+scan_w), '-+', color='b')
ax.axvline(x=boundary[0], color='y')
ax.axvline(x=boundary[1], color='y')
fig2.canvas.draw()
#%%
"""
Background correction
Extract QD spectra
Blinking threshold
"""
bg = np.mean(movie*bg_mask, axis=1)
boxmovie = np.zeros((frame, 2*scan_w, 2*scan_l, len(pts)), dtype=float)
spectra_bgcr = np.zeros((frame, 2*scan_l, len(pts)), dtype=float)
for n in range(len(pts)):
    boxmovie[:,:,:,n] = point1.mask3d(movie, pts[n,:], scan_w, scan_l)
    spectra_bgcr[:,:,n] = np.mean(boxmovie[:,:,:,n], axis=1)-np.array(bg[:,pts[n,0]-scan_l:pts[n,0]+scan_l])
tt = np.mean(spectra_bgcr, axis=1)
std = np.std(tt[frame_start:,:],axis=0,ddof=1,dtype='d')
threshold = np.mean(tt[frame_start:,:], axis=0)-std/2
dL = pd.Series()

if fit_gauss == 1:
    def gauss(x, A, mu, sigma, slope, b):
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))+slope*x+b
else:
    from lmfit.models import PolynomialModel

extent = [0,frame,0,scan_l*2]
for n in range(len(pts)):
    on_mean = np.mean([tt[i,n] for i in range(frame_start,frame) if tt[i,n] > threshold[n]], axis=0)
    on_std = np.std([tt[i,n] for i in range(frame_start,frame) if tt[i,n] > threshold[n]], axis=0, ddof=1,dtype='d')
    threshold1 = on_mean - on_std*nstd
    for i in range(iterations_to_find_threshold):
        if threshold1 != threshold[n]:
            threshold[n] = threshold1
            on_mean = np.mean([tt[i,n] for i in range(frame_start,frame) if tt[i,n] > threshold[n]], axis=0)
            on_std = np.std([tt[i,n] for i in range(frame_start,frame) if tt[i,n] > threshold[n]], axis=0, ddof=1,dtype='d')
            threshold1 = on_mean - on_std*nstd
            #print threshold[n]
            #ax[1].axhline(y = threshold[n], c='0.3', alpha=0.1)

    x = x_lambda[pts[n,0]-scan_l:pts[n,0]+scan_l]
    Von_spec = np.array([spectra_bgcr[i,:,n] for i in range(frame_start, frame) if i%2==1 and tt[i,n] > threshold[n]])
    Voff_spec = np.array([spectra_bgcr[i,:,n] for i in range(frame_start, frame) if i%2==0 and tt[i,n] > threshold[n]])
    Pon = np.sum(Von_spec*x, axis=1)/np.sum(Von_spec,axis=1)
    Poff = np.sum(Voff_spec*x, axis=1)/np.sum(Voff_spec,axis=1)
    Von_specm = np.mean(Von_spec, axis=0)
    Voff_specm = np.mean(Voff_spec, axis=0)

    if fit_gauss == 1:
        gmod = lmfit.Model(gauss)
        slope1 = (np.mean(Von_specm[scan_l*2-scan_l*2/10:])-np.mean(Von_specm[:scan_l*2/10]))/(x.max()-x.min())
        slope2 = (np.mean(Voff_specm[scan_l*2-scan_l*2/10:])-np.mean(Voff_specm[:scan_l*2/10]))/(x.max()-x.min())
        params1 = gmod.make_params()
        params1['A'].set(value=np.max(Von_specm), min=0)
        params1['mu'].set(value=x_lambda[pts[n,0]-scan_l+int(*np.where(Von_specm == Von_specm.max())[0])], min=400, max=800)
        params1['sigma'].set(value=8, max=100)
        params1['slope'].set(value=slope1)
        params1['b'].set(value=np.min(Von_specm))
        params2 = gmod.make_params()
        params2['A'].set(value=np.max(Voff_specm), min=0)
        params2['mu'].set(value=x_lambda[pts[n,0]-scan_l+int(*np.where(Voff_specm == Voff_specm.max()))], min=400, max=800)
        params2['sigma'].set(value=8, max=100)
        params1['slope'].set(value=slope2)
        params1['b'].set(value=np.min(Voff_specm))
        result1 = gmod.fit(Von_specm, x=x, **params1)
        result2 = gmod.fit(Voff_specm, x=x, **params2)
        fitpeak1 = result1.best_values['mu']
        fitpeak2 = result2.best_values['mu']
        deltaL = fitpeak1 -fitpeak2
    else:
        mod = PolynomialModel(7)
        pars1 = mod.guess(Von_specm, x=x)
        pars2 = mod.guess(Voff_specm, x=x)
        result1 = mod.fit(Von_specm, pars1, x=x)
        result2 = mod.fit(Voff_specm, pars2, x=x)
        fitpeak1 = x[np.where(result1.best_fit == np.max(result1.best_fit))]
        fitpeak2 = x[np.where(result2.best_fit == np.max(result2.best_fit))]
        deltaL = fitpeak1 -fitpeak2
    dL['NR#{}'.format(n)] = deltaL

    fig3, ax = plt.subplots(2,5, figsize=(18,5))
    ax[0,0] = plt.subplot2grid((2,5), (0,0), colspan=4, rowspan=1)
    ax[1,0] = plt.subplot2grid((2,5), (1,0), colspan=4, rowspan=1, sharex=ax[0,0])
    ax[0,4] = plt.subplot2grid((2,5), (0,4), colspan=1, rowspan=1)
    ax[1,4] = plt.subplot2grid((2,5), (1,4), colspan=1, rowspan=1, sharex=ax[0,1])

    ax[0,0].imshow((np.reshape(boxmovie[:,:,:,n], (frame*(2*scan_w),2*scan_l)).T), cmap='afmhot', vmin=np.min(boxmovie[:,:,:,n]), vmax=np.max(boxmovie[:,:,:,n]), extent=extent, aspect ='auto', interpolation='None')
    #ax[1].plot(x_frame, np.mean(np.mean(boxmovie[:,:,:,n], axis=1), axis=1), 'b', label='NR intensity')
    #ax[1].plot(x_frame, tt[:,n], 'g', label='NR -bg')
    #ax[1].plot(x_frame, np.mean(bg[:,pts[n,0]-scan_l:pts[n,0]+scan_l], axis=1), 'y', label='bg intensity')
    #ax[1].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax[1,0].plot(x_frame, tt[:,n],'g', label='bgcr')
    ax[1,0].axhline(y = threshold[n], c='0.3', alpha=0.1)
    ax[1,0].set_xlim([0,frame])
    ax[1,0].axhline(y = threshold[n], c='r')
    ax[1,0].set_ylim(np.min(tt[frame_start:,n]), np.max(tt[frame_start:,n]))
    ax[1,0].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)

    ax[0,4].plot(x, Von_specm, 'r.')#, label='Von data')
    ax[0,4].plot(x, Voff_specm, 'b.')#, label='Voff data')
    ax[0,4].plot(x, result1.best_fit, '-', label='Von ({} nm)'.format(round(fitpeak1,3)), color='r')
    ax[0,4].plot(x, result2.best_fit, '-', label='Voff ({} nm)'.format(round(fitpeak2,3)), color='b')
    ax[0,4].annotate('$\Delta$$\lambda$ = {} nm'.format(round(deltaL,3)), xy=(1,1), xytext=(0.02,1.05), xycoords='axes fraction', fontsize=12)
    ax[0,4].legend(bbox_to_anchor=(1.8, 1.2), frameon=False, fontsize=10)
    ax[0,4].set_xlabel('Wavelength (nm)')
    ax[0,4].set_ylabel('Intensity')
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    #, range=(x.min(), x.max())
    counts, bins, patches = ax[1,4].hist(Pon, bins=50, histtype='stepfilled', alpha=0.5, label='Von', color='r')
    counts, bins, patches = ax[1,4].hist(Poff, bins=bins, histtype='stepfilled', alpha=0.5, label='Voff', color='b')
    ax[1,4].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    fig3.canvas.draw()
    if savefig ==1:
        fig3.savefig(filePath+'results/batch analysis/'+fileName+' result{}.pdf'.format(n), format='pdf', bbox_inches = 'tight')
#%%
if savefig ==1:
    fig1.savefig(filePath+'results/batch analysis/'+'calibration.pdf', format='pdf',bbox_inches = 'tight')
    fig2.savefig(filePath+'results/batch analysis/'+fileName+'_QD selection.pdf', format='pdf', bbox_inches = 'tight')
    #fig4.savefig(filePath+'results/'+fileName+abc+'.fig4_spec.pdf', format='pdf', bbox_inches = 'tight')
    #fig5.savefig(filePath+'results/'+fileName+abc+'.fig5_pp hist.pdf', format='pdf', bbox_inches = 'tight')
    print dL
    f = open(filePath+'results/batch analysis/'+'_result.txt','a')
    f.write('{},'.format(fileName)+'\n{}\n'.format(dL)) # python will convert \n to os.linesep
    f.close()
#workbook = xlsxwriter.Workbook(filePath+'{}_dL.xlsx'.format(fileName))
#worksheet = workbook.add_worksheet()
#worksheet.write_string(0, 0, 'dL ')
#for n in range(len(dL)):
#    worksheet.write(n+1, 0, dL[n])
