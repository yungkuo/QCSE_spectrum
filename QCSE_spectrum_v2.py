# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
import matplotlib.animation as animation
from sub import point1, lambda_cali_v1
#from scipy.ndimage.filters import gaussian_filter1d
#from scipy.optimize import curve_fit
import lmfit
import pandas as pd
"""
Control Panel
"""
#filePath='E:/NPLs spectrum/150522/'
filePath = '/Users/yungkuo/Documents/Data/QCSE/101315 ZnSeCdS 10nm rod 928/8.83mW/'
fileName = '003-120V'
framerate = 8       # in unit of Hz
frame_start = 2
frame_stop = 600
scan_w = 4          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l = 35         # extract scan_l*2 pixels in length = spectral width
bg_scan = scan_w
displacement = [0,scan_w*(2)]
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
mean_fig = 2        # 1 = display mean movie image, 2 = display mean(log) image, else = display differencial image
nstd = 0            # set blinking threshold to mean(blink off)+std(blink off)*nstd
assignQDcoor = 0
savefig = 1         # 1 = Yes, save figures, else = No, don't save
abc = 'd'
"""
Import movie; Define parameters
"""
mov = libtiff.TiffFile(filePath+fileName+'.tif')
movie = mov.get_tiff_array()
movie = np.array(movie[:frame_stop,:,:],dtype='d')

frame = len(movie[:,0,0])
row = len(movie[0,:,0])
col = len(movie[0,0,:])
dt = 1/framerate
movie[0:frame_start,:,:] = np.zeros((row, col))
x = np.arange(0,col,1)
x_frame = np.arange(0,frame,1)
if assignQDcoor == 1:
    coor = np.array([])

#%%
"""
Calibrating wavelength
"""
c1 = filePath+'c1.tif'
mov = libtiff.TiffFile(c1)
c1 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c1[0,:,:]))
c1 = np.mean(c1[0,50:230,:],dtype='d', axis=0)-np.mean(c1[0,0:2,:],dtype='d', axis=0)

c2 = filePath+'c2.tif'
mov = libtiff.TiffFile(c2)
c2 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c2[0,:,:]))
c2 = np.mean(c2[0,50:230,:],dtype='d', axis=0)-np.mean(c2[0,0:2,:],dtype='d', axis=0)

c3 = filePath+'c3.tif'
mov = libtiff.TiffFile(c3)
c3 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c3[0,:,:]))
c3 = np.mean(c3[0,50:230,:],dtype='d', axis=0)-np.mean(c3[0,0:2,:],dtype='d', axis=0)

lamp = filePath+'lamp.tif'
mov = libtiff.TiffFile(lamp)
lamp = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(lamp[0,:,:]))
lamp = np.mean(lamp[0,20:230,:],dtype='d', axis=0)-np.mean(lamp[0,0:2,:],dtype='d', axis=0)

x_lambda, fig1 = lambda_cali_v1.x_lambda(lamp, c1, c2, c3, x)

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
"""
Define points (QDs) of interest, and their peak position
"""
fig2, ax = plt.subplots(figsize=(10,10))
fig2.tight_layout()
if assignQDcoor == 1:
    pts = coor
    pts_new = coor
    mean_I = np.mean(movie, axis=0)
    ax.imshow(mean_I, cmap='gray', interpolation='None')
    ax.set_title('Mean image (assign coordinates)')
else:
    if mean_fig == 1:
        mean_I = np.mean(movie, axis=0)
        ax.imshow(mean_I, cmap='gray', interpolation='None')
        ax.set_title('Mean image')
    elif mean_fig == 2:
        logmean_I = np.mean(np.log(movie[frame_start:,:,:]), axis=0)
        ax.imshow(logmean_I, cmap='gray', interpolation='None')
        ax.set_title('Mean(log) image')
    else:
        diff_I = np.mean(np.diff(movie, axis=0), axis=0)
        ax.imshow(diff_I, cmap='gray', interpolation='None')
        ax.set_title('Differential image')

    pts = np.array(plt.ginput(n=0, timeout=0, show_clicks=True))
    if mean_fig == 1:
        pts_new = point1.localmax(mean_I, pts, ax, fig2)
        pts_new = point1.localmax(mean_I, pts_new, ax, fig2)
        pts_new = point1.localmax(mean_I, pts_new, ax, fig2)
    elif mean_fig == 2:
        pts_new = point1.localmax(logmean_I, pts, ax, fig2)
        pts_new = point1.localmax(logmean_I, pts_new, ax, fig2)
        pts_new = point1.localmax(logmean_I, pts_new, ax, fig2)
    else:
        pts_new = point1.localmax(diff_I, pts, ax, fig2)
        pts_new = point1.localmax(diff_I, pts_new, ax, fig2)
        pts_new = point1.localmax(diff_I, pts_new, ax, fig2)
    ax.plot(pts[:,0], pts[:,1], 'r+')
    pts_bg = pts_new-displacement
ax.set_xlim(0, col)
ax.set_ylim(row, 0)
ax.plot((pts_new[:,0]-scan_l, pts_new[:,0]-scan_l, pts_new[:,0]+scan_l, pts_new[:,0]+scan_l,pts_new[:,0]-scan_l),
        (pts_new[:,1]+scan_w, pts_new[:,1]-scan_w, pts_new[:,1]-scan_w, pts_new[:,1]+scan_w,pts_new[:,1]+scan_w), '-+', color='b')
ax.plot((pts_bg[:,0]-scan_l, pts_bg[:,0]-scan_l, pts_bg[:,0]+scan_l, pts_bg[:,0]+scan_l,pts_bg[:,0]-scan_l),
        (pts_bg[:,1]+bg_scan, pts_bg[:,1]-bg_scan, pts_bg[:,1]-bg_scan, pts_bg[:,1]+bg_scan,pts_bg[:,1]+bg_scan), '-+', color='c', alpha=0.5)
fig2.canvas.draw()
#%%
"""
Extract spectra and background around points of interest (QD); Background correction
"""
boxmovie = np.zeros((frame, 2*scan_w, 2*scan_l, len(pts)))
bgmovie = np.zeros((frame, 2*bg_scan, 2*scan_l, len(pts)))
fig3, ax = plt.subplots(len(pts)*4, sharex=True, sharey=False)
extent = [0,frame,0,scan_l*2]
for n in range(len(pts)):
    boxmovie[:,:,:,n] = point1.mask3d(movie, pts_new[n,:], scan_w, scan_l)
    bgmovie[:,:,:,n] = point1.mask3d(movie, pts_bg[n,:], bg_scan, scan_l)
    #boxbg = point1.mask3d(movie, pts_new[n,:], bg_scan, scan_l)
    #bgmovie[:,:,:,n] = np.append(boxbg[:,:(bg_scan-scan_w),:],boxbg[:,(bg_scan+scan_w):,:], axis=1)
    ax[n*4].imshow((np.reshape(boxmovie[:,:,:,n], (frame*(2*scan_w),2*scan_l)).T), cmap='afmhot',
                    vmin=np.min(boxmovie[:,:,:,n]), vmax=np.max(boxmovie[:,:,:,n]), extent=extent,
                    aspect ='auto', interpolation='None')
    ax[n*4+1].imshow((np.reshape(bgmovie[:,:,:,n], (frame*(2*bg_scan),2*scan_l)).T), cmap='afmhot',
                      vmin=np.min(bgmovie[:,:,:,n]), vmax=np.max(bgmovie[:,:,:,n]), extent=extent,
                      aspect = 'auto', interpolation='None')
    ax[n*4+2].plot(x_frame[frame_start:], np.mean(np.mean(boxmovie[frame_start:,:,:,n], axis=1), axis=1), 'b', label='NR intensity')
    ax[n*4+2].plot(x_frame[frame_start:], np.mean(np.mean(bgmovie[frame_start:,:,:,n], axis=1), axis=1), 'y', label='bg intensity')
    ax[n*4+2].set_xlim([0,frame])
    ax[n*4+2].legend(bbox_to_anchor=(1.2, 1), frameon=False, fontsize=10)
fig3.canvas.draw()
"""
Blinking threshold
"""
spectra_bgcr = np.mean(boxmovie[:,:,:,:], axis=1)-np.mean(bgmovie[:,:,:,:], axis=1)
tt = np.mean(spectra_bgcr, axis=1)
std = np.std(tt[frame_start:,:],axis=0,ddof=1,dtype='d')
threshold = np.mean(tt[frame_start:,:], axis=0)-std/2
for n in range(len(pts)):
    ax[n*4+3].plot(tt[:,n],'g', label='bgcr')
    ax[n*4+3].axhline(y = threshold[n], c='0.3', alpha=0.1)
    off_mean = np.mean([tt[i,n] for i in range(frame_start,frame) if tt[i,n] < threshold[n]], axis=0)
    off_std = np.std([tt[i,n] for i in range(frame_start,frame) if tt[i,n] < threshold[n]], axis=0, ddof=1,dtype='d')
    threshold1 = off_mean + off_std*nstd
    while threshold1 != threshold[n]:
            threshold[n] = threshold1
            off_mean = np.mean([tt[i,n] for i in range(frame_start,frame) if tt[i,n] < threshold[n]], axis=0)
            off_std = np.std([tt[i,n] for i in range(frame_start,frame) if tt[i,n] < threshold[n]], axis=0, ddof=1,dtype='d')
            threshold1 = off_mean + off_std*nstd
            print threshold[n]
            ax[n*4+3].axhline(y = threshold[n], c='0.3', alpha=0.1)
    ax[n*4+3].axhline(y = threshold[n], c='r')
    ax[n*4+3].set_ylim(np.min(tt[frame_start:,n]), np.max(tt[frame_start:,n]))
    ax[n*4+3].legend(bbox_to_anchor=(1.2, 1), frameon=False, fontsize=10)
fig3.canvas.draw()

#%%
"""
Extract Von and Voff spectra and fit Gaussian
"""
import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)
def gauss(x, A, mu, sigma, slope, b):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+slope*x+b
dL = {}
d = {}
fig4, ax = plt.subplots(len(pts), squeeze=True)
for n in range(len(pts)):
    x = x_lambda[pts_new[n,0]-scan_l:pts_new[n,0]+scan_l]
    Von_spec = np.array([spectra_bgcr[i,:,n] for i in range(frame_start, frame) if i%2==1 and tt[i,n] > threshold[n]])
    Voff_spec = np.array([spectra_bgcr[i,:,n] for i in range(frame_start, frame) if i%2==0 and tt[i,n] > threshold[n]])
    Pon = np.sum(Von_spec*x, axis=1)/np.sum(Von_spec,axis=1)
    Poff = np.sum(Voff_spec*x, axis=1)/np.sum(Voff_spec,axis=1)
    d['Von peak position, NR#{}'.format(n)] = pd.Series(Pon)
    d['Voff peak position, NR#{}'.format(n)] = pd.Series(Poff)
    Von_specm = np.mean(Von_spec, axis=0)
    Voff_specm = np.mean(Voff_spec, axis=0)
    slope1 = (np.mean(Von_specm[scan_l*2-scan_l*2/10:])-np.mean(Von_specm[:scan_l*2/10]))/(x.max()-x.min())
    slope2 = (np.mean(Voff_specm[scan_l*2-scan_l*2/10:])-np.mean(Voff_specm[:scan_l*2/10]))/(x.max()-x.min())
    gmod = lmfit.Model(gauss)
    params1 = gmod.make_params()
    params1['A'].set(value=np.max(Von_specm), min=0)
    params1['mu'].set(value=x_lambda[pts_new[n,0]-scan_l+int(*np.where(Von_specm == Von_specm.max())[0])], min=400, max=800)
    params1['sigma'].set(value=10, max=100)
    params1['slope'].set(value=slope1)
    params1['b'].set(value=np.min(Von_specm))
    params2 = gmod.make_params()
    params2['A'].set(value=np.max(Voff_specm), min=0)
    params2['mu'].set(value=x_lambda[pts_new[n,0]-scan_l+int(*np.where(Voff_specm == Voff_specm.max()))], min=400, max=800)
    params2['sigma'].set(value=10, max=100)
    params2['slope'].set(value=slope2)
    params2['b'].set(value=np.min(Voff_specm))
    result1 = gmod.fit(Von_specm, x=x, **params1)
    result2 = gmod.fit(Voff_specm, x=x, **params2)
    deltaL = result1.best_values['mu']-result2.best_values['mu']
    dL['NR#{}'.format(n)] = deltaL
    ax[n].plot(x, Von_specm, 'r.', label='Von Data')
    ax[n].plot(x, Voff_specm, 'b.', label='Voff Data')
    ax[n].plot(x, gauss(x, result1.best_values['A'], result1.best_values['mu'], result1.best_values['sigma'], result1.best_values['slope'], result1.best_values['b']), '-', label='Von ({} nm)'.format(round(result1.best_values['mu'],3)), color='r')
    ax[n].plot(x, gauss(x, result2.best_values['A'], result2.best_values['mu'], result2.best_values['sigma'], result2.best_values['slope'], result2.best_values['b']), '-', label='Voff ({} nm)'.format(round(result2.best_values['mu'],3)), color='b')
    ax[n].annotate('$\Delta$$\lambda$ = {} nm'.format(round(deltaL,3)), xy=(1,1), xytext=(0.02,0.9), xycoords='axes fraction', fontsize=12)
    ax[n].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax[n].set_xlabel('Wavelength (nm)')
    ax[n].set_ylabel('Intensity')
    plt.subplots_adjust(hspace = 0.5)
#%%
fig5, ax = plt.subplots(len(pts), squeeze=True)
for n in range(len(pts)):
    Vonn, bins, patches = ax[n].hist(d['Von peak position, NR#{}'.format(n)], bins=50, histtype='stepfilled', alpha=0.5, label='Von', color='r')
    Voffn, bins, patches = ax[n].hist(d['Voff peak position, NR#{}'.format(n)], bins=50, histtype='stepfilled', alpha=0.5, label='Voff', color='b')

#%%
if savefig ==1:
    fig1.savefig(filePath+'fig1_calibration.pdf', format='pdf',bbox_inches = 'tight')
    fig2.savefig(filePath+fileName+abc+'.fig2_QD.pdf', format='pdf', bbox_inches = 'tight')
    fig3.savefig(filePath+fileName+abc+'.fig3_tt.pdf', format='pdf', bbox_inches = 'tight')
    fig4.savefig(filePath+fileName+abc+'.fig4_spec.pdf', format='pdf', bbox_inches = 'tight')
    fig5.savefig(filePath+fileName+abc+'.fig5_pp hist.pdf', format='pdf', bbox_inches = 'tight')
print dL
f = open(filePath+'_result.txt','a')
f.write('{},'.format(fileName)+'{},'.format(abc)+'{}\n'.format(dL)) # python will convert \n to os.linesep
f.close()
#workbook = xlsxwriter.Workbook(filePath+'{}_dL.xlsx'.format(fileName))
#worksheet = workbook.add_worksheet()
#worksheet.write_string(0, 0, 'dL ')
#for n in range(len(dL)):
#    worksheet.write(n+1, 0, dL[n])
