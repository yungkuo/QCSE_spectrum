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
#import lmfit
#import xlsxwriter

"""
Control Panel
"""
filePath='E:/ND/092315/'
#filePath = '/Users/yungkuo/Documents/Data/061115 CdSe(Te)@CdS/'
fileName='30V_0.4Hz'
abc = 'a'
savefig = 1         # 1 = Yes, save figures, else = No, don't save
frame_start = 2
scan_w = 6          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l = 100         # extract scan_l*2 pixels in length = spectral width
bg_scan = scan_w+12
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
mean_fig = 1        # 1 = display mean movie image, 2 = display mean(log) image, else = display differencial image
nstd = 1.8
assignQDcoor = 0
"""
Import movie; Define parameters
"""
mov = libtiff.TiffFile(filePath+fileName+'.tif')
movie = mov.get_tiff_array()
movie = np.array(movie[:,:,:],dtype='d')

frame = len(movie[:,0,0])
row = len(movie[0,:,0])
col = len(movie[0,0,:])
dt = 0.125
movie[0:frame_start,:,:] = movie[frame_start,:,:]
x = np.arange(0,col,1)
if assignQDcoor == 1:
    coor = np.array([[259, 332],
    [312, 164],
    
    
    ])

#%%
"""
Calibrating wavelength
"""
c1 = filePath+'c1.tif'
mov = libtiff.TiffFile(c1)
c1 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c1[0,:,:]))
c1 = np.mean(c1[0,50:,:],dtype='d', axis=0)-np.mean(c1[0,0:2,:],dtype='d', axis=0)

c2 = filePath+'c2.tif'
mov = libtiff.TiffFile(c2)
c2 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c2[0,:,:]))
c2 = np.mean(c2[0,50:,:],dtype='d', axis=0)-np.mean(c2[0,0:2,:],dtype='d', axis=0)

c3 = filePath+'c3.tif'
mov = libtiff.TiffFile(c3)
c3 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c3[0,:,:]))
c3 = np.mean(c3[0,50:,:],dtype='d', axis=0)-np.mean(c3[0,0:2,:],dtype='d', axis=0)

lamp = filePath+'lamp.tif'
mov = libtiff.TiffFile(lamp)
lamp = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(lamp[0,:,:]))
lamp = np.mean(lamp[0,20:,:],dtype='d', axis=0)-np.mean(lamp[0,0:2,:],dtype='d', axis=0)


x_lambda = lambda_cali_v1.x_lambda(lamp, c1, c2, c3, x)
if savefig == 1:
    plt.savefig(filePath+'fig1_calibration.pdf', format='pdf')

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
fig, ax = plt.subplots()
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
        logmean_I = np.mean(np.log(movie), axis=0)
        ax.imshow(logmean_I, cmap='gray', interpolation='None')
        ax.set_title('Mean(log) image')
    else:
        diff_I = np.mean(np.diff(movie, axis=0), axis=0)
        ax.imshow(diff_I, cmap='gray', interpolation='None')
        ax.set_title('Differential image')

    pts = np.array(plt.ginput(n=0, timeout=0, show_clicks=True))
    if mean_fig == 1:
        pts_new = point1.localmax(mean_I, pts, ax, fig)
        pts_new = point1.localmax(mean_I, pts_new, ax, fig)
        pts_new = point1.localmax(mean_I, pts_new, ax, fig)
    elif mean_fig == 2:
        pts_new = point1.localmax(logmean_I, pts, ax, fig)
        pts_new = point1.localmax(logmean_I, pts_new, ax, fig)
        pts_new = point1.localmax(logmean_I, pts_new, ax, fig)
    else:
        pts_new = point1.localmax(diff_I, pts, ax, fig)
        pts_new = point1.localmax(diff_I, pts_new, ax, fig)
        pts_new = point1.localmax(diff_I, pts_new, ax, fig)
    ax.plot(pts[:,0], pts[:,1], 'r+')
ax.set_xlim(0, col)
ax.set_ylim(row, 0)
ax.plot((pts_new[:,0]-scan_l, pts_new[:,0]-scan_l, pts_new[:,0]+scan_l, pts_new[:,0]+scan_l,pts_new[:,0]-scan_l),
        (pts_new[:,1]+scan_w, pts_new[:,1]-scan_w, pts_new[:,1]-scan_w, pts_new[:,1]+scan_w,pts_new[:,1]+scan_w), '-+', color='b')
ax.plot((pts_new[:,0]-scan_l, pts_new[:,0]-scan_l, pts_new[:,0]+scan_l, pts_new[:,0]+scan_l,pts_new[:,0]-scan_l),
        (pts_new[:,1]+bg_scan, pts_new[:,1]-bg_scan, pts_new[:,1]-bg_scan, pts_new[:,1]+bg_scan,pts_new[:,1]+bg_scan), '-+', color='c', alpha=0.5)

fig.canvas.draw()
if savefig ==1:
    plt.savefig(filePath+fileName+abc+'.fig2_QD.pdf', format='pdf')

#%%
"""
Extract spectra and background around points of interest (QD); Background correction
"""
boxmovie = np.zeros((frame, 2*scan_w, 2*scan_l, len(pts)))
bgmovie = np.zeros((frame, 2*(bg_scan-scan_w), 2*scan_l, len(pts)))
spectra = np.zeros((frame, 2*scan_l, len(pts)))
bgspectra = np.zeros((frame, 2*scan_l, len(pts)))
box_intensity = np.zeros((frame, len(pts)))
bg_intensity = np.zeros((frame, len(pts)))
boxtile = np.zeros((frame*(2*scan_w),2*scan_l, len(pts)))
bgtile = np.zeros((frame*(2*(bg_scan-scan_w)),2*scan_l, len(pts)))
for n in range(len(pts)):
    boxmovie[:,:,:,n] = point1.mask3d(movie, pts_new[n,:], scan_w, scan_l)
    spectra[:,:,n] = np.mean(boxmovie[:,:,:,n] , axis=1)
    box_intensity[:,n] = np.mean(spectra[:,:,n], axis=1)
    boxtile[:,:,n] = np.reshape(boxmovie[:,:,:,n], (frame*(2*scan_w),2*scan_l))

    boxbg = point1.mask3d(movie, pts_new[n,:], bg_scan, scan_l)
    bgmovie[:,:,:,n] = np.append(boxbg[:,:(bg_scan-scan_w),:],boxbg[:,(bg_scan+scan_w):,:], axis=1)
    bgspectra[:,:,n] = np.mean(bgmovie[:,:,:,n] , axis=1)
    bg_intensity[:,n] = np.mean(bgspectra[:,:,n], axis=1)
    bgtile[:,:,n] = np.reshape(bgmovie[:,:,:,n], ((frame*2*(bg_scan-scan_w)),2*scan_l))

spectra_bgcr = spectra - bgspectra
tt = box_intensity - bg_intensity
fig, ax = plt.subplots(len(pts)*4, sharex=True, sharey=False)
extent = [0,frame,0,scan_l*2]
for n in range(len(pts)):
    ax[n*4].imshow((boxtile[:,:,n].T), cmap='Reds', vmin=np.min(boxtile[:,:,n]), vmax=np.max(boxtile[:,:,n]), extent=extent, aspect ='auto', interpolation='None')
    ax[n*4+1].imshow((bgtile[:,:,n].T), cmap='Reds', vmin=np.min(boxtile[:,:,n]), vmax=np.max(boxtile[:,:,n]), extent=extent, aspect = 'auto', interpolation='None')
    ax[n*4+2].plot(box_intensity[:,n], 'b')
    ax[n*4+2].plot(bg_intensity[:,n], 'y')
    ax[n*4+2].set_xlim([0,frame])
ax[n*4+2].set_xlabel('frame')
if savefig ==1:
    fig.savefig(filePath+fileName+abc+'.fig3_tt.pdf', format='pdf', bbox_inches = 'tight')

#%%
"""
Extract Von and Voff spectra 
"""
fig, ax = plt.subplots(len(pts), squeeze=True)
for n in range(len(pts)):
    Von_spec = np.mean([spectra_bgcr[i,:,n] for i in range(frame) if i%2==0],  axis=0)
    Voff_spec = np.mean([spectra_bgcr[i,:,n] for i in range(frame) if i%2==0], axis=0)
    x = x_lambda[pts_new[n,0]-scan_l:pts_new[n,0]+scan_l]
    slope1 = (np.mean(Von_spec[scan_l*2-scan_l*2/10:])-np.mean(Von_spec[:scan_l*2/10]))/(x.max()-x.min())
    slope2 = (np.mean(Voff_spec[scan_l*2-scan_l*2/10:])-np.mean(Voff_spec[:scan_l*2/10]))/(x.max()-x.min())
    ax[n].plot(x, Von_spec, '-r.', label='Von Data')
    ax[n].plot(x, Voff_spec,  'b.', label='Voff Data')
    ax[n].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax[n].set_xlabel('Wavelength (nm)')
    ax[n].set_ylabel('Intensity')
    plt.subplots_adjust(hspace = 0.5)

if savefig ==1:
    fig.savefig(filePath+fileName+abc+'.fig4_spec.pdf', format='pdf', bbox_inches = 'tight')
