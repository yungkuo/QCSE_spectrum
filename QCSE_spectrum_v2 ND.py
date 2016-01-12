# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
from sub import point1, lambda_cali_v1

"""
Control Panel
"""
#filePath='E:/ND/092315/'
filePath = '/Users/yungkuo/Documents/Data/ND/092415/'
fileName='0_2V_4Hz'
abc = 'a'
savefig = 1         # 1 = Yes, save figures, else = No, don't save
frame_start = 2
scan_w = 6          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
displacement = scan_w*2
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
mean_fig = 1        # 1 = display mean movie image, 2 = display mean(log) image, else = display differencial image
assignQDcoor = 0
lambdax = 1
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
c1 = np.mean(c1[0,50:,:],dtype='d', axis=0)-np.mean(c1[0,0:20,:],dtype='d', axis=0)

c2 = filePath+'c2.tif'
mov = libtiff.TiffFile(c2)
c2 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c2[0,:,:]))
c2 = np.mean(c2[0,50:,:],dtype='d', axis=0)-np.mean(c2[0,0:20,:],dtype='d', axis=0)

c3 = filePath+'c3.tif'
mov = libtiff.TiffFile(c3)
c3 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c3[0,:,:]))
c3 = np.mean(c3[0,50:,:],dtype='d', axis=0)-np.mean(c3[0,0:20,:],dtype='d', axis=0)

lamp = filePath+'lamp.tif'
mov = libtiff.TiffFile(lamp)
lamp = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(lamp[0,:,:]))
lamp = np.mean(lamp[0,20:,:],dtype='d', axis=0)-np.mean(lamp[0,0:20,:],dtype='d', axis=0)


x_lambda = lambda_cali_v1.x_lambda(lamp, c1, c2, c3, x)
if savefig == 1:
    plt.savefig(filePath+'fig1_calibration.pdf', format='pdf')

#%%
"""
Play movie
"""
if playmovie == 1:
    import matplotlib.animation as animation
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
for i in range(len(pts)):
    ax.axhline(y=pts_new[i,1]-scan_w, xmin=0, xmax=col)
    ax.axhline(y=pts_new[i,1]+scan_w, xmin=0, xmax=col)
fig.canvas.draw()
if savefig ==1:
    plt.savefig(filePath+fileName+abc+'.fig2_ND.pdf', format='pdf')

#%%
"""
Extract spectra and background around points of interest (QD); Background correction
"""
spectra = np.zeros((frame, col, len(pts)))
bgspectra = np.zeros((frame, col, len(pts)))
for n in range(len(pts)):
    spectra[:,:,n] = np.mean(movie[:,pts_new[n,1]-scan_w:pts_new[n,1]+scan_w,:] , axis=1)
    bgspectra[:,:,n] = np.mean(movie[:,pts_new[n,1]-scan_w-displacement:pts_new[n,1]+scan_w-displacement,:] , axis=1)
spectra_bgcr = spectra - bgspectra
#%%
"""
Extract Von and Voff spectra
"""
fig, ax = plt.subplots(len(pts), squeeze=True)
for n in range(len(pts)):
    Von_spec = np.mean([spectra_bgcr[i,:,n] for i in range(frame) if i%2==1],  axis=0)
    Voff_spec = np.mean([spectra_bgcr[i,:,n] for i in range(frame) if i%2==0], axis=0)
    np.save(filePath+fileName+abc+'{}_'.format(n)+'specs.npy', np.append(Von_spec,Voff_spec).reshape(2,col))
    if lambdax == 1:
        x = x_lambda
    else:
        x = x
    ax[n].plot(x, Von_spec, '-r.', label='Von Data')
    ax[n].plot(x, Voff_spec,  'b.', label='Voff Data')
    ax[n].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    if lambdax == 1:
        ax[n].set_xlabel('Wavelength (nm)')
    else:
        ax[n].set_xlabel('pixel')
    ax[n].set_ylabel('Intensity')
    plt.subplots_adjust(hspace = 0.5)
if savefig ==1:
    fig.savefig(filePath+fileName+abc+'.fig3_spec.pdf', format='pdf', bbox_inches = 'tight')
