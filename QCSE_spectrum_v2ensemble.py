# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
#import libtiff
import matplotlib.animation as animation
from sub import lambda_cali_v2
#from scipy.ndimage.filters import gaussian_filter1d
#from scipy.optimize import curve_fit
#import lmfit
import tifffile as tff
from matplotlib.ticker import FormatStrFormatter
#import pandas as pd
"""
Import files
"""
#filePath='E:/NPLs spectrum/150522/'
filePath = '/Users/yungkuo/Google Drive/030716 sandwich device_13.4V_32onm thick/'
fileName = '40nmCdSeCdS_002'
c1 = '510.20.tif'
c2 = '590.80.tif'
c3 = '600.40.tif'
lamp = 'lamp.tif'
"""
Control panel
"""
framerate = 8       # in unit of Hz
frame_start = 2
frame_stop = 300
scan_w = 4          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l = 60         # extract scan_l*2 pixels in length = spectral width
bg_scan = scan_w
displacement = [0,scan_w*(2)]
plot_cali = 0
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
nstd = 1            # set blinking threshold to mean(blink off)+std(blink off)*nstd
savefig = 0         # 1 = Yes, save figures, else = No, don't save
abc = 'a'
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

#%%
"""
Calibrating wavelength
"""
c1 = datapath + c1
c2 = datapath + c2
c3 = datapath + c3
lamp = datapath + lamp
x = np.arange(0,data[2],1)
x_lambda, fig1 = lambda_cali_v2.x_lambda(lamp, c1, c2, c3, plot_cali, x)
x_lambda = x_lambda[::-1]
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

fig2, ax = plt.subplots(5,1, sharex=True)
ax[0] = plt.subplot2grid((5,1), (0,0), rowspan=4)
ax[1] = plt.subplot2grid((5,1), (4,0), rowspan=1, sharex=ax[0])
ax[0].imshow(movie_mean, cmap='gray', interpolation='None')
ax[1].plot(Von_spec_mean, 'r')
ax[1].plot(Voff_spec_mean, 'b')
ax[0].set_title('Mean image')
ax[0].set_xlim(0,512)
ax[1].set_xlabel('Pixels')
ax[1].set_ylabel('Intensity')
fig2.canvas.draw()

fig3, ax = plt.subplots(2,1)
ax[0].plot(x_lambda, Von_spec_mean-Voff_spec_mean)
ax[0].set_title('Von-Voff')
ax[0].set_ylabel('Differential intensity ')
counts, bins, patches = ax[1].hist(Voff_peaks, bins=20, histtype='stepfilled',color='b', alpha=0.2, label='Voff')
counts, bins, patches = ax[1].hist(Von_peaks, bins=bins, histtype='stepfilled',color='r', alpha=0.2, label='Von')
ax[1].set_xticks(bins)
ax[1].set_xticklabels(bins, rotation=45)
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].set_title('Peak "center of mass"')
ax[1].set_ylabel('Counts')
ax[1].set_xlabel('Wavelength (nm)')
fig3.tight_layout()
'''
def findparticle(image):
    pts = []
    for i in np.arange(scan_w,data[1]-scan_w,1, dtype='int'):
        for j in np.arange(scan_l,data[2]-scan_l,1, dtype='int'):
            if image[i,j] == np.max(image[(i-scan_w):(i+scan_w),(j-scan_l):(j+scan_l)]):
                #if (np.sum(image[(i-1):(i+2),(j-1):(j+2)])-image[i,j])/8 > (np.sum(image[(i-2):(i+3),(j-2):(j+3)])-np.sum(image[(i-1):(i+2),(j-1):(j+2)]))/16:
                if np.mean(image[(i-scan_w):(i+scan_w),(j-scan_l):(j+scan_l)]) > np.mean(image)+1*np.std(image):
                    pt = [i,j]
                    pts = np.append(pts, pt)
    return np.reshape(pts,[len(pts)/2,2])

pts = findparticle(movie_mean)
for i in range(len(pts[:,0])):
    ax[0].plot(pts[i,1],pts[i,0], 'b+')
ax[0].set_xlim(0, data[2])
ax[0].set_ylim(data[1], 0)
fig2.canvas.draw()

pts_new = pts
pts_bg = pts_new-displacement
x_frame = np.arange(0,frame,1)

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
    #ax[n].plot(x, gauss(x, result1.best_values['A'], result1.best_values['mu'], result1.best_values['sigma'], result1.best_values['slope'], result1.best_values['b']), '-', label='Von ({} nm)'.format(round(result1.best_values['mu'],3)), color='r')
    #ax[n].plot(x, gauss(x, result2.best_values['A'], result2.best_values['mu'], result2.best_values['sigma'], result2.best_values['slope'], result2.best_values['b']), '-', label='Voff ({} nm)'.format(round(result2.best_values['mu'],3)), color='b')
    #ax[n].annotate('$\Delta$$\lambda$ = {} nm'.format(round(deltaL,3)), xy=(1,1), xytext=(0.02,0.9), xycoords='axes fraction', fontsize=12)
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
    fig1.savefig(filePath+'results/'+'fig1_calibration.pdf', format='pdf',bbox_inches = 'tight')
    fig2.savefig(filePath+'results/'+fileName+abc+'.fig2_QD.pdf', format='pdf', bbox_inches = 'tight')
    fig3.savefig(filePath+'results/'+fileName+abc+'.fig3_tt.pdf', format='pdf', bbox_inches = 'tight')
    fig4.savefig(filePath+'results/'+fileName+abc+'.fig4_spec.pdf', format='pdf', bbox_inches = 'tight')
    fig5.savefig(filePath+'results/'+fileName+abc+'.fig5_pp hist.pdf', format='pdf', bbox_inches = 'tight')
print dL
f = open(filePath+'_result.txt','a')
f.write('{},'.format(fileName)+'{},'.format(abc)+'{}\n'.format(dL)) # python will convert \n to os.linesep
f.close()
#workbook = xlsxwriter.Workbook(filePath+'{}_dL.xlsx'.format(fileName))
#worksheet = workbook.add_worksheet()
#worksheet.write_string(0, 0, 'dL ')
#for n in range(len(dL)):
#    worksheet.write(n+1, 0, dL[n])
'''