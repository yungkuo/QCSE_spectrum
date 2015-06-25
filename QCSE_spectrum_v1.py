# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
import matplotlib.animation as animation 
from sub import point1, lambda_cali1
#from scipy.ndimage.filters import gaussian_filter1d
#from scipy.optimize import curve_fit
import lmfit

"""
Control Panel
"""
#filePath='E:/NPLs spectrum/150522/'
filePath = '/Users/yung/Documents/Data/QCSE/150615 CdSeCdS rod/'
fileName='120V-003'
abc = 'a'
savefig = 0         # 1 = Yes, save figures, else = No, don't save
backGND_corr = 1    # 1 = apply correction, else = no correction
Time_corr = 0       # 1 = apply polynomial fit, 2 = apply Gaussian filter, else = no correction
frame_start = 2
scan_w = 3          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l = 30         # extract scan_l*2 pixels in length = spectral width
scan_bg = scan_w*5
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
mean_fig = 1        # 1 = display mean movie image, else = display differencial image


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
t = frame*dt
T = np.arange(0,t,dt)
T_3d = np.tile(T[:,np.newaxis,np.newaxis],(1,row,col))
movie[0:frame_start,:,:] = movie[frame_start,:,:]
x = np.arange(0,col,1)
polydeg = 7
polydeg_bg = 33
polydeg_pb = 8


"""
Calibrating wavelength
"""
c1 = filePath+'c1.tif'
mov = libtiff.TiffFile(c1)
c1 = mov.get_tiff_array()
c1 = np.mean(c1[0,50:,:],dtype='d', axis=0)-np.mean(c1[0,0:50,:],dtype='d', axis=0)

c2 = filePath+'c2.tif'
mov = libtiff.TiffFile(c2)
c2 = mov.get_tiff_array()
c2 = np.mean(c2[0,50:,:],dtype='d', axis=0)-np.mean(c2[0,0:50,:],dtype='d', axis=0)

c3 = filePath+'c3.tif'
mov = libtiff.TiffFile(c3)
c3 = mov.get_tiff_array()
c3 = np.mean(c3[0,50:,:],dtype='d', axis=0)-np.mean(c3[0,0:50,:],dtype='d', axis=0)

lamp = filePath+'lamp.tif'
mov = libtiff.TiffFile(lamp)
lamp = mov.get_tiff_array()
lamp = np.mean(lamp[0,50:,:],dtype='d', axis=0)-np.mean(lamp[0,0:50,:],dtype='d', axis=0)


x_lambda = lambda_cali1.x(lamp, c1, c2, c3, x)
if savefig == 1:
    plt.savefig(filePath+'calibration.pdf', format='pdf')




"""
Click background positions
"""
fig, ax = plt.subplots()
if mean_fig == 1:
    mean_I = np.mean(movie, axis=0)
    ax.imshow(mean_I, cmap='gray')
    ax.set_title('Mean image')
else:
    diff_I = np.mean(np.diff(movie, axis=0), axis=0)
    ax.imshow(diff_I, cmap='gray')
    ax.set_title('Differential image')

bg_pt = np.array(plt.ginput(1, timeout=0))
ax.axhline(y = bg_pt[0,1]-scan_bg, xmin=0, xmax=col, c='w')
ax.axhline(y = bg_pt[0,1]+scan_bg, xmin=0, xmax=col, c='w')
fig.canvas.draw()
if savefig == 1:
    plt.savefig(filePath+fileName+abc+'.fig2.pdf', format='pdf')

"""
Background correction & time correction
"""
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

window_size=10

bg = np.mean(np.mean(movie[:, bg_pt[0,1]-scan_bg:bg_pt[0,1]+scan_bg,:], axis=0), axis=0)
bg_fit = np.polyfit(x, bg, polydeg_bg) 
bg_val = np.polyval(bg_fit, x)
#bg_val = movingaverage(bg, window_size)

bg_3d = np.tile(bg_val[np.newaxis,np.newaxis,:],(frame,row,1))
movie_bgcr = movie - bg_3d
movie_bgcrmean = np.mean(np.mean(movie_bgcr[:,bg_pt[0,1]-scan_bg:bg_pt[0,1]+scan_bg,:],axis=0),axis=0)
movie_t = np.mean(np.mean(movie_bgcr,axis=1),axis=1)
#movie_pb = movingaverage(movie_t,window_size)
pb_fit = np.polyfit(T[frame_start:len(T):1],movie_t[frame_start:len(T):1],polydeg_pb)
pb_val = np.polyval(pb_fit,T)
movie_pbcr = movie_t-pb_val
#pbc = pb_fit[1]/pb_val

#gaufil = gaussian_filter1d(movie_bgcr, sigma=80, axis=0)
#movie_fil = movie_bgcr-gaufil
#movie_fil_t = np.sum(np.sum(movie_fil,axis=1),axis=1)/(row*col)

fig,(ax1,ax2) = plt.subplots(2,sharex=False)
ax1.plot(x, bg, '.', label='background')
ax1.plot(x, bg_val, label='polyfit({})'.format(polydeg_bg))
ax1.plot(x,movie_bgcrmean, label='after bgcr')
ax2.plot(T[frame_start:len(T):1],movie_t[frame_start:len(T):1], label='time trace, I')
ax2.plot(T[frame_start:len(T):1],pb_val[frame_start:len(T):1], label='polyfit({})'.format(polydeg_pb))
ax2.plot(T[frame_start:len(T):1],movie_pbcr[frame_start:len(T):1], label='I- polyfit')
#ax3.plot(T[frame_start:len(T)-window_size:1],movie_fil_t[frame_start:len(T)-window_size:1], label='high pass filter=2')
ax1.set_title('Background correction')
ax2.set_title('Time trace correction')
ax1.set_xlabel('pixels')
ax2.set_xlabel('time (s)')
ax1.set_ylabel('fluorescence intensity')
ax2.set_ylabel('fluorescence intensity/pixel')
ax1.set_xlim([0,col])
ax1.legend(bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0, fontsize=10)
ax2.legend(bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0, fontsize=10)
ax1.annotate('Background correction={}'.format(backGND_corr),xy=(0,0), xytext=(0.7,1), xycoords='axes fraction', fontsize=10)
ax2.annotate('Time correction={}'.format(Time_corr),xy=(0,0), xytext=(0.7,1), xycoords='axes fraction', fontsize=10)
plt.subplots_adjust(hspace = 0.5)
if savefig ==1:
    plt.savefig(filePath+fileName+abc+'.fig3.pdf', format='pdf', bbox_inches = 'tight')    
    
    
if backGND_corr == 1:
    if Time_corr == 1:
        mov_f = np.zeros((frame,row,col))
        for i in range(frame):
            mov_f[i,:,:] = movie_bgcr[i,:,:]-pb_val[i] 
    #elif Time_corr ==2:
        #mov_f = movie_fil
    else: 
        mov_f=movie_bgcr
else:
    mov_f=movie


"""
Play movie
"""
if playmovie == 1:
    fig=plt.figure()
    ims = []
    for i in range(frame):
        im=plt.imshow(mov_f[i,:,:],vmin=mov_f.min(),vmax=mov_f.max(),cmap='hot')
        ims.append([im]) 
    ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True, repeat_delay=1000)
    plt.title('movie')  



"""
Define points (QDs) of interest, and their peak position
"""
fig, ax = plt.subplots()
if mean_fig == 1:
    new_mean_I = np.mean(mov_f, axis=0)
    ax.imshow(new_mean_I, cmap='gray', interpolation='None')
    ax.set_title('Mean image')
else:
    new_diff_I = np.mean(np.diff(mov_f, axis=0), axis=0)
    ax.imshow(new_diff_I, cmap='gray', interpolation='None')
    ax.set_title('Differential image')

pts = np.array(plt.ginput(n=0, timeout=0, show_clicks=True))
if mean_fig == 1:
    pts_new = point1.localmax(new_mean_I, pts, ax, fig)
    pts_new = point1.localmax(new_mean_I, pts_new, ax, fig)
    pts_new = point1.localmax(new_mean_I, pts_new, ax, fig)
else:
    pts_new = point1.localmax(diff_I, pts, ax, fig)
    pts_new = point1.localmax(diff_I, pts_new, ax, fig)    
    pts_new = point1.localmax(diff_I, pts_new, ax, fig)
ax.plot(pts[:,0], pts[:,1], 'r+')
ax.set_xlim(0, col)
ax.set_ylim(row, 0)
ax.plot((pts_new[:,0]-scan_l, pts_new[:,0]-scan_l, pts_new[:,0]+scan_l, pts_new[:,0]+scan_l,pts_new[:,0]-scan_l), 
        (pts_new[:,1]+scan_w, pts_new[:,1]-scan_w, pts_new[:,1]-scan_w, pts_new[:,1]+scan_w,pts_new[:,1]+scan_w), '-+', color='b')
fig.canvas.draw()
if savefig ==1:
    plt.savefig(filePath+fileName+abc+'.fig4.pdf', format='pdf')

#%%
"""
Extract spectra from (scan_w*scan_l) box around points of interest (QD)
"""
boxmovie = np.zeros((frame, 2*scan_w, 2*scan_l, len(pts)))
spectra = np.zeros((frame, 2*scan_l, len(pts)))
box_intensity = np.zeros((frame, len(pts)))
boxtile = np.zeros((frame*(2*scan_w),2*scan_l, len(pts)))
fig, ax = plt.subplots(len(pts)*2,1, sharex=True, sharey=False)
for n in range(len(pts)): 
    boxmovie[:,:,:,n]  = point1.mask3d(mov_f, pts_new[n,:], scan_w, scan_l)
    spectra[:,:,n] = np.mean(boxmovie[:,:,:,n] , axis=1)        
    box_intensity[:,n] = np.mean(np.mean(boxmovie[:,:,:,n], axis=1), axis=1) 
    boxtile[:,:,n] = np.reshape(boxmovie[:,:,:,n], (frame*(2*scan_w),2*scan_l))
    ax[n*2].imshow((boxtile[:,:,n].T), cmap='gray')
    ax[n*2+1].plot(np.arange(0,frame,1,dtype='int'),box_intensity[:,n], 'b')     
    ax[n*2+1].set_xlim([0,frame])
    ax[n*2+1].set_xlabel('frame')
    
"""
Blinking threshold
"""
std = np.std(box_intensity[frame_start:,:],axis=0,ddof=1,dtype='d')
threshold = np.mean(box_intensity[frame_start:,:], axis=0)-std/2
box_intensity_a = box_intensity[frame_start:,:]
for n in range(len(pts)):
    ax[n*2+1].axhline(y = threshold[n], c='0.3', alpha=0.1)
    print threshold[n]
    off_sum = []    
    for i in range(frame-frame_start):
        if box_intensity_a[i,n] < threshold[n]:
            off_sum = np.append(off_sum, box_intensity_a[i,n])   
    off_mean = np.mean(off_sum)
    off_std = np.std(off_sum,ddof=1,dtype='d')
    threshold1 = off_mean + off_std*1.3  
    while threshold1 != threshold[n]:
        threshold[n] = threshold1
        print threshold[n]
        ax[n*2+1].axhline(y = threshold[n], c='0.3', alpha=0.1)
        off_sum = []    
        for i in range(frame-frame_start):
            if box_intensity_a[i,n] < threshold[n]:
                off_sum = np.append(off_sum, box_intensity_a[i,n])   
        off_mean = np.mean(off_sum)
        off_std = np.std(off_sum,ddof=1,dtype='d')
        threshold1 = off_mean + off_std*1.35
    ax[n*2+1].axhline(y = threshold[n], c='r')
#%%
"""
Extract Von and Voff spectra and fit Gaussian
"""
def gauss(x, A, mu, sigma, slope, b):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+slope*x+b
fig, ax = plt.subplots(len(pts))
for n in range(len(pts)):
    Von_spec = np.mean([spectra[i,:,n] for i in range(frame) if i%2==0 and box_intensity[i,n] > threshold[n]], axis=0)
    Voff_spec = np.mean([spectra[i,:,n] for i in range(frame) if i%2==1 and box_intensity[i,n] > threshold[n]], axis=0)    
    x = x_lambda[pts_new[n,0]-scan_l:pts_new[n,0]+scan_l]
    slope1 = (np.mean(Von_spec[scan_l*2-scan_l*2/10:])-np.mean(Von_spec[:scan_l*2/10]))/(x.max()-x.min())
    slope2 = (np.mean(Voff_spec[scan_l*2-scan_l*2/10:])-np.mean(Voff_spec[:scan_l*2/10]))/(x.max()-x.min())
    gmod = lmfit.Model(gauss)
    params1 = gmod.make_params()
    params1['A'].set(value=np.max(Von_spec), min=0)
    params1['mu'].set(value=x_lambda[pts_new[n,0]-scan_l+int(*np.where(Von_spec == Von_spec.max()))], min=400, max=800)
    params1['sigma'].set(value=10, max=100)
    params1['slope'].set(value=slope1)
    params1['b'].set(value=np.min(Von_spec))    
    params2 = gmod.make_params()
    params2['A'].set(value=np.max(Voff_spec), min=0)
    params2['mu'].set(value=x_lambda[pts_new[n,0]-scan_l+int(*np.where(Voff_spec == Voff_spec.max()))], min=400, max=800)
    params2['sigma'].set(value=10, max=100)
    params2['slope'].set(value=slope2)
    params2['b'].set(value=np.min(Voff_spec)) 
    result1 = gmod.fit(Von_spec, x=x, **params1)
    result2 = gmod.fit(Voff_spec, x=x, **params2)
    ax[n].plot(x, Von_spec, 'ro', label='Von Data')
    ax[n].plot(x, Voff_spec,  'bo', label='Voff Data')
    ax[n].plot(x, gauss(x, result1.best_values['A'], result1.best_values['mu'], result1.best_values['sigma'], result1.best_values['slope'], result1.best_values['b']), '-', label='Von', color='r')
    ax[n].plot(x, gauss(x, result2.best_values['A'], result2.best_values['mu'], result2.best_values['sigma'], result2.best_values['slope'], result2.best_values['b']), '-', label='Voff', color='b')            
    ax[n].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax[n].set_xlabel('Wavelength (nm)')
    ax[n].set_ylabel('Intensity')
    plt.subplots_adjust(hspace = 0.5)
'''
    p02 = [np.max(Voff_spec), x_lambda[pts_new[n,1]-scan_l+int(*np.where(Voff_spec == Voff_spec.max()))] , 10, slope2, np.min(Voff_spec)]    
    coeff1, var_matrix1 = curve_fit(gauss, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l:1], Von_spec, p0=p01)
    coeff2, var_matrix2 = curve_fit(gauss, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l:1], Voff_spec, p0=p02)    
    Von_avg_gfit = gauss(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], *coeff1)
    Voff_avg_gfit = gauss(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], *coeff2)

    Von_avg_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1],Von_spec,polydeg)
    Voff_avg_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1],Voff_spec,polydeg)
    Von_avg_p = np.polyval(Von_avg_fit, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])
    Voff_avg_p = np.polyval(Voff_avg_fit, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])
    Von_avg_peak = x_lambda[pts_new[n,1]-scan_l+int(*np.where(Von_avg_p == Von_avg_p.max()))] 
    Voff_avg_peak = x_lambda[pts_new[n,1]-scan_l+int(*np.where(Voff_avg_p == Voff_avg_p.max()))]
   
    Von_avg_gpeak = x_lambda[pts_new[n,1]-scan_l+int(*np.where(Von_avg_gfit == Von_avg_gfit.max()))] 
    Voff_avg_gpeak = x_lambda[pts_new[n,1]-scan_l+int(*np.where(Voff_avg_gfit == Voff_avg_gfit.max()))]
      

    #fit Gaussian to every single frame

    Von_gpeak = []
    for j in range(len(Von_spec[:,0])):     
        slopes1 = (np.mean(Von_spec[j,scan_l*2+1-10:])-np.mean(Von_spec[j,0:10]))/(x_lambda[pts_new[n,1]+scan_l+1]-x_lambda[pts_new[n,1]-scan_l])
        ps01 = [np.max(Von_spec[j,:]), x_lambda[pts_new[n,1]-scan_l+int(*np.where(Von_spec[j,:] == Von_spec[j,:].max()))] , 10, slopes1, np.min(Von_spec[j,:])]
        try:        
            coeffs1, var_matrixs1 = curve_fit(gauss, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_spec[j,:], p0=ps01)
        except RuntimeError:
            print('Error - {}Von_curve_fit failed'.format(j))
        Von_gfit = gauss(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], *coeffs1)
        Von_gfit_max = x_lambda[int(pts_new[n,1]-scan_l+int(*np.where(Von_gfit == np.max(Von_gfit))))]
        Von_gpeak = np.append(Von_gpeak, Von_gfit_max)
        #fig, ax = plt.subplots()
        #ax.plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_gfit, 'b-')
        #ax.plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_spec[j,:], 'bo')
    Von_gpeak = Von_gpeak.reshape(len(Von_spec[:,0]), 1)
    
    Voff_gpeak = []
    for k in range(len(Voff_spec[:,0])):     
        slopes2 = (np.mean(Voff_spec[k,scan_l*2+1-10:])-np.mean(Voff_spec[k,0:10]))/(x_lambda[pts_new[n,1]+scan_l+1]-x_lambda[pts_new[n,1]-scan_l])
        ps02 = [np.max(Voff_spec[k,:]), x_lambda[pts_new[n,1]-scan_l+int(*np.where(Voff_spec[k,:] == Voff_spec[k,:].max()))] , 10, slopes2, np.min(Voff_spec[k,:])]
        try:            
            coeffs2, var_matrixs2 = curve_fit(gauss, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_spec[k,:], p0=ps02)
        except RuntimeError:
            print('Error - {}Voff_curve_fit failed'.format(k))
        Voff_gfit = gauss(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], *coeffs2)
        Voff_gfit_max = x_lambda[int(pts_new[n,1]-scan_l+int(*np.where(Voff_gfit == np.max(Voff_gfit))))]
        Voff_gpeak = np.append(Voff_gpeak, Voff_gfit_max)
    Voff_gpeak = Voff_gpeak.reshape(len(Voff_spec[:,0]), 1)


#    #fit polynomial to every single frame
#    Von_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], np.transpose(Von_spec), polydeg)
#    Voff_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], np.transpose(Voff_spec), polydeg)

#    Von_p = []
#    Voff_p = []  
#    Von_peak = []
#    Voff_peak = []
#    
#    for j in range(len(Von_fit[0,:])):    
#        a = np.polyval(Von_fit[:,j], x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])        
#        a_max = x_lambda[pts_new[n,1]-scan_l+int(*np.where(a == a.max()))]        
#        #Von_p = np.append(Von_p, a, axis=0)
#        Von_peak = np.append(Von_peak, a_max)
#        
#    #Von_p = Von_p.reshape(len(Von_fit[0,:]), scan_l*2+1)
#    Von_peak = Von_peak.reshape(len(Von_fit[0,:]), 1)
#    
#    for k in range(len(Voff_fit[0,:])): 
#        b = np.polyval(Voff_fit[:,k], x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])
#        b_max = x_lambda[pts_new[n,1]-scan_l+int(*np.where(b == b.max()))]        
#        #Voff_p = np.append(Voff_p, b,axis=0)
#        Voff_peak = np.append(Voff_peak, b_max)
#    #Voff_p = Voff_p.reshape(len(Voff_fit[0,:]), scan_l*2+1)
#    Voff_peak = Voff_peak.reshape(len(Voff_fit[0,:]), 1)

    
    ax2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg, 'bo', fillstyle='none')
    #ax2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg_p, 'b-', label='Von polyfit={}'.format(polydeg))
    ax2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg_gfit, 'b-', label='Von gsnfit')    
    ax2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg, 'ro', fillstyle='none')
    #ax2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg_p, 'r-', label='Voff polyfit={}'.format(polydeg))
    ax2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg_gfit, 'r-', label='Voff gsnfit')    
    ax2[n,0].set_xlim([x_lambda[pts_new[n,1]-scan_l],x_lambda[pts_new[n,1]+scan_l+1]])    
    handles, labels = ax2[n,0].get_legend_handles_labels()    
    ax2[n,0].legend(handles, labels,bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0, fontsize=10)
    #ax2[n,0].annotate('Polynomial', xytext=(0.7,1), xy=(Von_avg_peak,0), xycoords='axes fraction', fontsize=12)     
    #ax2[n,0].annotate(r'$\Delta\lambda=${}nm'.format(round(Von_avg_peak-Voff_avg_peak,3)),xy=(0,0), xytext=(0.7,0.9), xycoords='axes fraction', fontsize=12) 
    #ax2[n,0].annotate(r'$Von \lambda=${}nm'.format(round(Von_avg_peak,1)),xy=(Von_avg_peak,0), xytext=(0.7,0.8), xycoords='axes fraction', fontsize=10)    
    #ax2[n,0].annotate(r'$Voff \lambda=${}nm'.format(round(Voff_avg_peak,1)),xy=(Voff_avg_peak,0), xytext=(0.7,0.7), xycoords='axes fraction', fontsize=10)    
    #ax2[n,0].annotate('Gaussian', xytext=(0,1), xy=(Von_avg_peak,0), xycoords='axes fraction', fontsize=12)     
    ax2[n,0].annotate(r'$\Delta\lambda=${}nm'.format(round(Von_avg_gpeak-Voff_avg_gpeak,3)),xy=(0,0), xytext=(0,0.9), xycoords='axes fraction', fontsize=10) 
    ax2[n,0].annotate(r'$Von \lambda=${}nm'.format(round(Von_avg_gpeak,1)),xy=(Von_avg_gpeak,0), xytext=(0,0.8), xycoords='axes fraction', fontsize=10)    
    ax2[n,0].annotate(r'$Voff \lambda=${}nm'.format(round(Voff_avg_gpeak,1)),xy=(Voff_avg_gpeak,0), xytext=(0,0.7), xycoords='axes fraction', fontsize=10)    
   

    
    
    #ax2[n,1].hist(Von_peak, bins=len(Von_peak)/3, color='b', histtype='stepfilled',alpha=0.5, label='Von')
    #ax2[n,1].hist(Voff_peak, bins=len(Voff_peak)/3, color='r', histtype='stepfilled',alpha=0.5, label='Voff')
    ax2[n,1].hist(Von_gpeak, bins=len(Von_gpeak)/3, color='b', histtype='stepfilled',alpha=0.5, label='Von gsn')
    ax2[n,1].hist(Voff_gpeak, bins=len(Voff_gpeak)/3, color='r', histtype='stepfilled',alpha=0.5, label='Voff gsn')
    ax2[n,1].set_xlim([x_lambda[pts_new[n,1]-scan_l],x_lambda[pts_new[n,1]+scan_l+1]])    
    handles, labels = ax2[n,1].get_legend_handles_labels()    
    ax2[n,1].legend(handles, labels,bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0, fontsize=10)
if savefig == 1:
    fig.savefig(filePath+fileName+abc+'.fig5.pdf', format='pdf')
    fig2.savefig(filePath+fileName+abc+'.fig6.pdf', format='pdf', bbox_inches = 'tight')

splitrow = 2
for i in range(len(pts)):
    fig, ax = plt.subplots(splitrow,1)
    for k in range(splitrow):    
        ax[k].imshow((boxtile[len(boxtile[:,0,i])/splitrow*k:len(boxtile[:,0,i])/splitrow*(k+1),:,i].T), cmap='gray')
    if savefig ==1:
        fig.savefig(filePath+fileName+abc+'.fig{}'.format(7+i)+'.pdf', format='pdf', bbox_inches = 'tight')

#fig3, axarr3 = plt.subplots(npoint)
#for j in range(npoint):    
#    ims = []
#    for i in range(frame):
#        im=axarr3[j].imshow(boxmovie[i,:,:,j],vmin=boxmovie[:,:,:,j].min(),vmax=boxmovie[:,:,:,j].max(),cmap='hot')
#        ims.append([im]) 
#    ani = animation.ArtistAnimation(fig3, ims, interval=dt*1000, blit=True,repeat_delay=1000)
#    writer = animation.writers['ffmpeg'](fps=1/dt)
#ani.save(filePath+'movie.mp4',writer=writer,dpi=100)
'''