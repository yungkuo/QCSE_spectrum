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
<<<<<<< HEAD
=======
=======
#filePath='E:/NPLs spectrum/150522/'
<<<<<<< HEAD
filePath = '/Users/yungkuo/Documents/Data/QCSE/092115 ZnSeCdS 10nm rod/'
fileName='007-120V'
abc = 'a'
=======
filePath = '/Users/yungkuo/Documents/Data/061115 CdSe(Te)@CdS/'
fileName='4_120V'
abc = 'a1'
>>>>>>> origin/master
>>>>>>> origin/master
>>>>>>> origin/master
savefig = 1         # 1 = Yes, save figures, else = No, don't save
frame_start = 2
scan_w = 4          # extract scan_w*2 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l = 35         # extract scan_l*2 pixels in length = spectral width
bg_scan = scan_w+2
playmovie = 0       # 1 = Yes, play movie, else = No, don't play
mean_fig = 2        # 1 = display mean movie image, 2 = display mean(log) image, else = display differencial image
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
if assignQDcoor ==1:
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
<<<<<<< HEAD
c1 = np.mean(c1[0,50:,:],dtype='d', axis=0)-np.mean(c1[0,0:20,:],dtype='d', axis=0)
#ax.plot(x,c1)
=======
c1 = np.mean(c1[0,50:,:],dtype='d', axis=0)-np.mean(c1[0,0:2,:],dtype='d', axis=0)
>>>>>>> origin/master

c2 = filePath+'c2.tif'
mov = libtiff.TiffFile(c2)
c2 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c2[0,:,:]))
<<<<<<< HEAD
c2 = np.mean(c2[0,50:,:],dtype='d', axis=0)-np.mean(c2[0,0:20,:],dtype='d', axis=0)
#ax.plot(x,c2)
=======
c2 = np.mean(c2[0,50:,:],dtype='d', axis=0)-np.mean(c2[0,0:2,:],dtype='d', axis=0)
>>>>>>> origin/master

c3 = filePath+'c3.tif'
mov = libtiff.TiffFile(c3)
c3 = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(c3[0,:,:]))
<<<<<<< HEAD
c3 = np.mean(c3[0,50:,:],dtype='d', axis=0)-np.mean(c3[0,0:20,:],dtype='d', axis=0)
#ax.plot(x,c3)
=======
c3 = np.mean(c3[0,50:,:],dtype='d', axis=0)-np.mean(c3[0,0:2,:],dtype='d', axis=0)
>>>>>>> origin/master

lamp = filePath+'lamp.tif'
mov = libtiff.TiffFile(lamp)
lamp = mov.get_tiff_array()
#fig, ax = plt.subplots()
#ax.imshow(np.array(lamp[0,:,:]))
<<<<<<< HEAD
lamp = np.mean(lamp[0,20:,:],dtype='d', axis=0)-np.mean(lamp[0,0:20,:],dtype='d', axis=0)
#ax.plot(x,lamp)
=======
lamp = np.mean(lamp[0,20:,:],dtype='d', axis=0)-np.mean(lamp[0,0:2,:],dtype='d', axis=0)

>>>>>>> origin/master

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
    ax[n*4].imshow((boxtile[:,:,n].T), cmap='gray', vmin=np.min(boxtile[:,:,n]), vmax=np.max(boxtile[:,:,n]), extent=extent, aspect ='auto', interpolation='None')
    ax[n*4+1].imshow((bgtile[:,:,n].T), cmap='gray', vmin=np.min(boxtile[:,:,n]), vmax=np.max(boxtile[:,:,n]), extent=extent, aspect = 'auto', interpolation='None')
    ax[n*4+2].plot(box_intensity[:,n], 'b')
    ax[n*4+2].plot(bg_intensity[:,n], 'y')
    ax[n*4+2].set_xlim([0,frame])
ax[n*4+2].set_xlabel('frame')

"""
Blinking threshold
"""
std = np.std(tt[frame_start:,:],axis=0,ddof=1,dtype='d')
threshold = np.mean(tt[frame_start:,:], axis=0)-std/2
tt_a = tt[frame_start:,:]
for n in range(len(pts)):
    ax[n*4+3].plot(tt[:,n])
    ax[n*4+3].axhline(y = threshold[n], c='0.3', alpha=0.1)
    #print threshold[n]
    off_append = []
    for i in range(frame-frame_start):
        if tt_a[i,n] < threshold[n]:
            off_append = np.append(off_append, tt_a[i,n])
    off_mean = np.mean(off_append)
    off_std = np.std(off_append,ddof=1,dtype='d')
    threshold1 = off_mean + off_std*nstd
    while threshold1 != threshold[n]:
        threshold[n] = threshold1
        print threshold[n]
        ax[n*4+3].axhline(y = threshold[n], c='0.3', alpha=0.1)
        off_append = []
        for i in range(frame-frame_start):
            if tt_a[i,n] < threshold[n]:
                off_append = np.append(off_append, tt_a[i,n])
        off_mean = np.mean(off_append)
        off_std = np.std(off_append,ddof=1,dtype='d')
        threshold1 = off_mean + off_std*nstd
    ax[n*4+3].axhline(y = threshold[n], c='r')
fig.canvas.draw()
if savefig ==1:
    fig.savefig(filePath+fileName+abc+'.fig3_tt.pdf', format='pdf', bbox_inches = 'tight')

#%%
"""
Extract Von and Voff spectra and fit Gaussian
"""
def gauss(x, A, mu, sigma, slope, b):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+slope*x+b
dL = np.zeros((len(pts)))
fig, ax = plt.subplots(len(pts), squeeze=True)
for n in range(len(pts)):
    Von_spec = np.mean([spectra_bgcr[i,:,n] for i in range(frame) if i%2==0 and tt[i,n] > threshold[n]], axis=0)
    Voff_spec = np.mean([spectra_bgcr[i,:,n] for i in range(frame) if i%2==1 and tt[i,n] > threshold[n]], axis=0)
    x = x_lambda[pts_new[n,0]-scan_l:pts_new[n,0]+scan_l]
    slope1 = (np.mean(Von_spec[scan_l*2-scan_l*2/10:])-np.mean(Von_spec[:scan_l*2/10]))/(x.max()-x.min())
    slope2 = (np.mean(Voff_spec[scan_l*2-scan_l*2/10:])-np.mean(Voff_spec[:scan_l*2/10]))/(x.max()-x.min())
    gmod = lmfit.Model(gauss)
    params1 = gmod.make_params()
    params1['A'].set(value=np.max(Von_spec), min=0)
    params1['mu'].set(value=x_lambda[pts_new[n,0]-scan_l+int(*np.where(Von_spec == Von_spec.max())[0])], min=400, max=800)
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
    dL[n] = result1.best_values['mu']-result2.best_values['mu']
    ax[n].plot(x, Von_spec, 'r.', label='Von Data')
    ax[n].plot(x, Voff_spec,  'b.', label='Voff Data')
    ax[n].plot(x, gauss(x, result1.best_values['A'], result1.best_values['mu'], result1.best_values['sigma'], result1.best_values['slope'], result1.best_values['b']), '-', label='Von ({} nm)'.format(round(result1.best_values['mu'],3)), color='r')
    ax[n].plot(x, gauss(x, result2.best_values['A'], result2.best_values['mu'], result2.best_values['sigma'], result2.best_values['slope'], result2.best_values['b']), '-', label='Voff ({} nm)'.format(round(result2.best_values['mu'],3)), color='b')
    ax[n].annotate('$\Delta$$\lambda$ = {} nm'.format(round(dL[n],3)), xy=(1,1), xytext=(0.02,0.9), xycoords='axes fraction', fontsize=12)
    ax[n].legend(bbox_to_anchor=(1, 1), frameon=False, fontsize=10)
    ax[n].set_xlabel('Wavelength (nm)')
    ax[n].set_ylabel('Intensity')
    plt.subplots_adjust(hspace = 0.5)

if savefig ==1:
    fig.savefig(filePath+fileName+abc+'.fig4_spec.pdf', format='pdf', bbox_inches = 'tight')
print dL
f = open(filePath+'_result.txt','a')
for i in range(len(dL)):
    f.write('{}\n'.format(dL[i])) # python will convert \n to os.linesep
f.close()
#workbook = xlsxwriter.Workbook(filePath+'{}_dL.xlsx'.format(fileName))
#worksheet = workbook.add_worksheet()
#worksheet.write_string(0, 0, 'dL ')
#for n in range(len(dL)):
#    worksheet.write(n+1, 0, dL[n])



#%%
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