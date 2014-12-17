# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
#import matplotlib.animation as animation 
from sub import point, lambda_cali
#from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit
plt.close("all")

filePath='E:/QCSE data/'
fileName='617E-120V-4'


mov = libtiff.TiffFile(filePath+fileName+'.tif')
movie = mov.get_tiff_array()
movie=np.array(movie[:,:,:],dtype='d')

backGND_corr = 1          # 1 = apply correction, else = no correction
Time_corr = 1   # 1 = apply polynomial fit, 2 = apply Gaussian filter, else = no correction
frame=len(movie[:,0,0])
row=len(movie[0,:,0])
col=len(movie[0,0,:])
dt=0.125
frame_start=2
t=frame*dt
T = np.arange(0,t,dt)
T_3d = np.tile(T[:,np.newaxis,np.newaxis],(1,row,col))
movie[0:frame_start,:,:]=movie[frame_start,:,:]
scan_w=3     # extract 3*2+1=7 pixels in width(perpendicular to spectral diffusion line) around QD
scan_l=25    # extract 45*2+1=91 pixels in length = spectral width
x = np.arange(0,col,1)
polydeg = 7
polydeg_bg = 9
polydeg_pb = 8




"""
Calibration of wavelength
"""

cali1 = np.float64(plt.imread(filePath+'550.40.tif'))
cali2 = np.float64(plt.imread(filePath+'600lp.tif'))
cali3 = np.float64(plt.imread(filePath+'700lp.tif'))
bulb  = np.float64(plt.imread(filePath+'light bulb.tif'))

x_lambda = lambda_cali.lambda_cali(bulb,cali1,cali2,cali3)
plt.savefig(filePath+fileName+'d.fig1.pdf', format='pdf')



"""
Background and Gaussian filter correction
"""

abs_I_diff=np.zeros((row, col))
for i in range(frame-1):       
    c=movie[i,:,:]
    d=movie[i+1,:,:]
    abs_I_diff=abs_I_diff+np.absolute(d-c)  


fig, ax = plt.subplots()
ax.imshow(abs_I_diff, vmin=abs_I_diff.min(), vmax=abs_I_diff.max(),cmap='gray')
plt.title('Differential image')


print 'Choose background near point of interest'
bg_pt = np.array(plt.ginput(1))
row_pt = int(bg_pt[0,1])
col_pt = int(bg_pt[0,0])
ax.plot(x,np.tile(np.array(row_pt+scan_w), col),c='w')
ax.plot(x,np.tile(np.array(row_pt-scan_w+1), col),c='w')
ax.set_xlim([0,col])
ax.set_ylim([row,0])
fig.canvas.draw()
plt.savefig(filePath+fileName+'d.fig2.pdf', format='pdf')
bg = np.sum(np.sum(movie[:,row_pt-scan_w:row_pt+scan_w+1,:],axis=0),axis=0)/((scan_w*2+1)*frame)


bg_fit=np.polyfit(x, bg, polydeg_bg) #fitting to background
p=np.polyval(bg_fit,x)



def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

window_size=10

#p=movingaverage(bg,window_size)

bg_3d = np.tile(p[np.newaxis,np.newaxis,:],(frame,row,1))
movie_bgcr = movie[:,:,:]-bg_3d
movie_bgcr1 = np.sum(np.sum(movie_bgcr,axis=0),axis=0)/(row*frame)
movie_t = np.sum(np.sum(movie_bgcr,axis=1),axis=1)/(row*col)
#movie_pb = movingaverage(movie_t,window_size)
pb_constant = np.polyfit(T[frame_start:len(T):1],movie_t[frame_start:len(T):1],polydeg_pb)
pbleach = np.polyval(pb_constant,T)
movie_pbcr = movie_t-pbleach
#pbc = pb_constant[1]/pbleach

#gaufil = gaussian_filter1d(movie_bgcr, sigma=80, axis=0)
#movie_fil = movie_bgcr-gaufil
#movie_fil_t = np.sum(np.sum(movie_fil,axis=1),axis=1)/(row*col)

fig,(ax,ax2,ax3)=plt.subplots(3,1,sharex=False)

line_sbg=ax.plot(x, p,'m',label='polyfit({}) bg'.format(polydeg_bg))
line_bg=ax.plot(x, bg,'c',label='bg')
ax.set_title('Background')
ax.set_xlabel('pixels')
ax.set_xlim([0,col])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)

line_movie=ax2.plot(x,(np.sum(np.sum(movie,axis=0),axis=0)/(row*frame)),'g',label='before bgcr')
line_movie_bgcr=ax2.plot(x,movie_bgcr1,'y',label='after bgcr')
ax2.set_title('Background correction')
ax2.set_xlabel('pixels')
ax2.set_xlim([0,col])
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)
ax2.annotate('Background correction={}'.format(backGND_corr),xy=(0,0), xytext=(0.7,0.1), xycoords='axes fraction', fontsize=10)

ax3.plot(T[frame_start:len(T):1],movie_t[frame_start:len(T):1], label='original I')
ax3.plot(T[frame_start:len(T):1],pbleach[frame_start:len(T):1], label='polyfit({})'.format(polydeg_pb))
ax3.plot(T[frame_start:len(T):1],movie_pbcr[frame_start:len(T):1], label='I- polyfit=1')
#ax3.plot(T[frame_start:len(T)-window_size:1],movie_fil_t[frame_start:len(T)-window_size:1], label='high pass filter=2')
ax3.set_title('Time trace correction')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels,bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)
ax3.set_xlabel('time (s)')
ax3.annotate('Time correction={}'.format(Time_corr),xy=(0,0), xytext=(0.7,0.1), xycoords='axes fraction', fontsize=10)
plt.show()
plt.savefig(filePath+fileName+'d.fig3.pdf', format='pdf', bbox_inches = 'tight')    
    
    
if backGND_corr == 1:
    if Time_corr == 1:
        mov_f = np.zeros((frame,row,col))
        for i in range(frame):
            mov_f[i,:,:] = movie_bgcr[i,:,:]-pbleach[i] 
    #elif Time_corr ==2:
        #mov_f = movie_fil
    else: 
        mov_f=movie_bgcr
else:
    mov_f=movie



#fig=plt.figure()
#ims = []
#for i in range(frame):
#    im=plt.imshow(mov_f[i,:,:],vmin=mov_f.min(),vmax=mov_f.max(),cmap='hot')
#     ims.append([im]) 
#ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True,
#    repeat_delay=1000)
#plt.title('movie')  

newim=np.zeros((row,col))
for i in range(frame-1):       
    c=mov_f[i,:,:]
    d=mov_f[i+1,:,:]
    newim=newim+np.absolute(d-c)
#newim=np.sum(movie_bgcr,axis=0)

"""
Define points (QDs) of interest, and their peak position
"""

fig, ax = plt.subplots()
im = ax.imshow(newim,cmap='gray')

pts = point.pIO(mov_f, ax, fig)
pts = np.array(pts)
pts_new = point.localmax(abs_I_diff, pts, ax, fig)
npoint = np.size(pts_new[:,0])
ax.plot((pts_new[:,1]+scan_l, pts_new[:,1]-scan_l, pts_new[:,1]-scan_l,pts_new[:,1]+scan_l,pts_new[:,1]+scan_l), (pts_new[:,0]-scan_w, pts_new[:,0]-scan_w, pts_new[:,0]+scan_w, pts_new[:,0]+scan_w,pts_new[:,0]-scan_w), '-+', color='b')
plt.savefig(filePath+fileName+'d.fig4.pdf', format='pdf')

"""
Extracting spectra from 7 X 91 pixels around points of interest 
Thresholding
Fit spectra when voltage is on and off, respectively.
"""
def gauss(x, *p):
    A, mu, sigma, slope, b = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+slope*x+b



box_intensity = np.zeros((frame,npoint))
box_gaufil = np.zeros((frame,npoint))
box_bgcr = np.zeros((frame,npoint))
box_origin = np.zeros((frame,npoint))
spectra = np.zeros((frame,2*scan_l+1,npoint))
boxmovie = np.zeros((frame,2*scan_w+1,2*scan_l+1,npoint))
boxtile = np.zeros((frame*(2*scan_w+1),2*scan_l+1, npoint))
fig, axarr = plt.subplots(npoint*2,1, sharex=True, sharey=False)
fig2, axarr2 = plt.subplots(npoint,2)


for n in range(npoint):   
    for i in range(frame):
        boxmovie[i,:,:,n]  = point.mask(mov_f[i,:,:], pts_new[n,:], scan_w, scan_l)
        spectra[i,:,n] = np.mean(boxmovie[i,:,:,n] , axis=0)        
        
        box_intensity[i,n] = np.sum(np.sum(boxmovie[i,:,:,n], axis=0),axis=0)/((scan_w*2+1)*(scan_l*2+1)) 
    
        #tempg = point.mask(gaufil[i,:,:], pts_new[n,:], scan_w, scan_l)
        #box_gaufil[i,n] = tempg.sum()/((scan_w*2+1)*(scan_l*2+1))
        
        tempbgcr = point.mask(movie_bgcr[i,:,:], pts_new[n,:], scan_w, scan_l)
        box_bgcr[i,n] = tempbgcr.sum()/((scan_w*2+1)*(scan_l*2+1))        
        
        tempo = point.mask(movie[i,:,:], pts_new[n,:], scan_w, scan_l)
        box_origin[i,n] = tempo.sum()/((scan_w*2+1)*(scan_l*2+1))  
    
    
    boxtile[:,:,n] = np.reshape(boxmovie[:,:,:,n], (frame*(2*scan_w+1),2*scan_l+1))
    std = np.std(box_intensity[frame_start:,n],axis=0,ddof=1,dtype='d')
    thre_constant = box_intensity[frame_start:,n].mean()-std/2
    threshold = np.tile(thre_constant,frame)     
    
    axarr[n*2].imshow((spectra[:,:,n].T), cmap='gray')
    axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),box_intensity[:,n], 'b')
    axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),threshold, 'r')
    #axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),box_gaufil[:,n], 'g')
    #axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),box_bgcr[:,n], 'c')
    #axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),box_origin[:,n], 'y')
    axarr[n*2+1].set_xlim([0,300])
    axarr[n*2+1].set_xlabel('frame')    
        
    
    
    Von_spec = np.array([spectra[i,:,n] for i in range(frame) if i%2==0 and box_intensity[i,n] > threshold[i]])
    Voff_spec = np.array([spectra[i,:,n] for i in range(frame) if i%2==1 and box_intensity[i,n] > threshold[i]])    
    
    #axarr[n*4+2].imshow(np.transpose(Von_spec),vmax=spectra[:,:,n].max(),vmin=spectra[:,:,n].min(), cmap='gray')
    #axarr[n*4+3].imshow(np.transpose(Voff_spec),vmax=spectra[:,:,n].max(),vmin=spectra[:,:,n].min(), cmap='gray')
    #axarr[n*4+2].annotate('Von > threshold',xy=(0,0), xytext=(0.7,0.5), xycoords='axes fraction') 
    #axarr[n*4+3].annotate('Voff > threshold',xy=(0,0), xytext=(0.7,0.5), xycoords='axes fraction')
    


    Von_avg = np.mean(Von_spec, axis=0)    
    Voff_avg = np.mean(Voff_spec, axis=0)
    
    slope1 = (np.mean(Von_avg[int(len(Von_avg)-len(Von_avg)/10):len(Von_avg):1])-np.mean(Von_avg[0:int(len(Von_avg)/10):1]))/(x_lambda[pts_new[n,1]+scan_l+1]-x_lambda[pts_new[n,1]-scan_l])
    slope2 = (np.mean(Voff_avg[int(len(Voff_avg)-len(Voff_avg)/10):len(Voff_avg):1])-np.mean(Voff_avg[0:int(len(Voff_avg)/10):1]))/(x_lambda[pts_new[n,1]+scan_l+1]-x_lambda[pts_new[n,1]-scan_l])
    p01 = [np.max(Von_avg), x_lambda[pts_new[n,1]-scan_l+int(*np.where(Von_avg == Von_avg.max()))] , 10, slope1, np.min(Von_avg)]
    p02 = [np.max(Voff_avg), x_lambda[pts_new[n,1]-scan_l+int(*np.where(Voff_avg == Voff_avg.max()))] , 10, slope2, np.min(Voff_avg)]    
    coeff1, var_matrix1 = curve_fit(gauss, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg, p0=p01)
    coeff2, var_matrix2 = curve_fit(gauss, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg, p0=p02)    
    Von_avg_gfit = gauss(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], *coeff1)
    Voff_avg_gfit = gauss(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], *coeff2)

    Von_avg_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1],Von_avg,polydeg)
    Voff_avg_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1],Voff_avg,polydeg)
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

    
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg, 'bo', fillstyle='none')
    #axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg_p, 'b-', label='Von polyfit={}'.format(polydeg))
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg_gfit, 'b-', label='Von gsnfit')    
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg, 'ro', fillstyle='none')
    #axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg_p, 'r-', label='Voff polyfit={}'.format(polydeg))
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg_gfit, 'r-', label='Voff gsnfit')    
    axarr2[n,0].set_xlim([x_lambda[pts_new[n,1]-scan_l],x_lambda[pts_new[n,1]+scan_l+1]])    
    handles, labels = axarr2[n,0].get_legend_handles_labels()    
    axarr2[n,0].legend(handles, labels,bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0, fontsize=10)
    #axarr2[n,0].annotate('Polynomial', xytext=(0.7,1), xy=(Von_avg_peak,0), xycoords='axes fraction', fontsize=12)     
    #axarr2[n,0].annotate(r'$\Delta\lambda=${}nm'.format(round(Von_avg_peak-Voff_avg_peak,3)),xy=(0,0), xytext=(0.7,0.9), xycoords='axes fraction', fontsize=12) 
    #axarr2[n,0].annotate(r'$Von \lambda=${}nm'.format(round(Von_avg_peak,1)),xy=(Von_avg_peak,0), xytext=(0.7,0.8), xycoords='axes fraction', fontsize=10)    
    #axarr2[n,0].annotate(r'$Voff \lambda=${}nm'.format(round(Voff_avg_peak,1)),xy=(Voff_avg_peak,0), xytext=(0.7,0.7), xycoords='axes fraction', fontsize=10)    
    #axarr2[n,0].annotate('Gaussian', xytext=(0,1), xy=(Von_avg_peak,0), xycoords='axes fraction', fontsize=12)     
    axarr2[n,0].annotate(r'$\Delta\lambda=${}nm'.format(round(Von_avg_gpeak-Voff_avg_gpeak,3)),xy=(0,0), xytext=(0,0.9), xycoords='axes fraction', fontsize=10) 
    axarr2[n,0].annotate(r'$Von \lambda=${}nm'.format(round(Von_avg_gpeak,1)),xy=(Von_avg_gpeak,0), xytext=(0,0.8), xycoords='axes fraction', fontsize=10)    
    axarr2[n,0].annotate(r'$Voff \lambda=${}nm'.format(round(Voff_avg_gpeak,1)),xy=(Voff_avg_gpeak,0), xytext=(0,0.7), xycoords='axes fraction', fontsize=10)    
   

    
    
    #axarr2[n,1].hist(Von_peak, bins=len(Von_peak)/3, color='b', histtype='stepfilled',alpha=0.5, label='Von')
    #axarr2[n,1].hist(Voff_peak, bins=len(Voff_peak)/3, color='r', histtype='stepfilled',alpha=0.5, label='Voff')
    axarr2[n,1].hist(Von_gpeak, bins=len(Von_gpeak)/3, color='b', histtype='stepfilled',alpha=0.5, label='Von gsn')
    axarr2[n,1].hist(Voff_gpeak, bins=len(Voff_gpeak)/3, color='r', histtype='stepfilled',alpha=0.5, label='Voff gsn')
    axarr2[n,1].set_xlim([x_lambda[pts_new[n,1]-scan_l],x_lambda[pts_new[n,1]+scan_l+1]])    
    handles, labels = axarr2[n,1].get_legend_handles_labels()    
    axarr2[n,1].legend(handles, labels,bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0, fontsize=10)
fig.savefig(filePath+fileName+'d.fig5.pdf', format='pdf')
fig2.savefig(filePath+fileName+'d.fig6.pdf', format='pdf', bbox_inches = 'tight')

splitrow = 5
for i in range(npoint):
    fig, ax = plt.subplots(splitrow,1)
    for k in range(splitrow):    
        ax[k].imshow((boxtile[len(boxtile[:,0,i])/splitrow*k:len(boxtile[:,0,i])/splitrow*(k+1),:,i].T), cmap='gray')
    fig.savefig(filePath+fileName+'d.fig{}'.format(7+i)+'.pdf', format='pdf', bbox_inches = 'tight')

#fig3, axarr3 = plt.subplots(npoint)
#for j in range(npoint):    
#    ims = []
#    for i in range(frame):
#        im=axarr3[j].imshow(boxmovie[i,:,:,j],vmin=boxmovie[:,:,:,j].min(),vmax=boxmovie[:,:,:,j].max(),cmap='hot')
#        ims.append([im]) 
#    ani = animation.ArtistAnimation(fig3, ims, interval=dt*1000, blit=True,repeat_delay=1000)
#    writer = animation.writers['ffmpeg'](fps=1/dt)
#ani.save(filePath+'movie.mp4',writer=writer,dpi=100)
