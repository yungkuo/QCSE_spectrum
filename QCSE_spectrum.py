# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:30:02 2014

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
import matplotlib.animation as animation 
from sub import point, lambda_cali
plt.close("all")

filePath='/Users/yung/111414 QCSE/'
fileName='100617G-90V-5'

mov = libtiff.TiffFile(filePath+fileName+'.tif')
movie = mov.get_tiff_array()
movie=np.array(movie[:,:,:],dtype='d')

backGND_corr = 1          # 1 = apply correction, else = no correction
Photobleaching_corr = 0   # 1 = apply correction, else = no correction
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
scan_l=45    # extract 45*2+1=91 pixels in length = spectral width
x = np.arange(0,col,1)
polydeg = 9
polydeg_bg = 9
polydeg_pb = 1




"""
Calibration of wavelength
"""
cali1 = plt.imread('/Users/yung/111414 QCSE/calibration2.tif')
cali2 = plt.imread('/Users/yung/111414 QCSE/calibration.tif')
refimg = cali1+cali2
x_lambda = lambda_cali.lambda_cali(refimg)
plt.savefig(filePath+fileName+'fig1.pdf', format='pdf')



"""
Background and photobleaching correction
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
plt.savefig(filePath+fileName+'fig2.pdf', format='pdf')
bg = np.sum(np.sum(movie[:,row_pt-scan_w:row_pt+scan_w+1,:],axis=0),axis=0)/((scan_w*2+1)*frame)


bg_fit=np.polyfit(x, bg, polydeg_bg) #fitting to background
p=np.polyval(bg_fit,x)



def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

window_size=10

#p=movingaverage(bg,window_size)

bg_3d=np.tile(p[np.newaxis,np.newaxis,:],(frame,row,1))
movie_bgcr=movie[:,:,:]-bg_3d
movie_bgcr1=np.sum(np.sum(movie_bgcr,axis=0),axis=0)/(row*frame)
movie_t=np.sum(np.sum(movie_bgcr,axis=1),axis=1)/(row*col)
movie_pb=movingaverage(movie_t,window_size)
pb_constant=np.polyfit(T[frame_start:len(T)-window_size:1],movie_pb[frame_start:len(T)-window_size:1],polydeg_pb)
pbleach=np.polyval(pb_constant,T)
pbc=pb_constant[1]/pbleach

fig,(ax,ax2,ax3)=plt.subplots(3,1,sharex=False)

line_sbg=ax.plot(x, p,'m',label='smoothen bg')
line_bg=ax.plot(x, bg,'c',label='bg')
ax.set_title('Background')
ax.set_xlabel('pixels')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)

line_movie=ax2.plot(x,(np.sum(np.sum(movie,axis=0),axis=0)/(row*frame)),'g',label='before bgcr')
line_movie_bgcr=ax2.plot(x,movie_bgcr1,'y',label='after bgcr')
ax2.set_title('Background correction')
ax2.set_xlabel('pixels')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)
ax2.annotate('Background correction={}'.format(backGND_corr),xy=(0,0), xytext=(0.7,0.1), xycoords='axes fraction', fontsize=10)

line_smoothen, = ax3.plot(T[frame_start:len(T)-window_size:1],movie_pb[frame_start:len(T)-window_size:1], label="Smoothen I")
line_pbleaching, = ax3.plot(T[frame_start:len(T)-window_size:1],pbleach[frame_start:len(T)-window_size:1], label="Photobleaching")
line_pb_correct_I, = ax3.plot(T[frame_start:len(T)-window_size:1],np.multiply(movie_pb,pbc)[frame_start:len(T)-window_size:1], label="P.B. corrected I")
ax3.set_title('Photobleaching correction')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels,bbox_to_anchor=(0.93, 1), loc=2, borderaxespad=0, fontsize=12)
ax3.set_xlabel('time (s)')
ax3.annotate('Photobleaching correction={}'.format(Photobleaching_corr),xy=(0,0), xytext=(0.65,0.1), xycoords='axes fraction', fontsize=10)
plt.show()
plt.savefig(filePath+fileName+'fig3.pdf', format='pdf', bbox_inches = 'tight')    
    
    
if backGND_corr == 1:
    if Photobleaching_corr == 1:
        mov_f=np.zeros((frame,row,col))
        for i in range(frame):
            mov_f[i,:,:]=movie_bgcr[i,:,:]*pbc[i] 
    else: 
        mov_f=movie_bgcr
else:
    mov_f=movie

fig=plt.figure()
ims = []
for i in range(frame):
    im=plt.imshow(mov_f[i,:,:],vmin=mov_f.min(),vmax=mov_f.max(),cmap='hot')
    ims.append([im]) 
ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True,
    repeat_delay=1000)
plt.title('movie')  

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
ax.plot((pts[:,1]+scan_l, pts[:,1]-scan_l, pts[:,1]-scan_l,pts[:,1]+scan_l,pts[:,1]+scan_l), (pts[:,0]-scan_w, pts[:,0]-scan_w, pts[:,0]+scan_w, pts[:,0]+scan_w,pts[:,0]-scan_w), '-+', color='b')
plt.savefig(filePath+fileName+'fig4.pdf', format='pdf')

"""
Extracting spectra from 7 X 71 pixels around points of interest 
Thresholding
Fit spectra when voltage is on and off, respectively.
"""
box_intensity = np.zeros((frame,npoint))
spectra = np.zeros((frame,2*scan_l+1,npoint))
fig, axarr = plt.subplots(npoint*2,1, sharex=True)
fig2, axarr2 = plt.subplots(npoint,2)
for n in range(npoint):
    for i in range(frame):
        temp = point.mask(mov_f[i,:,:], pts_new[n,:], scan_w, scan_l)
        spectra[i,:,n] = np.mean(temp, axis=0)
        box_intensity[i,n] = temp.sum()/((scan_w*2+1)*(scan_l*2+1)) 


    std5 = np.std(box_intensity[frame_start:,n],axis=0,ddof=1,dtype='d')/5
    thre_constant = box_intensity[frame_start:,n].mean()-std5   
    threshold = np.tile(thre_constant,frame)     
    
    axarr[n*2].imshow(np.transpose(spectra[:,:,n]), cmap='gray')
    axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),box_intensity[:,n], 'b')
    axarr[n*2+1].plot(np.arange(0,frame,1,dtype='int'),threshold, 'r')    
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
    Von_avg_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1],Von_avg,polydeg)
    Voff_avg_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1],Voff_avg,polydeg)
    Von_avg_p = np.polyval(Von_avg_fit, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])
    Voff_avg_p = np.polyval(Voff_avg_fit, x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])
    Von_avg_peak = x_lambda[pts_new[n,1]-scan_l+int(*np.where(Von_avg_p == Von_avg_p.max()))] 
    Voff_avg_peak = x_lambda[pts_new[n,1]-scan_l+int(*np.where(Voff_avg_p == Voff_avg_p.max()))]
   
   
 
    #fit every single frame
    Von_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], np.transpose(Von_spec), polydeg)
    Voff_fit = np.polyfit(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], np.transpose(Voff_spec), polydeg)

    Von_p = []
    Voff_p = []  
    Von_peak = []
    Voff_peak = []
    
    for j in range(len(Von_fit[0,:])):    
        a = np.polyval(Von_fit[:,j], x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])        
        a_max = x_lambda[pts_new[n,1]-scan_l+int(*np.where(a == a.max()))]        
        #Von_p = np.append(Von_p, a, axis=0)
        Von_peak = np.append(Von_peak, a_max)
        
    #Von_p = Von_p.reshape(len(Von_fit[0,:]), scan_l*2+1)
    Von_peak = Von_peak.reshape(len(Von_fit[0,:]), 1)
    
    for k in range(len(Voff_fit[0,:])): 
        b = np.polyval(Voff_fit[:,k], x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1])
        b_max = x_lambda[pts_new[n,1]-scan_l+int(*np.where(b == b.max()))]        
        #Voff_p = np.append(Voff_p, b,axis=0)
        Voff_peak = np.append(Voff_peak, b_max)
    #Voff_p = Voff_p.reshape(len(Voff_fit[0,:]), scan_l*2+1)
    Voff_peak = Voff_peak.reshape(len(Voff_fit[0,:]), 1)


    
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg, 'bo', fillstyle='none')
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Von_avg_p, 'b-', label='Von polyfit={}'.format(polydeg))
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg, 'ro', fillstyle='none')
    axarr2[n,0].plot(x_lambda[pts_new[n,1]-scan_l:pts_new[n,1]+scan_l+1:1], Voff_avg_p, 'r-', label='Voff polyfit={}'.format(polydeg))
    axarr2[n,0].set_xlim([x_lambda[pts_new[n,1]-scan_l],x_lambda[pts_new[n,1]+scan_l+1]])    
    handles, labels = axarr2[n,0].get_legend_handles_labels()    
    axarr2[n,0].legend(handles, labels,bbox_to_anchor=(0.55, 0.2), loc=2, borderaxespad=0, fontsize=8)
    axarr2[n,0].annotate(r'$\Delta\lambda=${}nm'.format(round(Von_avg_peak-Voff_avg_peak,3)),xy=(0,0), xytext=(0.3,0.9), xycoords='axes fraction', fontsize=12) 
    axarr2[n,0].annotate(r'$Von \lambda=${}nm'.format(round(Von_avg_peak,1)),xy=(Von_avg_peak,0), xytext=(0.6,0.3), xycoords='axes fraction', fontsize=10)    
    axarr2[n,0].annotate(r'$Voff \lambda=${}nm'.format(round(Voff_avg_peak,1)),xy=(Voff_avg_peak,0), xytext=(0.6,0.23), xycoords='axes fraction', fontsize=10)    
    
    
    axarr2[n,1].hist(Von_peak, bins=20, color='b', histtype='stepfilled',alpha=0.5, label='Von')
    axarr2[n,1].hist(Voff_peak, bins=20, color='r', histtype='stepfilled',alpha=0.5, label='Voff')
    axarr2[n,1].set_xlim([x_lambda[pts_new[n,1]-scan_l],x_lambda[pts_new[n,1]+scan_l+1]])    
    handles, labels = axarr2[n,1].get_legend_handles_labels()    
    axarr2[n,1].legend(handles, labels,bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0, fontsize=12)
fig.savefig(filePath+fileName+'fig5.pdf', format='pdf')
fig2.savefig(filePath+fileName+'fig6.pdf', format='pdf')
  

