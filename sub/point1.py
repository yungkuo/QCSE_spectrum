# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:47:36 2014

@author: Philip
"""
import numpy as np
import matplotlib.pyplot as plt

scan_w = 2
scan_l = 7

def pIO(mov, ax, fig):
    print("Go to Figure 3, and click points of interests")
    print("If finished, press enter")
    row=len(mov[1,:,1])
    col=len(mov[1,1,:])
    pts = plt.ginput(0,0) 
    pts=np.array(pts)
    col_pts=np.around(pts[:,0])
    row_pts=np.around(pts[:,1])
    ax.plot(col_pts,row_pts, 'r+', markersize=20)
    ax.set_xlim([0,col])
    ax.set_ylim([row,0])
    fig.canvas.draw()    
    pts=pts.astype(int)
    pts_rc=zip(row_pts, col_pts)
    
    return pts_rc
    
def localmax(refimg, pts, ax, fig):
    row=len(refimg[:,0])
    col=len(refimg[0,:])
    pts_new = np.zeros((len(pts),2))
    for i in range(len(pts)):
        local = mask(refimg, pts[i,:],scan_w, scan_l)
        a = np.array(zip(*np.where(local == local.max())))   # * unpack the tuple, return its value as an input element of zip)
        pts_new[i,0] = int(pts[i,0]-scan_l+a[0,1])
        pts_new[i,1] = int(pts[i,1]-scan_w+a[0,0])       
        
    ax.plot(pts_new[:,0], pts_new[:,1], 'y+', markersize=25)
    ax.plot((pts[:,0]-scan_l, pts[:,0]-scan_l, pts[:,0]+scan_l, pts[:,0]+scan_l, pts[:,0]-scan_l), 
            (pts[:,1]+scan_w, pts[:,1]-scan_w, pts[:,1]-scan_w, pts[:,1]+scan_w, pts[:,1]+scan_w), '-+', color='w')
   
    for n in range(len(pts)):   
        ax.annotate(n, xy=(pts_new[n,0], pts_new[n,1]), xytext=(pts_new[n,0], pts_new[n,1]+20),color='w')   
    ax.set_xlim([0,col])
    ax.set_ylim([row,0])
    fig.canvas.draw()
    return pts_new
    
def mask(refimg, pts, scan_w, scan_l):
    local = refimg[pts[1]-scan_w:pts[1]+scan_w,pts[0]-scan_l:pts[0]+scan_l]
    return local

def mask3d(refimg, pts, scan_w, scan_l):
    local = refimg[:,pts[1]-scan_w:pts[1]+scan_w,pts[0]-scan_l:pts[0]+scan_l]
    return local

if __name__ == "__main__": 
    print("Your scan pixel is %w, x %l" %(scan_w, scan_l))  
    