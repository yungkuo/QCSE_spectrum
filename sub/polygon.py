# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:42:48 2014

@author: KyoungWon
"""
import numpy as np
import matplotlib.pyplot as plt 

def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = 0
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
    
    
def mean_polygon(movie, abs_I_diff, ax, fig):
    frame=len(movie[:,1,1])
    nrow=len(movie[1,:,1])
    ncol=len(movie[1,1,:])

    print("Choose ROI by clicking multiple points")
    print("If finished, press enter")
    pts = plt.ginput(0,0) #it will wait for multiple clicks
    pts = np.array(pts)
    col_pts = pts[:,0]
    row_pts = pts[:,1]
    col_pts=np.append(col_pts,col_pts[0])
    row_pts=np.append(row_pts,row_pts[0])
    poly=zip(col_pts,row_pts)
    ax.plot(col_pts, row_pts, '-o')
    ax.set_xlim([0,ncol])
    ax.set_ylim([nrow,0])
    fig.canvas.draw()
    mask = np.zeros((nrow, ncol), dtype=np.int)
    for i in range(nrow):
        for j in range(ncol):
            mask[i,j]=inside_polygon(j,i,poly)
    
    mask3d=np.tile(mask, (frame,1,1)) 
    movie=np.array(movie[:,:,:], dtype='d')
    mask3d=np.array(mask3d[:,:,:],dtype='d')
    bg_3d=np.multiply(movie,mask3d)
    bg=np.sum(np.sum(bg_3d, axis=1), axis=1)/mask.sum()

    return bg, bg_3d
