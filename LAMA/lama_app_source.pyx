##
##  lama_app_source.pyx
##  LAMA
##
##  Created by Sebastian Malkusch on 10.06.15.
##  <malkusch@chemie.uni-frankfurt.de>
##  Copyright (c) 2015 Single Molecule Biophysics. All rights reserved.
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.

cimport numpy as cnp
import numpy as np
import scipy as sp
import pylab as pl
import os
import math
import sys
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import spatial
from scipy import special
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from scipy.stats import nbinom
from PIL import Image, ImageDraw

def update_progress(float progress):
    '''
        receives progress status from application layer and visualizes progress in terminal.
    '''
    cdef str text
    cdef int barLength, block

    barLength = 10 # Modify this to change the length of the progress bar
    block = int(round(barLength*progress))
    text = "\rthe lama is processing: [{0}] {1}%".format( "#"*block + "-"*(barLength-block), int(progress*100)+1)
    sys.stdout.write(text)
    sys.stdout.flush()

def read_filenames(str name):
    '''
        receives path from communication layer, extracts filenames from path, returns them to communication layer.
    '''
    #cdef FILE *f
    cdef list files
    
    f = open(name)
    files = f.readlines()
    f.close
    for i in range (0,len(files)):
        try:
            files[i]=files[i].rstrip('\n')
        except ValueError:
            break
    return files

def read_locs(str name):
    '''
        receives path from communication layer,  loads a localization list from the received path and returns it.
    '''
    cdef cnp.ndarray[double, ndim=2] locs
    
    name=name.rstrip('\n')
    print ('the lama is pronking through '+ name)
    locs = np.loadtxt(name,skiprows = 0,comments='#')
    return locs


def read_locs_rs(str name):
    '''
        receives path from communication layer, loads a localization list in MALK format from the received path and returns it.
    '''
    cdef cnp.ndarray[double, ndim=2] locs
    
    name=name.rstrip('\n')
    print ('the lama is pronking through '+ name)
    locs = np.loadtxt(name,skiprows = 0,comments='#',usecols = (0,1,2,3))
    return locs

def make_dir(str name, str append):
    '''
        receives path to a folder from communication layer,  checks whether it exists, if not creates it and returns folder name.
    '''
    cdef str foldername
    
    foldername=name+append
    if os.path.isdir(foldername)==False:
        os.mkdir(foldername)
    return foldername


def create_roi(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] roi, str name):
    '''
        receives localization list, path to a loc-file, and roi parameters from communication layer. Filters localization list spatially and chronologically based on roi parameters. Saves filtered localization list. Returns filtered localization list.
    '''
    cdef str outfilename, hd
    cdef cnp.ndarray[double, ndim=2] locs1
    cdef cnp.ndarray idx1, idx2, idy1, idy2, idt1, idt2, idint1, idint2
    
    outfilename=name + '/locs_roi.txt'
    idx1=locs[:,0]>=roi[0,0]
    idx2=locs[:,0]<=roi[0,1]
    idy1=locs[:,1]>=roi[1,0]
    idy2=locs[:,1]<=roi[1,1]
    idt1=locs[:,2]>=roi[2,0]
    idt2=locs[:,2]<=roi[2,1]
    idint1=locs[:,3]>=roi[3,0]
    idint2=locs[:,3]<=roi[3,1]
    locs1 = (locs[(idx1&idx2&idy1&idy2&idt1&idt2&idint1&idint2),:])
    hd=str('localization roi file (Malk format)\nx[nm]\ty[nm]\t[frame]\tI[a.u.]')
    np.savetxt(outfilename, locs1, fmt='%.5e', delimiter='   ', header = hd, comments='# ' )
    return locs1
    
def create_roi_ripley(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] roi, str name):
    '''
        receives localization list, path to a loc-file, and roi parameters suitable for computing ripleys k-function from communication layer. Filters localization list spatially based on roi parameters. Saves filtered localization list. Returns filtered localization list.
    '''
    cdef str outfilename, hd
    cdef cnp.ndarray[double, ndim=2] locs1
    cdef cnp.ndarray idx1, idx2, idy1, idy2
    
    outfilename=name + '/roi_ripley.txt'
    idx1=locs[:,0]>=roi[0,0]
    idx2=locs[:,0]<=roi[0,1]
    idy1=locs[:,1]>=roi[1,0]
    idy2=locs[:,1]<=roi[1,1]
    locs1 = (locs[(idx1&idx2&idy1&idy2),:])
    hd=str('localization roi file (Malk format)\nx[nm]\ty[nm]\t[frame]\tI[a.u.]')
    np.savetxt(outfilename, locs1, fmt='%.5e', delimiter='   ',header = hd, comments='# ')
    return locs1

def createRoiBead(cnp.ndarray[double, ndim=2] fas,cnp.ndarray[double, ndim=2] roi, int r_min, int r_max, int on_min, int on_max, str outfilename):
    '''
        Input: FAS list MCA format
        Sorts for spatial coordinates (roi), FAS size (r_min, r_max), Localizations (on_min, on_max)
        saves FAS file MCA format
    '''
    cdef str hd
    cdef int cnum
    cdef cnp.ndarray[double, ndim=2] fas_sort
    cdef cnp.ndarray idx1, idx2, idy1, idy2, idr1, idr2, ido1, ido2

    idx1=fas[:,0]>=roi[0,0]
    idx2=fas[:,0]<=roi[0,1]
    idy1=fas[:,1]>=roi[1,0]
    idy2=fas[:,1]<=roi[1,1]
    idr1=fas[:,3]>=r_min
    idr2=fas[:,3]<=r_max
    ido1=fas[:,4]>=on_min
    ido2=fas[:,4]<=on_max
    fas_sort = (fas[(idx1&idx2&idy1&idy2&idr1&idr2&ido1&ido2),:])
    cnum=np.shape(fas_sort)[0]
    hd=str('Cluster analysis LAMA format\nnumber of cluster: %i \nx[nm]\ty[nm]\tsize[nm*nm]\tr[nm]\tI[a.u.]' %(cnum))
    np.savetxt(outfilename, fas_sort, fmt='%.5e', delimiter='   ', header = hd, comments='# ' )


def make_Image(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] roi, int pxl, int fixed_ind, int fix_val, str name):
    '''
        receives localization list, roi parameters and desired pxl size from communication layer. computes a 2d-histogram. transform histogram information into 8-bit gray scale image. saves image. returns raw histogram containing absolute number of localizations per bin.
    '''
    cdef str outfilename
    cdef cnp.ndarray[long, ndim=2] locs1
    cdef cnp.ndarray[double, ndim=2] BW0, BW1
    
    outfilename=name + '/roi_int.png'
    locs1=locs.astype(int)
    BW0=np.histogram2d(locs1[:,0], locs1[:,1],bins=[(roi[0,1]-roi[0,0])/pxl,(roi[1,1]-roi[1,0])/pxl], range=[[roi[0,0],roi[0,1]],[roi[1,0],roi[1,1]]])[0]
    BW0=np.rot90(BW0, k=1)
    BW0=np.flipud(BW0)
    BW1=BW0
    if fixed_ind==1:
        BW1=np.clip(BW1,0,fix_val)
        BW1=(np.divide(BW1,fix_val)*255)
    BW2 = Image.fromarray(np.uint8(BW1))
    BW2.save(outfilename)
    return (BW0)
    
def convolve_image(cnp.ndarray[double, ndim=2] BW0, float sigma, int fixed_ind, int fix_val, str name):
    '''
        receives super resolved image and localization accuracy from communication layer. Computes a convolution by Gaussian low pass filtering with a Gaussian function with sigma=localization accuracy. Returns convolved image.
    '''
    cdef str outfilename
    cdef cnp.ndarray[double, ndim=2] BW1
    
    outfilename=name + '/roi_iwm.png'
    BW1 = ndimage.gaussian_filter(BW0, sigma)
    if fixed_ind==1:
        BW1=np.clip(BW1,0,fix_val)
        BW1=(np.divide(BW1,fix_val)*255)
    else:
        BW1=(np.divide(BW1,(np.max(BW1)))*255)
    BW2 = Image.fromarray(np.uint8(BW1))
    BW2.save(outfilename)
    return (BW1)
    

def bud_locs(cnp.ndarray[double, ndim=2] BW, int MinR, int MaxR, int r, int pxl, str name):
    '''
        receives super resolved image and filter parameter from communication layer. Computes a binary mask from cohesive regions. Extracts morphological cluster information from binary map. Extracts intensity cluster information from a multiplication of binary mask with raw image. Saves binary image. Returns cluster information.
    '''
    cdef str Binariename
    cdef int i, length, nb_labels
    cdef float R
    cdef cnp.ndarray[double, ndim=1] sizes, mask_r
    cdef cnp.ndarray[list, ndim=1] x
    cdef cnp.ndarray[int, ndim=2] label_im
    cdef cnp.ndarray[double, ndim=2] BW1, coord, sort_coord
    cdef cnp.ndarray mask,mask1, mask2, mask_size, remove_pixel, BW0
    
    Binariename=name + '/First_channel_Bin.png'
    R=float(r/pxl)
    BW1 = ndimage.gaussian_filter(BW, 0.2)
    mask = BW1 > 0
    mask1=ndimage.morphology.binary_opening(mask,iterations=1)
    mask2=ndimage.morphology.binary_closing(mask1,iterations=1)
    
    label_im, nb_labels = ndimage.label(mask2)
    sizes = ndimage.sum(mask2, label_im, range(nb_labels + 1))
    mask_r=np.dot(np.sqrt(np.divide(sizes,np.pi)),pxl)
    mask_size = mask_r <= MinR
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    label_im, nb_labels = ndimage.label(label_im)
    sizes = ndimage.sum(mask2, label_im, range(nb_labels + 1))
    mask_r=np.dot(np.sqrt(np.divide(sizes,np.pi)),pxl)
    mask_size = mask_r > MaxR
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    BW0 = ndimage.binary_fill_holes(label_im)
    BW2 = Image.fromarray(np.uint8(BW0)*255)
    BW2.save(Binariename)

    label_im, nb_labels = ndimage.label(label_im)
    coord  =(np.array(ndimage.center_of_mass(BW,label_im, range(nb_labels+1))))[1:,[1, 0]]
    tree=spatial.KDTree(coord)
    x=tree.query_ball_point(coord, (R))
    sort_coord = np.zeros([1,np.shape(coord)[1]], dtype=np.float64)
    for i in range (0,len(x)):
        length = len(x[i])
        if length==1:
            sort_coord=np.append(sort_coord, [coord[i,:]], axis=0)
    sort_coord=np.delete(sort_coord,0,0)
    sort_coord=np.array(sort_coord)
    return (np.dot(sort_coord,pxl))

def make_binary(cnp.ndarray BW, float Thr):
    '''
        receives super resolved image and threshold parameter from communication layer.creates a binry from image based on a intensity theshold.returns binary mask.
    '''
    cdef cnp.ndarray mask
    
    mask = BW > Thr
    return mask

def imb_ana(cnp.ndarray mask, cnp.ndarray[double, ndim=2] BW0, cnp.ndarray[double, ndim=2] roi, float thr, float min_r, float max_r, int pxl, str dir_name):
    '''
        receives super resolved image and filter parameter from communication layer. Computes a binary mask from cohesive regions. Extracts morphological cluster information from binary map. Extracts intensity cluster information from a multiplication of binary mask with raw image. Saves binary image. Saves cluster information.
    '''
    cdef str maskname, im_name, outname, hd
    cdef int nb_labels, l
    cdef cnp.ndarray[int, ndim=2] label_im
    cdef cnp.ndarray[double, ndim=1] sizes, mask_r
    cdef cnp.ndarray[double, ndim=2] coord, mca_res
    cdef cnp.ndarray mask_size, remove_pixel
    
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_r=np.dot(np.sqrt(np.divide(sizes,np.pi)),pxl)
    mask_size=mask_r<=min_r
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    
    label_im, nb_labels = ndimage.label(label_im)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_r=np.dot(np.sqrt(np.divide(sizes,np.pi)),pxl)
    mask_size=mask_r>=max_r
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    
    binary = ndimage.binary_fill_holes(label_im)
    maskname=dir_name+'/mask'+'.png'
    Image.fromarray(np.uint8(binary)*255).save(maskname)
    
    im_name=dir_name+'/masked_roi.png'
    Image.fromarray(np.uint8(np.multiply(BW0,binary))).save(im_name)
    
    label_im, nb_labels = ndimage.label(label_im)
    coord  = (np.array(ndimage.center_of_mass(BW0,label_im, range(nb_labels+1))))[1:,[1, 0]]
    sizes  = (ndimage.sum(mask, label_im, range(nb_labels + 1)))[1:]
    l=len(coord[:,0])
    mca_res  = np.zeros([l,5])
    mca_res[:,0]  = roi[0,0]+np.dot(coord[:, 0],pxl)
    mca_res[:,1]  = roi[1,0]+np.dot(coord[:, 1],pxl)
    mca_res[:,2]  = np.dot(sizes,(pxl**2))
    mca_res[:,3]  = np.dot(np.sqrt(np.divide(sizes,np.pi)),pxl)
    mca_res[:,4]  = ndimage.sum(BW0, label_im, range(nb_labels))
    hd=str('morphological cluster analysis LAMA format\nx[nm]\ty[nm]\tsize[nm*nm]\tr[nm]\tI[a.u.]')
    outname=dir_name+'/mca_roi.txt'
    np.savetxt(outname, mca_res[1:,:], fmt='%.5e', delimiter='   ', header = hd, comments='# ')

def print_theo_acc(cnp.ndarray[double, ndim=1] acc_list, str name):
    '''
        Receives list of theoretical achieved. Computes a distribution of achieved localization accuracies. Plots histogram.
    '''
    cdef int Min, Max, Inc
    cdef float Int
    cdef cnp.ndarray[double, ndim=1] x,y
    
    Min=0
    Max=int(np.max(acc_list))
    Int=0.1
    Inc=int((Max-Min)/Int)
    x=np.arange(Min,Max,Int,dtype='float')
    y=np.histogram(acc_list, bins=Inc, range=(Min,Max), density=True)[0]
    f, axarr = plt.subplots(1, sharex=False)
    axarr.bar(x, y, color='gray', edgecolor='black',width=Int)
    axarr.set_xlim([Min,Max])
    axarr.set_xlabel('loc_acc [nm]')
    axarr.set_ylabel('Intensity [a.u.]')
    plt.savefig(name, format='pdf')
    plt.close()



def Thompson(cnp.ndarray[double, ndim=2] locs, float con_fac, float pxl_size, float noise, float sigma, str dir_name):
    '''
        Receives list of localizations in MALK format and experimental setup characteristics from communication layer. Computes the theoretical achieved localization accuracy after the method of Thompson et al. and Mortensen et al. Saves localization accuracy.
    '''
    cdef str outname, thom_name, mort_name, hd
    cdef int length, i
    cdef float progress, Tho_mean, Tho_med, Tho_std, Mor_mean, Mor_med, Mor_std
    cdef double sa
    cdef cnp.ndarray[double, ndim=2] theo_acc
    
    length=np.shape(locs)[0]
    sa=(np.square(sigma)+np.square(pxl_size)/12)
    theo_acc=np.zeros([length,2])
    for i in range (0,length):
        theo_acc[i,0]=np.sqrt((sa/(locs[i,3]/con_fac))+((8*np.pi*np.square(sigma)*np.square(sigma)*np.square(noise)))/(np.square(pxl_size)*np.square(locs[i,3]/con_fac)))
        theo_acc[i,1]=np.sqrt((sa/(locs[i,3]/con_fac))*((16.0/9)+((8*np.pi*sa*np.square(noise))/((locs[i,3]/con_fac)*np.square(pxl_size)))))
        progress=float(i)/length
        update_progress(progress)
    print ('\n')
    Tho_mean = np.mean(theo_acc[:,0])
    Tho_med = np.median(theo_acc[:,0])
    Tho_std =   np.std(theo_acc[:,0])
    Mor_mean = np.mean(theo_acc[:,1])
    Mor_med = np.median(theo_acc[:,1])
    Mor_std = np.std(theo_acc[:,1])
    hd='theoretical caluculated localization accuracy (Lama format)\nTho_mean = %.2f[nm]\tTho_med = %.2f[nm]\tTho_std = %.2f[nm]\nMor_mean = %.2f[nm]\tMor_med = %.2f[nm]\tMor_std = %.2f[nm]\nThompson [nm]\tMortensen [nm]' %(Tho_mean, Tho_med, Tho_std, Mor_mean, Mor_med, Mor_std)
    outname=dir_name+'/theo_lac.txt'
    np.savetxt(outname, theo_acc, fmt='%.5e', delimiter='   ', header = hd, comments='# ')
    thom_name=dir_name+'/thom_acc.pdf'
    print_theo_acc(theo_acc[:,0],thom_name)
    mort_name=dir_name+'/mort_acc.pdf'
    print_theo_acc(theo_acc[:,1],mort_name)

def min_dist(cnp.ndarray[double, ndim=1] point, cnp.ndarray[double, ndim=2]locs):
    '''
        Receives a single localization and a list of localizations. Computes all distance between given localization and list. Returns minimal distance.
    '''
    cdef double d
    
    d=np.min(np.sqrt(np.square(locs[:,0]-point[0])+np.square(locs[:,1]-point[1])))
    return d

def NeNA(cnp.ndarray[double, ndim=2] locs, str dir_name):
    '''
        Receives list of localizations in MALK format. Computes the nearest neighbor distance for every localization. Computes the nearest neighbor distribution. Fits the distribution to a probability density of a jump-distance of a Brownian motion with infinitesimal small diffusion coefficient and Gaussian localization accuracy. Saves the NeNA distribution and the fit result.
    '''
    cdef str outname, hd
    cdef int length, i, max_frame
    cdef double j,o,p,progress
    cdef cnp.ndarray[double, ndim=1] NeNA_dist, NeNA_acc
    cdef cnp.ndarray[double, ndim=2] frames, d, temp_locs, NeNA_err
    cdef cnp.ndarray idx
    
    max_frame=int(np.max(locs[:,2]))
    length=np.shape(locs)[0]
    frames=np.zeros([length,2])
    frames[:,1]=locs[:,2]
    tree=spatial.KDTree(frames[:,0:2])
    d=np.zeros([length,1])
    p=-1
    for i in range (0,length):
        o=locs[i,2]
        # j muss angeben,ob naechster Frame existent ist
        j=locs[i+1,2]-locs[i,2]
        if locs[i,2]<max_frame and o==p:
            d[i]=min_dist(locs[i,0:3],temp_locs)
            p=o
        elif locs[i,2]<max_frame and o>p:
            temp_locs=locs[tree.query_ball_point([0,o+1], 0.1),0:2]
            if np.shape(temp_locs)[0]>0:
                d[i]=min_dist(locs[i,0:3],temp_locs)
                p=o
            progress=float(locs[i,2])/(max_frame)
            update_progress(progress)
        elif locs[i,2]==max_frame:
            break
    print('\n')
    idx=d>0
    NeNA_dist=d[idx]
    NeNA_acc, NeNA_err=plot_NeNA(NeNA_dist,dir_name)
    hd=str('the average localization accuracy by NeNA is at %.1f [nm]' %(float(NeNA_acc[0])))
    outname=dir_name+'/NeNA_lac.txt'
    np.savetxt(outname, NeNA_dist, fmt='%.5e', delimiter='   ',header=hd,comments='# ')

def CFunc2dCorr(cnp.ndarray[double, ndim=1] r, double a, double rc, double w, double F, double A, double O):
    '''
        NeNA function with correction terms. Receives probability density parameters and a distance. Calculates the probability of the given distance to be measured in combination with given parameters. Returns probability.
    '''
    cdef cnp.ndarray[double, ndim=1] y
    
    y=(r/(2*a*a))*np.exp((-1)*r*r/(4*a*a))*A+(F/(w*np.sqrt(np.pi/2)))*np.exp(-2*((r-rc)/w)*((r-rc)/w))+O*r
    return y

def Area(cnp.ndarray[double, ndim=1] r, cnp.ndarray[double, ndim=1] y):
    '''
        Receives probability density function and distance range. Calculates the integrated area for the distance range. Returns area.
    '''
    cdef double A
    
    A=abs(np.trapz(y, r))
    return A

def CFit_resultsCorr(cnp.ndarray[double, ndim=1] r, cnp.ndarray[double, ndim=1] y):
    '''
        Receives probability density function and distance range. Calculates most probable NeNA parameters by fitting the function. Returns fit parameters.
    '''
    cdef double A
    cdef cnp.ndarray[double, ndim=1] p0, popt
    cdef cnp.ndarray[double, ndim=2] pcov
    
    A=Area(r,y)
    p0 = np.array([10.0,15,100,(A/2),(A/2),((y[98]/200))])
    popt, pcov = curve_fit(CFunc2dCorr,r,y,p0)
    return popt, pcov

def plot_NeNA(cnp.ndarray[double, ndim=1] NeNA_dist, str dir_name):
    '''
        Receives probability density function and distance range. Calls function to compute NeNA parameters by fitting. Plots received probability density function and overlays most fitting NeNA function with it. Saves figure.
    '''
    cdef str name
    cdef int Min, Max, Int, Inc
    cdef cnp.ndarray[double, ndim=1] x,y,acc,NeNA_func
    cdef cnp.ndarray[double, ndim=2] acc_err

    
    Min=0
    Max=150
    Int=1
    Inc=(Max-Min)/Int
    x=np.arange(Min,Max,Int,dtype='float')
    y=np.histogram(NeNA_dist, bins=Inc, range=(Min,Max), density=True)[0]
    acc, acc_err=CFit_resultsCorr(x,y)
    NeNA_func=CFunc2dCorr(x,acc[0],acc[1],acc[2],acc[3],acc[4],acc[5])
    name=dir_name+'/NeNA_lac.pdf'
    f, axarr = plt.subplots(1, sharex=False)
    axarr.bar(x, y, color='gray', edgecolor='black',width=Int)
    axarr.plot(x,NeNA_func, 'b')
    axarr.set_xlim([Min,Max])
    axarr.set_xlabel('loc_acc [nm]')
    axarr.set_ylabel('Intensity [a.u.]')
    plt.savefig(name, format='pdf')
    plt.close()
    return acc, acc_err


def detect_beads(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] buds, int r, float n):
    '''
        Receives localization list, a candidate list of detected budding sites, and threshold parameters. Proofs if a candidate site has enough localizations to be a fiducial marker. Returns list of fiducial marker positions.
        
    '''
    cdef double m
    cdef cnp.ndarray[list, ndim=1] x
    cdef cnp.ndarray[double, ndim=1] lengths
    cdef cnp.ndarray[double, ndim=2] sort_buds
    cdef cnp.ndarray idx
    
    m=(n*((max(locs[:,2])-min(locs[:,2]))/100.0))
    #m=max(locs[:,2])-(n*(max(locs[:,2])/100.0))
    tree=spatial.KDTree(locs[:,0:2])
    x=tree.query_ball_point(buds, r)
    lengths = np.asarray([len(j) for j in x]).astype('float')
    idx = lengths > m
    sort_buds = buds[(idx), :]
    return sort_buds

def write_sort_beads(cnp.ndarray[double, ndim=2] buds01, cnp.ndarray[double, ndim=2] buds02, str name01, str name02):
    '''
        Receives bud files and saves them.
    '''
    cdef str bud_name01, bud_name02, hd
    cdef int cnum
    
    bud_name01=name01[0:(len(name01)-4)]+'_Beads.txt'
    bud_name02=name02[0:(len(name02)-4)]+'_Beads.txt'
    cnum = np.shape(buds01)[0]
    hd=str('Cluster analysis LAMA format\nnumber of cluster: %i \nx[nm]\ty[nm]\tsize[nm*nm]\tr[nm]\tI[a.u.]' %(cnum))
    np.savetxt(bud_name01, buds01, fmt='%.5e', delimiter='   ',header=hd,comments='# ')
    np.savetxt(bud_name02, buds02, fmt='%.5e', delimiter='   ',header=hd,comments='# ')


def draw_roi(cnp.ndarray[double, ndim=2] BW, cnp.ndarray[double, ndim=2] buds, int r, int pxl, str name):
    '''
        Receives 2d histogram of localization list, a list of detected budding sites, and filtering conditions. Draws roi and outlines detected budding sites. Saves image.
    '''
    cdef str Clustername
    cdef float R
    
    Clustername=name +'/First_channel_Cluster.png'
    R=float(r/pxl)
    buds=np.divide(buds,pxl)
    Im = Image.fromarray(np.uint8(BW)*255)
    draw =ImageDraw.Draw(Im)
    for x,y in buds:
        draw.ellipse((x-R,y-R,x+R,y+R), outline=255)
        draw.point((x,y), fill=255)
    Im.save(Clustername)

def link_fuducials(cnp.ndarray[double, ndim=2] buds01, cnp.ndarray[double, ndim=2] buds02, int rmax):
    '''
        Receives two fiducial localization lists from corresponding channels and a distance threshold parameter. Sorts and links corresponding fiducials. Returns sorted  fiducial localization lists.
    '''
    cdef int i, length01, length02
    cdef cnp.ndarray[list, ndim=1] x01, x02
    cdef cnp.ndarray[long, ndim=1] idx01, idx02
    
    tree01=spatial.KDTree(buds01[:,0:2])
    tree02=spatial.KDTree(buds02[:,0:2])
    x01=tree01.query_ball_point(buds02[:,0:2], rmax)
    x02=tree02.query_ball_point(buds01[:,0:2], rmax)
    idx01=np.zeros([1], dtype=int)
    idx02=np.zeros([1], dtype=int)
    for i in range(0,len(x01)):
        length01 = len(x01[i])
        if length01==1:
            idx01=np.append(idx01,i).astype('int')
    idx01=np.delete(idx01,0)
    buds02=buds02[idx01,:]
    for i in range(0,len(x02)):
        length02 = len(x02[i])
        if length02==1:
            idx02=np.append(idx02,i).astype('int')
    idx02=np.delete(idx02,0)
    buds01=buds01[idx02,:]
    tree01=spatial.KDTree(buds01[:,0:2])
    x01=tree01.query_ball_point(buds02[:,0:2], rmax)
    idx01=np.zeros([1], dtype=int)
    for i in range(0,len(x01)):
        length01 = len(x01[i])
        if length01==1:
            idx01=np.append(idx01,i).astype('int')
    idx01=np.delete(idx01,0)
    buds02=buds02[idx01,:]
    return buds01,buds02

def plot_buds(cnp.ndarray[double, ndim=2] buds01, cnp.ndarray[double, ndim=2]buds02, str name):
    '''
        Receives bud localization lists and plots them.
    '''
    cdef str outname
    
    outname=name[0:(len(name)-4)] +'_buds.pdf'
    f, axarr = plt.subplots(1, sharex=False)
    axarr.plot(buds01[:,0],buds01[:,1], 'ob')
    axarr.plot(buds02[:,0],buds02[:,1], 'xr')
    axarr.set_xlabel('x [nm]')
    axarr.set_ylabel('y [nm]')
    plt.savefig(outname, bbox_inches='tight')
    plt.close()

def create_Affine_Matrix(cnp.ndarray[double, ndim=2] fp, cnp.ndarray[double, ndim=2] tp):
    '''
        Receives two fiducial localization lists from corresponding channels. Creates affine transformation matrix by least square fitting. returns matrix.
    '''
    cdef int BeadNumber, i
    cdef cnp.ndarray[double, ndim=2] M, B, x
    
    BeadNumber=len(fp)
    M=np.zeros ([2*BeadNumber,6], float)
    M[0:BeadNumber,0]=fp[:,0]
    M[0:BeadNumber,1]=fp[:,1]
    M[0:BeadNumber,2]=1
    M[BeadNumber:2*BeadNumber,3]=fp[:,0]
    M[BeadNumber:2*BeadNumber,4]=fp[:,1]
    M[BeadNumber:2*BeadNumber,5]=1
    B=np.zeros([2*BeadNumber,1], float)
    for i in range (0,BeadNumber):
        B[i]= tp[i,0]
        B[i+BeadNumber]=tp[i,1]
    x = np.linalg.lstsq(M, B)[0]
    return x

def create_lin_trans_Matrix(cnp.ndarray[double, ndim=2] fp, cnp.ndarray[double, ndim=2] tp):
    '''
        Receives two fiducial localization lists from corresponding channels. Creates linear transformation matrix by least square fitting. returns matrix.
    '''
    cdef int BeadNumber,i
    cdef cnp.ndarray[double, ndim=2] M, B, x
    
    BeadNumber=len(fp)
    M=np.zeros([2*BeadNumber,4], float)
    y=np.zeros([9,1], float)
    M[0:BeadNumber,0]=fp[:,0]
    M[0:BeadNumber,1]=1
    M[BeadNumber:2*BeadNumber,2]=fp[:,1]
    M[BeadNumber:2*BeadNumber,3]=1
    B=np.zeros([2*BeadNumber,1], float)
    for i in range (0,BeadNumber):
        B[i]= tp[i,0]
        B[i+BeadNumber]=tp[i,1]
    x = np.linalg.lstsq(M, B)[0]
    y[0]=x[0]
    y[2]=x[1]
    y[4]=x[2]
    y[5]=x[3]
    y[8]=1
    return y

def translate_locs(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] H):
    '''
        Receives localization list MALK format and translation matrix. Performs translation. Returns translated localization list.
    '''
    cdef int i
    cdef cnp.ndarray[double, ndim=2] M,p,o
    
    # Form Affine Matrix from H
    M=np.zeros([3,3],float)
    M[0,0]=H[0]
    M[0,1]=H[1]
    M[0,2]=H[2]
    M[1,0]=H[3]
    M[1,1]=H[4]
    M[1,2]=H[5]
    M[2,2]=1
    # Translate
    p=np.zeros([3,1])
    p[2]=1
    for i in range (0,len(locs)):
        p[0]=locs[i,0]
        p[1]=locs[i,1]
        o=np.dot(M,p)
        locs[i,0]=o[0]
        locs[i,1]=o[1]
    return locs
    
def register_channels(cnp.ndarray[double, ndim=2] locs02, cnp.ndarray[double, ndim=2] buds02, cnp.ndarray[double, ndim=2] buds01, str name):
    '''
        Receives localization list in MALK format and bud lists from corresponding channles. Calls routine to create affine transformation matrix. Calls function to translate localization list. Saves translated localization list.
    '''
    cdef str outname, hd
    cdef cnp.ndarray[double, ndim=2] H, locs025
    
    H=create_Affine_Matrix(buds02,buds01)
    locs025=translate_locs(locs02,H)
    outname=name[0:(len(name)-4)] +'_channel_registered.txt'
    hd = str('localization roi file (Malk format)\nx[nm]\ty[nm]\t[frame]\tI[a.u.]')
    np.savetxt(outname, locs025, fmt='%.5e', delimiter='   ',header=hd,comments='# ')


def register_channels_trans(cnp.ndarray[double, ndim=2] locs02, cnp.ndarray[double, ndim=2] buds02, cnp.ndarray[double, ndim=2] buds01, str name):
    '''
        Receives localization list in MALK format and bud lists from corresponding channles. Creates linear transformation. Saves translated localization list.
    '''
    cdef str outname, hd
    cdef cnp.ndarray[double, ndim=2] H, locs025
    
    H=buds02-buds01
    locs025=locs02
    locs025[:,0]=locs02[:,0]-np.mean(H[:,0])
    locs025[:,1]=locs02[:,1]-np.mean(H[:,1])
    outname=name[0:(len(name)-4)] +'_channel_registered.txt'
    hd = str('localization roi file (Malk format)\nx[nm]\ty[nm]\t[frame]\tI[a.u.]')
    np.savetxt(outname, locs025, fmt='%.5e', delimiter='   ',header=hd,comments='# ')


def RipleysEdge(cnp.ndarray[double, ndim=2] BoxOR, float Dist, str name):
    '''
        Receives quadratic roi of localization list and edge length of the roi. Projects the localizations of the roi on a surface of a sphere to receive a toroidal edge correction. Returns sphrere surface as edge effect corrected roi.
    '''
    cdef str Boxname, hd
    cdef int Length, NMR
    cdef cnp.ndarray[double, ndim=2] BoxMR
    
    Length=len(BoxOR)
    NMR=0
    BoxMR=np.zeros([(9*Length),2])
    BoxMR[0:Length,0]=BoxOR[:,0]
    BoxMR[0:Length,1]=BoxOR[:,1];
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]-Dist
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]+Dist
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]+Dist;
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]+Dist
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]+Dist
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]-Dist
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]+Dist
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]-Dist
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]-Dist
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]-Dist
    
    NMR=NMR+Length
    BoxMR[NMR:(NMR+Length),0]=BoxOR[:,0]+Dist
    BoxMR[NMR:(NMR+Length),1]=BoxOR[:,1]-Dist
    
    Boxname=name + '/TEC_locs.txt'
    hd = str('localization roi file (Malk format)\nx[nm]\ty[nm]\t[frame]\tI[a.u.]')
    np.savetxt(Boxname, BoxMR, fmt='%.3e', delimiter='   ',header=hd,comments='# ')
    return BoxMR

def RipleysK(cnp.ndarray[double, ndim=2] BoxOR, cnp.ndarray[double, ndim=2] BoxMR, int Ink, float R, float d, str name):
    '''
        Receives quadratic roi of localization list and sphere surface as edge corrected roi, as threshold parameters to calculate Ripleys k-function. Calculates Ripleys k-, l-, and h-function. Detects Rmax of Ripleys h-function. Plots and saves result.
    '''
    cdef str RIPname, Plotname, hd
    cdef int end_val, length, i, n
    cdef long m
    cdef float r, progress, Ro, est, lam
    cdef cnp.ndarray[double, ndim=2] Distx, Disty, Dist, Kr, K
    
    Distx=np.zeros([(len(BoxMR)),1])
    Disty=np.zeros([(len(BoxMR)),1])
    Dist=np.zeros([(len(BoxMR)),1])
    Kr=np.zeros([Ink,3])
    end_val=np.shape(BoxOR)[0]
    for i in range(0, (len(BoxOR))):
        Distx[:,0]=BoxOR[i,0]-BoxMR[:,0]
        Disty[:,0]=BoxOR[i,1]-BoxMR[:,1]
        Dist =(np.sqrt(np.power(Distx, 2)+np.power(Disty, 2)))
        Dist =np.delete(Dist, i, 0)
        for n in range(0,Ink):
            r=n*(R/Ink)
            Kr[n,0]=len(np.nonzero(Dist<r)[0])
        Kr[:,1]=Kr[:,1]+Kr[:,0]
        progress=float(i)/end_val
        update_progress(progress)
    print('\n')

    Ro=len(BoxOR)/(d**2)
    est=Ro*(math.pi*(R**2))*len(BoxOR)
    lam=1/(math.pi*(R**2))
    Kr[:,2]=Kr[:,1]/(lam*est)
    
    K=np.zeros([Ink,7])
    for n in range(0,Ink):
        r=n*(R/Ink)
        K[n,0]=r
        K[n,1]=np.pi*r*r
        K[n,2]=Kr[n,2]
        K[n,3]=r
        K[n,4]=np.sqrt(K[n,2]/np.pi)
        K[n,5]=K[n,3]-r
        K[n,6]=K[n,4]-r
    m=np.argmax(K[:,6])
    hd=str('Ripley s K-Function Maximum %.1f at r = %.1f [nm].' %(K[m,6],K[m,0]))
    RIPname=name + '/ripley.txt'
    np.savetxt(RIPname, K, fmt='%.3e', delimiter='   ', header=hd, comments='# ')


    Plotname=name + '/ripley.pdf'
    f, axarr = plt.subplots(1, sharex=False)
    axarr.plot(K[:,0], K[:,5], 'b')
    axarr.plot(K[:,0], K[:,6], 'r')
    axarr.set_xlim([0,R])
    axarr.set_xlabel('r [nm]')
    axarr.set_ylabel('L(r)-r')
    plt.savefig(Plotname, bbox_inches='tight')
    plt.close()


def create_cbc_roi(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] roi, str dir_name, str name):
    '''
        receives localization list, path to a loc-file, and roi parameters from communication layer. Filters localization list spatially and chronologically based on roi parameters. Saves filtered localization list. Returns filtered localization list.
    '''
    cdef str outfilename, hd
    cdef cnp.ndarray[double, ndim=2] locs1
    cdef cnp.ndarray idx1, idx2, idy1, idy2, idt1, idt2
    
    outfilename=dir_name + name
    idx1=locs[:,0]>=roi[0,0]
    idx2=locs[:,0]<=roi[0,1]
    idy1=locs[:,1]>=roi[1,0]
    idy2=locs[:,1]<=roi[1,1]
    idt1=locs[:,2]>=roi[2,0]
    idt2=locs[:,2]<=roi[2,1]
    locs1 = (locs[(idx1&idx2&idy1&idy2&idt1&idt2),:])
    hd=str('localization roi file (Malk format)\nx[nm]\ty[nm]\t[frame]\tcbc[a.u.]')
    np.savetxt(outfilename, locs1, fmt='%.5e', delimiter='   ', header = hd, comments='# ' )
    return locs1

def cbc(cnp.ndarray[double, ndim=2] locsA, cnp.ndarray[double, ndim=2] locsB, int rmax, int inc, float w):
    '''
        Receives two localization lists in MALK format and threshold parameters. Calculates cbc values for every localization in first channel with the distribution of the second channel. Returns cbc value list for first channel.
    '''
    cdef int rp, i, r1, r2, end_val, n, m
    cdef float progress
    cdef double S, D
    cdef list roi_idx_A, roi_idx_B
    cdef cnp.ndarray[double, ndim=1] loc, rA, rB, distA, distB
    cdef cnp.ndarray[long, ndim=1] histA, histB
    cdef cnp.ndarray[double, ndim=2] area, dist, C, roi
    
    area=np.zeros([inc,1])
    dist=np.zeros([inc,3])
    C=np.zeros([len(locsA),1])
    rp=rmax/inc
    for i in range(0,inc):
        r1=rp*i
        r2=rp*(i+1)
        area[i]=math.pi*((r2**2)-(r1**2))
    treeB=spatial.KDTree(locsB[:,0:2])
    treeA=spatial.KDTree(locsA[:,0:2])
    end_val=np.shape(locsA)[0]
    for i in range (0,end_val):
        loc=locsA[i,0:2]
        #Verteilung B
        #roi=[]
        roi_idx_B=treeB.query_ball_point(loc,rmax)
        roi=locsB[roi_idx_B,:]
        distB=np.sqrt(np.square(roi[:,0]-loc[0])+np.square(roi[:,1]-loc[1]))
        histB, rB=np.histogram(distB, bins=inc, range=(0,rmax))
        #dist[:,0]=histB[1][1:]
        dist[:,0]=rB[1:]
        #dist[:,1]=np.divide(histB[0],area.T)
        dist[:,1]=np.divide(histB,area.T)
        n=len(roi)
        #Verteilung A
        #roi=[]
        roi_idx_A=treeA.query_ball_point(loc,rmax)
        roi=locsA[roi_idx_A,:]
        distA=np.sqrt(np.square(roi[:,0]-loc[0])+np.square(roi[:,1]-loc[1]))
        histA, rA=np.histogram(distA, bins=inc, range=(0,rmax))
        dist[:,2]=np.divide(histA,area.T)
        m=len(roi)
        # Correlation und Wichtung
        if n>5 and m>5:
            S=spearmanr(dist[:,1], dist[:,2])[0]
            D=min(distB)
            C[i]=S*np.e**(-w*D/rmax)
        else:
            C[i]=0
        progress=float(i)/end_val
        update_progress(progress)
    print ('\n')
    return C


def split_cbc_locs(cnp.ndarray[double, ndim=2] locs):
    '''
        Receives localization list in MALK format. Splits list into two lists containing only frames with even or odd frame numbers. Returns two localization lists.
    '''
    cdef cnp.ndarray[double, ndim=2] locs_A, locs_B
    cdef cnp.ndarray ind
    
    ind=locs[:,2]%2==0
    locs_A=locs[ind,:]
    ind=locs[:,2]%2!=0
    locs_B=locs[ind,:]
    return (locs_A,locs_B)

def combine_split_cbc_locs(cnp.ndarray[double, ndim=2] locs_A, cnp.ndarray[double, ndim=2] locs_B):
    '''
        Receives two localization lists in MALK format. Recombines lists into single list. Sorts list concerning frame numbers. Returns list.
    '''
    cdef cnp.ndarray[double, ndim=2] locs
    locs=np.append(locs_A,locs_B,axis=0)
    locs=locs[locs[:,2].argsort(axis=0)]
    return locs

def make_CBC_image(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] roi, int pxl, str dir_name):
    '''
        receives localization list, roi parameters and desired pxl size from communication layer. computes a 2d-histogram. transform histogram information into 8-bit gray scale image. saves image. returns raw histogram containing absolute number of localizations per bin.
    '''
    cdef str outfilename_cbc, outfilename_int
    cdef cnp.ndarray[double, ndim=1] Z
    cdef cnp.ndarray[double, ndim=2] BW0, BW1, BW2, BW3
    
    outfilename_cbc=dir_name+'/cbc_ind.png'
    outfilename_int=dir_name+'/cbc_locs.png'
    Z=(locs[:,3]+1)/2
    BW0=np.histogram2d(locs[:,0], locs[:,1],bins=[(roi[0,1]-roi[0,0])/pxl,(roi[1,1]-roi[1,0])/pxl], range=[[roi[0,0],roi[0,1]],[roi[1,0],roi[1,1]]])[0]
    BW0=np.rot90(BW0, k=1)
    BW0=np.flipud(BW0)
    BW0=np.clip(BW0,0,255)
    BW1=np.clip(BW0, 1, 255)
    BW2=np.histogram2d(locs[:,0], locs[:,1],bins=[(roi[0,1]-roi[0,0])/pxl,(roi[1,1]-roi[1,0])/pxl], range=[[roi[0,0],roi[0,1]],[roi[1,0],roi[1,1]]],weights=Z)[0]
    BW2=np.rot90(BW2, k=1)
    BW2=np.flipud(BW2)
    BW3=np.dot(np.divide(BW2, BW1),255)
    BW4 = Image.fromarray(np.uint8(BW3))
    BW4.save(outfilename_cbc)
    BW5 = Image.fromarray(np.uint8(BW0))
    BW5.save(outfilename_int)
    return BW0

# DBSCAN



def write_offset(cnp.ndarray[double, ndim=2] roi, int pxl, str dir_name):
    '''
        calculates the offset of the claculated super resolved image and saves it.
    '''
    cdef str outname, hd
    cdef list o
    
    o=[roi[0,0],roi[1,0],pxl]
    hd='image offset: xmin [nm], ymin[nm], pxl[nm]'
    outname=dir_name+'/offset.txt'
    np.savetxt(outname, o, fmt='%.5e', delimiter='   ',header=hd,comments='# ')

def roi_from_file(list raw):
    '''
        Receives raw LAMA settings file and extracts roi information. Farmats them. Returns them.
    '''
    cdef int i
    cdef cnp.ndarray[double, ndim=2] roi
    
    roi=np.zeros([5,2])
    for i in range (0,5):
        roi_line=np.array(raw[(5+i)].split())
        roi[i,:]=roi_line.astype(np.float, copy=False)
    return roi

def rest_from_file(list raw):
    '''
        Receives raw LAMA settings file and extracts information except roi information. Farmats them. Returns them.
    '''
    cdef int i,j,length
    cdef str first
    cdef cnp.ndarray rest_line
    cdef cnp.ndarray[double, ndim=2] rest_para

    rest_para=np.zeros([42,1])
    length = np.shape(raw)[0]
    j=0
    for i in range (11,length):
        rest_line=np.array(raw[i].split())
        first = str(rest_line[0])
        if first != '#':
            rest_para[j]=rest_line.astype(np.float, copy=False)
            j+=1
    return rest_para
    
def read_import_file(str name):
    '''
        Recieves file name for saved LAMA settings. Loads several functions to extract the setting information from file. Returns information to gui.
    '''
    cdef int data_type
    cdef list raw
    cdef cnp.ndarray[double, ndim=2] roi, rest_para
    
    file = open(name)
    raw=file.read().splitlines()
    file.close
    data_type=int(raw[3])
    roi=roi_from_file(raw)
    rest_para=rest_from_file(raw)
    return data_type,roi,rest_para

    
def write_settings(cnp.ndarray[double, ndim=2] settings, str dir_name):
    '''
        saves settings to a LAMA settings file.
    '''
    cdef str out_name
    cdef int ver
    
    ver=1510
    out_name=dir_name+'/settings.txt'
    out_file = open(out_name, "w")
    out_file.write('# Settings file Lama v.%i format:\n' %(ver))
    out_file.write('# Main\n')
    out_file.write('# file name type:\n')
    out_file.write('%i\n' %(settings[0]))
    out_file.write('# roi:\n')
    out_file.write('%.3f\t%.3f\n' %(settings[1], settings[2]))
    out_file.write('%.3f\t%.3f\n' %(settings[3], settings[4]))
    out_file.write('%.3f\t%.3f\n' %(settings[5], settings[6]))
    out_file.write('%.3f\t%.3f\n' %(settings[7], settings[8]))
    out_file.write('%.3f\t%.3f\n' %(settings[9], settings[10]))
    out_file.write('# setup:\n')
    out_file.write('%.3f\n' %(settings[11]))
    out_file.write('%.3f\n' %(settings[12]))
    out_file.write('%.3f\n' %(settings[13]))
    out_file.write('# Visualize\n')
    out_file.write('# image:\n')
    out_file.write('%.3f\n' %(settings[14]))
    out_file.write('%.3f\n' %(settings[15]))
    out_file.write('%.3f\n' %(settings[16]))
    out_file.write('# image type:\n')
    out_file.write('%i\n' %(settings[17]))
    out_file.write('%i\n' %(settings[18]))
    out_file.write('%i\n' %(settings[19]))
    out_file.write('# mca:\n')
    out_file.write('# enable mca:\n')
    out_file.write('%i\n' %(settings[20]))
    out_file.write('# mca specs:\n')
    out_file.write('%.3f\n' %(settings[21]))
    out_file.write('%.3f\n' %(settings[22]))
    out_file.write('%.3f\n' %(settings[23]))
    out_file.write('# Accuracy:\n')
    out_file.write('%.3f\n' %(settings[24]))
    out_file.write('%.3f\n' %(settings[25]))
    out_file.write('%.3f\n' %(settings[26]))
    out_file.write('# Ripley:\n')
    out_file.write('%.3f\n' %(settings[27]))
    out_file.write('%.3f\n' %(settings[28]))
    out_file.write('%.3f\n' %(settings[29]))
    out_file.write('# Register\n')
    out_file.write('# Bead Detection:\n')
    out_file.write('%.3f\n' %(settings[30]))
    out_file.write('%.3f\n' %(settings[31]))
    out_file.write('%.3f\n' %(settings[32]))
    out_file.write('%.3f\n' %(settings[33]))
    out_file.write('# Registeration:\n')
    out_file.write('%.3f\n' %(settings[34]))
    out_file.write('%i\n' %(settings[35]))
    out_file.write('# CBC:\n')
    out_file.write('%.3f\n' %(settings[36]))
    out_file.write('%.3f\n' %(settings[37]))
    out_file.write('%.3f\n' %(settings[38]))
    out_file.write('# CBC type:\n')
    out_file.write('%i\n' %(settings[39]))
    out_file.write('# Hirarchical Clustering:\n')
    out_file.write('# sort:\n')    
    out_file.write('%i\n' %(settings[40]))
    out_file.write('%.3f\n' %(settings[41]))
    out_file.write('%.3f\n' %(settings[42]))
    out_file.write('%.3f\n' %(settings[43]))
    out_file.write('# MCA:\n')
    out_file.write('%i\n' %(settings[44]))
    out_file.write('# Stoichiometry:\n')
    out_file.write('%.3f\n' %(settings[45]))
    out_file.write('%.3f\n' %(settings[46]))
    out_file.write('%.3f\n' %(settings[47]))
    out_file.write('%.3f\n' %(settings[48]))
    out_file.write('%.3f\n' %(settings[49]))
    out_file.write('%.3f\n' %(settings[50]))
    out_file.write('%i\n' %(settings[51]))
    out_file.write('%i\n' %(settings[52]))
    out_file.write('# go Lama!')
    out_file.close()

class hierarchical_cluster:
    def __init__(self,cnp.ndarray[double, ndim=2] locs, double eps, int pmin, int dim):
        
        cdef int cnum, length
        cdef cnp.ndarray[long, ndim=1] cluster, marked,order
        cdef cnp.ndarray[double, ndim=1] reachDist, coreDist
        
        self.locs=locs
        self.eps = eps
        self.dim = dim
        self.pmin = pmin
        cnum = 0
        self.cnum = cnum
        length = np.shape(locs)[0]
        self.length = length
        cluster = np.zeros(self.length, dtype=int)
        self.cluster = cluster
        marked = np.zeros(self.length, dtype=int)
        self.marked = marked
        self.tree = spatial.KDTree(self.locs[:,0:self.dim])
        reachDist = np.ones(self.length, dtype=float)*1E10
        self.reachDist = reachDist
        coreDist = np.zeros(self.length, dtype=float)
        self.coreDist = coreDist
        order = np.zeros(self.length, dtype=int)
        self.order = order
    
    
    def update_progress(self, float i):
        '''
            receives progress status from application layer and visualizes progress in terminal.
            '''
        cdef str text
        cdef int barLength, block
        cdef float progress
        
        progress = float(i)/float(self.length)
        barLength = 10 # Modify this to change the length of the progress bar
        block = int(round(barLength*progress))
        text = "\rthe lama is processing: [{0}] {1}%".format( "#"*block + "-"*(barLength-block), int(progress*100)+1)
        sys.stdout.write(text)
        sys.stdout.flush()
    
    def regionQuery(self, cnp.ndarray[double, ndim=1] loc, int l):
        cdef list ptemp
        
        ptemp = self.tree.query_ball_point(loc,self.eps+l)
        return (ptemp)
    
    def expandCluster(self, list pnhood, int nhood):
        
        cdef int i, exp_nhood
        cdef list exp_pnhood
        cdef cnp.ndarray[double, ndim=1] exp_loc
        
        i=1
        while i > 0:
            if self.marked[pnhood[i]] == 0:
                self.marked[pnhood[i]] = 1
                exp_loc = self.locs[pnhood[i],0:self.dim]
                exp_pnhood = self.regionQuery(exp_loc,0)
                exp_nhood = len(exp_pnhood)
                if exp_nhood >= self.pmin:
                    #pnhood = np.append(pnhood,exp_pnhood, axis=0)
                    pnhood.extend(exp_pnhood)
                    nhood += exp_nhood
            if self.cluster[pnhood[i]] == 0:
                self.cluster[pnhood[i]] = self.cnum
            if i < nhood-1:
                i += 1
            else:
                i = 0

    def core_distance(self, cnp.ndarray[double, ndim=1] loc, list pnhood, int nhood):
        
        cdef int i, j
        cdef double temp_coreDist
        cdef cnp.ndarray[double, ndim=2] dist
        
        dist = np.zeros([nhood,1])
        if nhood > self.pmin:
            for i in range (0,nhood):
                j=pnhood[i]
                dist[i] = self.euc_dist(loc, self.locs[j,0:self.dim])
            dist = np.sort(dist, axis=0)
            temp_coreDist=dist[self.pmin]
        else:
            temp_coreDist=0
        return temp_coreDist

    def euc_dist(self,cnp.ndarray[double, ndim=1] p, cnp.ndarray[double, ndim=1] o):
        
        cdef int i
        cdef double dist
        
        dist = 0
        for i in range (0,self.dim):
            dist += ((p[i]-o[i])**2)
        dist = np.sqrt(dist)
        return dist

    def DBSCAN(self):
        cdef int i, nhood
        cdef list pnhood
        cdef cnp.ndarray[double, ndim=1] loc
        
        self.cnum = 0
        for i in range(0, self.length):
            self.update_progress(i)
            loc = self.locs[i,0:self.dim]
            if self.marked[i] == 0:
                self.marked[i] = 1
                pnhood = self.regionQuery(loc,0)
                nhood = len(pnhood)
                if nhood <= self.pmin:
                    self.cluster[i] = 0
                else:
                    self.cnum += 1
                    self.cluster[i] = self.cnum
                    self.expandCluster(pnhood, nhood)
        print('\n')


    def OPTICS(self):
        cdef int i, l, nhood, ind
        cdef long ob
        cdef list order_grow, pnhood
        cdef cnp.ndarray[long, ndim=1] seed
        cdef cnp.ndarray[double, ndim=1] loc
        
        for i in range (0,self.length):
            self.update_progress(i)
            loc=self.locs[i,0:self.dim]
            nhood = 0
            l=0
            while nhood <= self.pmin:
                pnhood = self.regionQuery(loc, l)
                nhood = len(pnhood)
                l+=1
            self.coreDist[i] = self.core_distance(loc,pnhood,nhood)
        print ('\n')
        
        order_grow = list()
        seed = np.arange(self.length, dtype = int)
        ind = 0
        
        while len(seed) != 1:
            ob = seed[ind]
            seedInd = np.where(seed != ob)
            seed = seed[seedInd]
            
            #order_grow.append(ob)
            order_grow.extend([int(ob)])
            tempX = np.ones(len(seed))*self.coreDist[ob]
            tempD = [self.euc_dist(self.locs[ob,0:self.dim],self.locs[seed[i],0:self.dim]) for i in range(len(seed))]
            
            temp = np.column_stack((tempX, tempD))
            mm = np.max(temp, axis = 1)
            ii = np.where(self.reachDist[seed]>mm)[0]
            self.reachDist[seed[ii]] = mm[ii]
            ind = np.argmin(self.reachDist[seed])
                
        #order_grow.append(seed[0])
        order_grow.extend([seed[0]])
        self.order = order_grow
        self.reachDist[0] = 0

    def extract_DBSCAN(self):
        cdef int i
        
        self.cnum = 1
        for i in range (0,self.length):
            if self.reachDist[self.order[i]]>self.eps:
                if self.coreDist[self.order[i]]<= self.eps:
                    self.cnum += 1
                    self.cluster[self.order[i]] = self.cnum
                else:
                    self.cluster[self.order[i]] = 0
            else:
                self.cluster[self.order[i]] = self.cnum


# Class analyze hc_cluster
class hcAnalysis:
    def __init__(self, cnp.ndarray[double, ndim=2] hc_locs, double eps, int pmin):
        cdef cnp.ndarray[double, ndim=2] cluster_ana
        cdef int cnum
        
        self.hc_locs = hc_locs
        self.cnum = np.max(self.hc_locs[:,3])
        cluster_ana = np.zeros([self.cnum-1,5])
        self.cluster_ana = cluster_ana
        blinks = np.zeros([self.cnum-1,1])
        self.blinks = blinks
        self.eps = eps
        self.pmin = pmin
    
    def count_blinks(self, locs):
        
        
        length = np.shape(locs)[0]
        n=0
        for i in range (1,length):
            if locs[i,2]>(locs[i-1,2]+1):
                n+=1
        return n
    
    def create_roi_hc(self, int cluster):
        #cdef cnp.ndarray[int, ndim=1] index
        cdef cnp.ndarray[double, ndim=2] locs1
        
        index = self.hc_locs[:,3] == cluster
        locs1 = (self.hc_locs[index,:])
        if np.shape(locs1)[0]>1:
            locs1 = locs1[np.argsort(locs1[:,2]),:]
            self.blinks[cluster-1] = self.count_blinks(locs1)
        else:
            self.blinks[cluster-1] = 1
        return locs1

    def ShoelaceArea(self, corners):
        n = len(corners) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2.0
        return area

    def create_polygon_hc(self, locs):
        
        if np.shape(locs)[0]>3:
            #get area
            hull = sp.spatial.ConvexHull(locs)
            shape=np.zeros([np.shape(hull.vertices)[0],2])
            shape[:,0]=locs[hull.vertices,0]
            shape[:,1]=locs[hull.vertices,1]
            #get centroid
            cx = np.mean(shape[:,0])
            cy = np.mean(shape[:,1])
            area = self.ShoelaceArea(shape)
        else:
            cx = np.mean(locs[:,0])
            cy = np.mean(locs[:,1])
            area = 0.0
        #get radius
        r=np.sqrt(area/np.pi)
        #get intensity
        l = np.shape(locs)[0]
        return [cx,cy,area,r,l]

    def analyze_hc_cluster(self):
        '''analyze clusters from hierarcical clustering for ther center of mass, area, perimeter, and number of comprised localizations
            import: localizations with cluster indices (hc_locs), number of clusters (cnum), and path name (dir_name)
            no output
        '''
        cdef int i,j
        cdef cnp.ndarray[double, ndim=2] cluster_locs
        cdef str hd, outfilename

        for i in range (0,self.cnum-1):
            j=i+1
            # sub list locs of a distinct cluster
            cluster_locs = self.create_roi_hc(j)
            # find edge localizations
            self.cluster_ana[i,:] = self.create_polygon_hc(cluster_locs[:,0:2])

# create Cluster images
def make_hc_Images(cnp.ndarray[double, ndim=2] locs, cnp.ndarray[double, ndim=2] cluster_ana, int cnum, cnp.ndarray[double, ndim=2] roi, int pxl, int fixed_ind, int fix_val, str dir_name):
    '''
        receives localization list, roi parameters and desired pxl size from communication layer. computes a 2d-histogram. transform histogram information into 8-bit gray scale image. saves image. returns raw histogram containing absolute number of localizations per bin.
    '''
    cdef str outfilename
    cdef cnp.ndarray[long, ndim=2] locs1
    cdef cnp.ndarray[double, ndim=2] BW0, BW1
    
    outfilename=dir_name + '/roi_hc.png'
    locs1=locs.astype(int)
    BW0=np.histogram2d(locs1[:,0], locs1[:,1],bins=[(roi[0,1]-roi[0,0])/pxl,(roi[1,1]-roi[1,0])/pxl], range=[[roi[0,0],roi[0,1]],[roi[1,0],roi[1,1]]])[0]
    BW0=np.rot90(BW0, k=1)
    BW0=np.flipud(BW0)
    BW1=BW0
    if fixed_ind==1:
        BW1=np.clip(BW1,0,fix_val)
        BW1=(np.divide(BW1,fix_val)*255)
    BW2 = Image.fromarray(np.uint8(BW1))
    draw =ImageDraw.Draw(BW2)
    for i in range (0,cnum-1):
        x= np.divide(cluster_ana[i,0]-roi[0,0],pxl)
        y= np.divide(cluster_ana[i,1]-roi[1,0],pxl)
        r= np.divide(cluster_ana[i,3],pxl)
        draw.ellipse((x-r,y-r,x+r,y+r), outline=255)
        draw.point((x,y), fill=255)
    BW2.save(outfilename)


# make hc-color images
def make_hc_color_Images(cnp.ndarray[double, ndim=2] locs, int cnum, cnp.ndarray[double, ndim=2] roi, int pxl, str dir_name):
    '''
        receives localization list, roi parameters and desired pxl size from communication layer. computes a 2d-histogram. transform histogram information into 8-bit gray scale image. saves image. returns raw histogram containing absolute number of localizations per bin.
        '''
    cdef int max
    cdef str outfilename
    cdef cnp.ndarray[double, ndim=1] Z
    cdef cnp.ndarray[double, ndim=2] BW0, BW1, BW2
    
    outfilename=dir_name+'/roi_hc_color.png'
    
    max = np.int(np.ceil(np.divide(cnum,255)))
    Z= np.ceil(np.divide(locs[:,3],max))
    BW0=np.histogram2d(locs[:,0], locs[:,1],bins=[(roi[0,1]-roi[0,0])/pxl,(roi[1,1]-roi[1,0])/pxl], range=[[roi[0,0],roi[0,1]],[roi[1,0],roi[1,1]]])[0]
    BW0=np.rot90(BW0, k=1)
    BW0=np.flipud(BW0)
    #BW0=np.clip(BW0,0,255)
    #BW1=np.clip(BW0, 1, 255)
    BW1=np.histogram2d(locs[:,0], locs[:,1],bins=[(roi[0,1]-roi[0,0])/pxl,(roi[1,1]-roi[1,0])/pxl], range=[[roi[0,0],roi[0,1]],[roi[1,0],roi[1,1]]],weights=Z)[0]
    BW1=np.rot90(BW1, k=1)
    BW1=np.flipud(BW1)
    BW2=np.divide(BW1, BW0)
    BW3 = Image.fromarray(np.uint8(BW2))
    BW3.save(outfilename)


# plot da stuff
def plot_dbscan(cnp.ndarray[double, ndim=2] hc_locs, int cnum, str dir_name):
    cdef int Min_size, Max_size
    cdef cnp.ndarray[double, ndim=1] x,y
    cdef str outfilename

    sizes=np.histogram(hc_locs[:,3], bins=cnum, range=(1,cnum), density=False)[0]
    Min_size = np.min(sizes)
    Max_size = np.max(sizes)
    x = np.arange(Min_size,Max_size,1,dtype='float')
    y = np.histogram(sizes, bins=Max_size-Min_size, range=(Min_size,Max_size), density=True)[0]
    outfilename = dir_name + '/DBSCAN.pdf'
    f, axarr = plt.subplots(1, sharex=False)
    axarr.plot(x, y, 'b')
    axarr.set_xlim([Min_size,Max_size])
    axarr.set_xlabel('cluster size [localizations]')
    axarr.set_ylabel('Intensity [a.u.]')
    plt.savefig(outfilename, format='pdf')
    plt.close()

def plot_optics(cnp.ndarray[double, ndim=2] hc_locs, float noise_ind, str dir_name):
    cdef str outfilename
    cdef int i
    cdef float noise, sum_y, new_eps
    cdef cnp.ndarray[double, ndim=1] x,y, z
    
    hc_locs = hc_locs[np.argsort(hc_locs[:, 2])]

    outfilename = dir_name + '/OPTICS_hist.pdf'
    x = np.arange(1,np.max(hc_locs[:,3]),1,dtype='float')
    y = np.histogram(hc_locs[:,3], bins=np.max(hc_locs[:,3]), range=(1,np.max(hc_locs[:,3])), density=True)[0]
    z = x-x
    noise = 1.0-(noise_ind/100.0)
    i=0
    sum_y = 0.0
    while sum_y < noise:
        new_eps = x[i]
        sum_y += y[i]
        i += 1
    f, axarr = plt.subplots(1, sharex=False)
    axarr.plot(x, y, 'b')
    axarr.axvline(new_eps, color='r')
    axarr.set_xlim([0,200])
    axarr.set_xlabel('reach distance [nm]')
    axarr.set_ylabel('Intensity [a.u.]')
    plt.savefig(outfilename, format='pdf')
    plt.close()

    outfilename = dir_name + '/OPTICS.pdf'
    z = hc_locs[:,3]-hc_locs[:,3]+new_eps
    f, axarr = plt.subplots(1, sharex=False)
    axarr.plot(hc_locs[:,2], hc_locs[:,3], 'b')
    axarr.plot(hc_locs[:,2], z, 'r')
    axarr.set_xlim([0,len(hc_locs[:,2])])
    axarr.set_ylim([1,np.max(hc_locs[:,3])])
    axarr.set_yscale('log')
    axarr.set_xlabel('cluster order [a.u.]')
    axarr.set_ylabel('reach distance [nm]')
    plt.savefig(outfilename, format='pdf')
    plt.close()

    return (new_eps)



class condence_locs:
    def __init__(self, locs, dir_name):
        
        self.locs = locs
        self.dir_name = dir_name
        self.append = '/condensed_locs_roi.txt'
        self.condensed_locs = np.zeros([2,4])
        self.cnum = int(np.max(self.locs[:,3]))
    
    def print_progress(self, progress):
        
        barLength = 10 # Modify this to change the length of the progress bar
        block = int(round(barLength*progress))
        text = "\rthe lama is condensing localizations: [{0}] {1}%".format( "#"*block + "-"*(barLength-block), int(progress*100)+1)
        sys.stdout.write(text)
        sys.stdout.flush()
    
    def count_blinks(self, cluster):
        
        
        length = np.shape(cluster)[0]
        blinks = np.zeros([length,1])
        n=0
        for i in range (1,length):
            if cluster[i,2]!=(cluster[i-1,2]+1):
                n+=1
            blinks[i]=n
        return blinks
    
    
    def condense_cluster(self, cluster, blinks):
        
        nBlinks = int(np.max(blinks)+1)
        cCluster = np.zeros([nBlinks,4])
        for i in range (0,nBlinks):
            idx = blinks[:,0]==i
            length = np.shape(cluster[idx])[0]
            cCluster[i,0] = np.sum(cluster[idx,0])/float(length)
            cCluster[i,1] = np.sum(cluster[idx,1])/float(length)
            cCluster[i,2] = np.min(cluster[idx,2])
            cCluster[i,3] = cluster[0,3]
        self.condensed_locs = np.append(self.condensed_locs, cCluster, axis=0)
    
    def sort_cluster(self):
        
        for i in range (1,self.cnum+1):
            self.print_progress(i/self.cnum)
            idx = self.locs[:,3]==i
            cluster_locs = np.zeros([np.shape(self.locs[idx,:])[0],4])
            cluster_locs = self.locs[idx,:]
            cluster_locs = cluster_locs[np.argsort(cluster_locs[:,2]),:]
            blinks = self.count_blinks(cluster_locs)
            self.condense_cluster(cluster_locs, blinks)
        self.print_progress(0.99)
        print('\n')
        self.condensed_locs = np.delete(self.condensed_locs,[0,1],axis=0)
    
    def save_cond_locs(self):
        
        hd = str('Condensed hc-file (LAMA format)\nnumber of cluster: %i\nx[nm]\ty[nm]\tt[frame]\cluster' %(self.cnum))
        outfilename = self.dir_name + self.append
        np.savetxt(outfilename, self.condensed_locs, fmt='%.5e', delimiter='   ', header = hd, comments ='# ')

def saveRawCluster(cluster, dir_name):
    
    outfilename = dir_name + '/batch_CA.txt'
    cnum = np.shape(cluster)[0]
    hd=str('Cluster analysis LAMA format\nnumber of cluster: %i \nx[nm]\ty[nm]\tsize[nm*nm]\tr[nm]\tI[a.u.]' %(cnum))
    np.savetxt(outfilename, cluster, fmt='%.5e', delimiter='   ', header = hd, comments='# ' )

class SMCounting:
    def __init__(self, cluster,cnum, roi, pType, p, dir_name):
        self.cluster = cluster
        self.roi = roi
        self.dir_name = dir_name
        self.pType = pType
        self.p = p
        self.n = 3.0
        self.cnum = cnum
        self.clustHist = np.zeros([int(self.roi[3,1]-self.roi[3,0]), 3])
        self.clustHist[:,0] = np.arange(self.roi[3,0], self.roi[3,1], 1, dtype='int')
    

    def roi_cluster(self):
        idX1=self.cluster[:,0]>=self.roi[0,0]
        idX2=self.cluster[:,0]<=self.roi[0,1]
        idY1=self.cluster[:,1]>=self.roi[1,0]
        idY2=self.cluster[:,1]<=self.roi[1,1]
        idR1=self.cluster[:,3]>=self.roi[2,0]
        idR2=self.cluster[:,3]<=self.roi[2,1]
        idI1=self.cluster[:,4]>=self.roi[3,0]
        idI2=self.cluster[:,4]<=self.roi[3,1]
        self.cluster = self.cluster[(idX1&idX2&idY1&idY2&idR1&idR2&idI1&idI2),:]
        self.cnum = np.shape(self.cluster)[0]

    def saveRoiCluster(self):
        append = '/ROI_hc_cluster.txt'
        hd = str('hc-roi-file (LAMA format)\nnumber of cluster: %i \nx[nm]\ty[nm]\tsize[nm*nm]\tr[nm]\tI[a.u.]' %(self.cnum))
        outfilename = self.dir_name + append
        np.savetxt(outfilename, self.cluster, fmt='%.5e', delimiter='   ', header = hd, comments ='# ')

    def blinkHistogram(self):
        #append = '/ROI_hc_histogram.txt'
        self.clustHist[:,1] = np.histogram(self.cluster[:,4], bins=len(self.clustHist[:,0]), range=(self.roi[3,0],self.roi[3,1]), density=True)[0]
        #hd = str('cluster intensity histogram (LAMA format)\nnumber of localizations/cluster\t relative fraquency [a.u.]')
        #outfilename = self.dir_name + append
        #np.savetxt(outfilename, self.clustHist, fmt='%.5e', delimiter='   ', header = hd, comments ='# ')
    
    def Nbinom(self, x, n, p):
        numer = n+x-1
        denom = n-1
        y=(special.gamma(numer+1)/(special.gamma(numer-denom+1)*special.gamma(denom+1)))*(p**(n))*((1-p)**(x))
        return y

    def estimateNP(self, x,y,n_guess,p_guess):
        p0=np.array([n_guess, p_guess])
        popt, pcov = curve_fit(self.Nbinom, x, y, p0)
        self.n = popt[0]
        self.p = popt[1]

    def estimateN(self, x,y,n_guess,p_guess):
        f=lambda x,n: self.Nbinom(x,n,p_guess)
        popt, pcov = curve_fit(f,x,y,n_guess)
        self.n = popt[0]
        self.p = p_guess

    def fitNegBin(self):
        append = '/ROI_hc_histogram.pdf'
        if self.pType == 0:
            self.estimateNP(self.clustHist[:,0], self.clustHist[:,1],self.n,self.p)
        else:
            self.estimateN(self.clustHist[:,0], self.clustHist[:,1],self.n,self.p)
        self.clustHist[:,2]=self.Nbinom(self.clustHist[:,0], self.n, self.p)

    def save_counting(self):
        appendTXT = '/ROI_hc_histogram.txt'
        hd = str('cluster intensity histogram (LAMA format)\nn= %.3f\tp= %.3f\nnumber of localizations/cluster\t relative fraquency [a.u.]\t fit[a.u.]'%(self.n, self.p))
        outfilenameTXT = self.dir_name + appendTXT
        np.savetxt(outfilenameTXT, self.clustHist, fmt='%.5e', delimiter='   ', header = hd, comments ='# ')
        appendPDF = '/ROI_hc_histogram.pdf'
        outfilenamePDF = self.dir_name + appendPDF
        f, axarr = plt.subplots(1, sharex=False)
        axarr.bar(self.clustHist[:,0], self.clustHist[:,1], color='gray', edgecolor='black',width=1, align='center')
        axarr.plot(self.clustHist[:,0], self.clustHist[:,2], 'b')
        axarr.set_xlim([self.roi[3,0]-1,self.roi[3,1]])
        axarr.set_xlabel('cluster intensity [localizations]')
        axarr.set_ylabel('frequency [a.u.]')
        plt.savefig(outfilenamePDF, format='pdf')
        plt.close()