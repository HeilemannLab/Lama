##
##  lama_communicate.pyx
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
import lama_app as la

def statement():
    '''
        Communicates a message of success to the terminal.
    '''
    print ('the lama got the job done and is going to rest!')

def analyze_roi_int(cnp.ndarray[double, ndim=2] roi, int pxl, str file, int conv_ind, int acc, int fixed_ind, int fix_val, int cbc_ind, int mca_ind, float thr, float min_r, float max_r, str append):
    '''
        receives necessary parameters from communication layer and calls algorithms to compute a super-resolved image from coordinate list within pre-defined roi from application layer. Checks if cluster analysis is requested and calls cluster analysis.
    '''
    cdef str dir_name
    cdef float sigma
    cdef cnp.ndarray[double, ndim=2] locs, roi_locs, BW0, BW1
    cdef cnp.ndarray mask
    
    locs=la.read_locs_rs(file)
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    roi_locs=la.create_roi(locs,roi,dir_name)
    la.write_offset(roi,pxl,dir_name)
    if cbc_ind==0:
        BW0=la.make_Image(roi_locs,roi,pxl,fixed_ind,fix_val,dir_name)
    else:
        BW0=la.make_CBC_image(roi_locs,roi,pxl,dir_name)
    if conv_ind==1:
        sigma=float(acc/pxl)
        BW1= la.convolve_image(BW0,sigma,fixed_ind,fix_val,dir_name)
    if mca_ind==1 and conv_ind==1:
        mask=la.make_binary(BW1, thr)
        la.imb_ana(mask,BW0,roi,thr,min_r,max_r,pxl, dir_name)
    elif mca_ind==1 and conv_ind==0:
        mask=la.make_binary(BW0, thr)
        la.imb_ana(mask,BW0,roi,thr,min_r,max_r,pxl, dir_name)


def calc_Thompson(str file, cnp.ndarray[double, ndim=2] roi, float con_fac, float pxl_size, float noise, float sigma, str append):
    '''
        receives necessary parameters from communication layer and calls algorithms to compute the localization precision theoretically after the method of Thompson et al. and Mortensen et al. based on Setup conditions and fluorescent probe photon emission.
    '''
    cdef str dir_name
    cdef cnp.ndarray[double, ndim=2] locs, roi_locs
    
    locs=la.read_locs_rs(file)
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    roi_locs=la.create_roi(locs,roi,dir_name)
    la.Thompson(roi_locs,con_fac,pxl_size,noise,sigma,dir_name)

def calc_NeNA(str file, cnp.ndarray[double, ndim=2] roi, str append):
    '''
        receives necessary parameters from communication layer and calls algorithms to compute the localization precision experimentally after the method of Endesfelder et al. based on the mean localization error determined from multiple localizations of fluorescent probes.
    '''
    cdef str dir_name
    cdef cnp.ndarray[double, ndim=2] locs, roi_locs
    
    locs=la.read_locs_rs(file)
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    roi_locs=la.create_roi(locs,roi,dir_name)
    la.NeNA(roi_locs,dir_name)

def calc_ripley_rois(str file):
    '''
        receives necessary parameters from communication layer to load multiple rois for Ripleys K-function analysis from external file and returns roi information.
    '''
    cdef cnp.ndarray[double, ndim=2] rois
    
    rois = np.dot(la.read_locs(file),1000)
    return rois

def calc_ripley(str file, cnp.ndarray[double, ndim=2] rois, float edge, float radius, int inc_num, int pxl, int fixed_ind, int fix_val,str append):
    '''
        receives necessary parameters from communication layer and calls algorithms load locs-files, to filter locs spatially and chronically by given roi parameters, to compute an image from the filtered coordinates, and calculate Ripleys K-function from the filtered coordinates.
    '''
    cdef str dir_name, roi_append, roi_dir_name
    cdef int i
    cdef cnp.ndarray[double, ndim=2] locs, roi, roi_locs, roi_box
    
    locs=la.read_locs_rs(file)
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    for i in range (0, len(rois[:,0])):
        roi_append='/roi_' + str(i)
        roi_dir_name=la.make_dir(dir_name,roi_append)
        roi=np.zeros([2,2])
        roi[0,0]=rois[i,0]
        roi[0,1]=rois[i,0]+edge
        roi[1,0]=rois[i,1]
        roi[1,1]=rois[i,1]+edge
        roi_locs=la.create_roi_ripley(locs,roi,roi_dir_name)
        la.make_Image(roi_locs,roi,pxl,fixed_ind,fix_val,roi_dir_name)
        roi_box=la.RipleysEdge(roi_locs,edge,roi_dir_name)
        la.RipleysK(roi_locs,roi_box,inc_num,radius,edge,roi_dir_name)

def callBeadDetection(list files, cnp.ndarray[double, ndim=2] roi, int r_min, int r_max, int on_min, int on_max):
    '''
        Input Data: Cluster MCA Format
        Sort for roi (spatial)
        Sort for Beads
        Returns Bead Data MCA Format
    '''
    cdef str dir_name, append
    cdef int i
    cdef cnp.ndarray[double, ndim=2] fas

    append = str('_roi.txt')
    for i in range (0, len(files)):
        fas=la.read_locs(files[i])
        file_name=files[i][0:(len(files[i])-4)]+append
        fas=la.createRoiBead(fas,roi,r_min, r_max, on_min, on_max, file_name)


def calc_registration(list files, int disp, int reg_ind):
    '''
        receives necessary parameters from communication layer and calls algorithms to compute fiducial marker based coordinate registration. Loads locs, filters locs by given roi parameters, calculates an image of the roi, detects fiducial markers, repeats routine for 2nd channel, links fiducial markers detected in both images. If more than two fiducial marker pairs could be linked calls registration by affine matrix. Else it calls registration by linear registration.
    '''
    cdef str dir_name
    cdef int i
    cdef cnp.ndarray[double, ndim=2] locs_temp, buds_temp, sort_buds_temp, locs, BW1, buds, sort_buds, fids01, fids02
    
    for i in range (0,len(files),3):
        fids01 = la.read_locs(files[i])
        fids02 = la.read_locs(files[i+1])
        fids01,fids02=la.link_fuducials(fids01,fids02,disp)
        la.write_sort_beads(fids01,fids02,files[i],files[i+1])
        la.plot_buds(fids01,fids02,files[i])
        locs=la.read_locs_rs(files[i+2])
        if reg_ind==1:
            print('the lama is calculating an affine registration')
            la.register_channels(locs,fids02,fids01,files[i+2])
        else:
            print('the lama is calculating a linear registration')
            la.register_channels_trans(locs,fids02,fids01,files[i+2])



def cbc_coloc(str file_A, str file_B, cnp.ndarray[double, ndim=2] roi, int r_max, int inc_num, float wf, str append):
    '''
        receives necessary parameters from gui layer and calls algorithms to compute a coordinate based colocalization from two channels. Loads channel A, filters localizations from channel A spatially and chronologically by given roi parameters. Loads channel B, filters localizations from channel B spatially and chronologically by given roi parameters. Characterizes the clustering behavior of every fluorescent probe localization from both channels by a CBC value.
    '''
    cdef str dir_name, name_int_A, name_int_B, name_cbc_A, name_cbc_B
    cdef cnp.ndarray[double, ndim=2] locs_A, roi_locs_A, locs_B, roi_locs_B, cbc_origin , cbc_partner
    
    dir_name=la.make_dir(file_A[0:(len(file_A)-4)],append)
    name_int_A='/origin_int_roi.txt'
    locs_A=la.read_locs_rs(file_A)
    roi_locs_A=la.create_cbc_roi(locs_A,roi,dir_name,name_int_A)
    name_int_B='/partner_int_roi.txt'
    locs_B=la.read_locs_rs(file_B)
    roi_locs_B=la.create_cbc_roi(locs_B,roi,dir_name, name_int_B)
    name_cbc_A='/origin_cbc_roi.txt'
    cbc_origin=la.cbc(roi_locs_A,roi_locs_B,r_max,inc_num,wf)
    roi_locs_A[:,3]=cbc_origin[:,0]
    la.create_cbc_roi(roi_locs_A,roi,dir_name,name_cbc_A)
    name_cbc_B='/partner_cbc_roi.txt'
    cbc_partner=la.cbc(roi_locs_B,roi_locs_A,r_max,inc_num,wf)
    roi_locs_B[:,3]=cbc_partner[:,0]
    la.create_cbc_roi(roi_locs_B,roi,dir_name,name_cbc_B)

def cbc_cluster(str file, cnp.ndarray roi, int r_max, int inc_num, float wf, str append):
    '''
       receives necessary parameters from gui layer and calls algorithms to compute a coordinate based colocalization from a single channel. Loads localizations, filters localizations spatially and chronologically by given roi parameters. Splits localizations by frame number (odd and even) and creates two channels. Characterizes the clustering behavior of every fluorescent probe localization from both channels by a CBC value.
    '''
    cdef str dir_name,name_cbc
    cdef cnp.ndarray[double, ndim=2] locs, roi_locs, locs_A, locs_B, cbc_locs, cbc_origin, cbc_partner
    
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    locs=la.read_locs_rs(file)
    roi_locs=la.create_roi(locs,roi,dir_name)
    locs_A,locs_B=la.split_cbc_locs(roi_locs)
    cbc_origin=la.cbc(locs_A,locs_B,r_max,inc_num,wf)
    locs_A[:,3]=cbc_origin[:,0]
    cbc_partner=la.cbc(locs_B,locs_A,r_max,inc_num,wf)
    locs_B[:,3]=cbc_partner[:,0]
    cbc_locs=la.combine_split_cbc_locs(locs_A,locs_B)
    name_cbc='/cluster_CBC.txt'
    la.create_cbc_roi(cbc_locs,roi,dir_name,name_cbc)

def DBSCAN_based_clustering(str file, cnp.ndarray[double, ndim=2] roi, float eps, int pmin, int pxl, str append):
    '''
        receives necessary parameters from gui layer and calls algorithms to compute a hierarchical clustering for a single channel. Loads localizations, filters localizations spatially and chronologically by given roi parameters. Calculates DBSCAN for roi. Saves ROI with clustering parameters in 2nd line. 0 characterizes noise.
    '''
    cdef str dir_name, hd, outfilename
    cdef int dim, cnum
    cdef cnp.ndarray[double, ndim=2] locs, roi_locs, cluster_ana
    
    dim = 2
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    locs=la.read_locs_rs(file)
    roi_locs=la.create_roi(locs,roi,dir_name)
    hc=la.hierarchical_cluster(roi_locs,eps,pmin,dim)
    hc.DBSCAN()
    hc_locs = hc.locs
    hc_locs[:,3] = hc.cluster
    outfilename_01 = dir_name + '/DBSCAN.txt'
    hd_01 = 'DBSCAN (LAMA format)\neps=%.2f[nm], pmin=%i, cluster=%i\nx[nm]\ty[nm]\tt[frame]\tcluster' %(float(hc.eps), int(hc.pmin), int(hc.cnum-1))
    np.savetxt(outfilename_01, hc_locs, fmt='%.5e', delimiter='   ', header = hd_01, comments='# ')
    #plot pdf with cluster histogram
    la.plot_dbscan(hc_locs,hc.cnum,dir_name)
    



def OPTICS_based_clustering(str file, cnp.ndarray roi, float eps, float noise_ind, int pmin, int pxl, str append):
    '''
        receives necessary parameters from gui layer and calls algorithms to compute a hierarchical clustering for a single channel. Loads localizations, filters localizations spatially and chronologically by given roi parameters. Calculates DBSCAN for roi. Saves ROI with clustering parameters in 2nd line. 0 characterizes noise.
        '''
    cdef str dir_name, hd_01, outfilename_01, hd_02, outfilename_02
    cdef int dim
    cdef float new_eps
    cdef cnp.ndarray[double, ndim=2] locs, roi_locs, hc_locs, DB_locs
    
    dim = 2
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    locs=la.read_locs_rs(file)
    roi_locs=la.create_roi(locs,roi,dir_name)
    hc=la.hierarchical_cluster(roi_locs,eps,pmin,dim)
    hc.OPTICS()
    #write new_eps to file
    hc_locs = np.zeros([np.shape(hc.order)[0],4])
    hc_locs[:,0:1] = hc.locs[:,0:1]
    hc_locs[:,2] = hc.order
    hc_locs[:,3] = hc.reachDist
    outfilename_01 = dir_name + '/OPTICS.txt'
    hd_01 = 'DBSCAN (LAMA format)\npmin=%i\nx[nm]\ty[nm]\tt[frame]\torder[a.u.]\treach distance[nm]' %(int(hc.pmin))
    np.savetxt(outfilename_01, hc_locs, fmt='%.5e', delimiter='   ', header = hd_01, comments='# ')
    #plot to extract eps
    new_eps = la.plot_optics(hc_locs, noise_ind, dir_name)
    hc.eps = new_eps
    hc.extract_DBSCAN()
    DB_locs = hc.locs
    DB_locs[:,3] = hc.cluster
    outfilename_02 = dir_name + '/OPTICS_DBSCAN.txt'
    hd_02 = 'DBSCAN (LAMA format)\neps=%.2f[nm], pmin=%i, cluster=%i\nx[nm]\ty[nm]\tt[frame]\tcluster' %(float(hc.eps), int(hc.pmin), int(hc.cnum-1))
    np.savetxt(outfilename_02, DB_locs, fmt='%.5e', delimiter='   ', header = hd_02, comments='# ')
    #plot pdf with cluster histogram
    la.plot_dbscan(DB_locs,hc.cnum,dir_name)

def hierarchical_cluster_Analysis(file, roi, pxl, eps, pmin, cond, append, hcaType):
    
    locs=la.read_locs_rs(file)
    dir_name=la.make_dir(file[0:(len(file)-4)],append)
    roi_locs=la.create_roi(locs,roi,dir_name)
    #analyze hc_locs
    hc_ana=la.hcAnalysis(roi_locs, eps, pmin)
    hc_ana.analyze_hc_cluster()
    # extract cluster
    cluster_ana = hc_ana.cluster_ana
    # condense locs
    if cond==1:
        cluster_ana[:,4] = hc_ana.blinks[:,0]
    # save cluster
    outfilename = dir_name + hcaType
    hd=str('hierarchical cluster analysis (LAMA format)\ncluster=%i\nx[nm]\ty[nm]\tsize[nm*nm]\tr[nm]\tI[a.u.]' %(int(hc_ana.cnum-1)))
    np.savetxt(outfilename, cluster_ana, fmt='%.5e', delimiter='   ', header = hd, comments='# ' )
    # plot cluster
    fixed_ind = 1
    fix_val = 255
    la.make_hc_Images(roi_locs, cluster_ana, hc_ana.cnum, roi, pxl, fixed_ind, fix_val, dir_name)
    la.make_hc_color_Images(roi_locs, hc_ana.cnum, roi, pxl, dir_name)


def count_emitters(filename, files, roi, rMin, rMax, iMin, iMax, pType, p, eps, pmin, append):
    cluster = np.zeros([2,5])
    for i in range (0,len(files)):
        cluster=np.append(cluster, la.read_locs(files[i]), axis=0)
    cluster = np.delete(cluster, [0,1], 0)
    # create directory
    dir_name=la.make_dir(filename[0:(len(filename)-4)],append)
    #save raw batch
    la.saveRawCluster(cluster, dir_name)
    #analyze Cluster by SMCounting
    #roi = np.dot(args.roi,1000)
    #roi = np.append(roi, [[args.Rmin,args.Rmax],[args.Imin,args.Imax]], axis=0)
    #print (roi)
    roi[2,:] = [rMin,rMax]
    roi[3,:] = [iMin,iMax]
    #print (roi)
    #create cluster counter
    cnum = np.shape(cluster)[0]
    smcount = la.SMCounting(cluster,cnum, roi, pType, p, dir_name)
    smcount.roi_cluster()
    smcount.saveRoiCluster()
    #create histogram
    smcount.blinkHistogram()
    smcount.fitNegBin()
    smcount.save_counting()


def load_settings(str name):
    '''
        receives necessary parameters from gui layer and reads in lama parameters of a previous LAMA session. Passes them back to gui layer and updates the layer.
    '''
    cdef int data_type
    cdef cnp.ndarray[double, ndim=2] settings, roi, rest_para
    
    name=name.rstrip('\n')
    print ('the lama is pronking through '+ name)
    settings=np.zeros([53,1])
    data_type,roi,rest_para=la.read_import_file(name)
    settings[0]=data_type
    settings[1:11]=np.array([[roi[0,0]],[roi[0,1]],
                             [roi[1,0]],[roi[1,1]],
                             [roi[2,0]],[roi[2,1]],
                             [roi[3,0]],[roi[3,1]],
                             [roi[4,0]],[roi[4,1]]])
    settings[11:]=rest_para
    return settings
