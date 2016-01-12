##
##  mainwindow.py
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

from PyQt4 import uic, QtGui
from PyQt4.QtCore import *
import lama_app as la
import lama_communicate as lc
import numpy as np

( Ui_MainWindow, QMainWindow ) = uic.loadUiType( 'mainwindow.ui' )
def define_file_type(self):
        files=[]
        if self.ui.radioButton_01.isChecked():
            files=[self.filename]
            self.file_type=0
        else:
            files=la.read_filenames(self.filename)
            self.file_type=1
        return files

def define_roi(self):
    '''
    defines the edges of the roi
    '''
    roi=np.zeros([4,2])
    roi[0,0]=float(self.ui.lineEdit_02.text())*1000
    roi[0,1]=float(self.ui.lineEdit_03.text())*1000
    roi[1,0]=float(self.ui.lineEdit_04.text())*1000
    roi[1,1]=float(self.ui.lineEdit_05.text())*1000
    roi[2,0]=float(self.ui.lineEdit_06.text())
    roi[2,1]=float(self.ui.lineEdit_07.text())
    roi[3,0]=float(self.ui.lineEdit_08.text())
    roi[3,1]=float(self.ui.lineEdit_09.text())
    return roi

def define_cbc_roi(self):
    '''
        defines the edges of the roi
        '''
    roi=np.zeros([4,2])
    roi[0,0]=float(self.ui.lineEdit_02.text())*1000
    roi[0,1]=float(self.ui.lineEdit_03.text())*1000
    roi[1,0]=float(self.ui.lineEdit_04.text())*1000
    roi[1,1]=float(self.ui.lineEdit_05.text())*1000
    roi[2,0]=float(self.ui.lineEdit_06.text())
    roi[2,1]=float(self.ui.lineEdit_07.text())
    roi[3,0]=float(self.ui.lineEdit_10.text())
    roi[3,1]=float(self.ui.lineEdit_11.text())
    return roi

    
class MainWindow ( QMainWindow ):
    """MainWindow inherits QMainWindow"""

    def __init__ ( self, parent = None ):
        QMainWindow.__init__( self, parent )
        self.toolbox = QtGui.QToolBox(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi( self )
        self.toolbox.resize(300,500)
        
    # Main Window actions
    # get filename
        self.ui.lineEdit_01.setDragEnabled(True)
        self.ui.lineEdit_01.setAcceptDrops(True)
        self.ui.pushButton_01.clicked.connect(self.browse_input_path)
        self.ui.lineEdit_01.textChanged.connect(self.drop_input_path)
    # create image and MCA 
        self.ui.pushButton_02.clicked.connect(self.visualize)
    # calculate Thompson precision
        self.ui.pushButton_03.clicked.connect(self.Thompson_acc)
    # calculate NeNA precision
        self.ui.pushButton_04.clicked.connect(self.NeNA_acc)
    # Ripley's K-function
        self.ui.pushButton_05.clicked.connect(self.ripley)
    # Register channels
        self.ui.pushButton_06.clicked.connect(self.detectBeads)
        self.ui.pushButton_07.clicked.connect(self.register)
    # Calculate CBC
        self.ui.pushButton_08.clicked.connect(self.cbc_procedure)
    # Calculate DBSCAN
        self.ui.pushButton_09.clicked.connect(self.DBSCAN_procedure)
        self.ui.pushButton_10.clicked.connect(self.analyze_hc_cluster)
    # Calculate Stoichiometry
        self.ui.pushButton_11.clicked.connect(self.smEmiter_procedure)
    # Load Settings
        self.ui.lineEdit_45.setDragEnabled(True)
        self.ui.lineEdit_45.setAcceptDrops(True)
        self.ui.pushButton_12.clicked.connect(self.browse_import_path)
        self.ui.lineEdit_45.textChanged.connect(self.drop_import_path)
        self.ui.pushButton_13.clicked.connect(self.import_settings)
    # Save Settings
        self.ui.lineEdit_46.setDragEnabled(True)
        self.ui.lineEdit_46.setAcceptDrops(True)
        self.ui.pushButton_14.clicked.connect(self.browse_save_path)
        self.ui.lineEdit_46.textChanged.connect(self.drop_save_path)
        self.ui.pushButton_15.clicked.connect(self.save_settings)
        
        

    def __del__ ( self ):
        self.ui = None

        
 # Main Window funcions       
    def browse_input_path(self):
        '''
            get path name from pyQT
            Update lineEdit
        '''
        self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
        self.ui.lineEdit_01.setText(self.filename)
    
    def drop_input_path(self):
        '''
            update lineEdit_01 for new filename
        '''
        self.filename=str(self.ui.lineEdit_01.text())
        self.ui.lineEdit_01.setText(self.filename)

    def visualize(self):
        pxl=int(self.ui.lineEdit_15.text())
        acc=int(self.ui.lineEdit_16.text())
        if self.ui.checkBox_01.isChecked():
            conv_ind=1
        else:
            conv_ind=0
        if self.ui.checkBox_02.isChecked():
            fixed_ind=1
        else:
            fixed_ind=0
        fix_val=int(self.ui.lineEdit_17.text())
        if self.ui.checkBox_03.isChecked():
            mca_ind=1
        else:
            mca_ind=0
        if self.ui.radioButton_03.isChecked():
            cbc_ind=0
            roi=define_roi(self)
        else:
            cbc_ind=1
            roi=define_cbc_roi(self)
        thr=float(self.ui.lineEdit_18.text())
        min_r=float(self.ui.lineEdit_19.text())
        max_r=float(self.ui.lineEdit_20.text())
        files=define_file_type(self)        
        append='_statMIA'
        for i in range (0,len(files)):
            lc.analyze_roi_int(roi,pxl,files[i],conv_ind,acc,fixed_ind,fix_val,cbc_ind,mca_ind,thr,min_r,max_r,append)
        lc.statement()

    def Thompson_acc(self):
        append='_statMIA'
        files=define_file_type(self)
        con_fac=float(self.ui.lineEdit_12.text())
        pxl_size=float(self.ui.lineEdit_21.text())
        noise=float(self.ui.lineEdit_23.text())
        sigma=float(self.ui.lineEdit_13.text())
        roi=define_roi(self)
        if self.ui.radioButton_01.isChecked():
            lc.calc_Thompson(files[0],roi,con_fac,pxl_size,noise,sigma,append)
        else:
            for i in range (0,len(files)):
                lc.calc_Thompson(files[i],roi,con_fac,pxl_size,noise,sigma,append)
        lc.statement()

    def NeNA_acc(self):
        append='_statMIA'
        files=define_file_type(self)
        roi=define_roi(self)
        if self.ui.radioButton_01.isChecked():
            lc.calc_NeNA(files[0],roi,append)
        else:
            for i in range (0,len(files)):
                lc.calc_NeNA(files[i],roi,append)
        lc.statement()
    
    def ripley(self):
        append='_statMIA'
        files=define_file_type(self)
        pxl=int(self.ui.lineEdit_15.text())
        if self.ui.checkBox_02.isChecked():
            fixed_ind=1
        else:
            fixed_ind=0
        fix_val=int(self.ui.lineEdit_17.text())
        edge=float(self.ui.lineEdit_24.text())*1000
        radius=float(self.ui.lineEdit_25.text())*1000
        inc_num=int(self.ui.lineEdit_26.text())
        if self.file_type==0:
            rois=roi=np.zeros([1,2])
            rois[0,0]=float(self.ui.lineEdit_02.text())*1000
            rois[0,1]=float(self.ui.lineEdit_04.text())*1000
            lc.calc_ripley(files[0],rois,edge,radius,inc_num,pxl,fixed_ind,fix_val,append)
        else:
            for i in range (0,len(files),2):
                rois=lc.calc_ripley_rois(files[i+1])
                lc.calc_ripley(files[i],rois,edge,radius,inc_num,pxl,fixed_ind,fix_val,append)
        lc.statement()

    def detectBeads(self):
        files=define_file_type(self)
        roi=define_roi(self)
        r_min=int(self.ui.lineEdit_28.text())
        r_max=int(self.ui.lineEdit_29.text())
        on_min=int(self.ui.lineEdit_30.text())
        on_max=int(self.ui.lineEdit_31.text())
        lc.callBeadDetection(files, roi, r_min, r_max, on_min, on_max)
        lc.statement()

    def register(self):
        if self.ui.radioButton_01.isChecked():
            print ('the lama can only operate registrations with multiple localization files')
        else:
            files=define_file_type(self)
            pxl=int(self.ui.lineEdit_15.text())
            if self.ui.checkBox_02.isChecked():
                fixed_ind=1
            else:
                fixed_ind=0
            fix_val=int(self.ui.lineEdit_17.text())
            roi=define_roi(self)
            disp=int(self.ui.lineEdit_32.text())
            if self.ui.radioButton_05.isChecked():
                reg_ind=1
            else:
                reg_ind=0
            if (len(files)%3)!=0:
                print('Wrong number of files! The lama always needs 3 files to calculate a registration: A bead file from the first chanel (MCA format), a bead file from the second channel (MCA format), and a localization file from the second channel (MALK format).')
            else:
                lc.calc_registration(files,disp,reg_ind)
            lc.statement()
        
    def cbc_procedure(self):
        append='_statMIA'
        files=define_file_type(self)
        r_max=int(self.ui.lineEdit_33.text())
        inc_num=int(self.ui.lineEdit_34.text())
        wf=float(self.ui.lineEdit_35.text())
        roi=define_roi(self)
        if self.ui.radioButton_01.isChecked() and self.ui.radioButton_07.isChecked():
            print ('the lama can only operate cbc analysis with multiple localization files')
        
        elif self.ui.radioButton_01.isChecked():
            lc.cbc_cluster(files[0],roi,r_max,inc_num,wf,append)
        else:
            if self.ui.radioButton_07.isChecked():
                for i in range (0,len(files),2):
                    print ('the lama calculates colocalization maps via CBC')
                    lc.cbc_coloc(files[i],files[i+1],roi,r_max,inc_num,wf,append)
            else:
                for i in range (0,len(files)):
                    print ('the lama calculates cluster maps via CBC')
                    lc.cbc_cluster(files[i],roi,r_max,inc_num,wf,append)
        lc.statement()
    
    def DBSCAN_procedure(self):
        append='_statMIA'
        files=define_file_type(self)
        roi=define_roi(self)
        eps=float(self.ui.lineEdit_36.text())
        pmin=int(self.ui.lineEdit_37.text())
        pxl=int(self.ui.lineEdit_15.text())
        if self.ui.radioButton_10.isChecked():
            noise_ind=float(self.ui.lineEdit_38.text())
            for i in range (0,len(files)):
                lc.OPTICS_based_clustering(files[i], roi, eps, noise_ind, pmin, pxl, append)
        else:
            for i in range (0,len(files)):
                lc.DBSCAN_based_clustering(files[i],roi,eps,pmin, pxl, append)
        lc.statement()

    def analyze_hc_cluster(self):
        append = '_statMIA'
        files=define_file_type(self)
        roi=define_roi(self)
        eps=float(self.ui.lineEdit_36.text())
        pmin=int(self.ui.lineEdit_37.text())
        pxl=int(self.ui.lineEdit_15.text())
        cond=0
        if self.ui.checkBox_04.isChecked():
            cond=1
        for i in range (0, len(files)):
            print ('condense locs')
            hcaType = '/hca_analysis.txt'
            lc.hierarchical_cluster_Analysis(files[i], roi, pxl, eps, pmin, cond, append, hcaType)
        '''
        if self.ui.checkBox_04.isChecked():
            for i in range (0, len(files)):
                print ('condense locs')
                hcaType = '/hca_condensed_roi.txt'
                lc.hierarchical_cluster_Analysis_condensed(files[i], roi, pxl, eps, pmin, append, hcaType)
        else:
            for i in range (0,len(files)):
                hcaType = '/hca_roi.txt'
                lc.hierarchical_cluster_Analysis(files[i], roi, pxl, eps, pmin, append, hcaType)
        '''
        lc.statement()

    def smEmiter_procedure(self):
        append = '_statMIA'
        files=define_file_type(self)
        roi=define_roi(self)
        rMin = int(self.ui.lineEdit_41.text())
        rMax = int(self.ui.lineEdit_42.text())
        iMin = int(self.ui.lineEdit_43.text())
        iMax = int(self.ui.lineEdit_44.text())
        if self.ui.checkBox_05.isChecked():
            pType = 1
        else:
            pType = 0
        p = float(self.ui.lineEdit_45.text())
        eps=float(self.ui.lineEdit_36.text())
        pmin=int(self.ui.lineEdit_37.text())
        lc.count_emitters(self.filename, files, roi, rMin, rMax, iMin, iMax, pType, p, eps, pmin, append)
        lc.statement()

    def browse_import_path(self):
        '''
            get path name from pyQT
            Update lineEdit
        '''
        self.import_filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
        self.ui.lineEdit_45.setText(self.import_filename)
    
    def drop_import_path(self):
        '''
            update lineEdit_45 for new filename
        '''
        self.import_filename=str(self.ui.lineEdit_45.text())
        self.ui.lineEdit_45.setText(self.import_filename)
    
    def import_settings(self):
        settings=lc.load_settings(self.import_filename)
        # Main / Input
        if settings[0,0]==1.0:
            self.ui.radioButton_01.setChecked(True)
        else:
            self.ui.radioButton_02.setChecked(True)
        # Main / ROI
        self.ui.lineEdit_02.setText(str(settings[1,0]))
        self.ui.lineEdit_03.setText(str(settings[2,0]))
        self.ui.lineEdit_04.setText(str(settings[3,0]))
        self.ui.lineEdit_05.setText(str(settings[4,0]))
        self.ui.lineEdit_06.setText(str(settings[5,0]))
        self.ui.lineEdit_07.setText(str(settings[6,0]))
        self.ui.lineEdit_08.setText(str(settings[7,0]))
        self.ui.lineEdit_09.setText(str(settings[8,0]))
        self.ui.lineEdit_10.setText(str(settings[9,0]))
        self.ui.lineEdit_11.setText(str(settings[10,0]))
        # Main / Setup
        self.ui.lineEdit_12.setText(str(settings[11,0]))
        self.ui.lineEdit_13.setText(str(settings[12,0]))
        self.ui.lineEdit_14.setText(str(settings[13,0]))
        # Visualize / Iamge
        self.ui.lineEdit_15.setText(str(int(settings[14,0])))
        self.ui.lineEdit_16.setText(str(int(settings[15,0])))
        self.ui.lineEdit_17.setText(str(int(settings[16,0])))
        if settings[17,0]==1.0:
            self.ui.radioButton_03.setChecked(True)
        else:
            self.ui.radioButton_04.setChecked(True)
        if settings[18,0]==1.0:
            self.ui.checkBox_01.setChecked(True)
        else:
            self.ui.checkBox_01.setChecked(False)
        if settings[19,0]==1.0:
            self.ui.checkBox_02.setChecked(True)
        else:
            self.ui.checkBox_02.setChecked(False)
        # Visualize / MCA
        if settings[20,0]==1.0:
            self.ui.checkBox_03.setChecked(True)
        else:
            self.ui.checkBox_03.setChecked(False)
        self.ui.lineEdit_18.setText(str(int(settings[21,0])))
        self.ui.lineEdit_19.setText(str(int(settings[22,0])))
        self.ui.lineEdit_20.setText(str(int(settings[23,0])))
        # Precision / Theoretical
        self.ui.lineEdit_21.setText(str(settings[24,0]))
        self.ui.lineEdit_22.setText(str(settings[25,0]))
        self.ui.lineEdit_23.setText(str(settings[26,0]))
        # Ripley's K-function
        self.ui.lineEdit_24.setText(str(settings[27,0]))
        self.ui.lineEdit_25.setText(str(settings[28,0]))
        self.ui.lineEdit_26.setText(str(int(settings[29,0])))
        # Register / Bead Detection
        self.ui.lineEdit_28.setText(str(int(settings[30,0])))
        self.ui.lineEdit_29.setText(str(int(settings[31,0])))
        self.ui.lineEdit_30.setText(str(int(settings[32,0])))
        self.ui.lineEdit_31.setText(str(int(settings[33,0])))
        # Register / Registration
        self.ui.lineEdit_32.setText(str(int(settings[34,0])))
        if settings[35,0]==1.0:
            self.ui.radioButton_05.setChecked(True)
        else:
            self.ui.radioButton_06.setChecked(True)
        # CBC
        self.ui.lineEdit_33.setText(str(int(settings[36,0])))
        self.ui.lineEdit_34.setText(str(int(settings[37,0])))
        self.ui.lineEdit_35.setText(str(settings[38,0]))
        if settings[39,0]==1.0:
            self.ui.radioButton_07.setChecked(True)
        else:
            self.ui.radioButton_08.setChecked(True)
        # Hierarchical Clustering / sort
        if settings[40,0]==1.0:
            self.ui.radioButton_10.setChecked(True)
        else:
            self.ui.radioButton_09.setChecked(True)
        self.ui.lineEdit_36.setText(str(int(settings[41,0])))
        self.ui.lineEdit_37.setText(str(int(settings[42,0])))
        self.ui.lineEdit_38.setText(str(settings[43,0]))
        # Hierarchical Clustering / MCA
        if settings[44,0]==1.0:
            self.ui.chekBox_04.setChecked(True)
        else:
            self.ui.checkBox_04.setChecked(False)
        # Stoichiometry
        self.ui.lineEdit_39.setText(str(int(settings[45,0])))
        self.ui.lineEdit_40.setText(str(int(settings[46,0])))
        self.ui.lineEdit_41.setText(str(int(settings[47,0])))
        self.ui.lineEdit_42.setText(str(int(settings[48,0])))
        self.ui.lineEdit_43.setText(str(settings[49,0]))
        self.ui.lineEdit_44.setText(str(settings[50,0]))
        if settings[51,0]==1.0:
            self.ui.chekBox_05.setChecked(True)
        else:
            self.ui.checkBox_05.setChecked(False)
        if settings[52,0]==1.0:
            self.ui.chekBox_06.setChecked(True)
        else:
            self.ui.checkBox_06.setChecked(False)
        lc.statement()
        
    def browse_save_path(self):
        '''
            get path name from pyQT
            Update lineEdit
        '''
        self.save_filename = QtGui.QFileDialog.getExistingDirectory(self, 'Save Settings', '.')
        self.ui.lineEdit_46.setText(self.save_filename)
    
    def drop_save_path(self):
        '''
            update lineEdit_46 for new filename
        '''
        self.save_filename=str(self.ui.lineEdit_46.text())
        self.ui.lineEdit_46.setText(self.save_filename)
    
    def save_settings(self):
        dir_name=self.save_filename
        settings=np.zeros([53,1])
        # Main / Input
        if self.ui.radioButton_01.isChecked():
            settings[0]=1
        # Main / ROI
        settings[1]=float(self.ui.lineEdit_02.text())
        settings[2]=float(self.ui.lineEdit_03.text())
        settings[3]=float(self.ui.lineEdit_04.text())
        settings[4]=float(self.ui.lineEdit_05.text())
        settings[5]=float(self.ui.lineEdit_06.text())
        settings[6]=float(self.ui.lineEdit_07.text())
        settings[7]=float(self.ui.lineEdit_08.text())
        settings[8]=float(self.ui.lineEdit_09.text())
        settings[9]=float(self.ui.lineEdit_10.text())
        settings[10]=float(self.ui.lineEdit_11.text())
        # Main / Setup
        settings[11]=float(self.ui.lineEdit_12.text())
        settings[12]=float(self.ui.lineEdit_13.text())
        settings[13]=float(self.ui.lineEdit_14.text())
        # Visualize / Iamge
        settings[14]=float(self.ui.lineEdit_15.text())
        settings[15]=float(self.ui.lineEdit_16.text())
        settings[16]=float(self.ui.lineEdit_17.text())
        if self.ui.radioButton_03.isChecked():
            settings[17]=1
        if self.ui.checkBox_01.isChecked():
            settings[18]=1
        if self.ui.checkBox_02.isChecked():
            settings[19]=1
        # Visualize / MCA
        if self.ui.checkBox_03.isChecked():
            settings[20]=1
        settings[21]=float(self.ui.lineEdit_18.text())
        settings[22]=float(self.ui.lineEdit_19.text())
        settings[23]=float(self.ui.lineEdit_20.text())
        # Precision / Theoretical
        settings[24]=float(self.ui.lineEdit_21.text())
        settings[25]=float(self.ui.lineEdit_22.text())
        settings[26]=float(self.ui.lineEdit_23.text())
        # Ripley's K-function
        settings[27]=float(self.ui.lineEdit_24.text())
        settings[28]=float(self.ui.lineEdit_25.text())
        settings[29]=float(self.ui.lineEdit_26.text())
        # Register / Bead Detection
        settings[30]=float(self.ui.lineEdit_28.text())
        settings[31]=float(self.ui.lineEdit_29.text())
        settings[32]=float(self.ui.lineEdit_30.text())
        settings[33]=float(self.ui.lineEdit_31.text())
        # Register / Registration
        settings[34]=float(self.ui.lineEdit_32.text())
        if self.ui.radioButton_05.isChecked():
            settings[35]=1
        # CBC
        settings[36]=float(self.ui.lineEdit_33.text())
        settings[37]=float(self.ui.lineEdit_34.text())
        settings[38]=float(self.ui.lineEdit_35.text())
        if self.ui.radioButton_07.isChecked():
            settings[39]=1
        # Hierarchical Clustering / sort
        if self.ui.radioButton_10.isChecked():
            settings[40]=1
        settings[41] = float(self.ui.lineEdit_36.text())
        settings[42] = float(self.ui.lineEdit_37.text())
        settings[43] = float(self.ui.lineEdit_38.text())
        # Hierarchical Clustering / MCA
        if self.ui.checkBox_04.isChecked():
            settings[44] = 1
        # Stoichiometry
        settings[45] = float(self.ui.lineEdit_39.text())
        settings[46] = float(self.ui.lineEdit_40.text())
        settings[47] = float(self.ui.lineEdit_41.text())
        settings[48] = float(self.ui.lineEdit_42.text())
        settings[49] = float(self.ui.lineEdit_43.text())
        settings[50] = float(self.ui.lineEdit_44.text())
        if self.ui.checkBox_05.isChecked():
            settings[51] = 1
        if self.ui.checkBox_06.isChecked():
            settings[52] = 1
        la.write_settings(settings,dir_name)
        lc.statement()
