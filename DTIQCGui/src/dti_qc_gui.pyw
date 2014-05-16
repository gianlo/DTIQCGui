#!/usr/bin/env python
'''
A graphical interface to perform Quality Control (QC) on DTI acquisition.
Allows the user to remove from the dataset volumes corresponding to directions corrupted by artifacts.
Saves the QC data into the same folder as the original data adding the '.exc' suffix and writes a txt file describing
what action was taken on the data (for data provenance purposes).

Created on 10 Apr 2013

@author: gfagiolo

The MIT License (MIT)

Copyright (c) 2013 Gianlorenzo Fagiolo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os
import sys
import argparse
from numpy import uint8, flatnonzero, percentile, zeros
from PyQt4 import QtGui, QtCore
from datetime import datetime
import logging
logging.basicConfig(level=logging.WARNING)

from dti_processing import ProcessDTI
from dtiqa_ui_manual import Ui_FormDTIQA

#===============================================================================
# CONSTANTS/GLOBALS
#===============================================================================

__version__ = "0.90beta"
SIGNAL = QtCore.SIGNAL
tr = QtCore.QString
APP_NAME = "Exclude DTI dirs"
SETTING_PATH = "CWD"
GREY_COLORTABLE = [QtGui.qRgb(i, i, i) for i in range(256)]
DEFAULT_PREVIEW_SIZE = QtCore.QSize(256, 256)

#===============================================================================
# HELPER FUNCTIONS/CLASSES
#===============================================================================

def rescaleImageData(data, quantile=99.5, fastest=False, quantileFunc=percentile, signal=None):
    """robustly rescales data intensities to fit into QImage dynamic range
        @type data: numpy.ndarray
        @type quantile: float
        @type fastest: bool
        @type quantilFunc: callable
        @type signal: QtCore.Object
    """
    def cast_uint8(data, normConst):
        return uint8(255.*data/normConst)
    if fastest:
        #original data (dicom) is in 12bit unsigned format
        #maxValue is 2^12 - 1 = 4095
        maxValue = 4095.
        return cast_uint8(data,maxValue)
    else:
        output = zeros(data.shape, dtype=uint8)
        #rescales each direction to fit the 255 range
        for dir_no in range(data.shape[-1]):
            if not signal is None:
                signal.emit(dir_no + 1)
            cdata = data[:, :, :, dir_no].copy()
            if not quantileFunc is None:
                #do robust maximum
                maxValue = quantileFunc(cdata.flatten(), quantile)
                #in order to prevent wrapping when converting to uint8,
                #everything above maxValue needs to be set to max value
                cdata.flat[flatnonzero(cdata.flatten() >= maxValue)] = maxValue
                maxValue += 0.0
            else:
                #use direction maximum
                maxValue = cdata.flatten().max() + 0.0            
            output[:, :, :, dir_no] = cast_uint8(cdata, maxValue)
        return output

class RescaleSignal(QtCore.QObject):
    volumeBeingRescaled = QtCore.pyqtSignal(int, name='volumeBeingRescaled')
#===============================================================================
# RESCALE PERFORMANCE 
# 
# MODE: Fastest (ie rescale 4096 to 255)
# open dataset 0:00:00.515000
# rescale image data 0:00:00.931000
# update preview 0:00:00.003000
# loading completed 0:00:01.449000
#
#MODE: Direction maximum
#open dataset 0:00:00.512000
#rescale image data 0:00:01.571000
#update preview 0:00:00.002000
#loading completed 0:00:02.085000
#
#MODE: Direction robust maximum [mquantiles from scipy]
#open dataset 0:00:00.522000
#rescale image data 0:00:03.467000
#update preview 0:00:00.002000
#loading completed 0:00:03.992000
#
#MODE: Direction robust maximum using numpy [use this to reduce load up time, since scipy is not required]
#open dataset 0:00:00.501000
#rescale image data 0:00:03.426000
#update preview 0:00:00.002000
#loading completed 0:00:03.930000
#===============================================================================

def numpy2grayscaleqimage(data, res, 
                          previewsize=DEFAULT_PREVIEW_SIZE, 
                          order='F', label=None, 
                          flipstate=(False, False),
                          rotate=False):
    """rescales spatially the data to fit into the previewsize
    Input parameters:
        @type data: numpy.ndarray
        @type res: tuple
        @type previewsize: QtCore.QSize
        @type order: 'F', 'C' 
        @type label: str
        @type flipstate: tuple(bool, bool)
        @type rotate: bool
    Returns:
        @rtype: QImage
    """
    
    qImage = QtGui.QImage(data.tostring(order), data.shape[0], data.shape[1], QtGui.QImage.Format_Indexed8)
    qImage.setColorTable(GREY_COLORTABLE)
    verticalSize = res[1]*data.shape[1]
    horizontalSize = res[0]*data.shape[0]
    #decide rescaling
    rescaleSize = QtCore.QSize(previewsize)
    if verticalSize >= horizontalSize:
        # vertical is larger, reduce width
        rescaleSize.setWidth(int(horizontalSize/verticalSize*previewsize.height()))
    else:
        #horizontal is larger, reduce height
        rescaleSize.setHeight(int(verticalSize/horizontalSize*previewsize.width()))
    logging.debug(" ".join(map(str, [data.shape, res, verticalSize, horizontalSize, str(label), rescaleSize.height(), rescaleSize.width()])))
    if rotate:
        #90 degree rotation
        rot90 = QtGui.QTransform()
        rot90.rotate(90)
        #QtGui.QMatrix(0.,1.,-1.,0.,1.,1.)
        return qImage.scaled(rescaleSize).transformed(rot90).mirrored(flipstate[0], flipstate[1])
    else:
        return qImage.scaled(rescaleSize).mirrored(flipstate[0], flipstate[1])

class PreviewObject(object):
    """contains references  for both numpy (.data_) QImage (.qimag_) and QOixmap (.qpixmap_) data for the preview panes"""
    
    data_x = None
    data_y = None
    data_z = None
    qimage_x = None
    qimage_y = None
    qimage_z = None
    qpixmap_x = None
    qpixmap_y = None
    qpixmap_z = None

    
#===============================================================================
# APPLICATION CLASS
#===============================================================================

class DtiQAGui(QtGui.QWidget):
    '''
    The dti QC app
    '''
    #dti object
    dti_obj = None
    #nifti object
    nifti_data = None
    #image data (numpy)
    image_data = None
    #preview container
    preview_data = None
    #pixel sizes
    pixel_sizes = None
    
    def __init__(self):
        '''
        Constructor
        '''
        super(DtiQAGui, self).__init__()
                
        #setup ui
        self.ui = Ui_FormDTIQA()
        self.ui.setupUi(self)
        self.statusbar = QtGui.QStatusBar(self)
        self.rescaleProgressBar = QtGui.QProgressBar(self.statusbar)
        self.rescaleProgressBar.setTextVisible(False)
        self.rescaleProgressBar.setFixedSize(80, 20)
        self.statusbar.addPermanentWidget(self.rescaleProgressBar)
        self.ui.gridLayout.addWidget(self.statusbar)
        
        #app settings (i.e. current path etc)
        self.settings = QtCore.QSettings(APP_NAME)
        
        #preview data
        self.preview_data = PreviewObject()
        
        #keyboard shortcuts
        self.keyboard_exclude = QtGui.QShortcut(QtGui.QKeySequence(tr("e")), self, )
                
        #signals
        self.rescaleSignal = RescaleSignal()
        #set up connections
        #dataset
        self.ui.pushButton_nifti.clicked.connect(self.__load_nifti)
        self.ui.pushButton_bvals.clicked.connect(self.__load_bvals)
        self.ui.pushButton_bvecs.clicked.connect(self.__load_bvecs)
        self.ui.commandLinkButton_load.clicked.connect(self.load_data)
        #actions
        self.ui.lineEdit_exclude.editingFinished.connect(self.__validate_text_excluded)
        self.ui.horizontalSlider_imageno.valueChanged.connect(self.__update_previews)
        self.ui.horizontalSlider_imageno.valueChanged.connect(self.__update_bvalue_label)
        self.ui.pushButton_exclude.clicked.connect(self.__exclude_dir)
        self.ui.pushButton_clear.clicked.connect(self.__reset_excluded)
        self.ui.commandLinkButton_save.clicked.connect(self.save_data)
        #preview
        self.ui.horizontalSlider_x.valueChanged.connect(self.__update_preview_x)       
        self.ui.horizontalSlider_y.valueChanged.connect(self.__update_preview_y)       
        self.ui.horizontalSlider_z.valueChanged.connect(self.__update_preview_z)
        self.ui.checkBox_preview_xy_vflip.stateChanged.connect(self.__update_preview_z)
        self.ui.checkBox_preview_xy_hflip.stateChanged.connect(self.__update_preview_z)
        self.ui.checkBox_preview_yz_vflip.stateChanged.connect(self.__update_preview_x)
        self.ui.checkBox_preview_yz_hflip.stateChanged.connect(self.__update_preview_x)
        self.ui.checkBox_preview_yz_rotate.stateChanged.connect(self.__update_preview_x)
        self.ui.checkBox_preview_zx_vflip.stateChanged.connect(self.__update_preview_y)
        self.ui.checkBox_preview_zx_hflip.stateChanged.connect(self.__update_preview_y)
        self.ui.checkBox_preview_zx_rotate.stateChanged.connect(self.__update_preview_y)
        #status bar
        self.rescaleSignal.volumeBeingRescaled.connect(self.__show_scaling_statusbar)
        #preview interaction
        self.ui.label_preview_xy.previewClicked.connect(self.__preview_clicked)
        self.ui.label_preview_yz.previewClicked.connect(self.__preview_clicked)
        self.ui.label_preview_zx.previewClicked.connect(self.__preview_clicked)
        
        
#===============================================================================
# HELPERS
#===============================================================================
    def __show_scaling_statusbar(self, dir_no):
        """Updates status of scaling using the status bar
            @type dir_no: int 
        """
        self.statusbar.showMessage("Rescaling volume %d (of %d)"%(dir_no, self.dti_obj.getSignalLength()))
        self.rescaleProgressBar.setValue(dir_no)
        
    def __validate_dataset(self):
        if self.get_nifti() and self.get_bvals() and self.get_bvecs():
            self.ui.commandLinkButton_load.setEnabled(True)
        else:
            self.ui.commandLinkButton_load.setEnabled(False)
    
    def __update_bvalue_label(self):
        self.ui.label_bvalue.setText(str(self.dti_obj.getBValues()[self.get_current_dir()]))
        
    def __update_preview_section(self):
        'toggles preview section on/off'
        if not self.image_data is None:                    
            sh = self.image_data.shape[:-1]            

            self.ui.horizontalSlider_x.setEnabled(True)
            self.ui.horizontalSlider_x.setMaximum(sh[0]-1)
            if self.ui.horizontalSlider_x.value() == sh[0]/2:
                self.ui.horizontalSlider_x.valueChanged.emit(sh[0]/2)
            self.ui.horizontalSlider_x.setValue(sh[0]/2)
                        
            self.ui.horizontalSlider_y.setEnabled(True)
            self.ui.horizontalSlider_y.setMaximum(sh[1]-1)
            if self.ui.horizontalSlider_y.value() == sh[1]/2:
                self.ui.horizontalSlider_y.valueChanged.emit(sh[1]/2)            
            self.ui.horizontalSlider_y.setValue(sh[1]/2)
            
            self.ui.horizontalSlider_z.setEnabled(True)
            self.ui.horizontalSlider_z.setMaximum(sh[2]-1)
            if self.ui.horizontalSlider_z.value() == sh[2]/2:
                self.ui.horizontalSlider_z.valueChanged.emit(sh[2]/2)
            self.ui.horizontalSlider_z.setValue(sh[2]/2)        
        else:
            #turn off preview
            self.ui.horizontalSlider_x.setEnabled(False)
            self.ui.horizontalSlider_x.setMaximum(0)
            self.ui.horizontalSlider_x.setValue(0)
            self.ui.horizontalSlider_y.setEnabled(False)
            self.ui.horizontalSlider_y.setMaximum(0)
            self.ui.horizontalSlider_y.setValue(0)
            self.ui.horizontalSlider_z.setEnabled(False)
            self.ui.horizontalSlider_z.setMaximum(0)
            self.ui.horizontalSlider_z.setValue(0)
            self.ui.label_preview_xy.clear()
            self.ui.label_preview_yz.clear()
            self.ui.label_preview_zx.clear()

    def __update_previews(self):
        'updates preview contents'
        self.__update_preview_x()
        self.__update_preview_y()
        self.__update_preview_z()
    
    def __update_preview_z(self):
        'updates label_preview_xy (i.e. slicing direction z)'
        res = self.get_pixel_sizes()
        mir = (self.ui.checkBox_preview_xy_hflip.isChecked(), 
               self.ui.checkBox_preview_xy_vflip.isChecked()) 
        self.preview_data.data_z = self.image_data[:, :, self.get_current_z(), self.get_current_dir()].squeeze()
        self.preview_data.qimage_z = numpy2grayscaleqimage(self.preview_data.data_z, (res[1], res[2]),
                                                           label='preview_z',
                                                           flipstate=mir)
        self.preview_data.qpixmap_z = QtGui.QPixmap(self.preview_data.qimage_z)
        self.ui.label_preview_xy.setPixmap(self.preview_data.qpixmap_z)       
        
    def __update_preview_x(self):
        'updates label_preview_yz (i.e. slicing direction x)'
        res = self.get_pixel_sizes()
        mir = (self.ui.checkBox_preview_yz_hflip.isChecked(), 
               self.ui.checkBox_preview_yz_vflip.isChecked())        
        self.preview_data.data_x = self.image_data[self.get_current_x(), :, :, self.get_current_dir()].squeeze()
        self.preview_data.qimage_x = numpy2grayscaleqimage(self.preview_data.data_x, (res[2], res[3]),
                                                           label='preview_x',
                                                           flipstate=mir,
                                                           rotate=self.ui.checkBox_preview_yz_rotate.isChecked())
        self.preview_data.qpixmap_x = QtGui.QPixmap(self.preview_data.qimage_x)
        self.ui.label_preview_yz.setPixmap(self.preview_data.qpixmap_x)       

    def __update_preview_y(self):
        'updates label_preview_zx (i.e. slicing direction y)'
        res = self.get_pixel_sizes()
        mir = (self.ui.checkBox_preview_zx_hflip.isChecked(), 
               self.ui.checkBox_preview_zx_vflip.isChecked()) 
        self.preview_data.data_y = self.image_data[:, self.get_current_y(), :, self.get_current_dir()].squeeze()
        self.preview_data.qimage_y = numpy2grayscaleqimage(self.preview_data.data_y, (res[1], res[3]),
                                                           label='preview_y',
                                                           flipstate=mir,
                                                           rotate=self.ui.checkBox_preview_zx_rotate.isChecked())
        self.preview_data.qpixmap_y = QtGui.QPixmap(self.preview_data.qimage_y)
        self.ui.label_preview_zx.setPixmap(self.preview_data.qpixmap_y)

    def __reset_excluded(self):
        if not self.dti_obj is None:
            self.dti_obj.EXCLUDED_DIRS = set()
            self.__update_excluded_list()
    
    def __validate_text_excluded(self):
        ERR_MSG = "The text entered must be in the [i, j, ...] format where i, j are integers, indicating which volume to exclude."
        if not self.dti_obj is None:
            lastdir = self.dti_obj.getSignalLength() - 1
            txt = self.get_excluded()
            try:
                exc = set(eval(txt))
            except SyntaxError:
                QtGui.QMessageBox.critical(self, "Editing Error", ERR_MSG)
                self.__update_excluded_list()
                return None
            except NameError:
                QtGui.QMessageBox.critical(self, "Editing Error", ERR_MSG)
                self.__update_excluded_list()
                return None
            rem = set()           
            for i in exc:
                if not isinstance(i, int):
                    QtGui.QMessageBox.critical(self, "Editing Error", ERR_MSG)
                    self.__update_excluded_list()
                    return None
                if i < 0 or i > lastdir:
                    #remove out of range integers
                    rem.add(i)
                     
            self.dti_obj.EXCLUDED_DIRS = exc.difference(rem)
            self.__update_excluded_list()
    
    def __exclude_dir(self):
        if not self.dti_obj is None:
            self.dti_obj.addExcluded(self.get_current_dir())
            self.__update_excluded_list()
            
    def __update_excluded_list(self):
        if not self.dti_obj is None:
            self.set_excluded(str(sorted(list(self.dti_obj.EXCLUDED_DIRS))))
        else:
            self.set_excluded("[]")

    def __dataset_ok(self):
        self.ui.lineEdit_loadedpath.setText(self.dti_obj.FILENAME)
        self.ui.lineEdit_loadedpath.setEnabled(True)
        self.ui.horizontalSlider_imageno.setEnabled(True)
        self.ui.horizontalSlider_imageno.setMinimum(0)
        self.ui.horizontalSlider_imageno.setMaximum(self.dti_obj.getSignalLength()-1)
        self.ui.horizontalSlider_imageno.setValue(0)
        self.ui.commandLinkButton_save.setEnabled(True)
        self.ui.pushButton_exclude.setEnabled(True)
        self.ui.pushButton_clear.setEnabled(True)
        self.ui.lineEdit_exclude.setEnabled(True)
        self.ui.checkBox_preview_xy_vflip.setEnabled(True)
        self.ui.checkBox_preview_xy_hflip.setEnabled(True)
        self.ui.checkBox_preview_yz_vflip.setEnabled(True)
        self.ui.checkBox_preview_yz_hflip.setEnabled(True)
        self.ui.checkBox_preview_yz_rotate.setEnabled(True)
        self.ui.checkBox_preview_zx_vflip.setEnabled(True)
        self.ui.checkBox_preview_zx_hflip.setEnabled(True)
        self.ui.checkBox_preview_zx_rotate.setEnabled(True)
        self.__update_excluded_list()
        self.__update_bvalue_label()
        self.keyboard_exclude.activated.connect(self.__exclude_dir)
                
    def __dataset_not_ok(self):
        self.ui.lineEdit_loadedpath.clear()
        self.ui.lineEdit_loadedpath.setEnabled(False)
        self.ui.horizontalSlider_imageno.setEnabled(False)
        self.ui.horizontalSlider_imageno.setValue(0)
#        self.ui.horizontalSlider_imageno.setMinimum(0)
        self.ui.horizontalSlider_imageno.setMaximum(0)
        self.ui.commandLinkButton_save.setEnabled(False)
        self.ui.pushButton_exclude.setEnabled(False)
        self.ui.pushButton_clear.setEnabled(False)
        self.ui.lineEdit_exclude.setEnabled(False)
        self.ui.checkBox_preview_xy_vflip.setEnabled(False)
        self.ui.checkBox_preview_xy_hflip.setEnabled(False)
        self.ui.checkBox_preview_yz_vflip.setEnabled(False)
        self.ui.checkBox_preview_yz_hflip.setEnabled(False)
        self.ui.checkBox_preview_yz_rotate.setEnabled(False)
        self.ui.checkBox_preview_zx_vflip.setEnabled(False)
        self.ui.checkBox_preview_zx_hflip.setEnabled(False)
        self.ui.checkBox_preview_zx_rotate.setEnabled(False)
        self.__update_excluded_list()
        self.keyboard_exclude.activated.disconnect()

    def is_data_loaded(self):
        '@rtype bool'
        return self.dti_obj is not None
    
    def __get_previewqlabel(self, slaxis):
        '@rtype PreviewQLabel'
        out = None
        if slaxis == 'z':
            out = self.ui.label_preview_xy
        elif slaxis == 'x':
            out = self.ui.label_preview_yz
        if slaxis == 'y':
            out = self.ui.label_preview_zx            
        return out
    
    def __get_previewpixmap_boundingbox(self, slaxis):
        """returns qpixmap bounding box for slaxis preview
        @type slaxis: str
        @rtype boundingbox: QtCore.QRect 
        """
        #see http://www.qtcentre.org/threads/34368-QLabel-and-its-content-%28pixmap%29-position-problem
        prevlab = self.__get_previewqlabel(slaxis)
        curr_qpixmap = getattr(self.preview_data, 'qpixmap_' + slaxis)
        cr = prevlab.contentsRect()
        cr.adjust(prevlab.margin(), prevlab.margin(), -prevlab.margin(), -prevlab.margin())
        align = prevlab.alignment()
        style = prevlab.style()
        aligned = style.alignedRect(QtGui.QApplication.layoutDirection(), align, curr_qpixmap.size(), cr)
        boundingbox = aligned.intersected(cr)
        return boundingbox
           
    
    def __preview_clicked(self, mouseclicked):
        """updates previews location by interpreting current mouseclick position 
        @type mouseclicked: PreviewMouseClick
        """
        slaxis = mouseclicked.SLICING_AXIS
        if self.is_data_loaded() and slaxis in ('x', 'y', 'z'):
            position = mouseclicked.EVENT.pos()            
            logging.debug('Mouse position: ' + str((position.x(), position.y())))
            curr_qpixmap = getattr(self.preview_data, 'qpixmap_' + slaxis)
            #find current bounding box of qpixmap
            pixmap_rect = self.__get_previewpixmap_boundingbox(slaxis)
            pmY = curr_qpixmap.height() + 0.
            pmX = curr_qpixmap.width() + 0.        
            logging.debug('Preview: ' + slaxis + str((pmY, pmX)))
            x_val, y_val, z_val = None, None, None
            #need to convert qlabel position to index in image        
            #get nifti pixel count
            nifti_shape = self.dti_obj.DIMS
            if slaxis == 'z':
                #xy preview
                #rescale position x,y to nifti sizes                
                x_val = int(position.x()/pmX*nifti_shape[1])
                y_val = int(position.y()/pmY*nifti_shape[0])
                if self.ui.checkBox_preview_xy_vflip.isChecked():
                    #invert vertical index: y
                    y_val = self.ui.horizontalSlider_y.maximum()-y_val
                if self.ui.checkBox_preview_xy_hflip.isChecked():
                    #invert horizontal index: x
                    x_val = self.ui.horizontalSlider_x.maximum()-x_val
                self.ui.horizontalSlider_x.setValue(x_val)
                self.ui.horizontalSlider_y.setValue(y_val)
            elif slaxis == 'x':
                #yz preview
                #rescale position x,y to nifti sizes
                if self.ui.checkBox_preview_yz_rotate.isChecked():
                    y_val = int(position.y()/pmY*nifti_shape[1])
                    z_val = self.ui.horizontalSlider_z.maximum()-int((position.x()-pixmap_rect.left())/pmX*nifti_shape[2])                    
                else:
                    y_val = int(position.x()/pmX*nifti_shape[1])
                    z_val = int((position.y()-pixmap_rect.top())/pmY*nifti_shape[2])
                if self.ui.checkBox_preview_yz_vflip.isChecked():
                    #invert vertical index: z
                    z_val = self.ui.horizontalSlider_z.maximum()-z_val
                if self.ui.checkBox_preview_yz_hflip.isChecked():
                    #invert horizontal index: y
                    y_val = self.ui.horizontalSlider_y.maximum()-y_val
                self.ui.horizontalSlider_z.setValue(z_val)
                self.ui.horizontalSlider_y.setValue(y_val)
            elif slaxis == 'y':
                #zx preview
                #rescale position x,y to nifti sizes
                z_val = int((position.y()-pixmap_rect.top())/pmY*nifti_shape[2])
                x_val = int(position.x()/pmX*nifti_shape[0])
                if self.ui.checkBox_preview_zx_vflip.isChecked():
                    #invert vertical index: z
                    z_val = self.ui.horizontalSlider_z.maximum()-z_val
                if self.ui.checkBox_preview_zx_hflip.isChecked():
                    #invert horizontal index: x
                    x_val = self.ui.horizontalSlider_x.maximum()-x_val
                if self.ui.checkBox_preview_zx_rotate.isChecked():
                    #swap indexes since it's rotated
                    tmp = int(x_val)                    
                    x_val = z_val
                    z_val = tmp                    
                self.ui.horizontalSlider_z.setValue(z_val)
                self.ui.horizontalSlider_x.setValue(x_val)
            
#===============================================================================
# GETTERS/SETTERS
#===============================================================================
    def set_app_path(self, path):
        self.settings.setValue(SETTING_PATH, tr(path))
        
    def get_app_path(self):
        return unicode(self.settings.value(SETTING_PATH).toString())
                                
    def get_current_dir(self):
        return self.ui.horizontalSlider_imageno.value()
    
    def get_current_x(self):
        return self.ui.horizontalSlider_x.value()

    def get_current_y(self):
        return self.ui.horizontalSlider_y.value()

    def get_current_z(self):
        return self.ui.horizontalSlider_z.value()

    def set_excluded(self, txt):
        self.ui.lineEdit_exclude.setText(txt)

    def get_excluded(self):
        return str(self.ui.lineEdit_exclude.text())
    
    def set_nifti(self, fname):
        self.ui.lineEdit_nifti.setText(fname)
    
    def get_nifti(self):
        return unicode(self.ui.lineEdit_nifti.text())
    
    def set_pixel_sizes(self, pixdims):
        self.pixel_sizes = tuple(pixdims)
        
    def get_pixel_sizes(self):
        return self.pixel_sizes
        
    def set_bvals(self, fname):
        self.ui.lineEdit_bvals.setText(fname)
    
    def get_bvals(self):
        return unicode(self.ui.lineEdit_bvals.text())

    def set_bvecs(self, fname):
        self.ui.lineEdit_bvecs.setText(fname)
    
    def get_bvecs(self):
        return unicode(self.ui.lineEdit_bvecs.text())

#===============================================================================
# ACTIONS
#===============================================================================
    def __load_nifti(self):
        cwd = self.get_app_path()
        if not os.path.isdir(cwd):
            cwd = os.path.abspath(os.path.curdir)
            self.set_app_path(cwd)
            
            
        ret = QtGui.QFileDialog.getOpenFileName(self, tr('Load DTI nifti'), cwd, filter=tr("Nifti files (*.nii.gz *.nii)"))
        if ret:
            self.set_app_path(os.path.dirname(unicode(ret)))
            self.set_nifti(ret)
            self.__validate_dataset()
                    
    def __load_bvals(self):
        cwd = self.get_app_path()
        if not os.path.isdir(cwd):
            cwd = os.path.abspath(os.path.curdir)
            self.set_app_path(cwd)
            
        ret = QtGui.QFileDialog.getOpenFileName(self, tr('Load FSL b-vals file'), cwd, filter=tr("bvals files (*.bvals*);; All files (*.*)"))
        if ret:
            self.set_app_path(os.path.dirname(unicode(ret)))
            self.set_bvals(ret)
            self.__validate_dataset()
    
    def __load_bvecs(self):
        cwd = self.get_app_path()
        if not os.path.isdir(cwd):
            cwd = os.path.abspath(os.path.curdir)
            self.set_app_path(cwd)
            
        ret = QtGui.QFileDialog.getOpenFileName(self, tr('Load FSL b-vecs file'), cwd, filter=tr("bvecs files (*.bvecs*);; All files (*.*)"))
        if ret:
            self.set_app_path(os.path.dirname(unicode(ret)))
            self.set_bvecs(ret)
            self.__validate_dataset()
    
    def load_data(self):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.statusbar.showMessage("Loading dataset...")
        t0 = datetime.now()                
        dti_obj = ProcessDTI(self.get_nifti(), self.get_bvecs(), self.get_bvals())
        dt = datetime.now() - t0
        logging.debug('open dataset %s'%str(dt))
        #sanity check
        if not dti_obj is None:
            ndirs = dti_obj.getSignalLength()
            try:
                #check that #of bvals and bves are the same as the image 4th dimension
                is_data_correct = dti_obj.getBValues().shape[0] == ndirs and dti_obj.getBVectors().shape[1] == ndirs                
            except:
                is_data_correct = False
        else:
            is_data_correct = False
            
        if is_data_correct:
            self.dti_obj = dti_obj
            self.dti_obj.clearExcluded()
            self.nifti_data = self.dti_obj.getNifti()
            self.rescaleProgressBar.setMinimum(1)
            self.rescaleProgressBar.setMaximum(self.dti_obj.getSignalLength())
            t1 = datetime.now()            
            self.image_data = rescaleImageData(self.dti_obj.getImageData(),signal=self.rescaleSignal.volumeBeingRescaled)
            self.rescaleProgressBar.reset()
            logging.debug('rescale image data %s'%str(datetime.now()-t1))
            self.set_pixel_sizes(self.nifti_data.get_header()['pixdim'])
            self.__dataset_ok()
            t1 = datetime.now()
            self.__update_preview_section()
            logging.debug('update preview %s'%str(datetime.now()-t1))
        else:
            QtGui.QMessageBox.critical(self, "Error Loading", "Please check that the nifti, the bvals and bvecs are correct")
            if self.dti_obj is None:
                self.__dataset_not_ok()
        logging.debug('loading completed %s'%str(datetime.now() - t0)) 
        self.statusbar.showMessage("dataset loaded", 1000)       
        QtGui.QApplication.restoreOverrideCursor()                
        
    def save_data(self):        
        saved_files = None
        if not self.dti_obj is None:
            if self.dti_obj.EXCLUDED_DIRS:
                QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
                did_exception_happen = False
                try:
                    saved_files = self.dti_obj.saveData()
                except:
                    did_exception_happen = True
                QtGui.QApplication.restoreOverrideCursor()
                if did_exception_happen:
                    QtGui.QMessageBox.critical(self, "SAVE ERROR", "Couldn't save data. Please check disk space (or writing privileges) on "+os.path.dirname(str(self.parameter_box.dataset.dti_nifti_name)))
                else:
                    QtGui.QMessageBox.information(self, "SAVE OK", "File saved: "+' '.join(saved_files))

if __name__ == "__main__":
    #parse arguments (the python way)
    parser = argparse.ArgumentParser()
#     follow FSL dtifti standard
#     -k,--data    dti data file
#     -o,--out    Output basename
#     -m,--mask    Bet binary mask file
#     -r,--bvecs    b vectors file
#     -b,--bvals    b values file
    parser.add_argument("-k", "--data", help="Nifti file containing dti data")
    parser.add_argument("-b", "--bvals", help="Txt file containing b-values")
    parser.add_argument("-r", "--bvecs", help="Txt file containing b-vectors")
    args = parser.parse_args()
    
    app = QtGui.QApplication(sys.argv)
    myapp = DtiQAGui()
    myapp.show()
    
    if args.data is not None and args.bvals is not None and args.bvecs is not None:
        #if a complete dti dataset is specified, then load it!
        myapp.set_nifti(args.data)
        myapp.set_bvals(args.bvals)
        myapp.set_bvecs(args.bvecs)
        myapp.load_data()
        
    #start event loop
    exit_code = app.exec_()
    sys.exit(exit_code)

        