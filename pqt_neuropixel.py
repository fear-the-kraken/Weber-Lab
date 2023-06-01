#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:28:52 2023

@author: fearthekraken
"""

import sys
import os
import scipy.io as so
import re
import scipy
import h5py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from colour import Color
import colorsys
import pyautogui
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pdb
# custom modules
import sleepy
import pwaves
import AS
import pqt_items as pqi

                
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ppath, name):
        """
        Instantiate main window
        """
        QtWidgets.QMainWindow.__init__(self)
        self.setObjectName('pqt_neuropixel')
        self.WIDTH, self.HEIGHT = pyautogui.size()
        self.coors = pd.Series(data=[20, 20, self.WIDTH-80, self.HEIGHT-80], 
                                index=['x','y','w','h'])
        self.setGeometry(QtCore.QRect(self.coors.x, self.coors.y, self.coors.w, self.coors.h))
        self.setFixedHeight(self.coors.h)
        self.settings_width = pqi.px_w(200, self.WIDTH)
        self.graph_width = self.coors.w - self.settings_width - 40
        
        self.ppath = ppath
        self.name  = name
        
        self.index = 15                 # index (FFT) of central timepoint for current EEG/LFP plot
        self.pcollect_index = False     # ignore or collect visited brain state indices?
        self.index_list = [self.index]  # collected brain state indices for bulk annotation
        self.bulk_annot = False         # if True, annotate all indices between two mouse clicks
        self.bulk_index1 = None
        
        self.fft_load = 25  # no. of FFT bins loaded into memory
        self.fft_view = 5   # no. of FFT bins shown in window
        self.open_fid = []  # open h5py files
        
        self.lfp_plotmode = 0  # plot local LFP signal (0) or session standard deviation (1)
        self.np_plotmode = 1   # plot single unit spike trains (0) or session firing rate (1)
        self.annotation_mode = False  # computationally efficient setting for annotating brain states
        
        self.f0 = 5.   # lowest freq in bandpass filter
        self.f1 = 30.  # highest freq in bandpass filter
        
        self.thres = 4.5         # threshold value for P-wave detection
        self.thres_type = 'std'  # thresholding method (raw value, st. deviations, or %iles)
        self.dup_win = 40  # eliminate duplicate waves (within X ms of each other)
        
        self.defaults = self.dict_from_vars()
        
        self.unit_state_df = None
        self.unit_PFR_df = None
        
        #######################################################################
        
        # set up window layout, instantiate items, connect widgets
        self.gen_layout()
        self._createMenuBar()
        self.connect_buttons()
        # load recording info, draw initial data plots
        self.load_recording()
        self.plot_treck()
        self.plot_brainstate()
        self.plot_session()
        self.plot_signals()
        [graph.updated_data() for graph in self.LFP_graphs]
    
    
    def gen_layout(self):
        """
        Layout for main window
        """
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.centralLayout.setContentsMargins(0,0,0,0)
        self.centralLayout.setSpacing(0)
        
        
        ### LIVE GRAPHS LAYOUT ###
        
        self.pgraphWidget = QtWidgets.QWidget()
        self.pgraphLayout = QtWidgets.QVBoxLayout(self.pgraphWidget)
        self.pgraphLayout.setContentsMargins(0,0,0,0)
        self.pgraphLayout.setSpacing(0)
        
        ### graphs for entire recording
        self.plotView = pg.GraphicsLayoutWidget(parent=self)
        self.plotView.setFixedHeight(int(self.coors.h*0.30))
        self.lay_brainstate = self.plotView.addLayout()
        
        # laser / annotation history / current timepoint
        self.graph_treck = pg.PlotItem()
        self.graph_treck.setFixedWidth(self.graph_width)
        # set axis params
        self.graph_treck.vb.setMouseEnabled(x=True, y=False)
        self.graph_treck.setYRange(-0.2, 0.5, padding=None)
        yax = self.graph_treck.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('', units='', **labelStyle)
        yax.setTicks([[(0, ''), (1, '')]])
        yax.setWidth(int(pqi.px_w(50, self.WIDTH)))
        self.graph_treck.hideAxis('bottom')
        self.lay_brainstate.addItem(self.graph_treck)
        self.lay_brainstate.nextRow()
        
        # color-coded brain state
        self.graph_brainstate = pg.PlotItem()
        self.graph_brainstate.setFixedWidth(self.graph_width)
        self.image_brainstate = pg.ImageItem() 
        self.graph_brainstate.addItem(self.image_brainstate)
        # brain state colormap
        pos = np.linspace(0, 1, 8)
        color = np.array([[0, 0, 0, 200], [0, 255, 255, 200], 
                          [150, 0, 255, 200], [150, 150, 150, 200], 
                          [66,86,219,200], [255,20,20,200], 
                          [0,255,43,200], [255,255,0,200]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.lut_brainstate = cmap.getLookupTable(0.0, 1.0, 8)
        # set axis params
        self.graph_brainstate.vb.setMouseEnabled(x=True, y=False)
        yax = self.graph_brainstate.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('State', units='', **labelStyle)
        yax.setTicks([[(0, ''), (1, '')]])
        yax.setWidth(int(pqi.px_w(50, self.WIDTH)))
        self.graph_brainstate.hideAxis('bottom')
        self.lay_brainstate.addItem(self.graph_brainstate)
        self.lay_brainstate.nextRow()
        
        # EEG spectrogram
        self.graph_spectrum = pg.PlotItem()
        self.graph_spectrum.setFixedWidth(self.graph_width)
        self.image_spectrum = pg.ImageItem()     
        self.graph_spectrum.addItem(self.image_spectrum)
        # spectrogram colormap
        pos = np.array([0., 0.05, .2, .4, .6, .9])
        color = np.array([[0, 0, 0, 255], [0,0,128,255], [0,255,0,255], [255,255,0, 255], 
                          (255,165,0,255), (255,0,0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.lut_spectrum = cmap.getLookupTable(0.0, 1.0, 256)
        # set axis params
        self.graph_spectrum.vb.setMouseEnabled(x=True, y=False)
        yax = self.graph_spectrum.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('Freq', units='Hz', **labelStyle)
        yax.setTicks([[(0, '0'), (10, '10'), (20, '20')]])
        yax.setWidth(int(pqi.px_w(50, self.WIDTH)))
        self.graph_spectrum.hideAxis('bottom')
        self.lay_brainstate.addItem(self.graph_spectrum)
        self.lay_brainstate.nextRow()
        
        # session data
        self.graph_session = pg.PlotItem()
        self.graph_session.setFixedWidth(self.graph_width)
        self.session_item = pg.PlotDataItem()
        self.session_item.setObjectName('session data item')
        self.session_item.setPen((255,255,255),width=1)
        self.graph_session.addItem(self.session_item)
        # set axis params
        self.graph_session.vb.setMouseEnabled(x=True, y=True)
        yax = self.graph_session.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('EMG Amp.', units='', **labelStyle)
        yax.setWidth(int(pqi.px_w(50, self.WIDTH)))
        self.lay_brainstate.addItem(self.graph_session)
        
        ### current raw EEG/EMG/P-wave signals
        self.intanView = pg.GraphicsLayoutWidget(parent=self)
        #self.intanView.setFixedHeight(int(self.coors.h*0.25))
        # EEG/EMG
        self.graph_intan = pg.PlotItem()
        self.graph_intan.setFixedWidth(self.graph_width)
        self.curIntan = pg.PlotDataItem()
        self.curIntan.setObjectName('current Intan signal')
        self.curIntan.setPen((255,255,255),width=1)
        self.graph_intan.addItem(self.curIntan)
        # set axis params
        self.graph_intan.setYRange(-700, 700, padding=None)
        yax = self.graph_intan.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('EEG1', units='V', **labelStyle)
        yax.setWidth(int(pqi.px_w(50, self.WIDTH)))
        self.intanView.addItem(self.graph_intan)
        self.intanView.nextRow()
        # LFP with P-waves
        self.graph_pwaves = pqi.GraphEEG2(parent=self)
        self.graph_pwaves.setFixedWidth(self.graph_width)
        self.graph_pwaves.setFixedHeight(int(self.coors.h*0.15))
        # set axis params
        self.graph_pwaves.setYRange(-700, 700, padding=None)
        self.intanView.addItem(self.graph_pwaves)
        self.graph_pwaves.hide()
        
        ### Neuropixel LFP signals
        self.lfpView = pqi.LFPView(parent=self)
        self.lfpView.qscroll.setFixedHeight(int(self.coors.h*0.45))
        
        ### Neuropixel single unit data
        self.unitView = pqi.LFPView(parent=self)
        self.unitView.qscroll.setFixedHeight(int(self.coors.h*0.45))
        self.unitView.qscroll.hide()
        
        self.pgraphLayout.addWidget(self.plotView)
        self.pgraphLayout.addWidget(self.intanView)
        self.pgraphLayout.addWidget(self.lfpView.qscroll)
        self.pgraphLayout.addWidget(self.unitView.qscroll)
        
        
        
        ### SETTINGS LAYOUT ###
        
        self.settingsWidget = QtWidgets.QFrame()
        self.settingsWidget.setFrameShape(QtWidgets.QFrame.Box)
        self.settingsWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.settingsWidget.setLineWidth(4)
        self.settingsWidget.setMidLineWidth(2)
        self.settingsLayout = QtWidgets.QVBoxLayout(self.settingsWidget)
        # set fonts
        headerFont = QtGui.QFont()
        headerFont.setPointSize(9)
        headerFont.setBold(True)
        headerFont.setUnderline(True)
        font = QtGui.QFont()
        font.setPointSize(8)
        # set contents margins
        cmargins = QtCore.QMargins(pqi.px_w(5, self.WIDTH),
                                   pqi.px_h(5, self.HEIGHT),
                                   pqi.px_w(5, self.WIDTH),
                                   pqi.px_h(5, self.HEIGHT))
        # get set of pixel widths and heights, standardized by monitor dimensions
        titleHeight = pqi.px_h(15, self.HEIGHT)
        wspace1, wspace5, wspace10, wspace15, wspace20, wspace50 = [pqi.px_w(w, self.WIDTH) for w in [1,5,10,15,20,50]]
        hspace1, hspace5, hspace10, hspace15, hspace20, hspace50 = [pqi.px_h(h, self.HEIGHT) for h in [1,5,10,15,20,50]]
        
        self.settingsLayout.setContentsMargins(cmargins.left(), 0, 
                                               cmargins.right(), 0)
        self.settingsLayout.setSpacing(hspace10)
        self.settingsWidget.setFixedWidth(self.settings_width)
        
        ### LFP band-pass filtering ###
        self.filterWidget = QtWidgets.QWidget()
        self.filterWidget.setFixedHeight(pqi.px_h(120, self.HEIGHT))
        self.filterLayout = QtWidgets.QVBoxLayout(self.filterWidget)
        self.filterLayout.setContentsMargins(cmargins)
        self.filterLayout.setSpacing(hspace10)
        lay1 = QtWidgets.QVBoxLayout()
        lay1.setContentsMargins(0,0,0,0)
        lay1.setSpacing(hspace20)
        title1 = QtWidgets.QLabel('Band-pass Filter')
        title1.setAlignment(QtCore.Qt.AlignCenter)
        title1.setFixedHeight(titleHeight)
        title1.setFont(headerFont)
        r1 = QtWidgets.QHBoxLayout()  # low freq - hi freq filter boxes
        r1.setSpacing(wspace5)
        r1.setContentsMargins(0,0,0,0)
        self.lfpFreqLo_val = QtWidgets.QDoubleSpinBox()
        self.lfpFreqLo_val.setFont(font)
        self.lfpFreqLo_val.setMinimum(1)
        self.lfpFreqLo_val.setMaximum(499)
        self.lfpFreqLo_val.setDecimals(1)
        self.lfpFreqLo_val.setSingleStep(0.5)
        self.lfpFreqLo_val.setSuffix(' Hz')
        lfpFreq_dash = QtWidgets.QLabel('-')
        lfpFreq_dash.setAlignment(QtCore.Qt.AlignCenter)
        self.lfpFreqHi_val = QtWidgets.QDoubleSpinBox()
        self.lfpFreqHi_val.setFont(font)
        self.lfpFreqHi_val.setMinimum(2)
        self.lfpFreqHi_val.setMaximum(500)
        self.lfpFreqHi_val.setDecimals(1)
        self.lfpFreqHi_val.setSingleStep(0.5)
        self.lfpFreqHi_val.setSuffix(' Hz')
        r1.addWidget(self.lfpFreqLo_val, stretch=2)
        r1.addWidget(lfpFreq_dash, stretch=0)
        r1.addWidget(self.lfpFreqHi_val, stretch=2)
        r2 = QtWidgets.QHBoxLayout()  # 'all signals' checkbox and 'go' button
        r2.setSpacing(wspace5)
        r2.setContentsMargins(0,0,0,0)
        self.filtAll_chk = QtWidgets.QCheckBox('All')
        self.filtAll_chk.setFont(font)
        self.filtGo_btn = QtWidgets.QPushButton('Filter')
        self.filtGo_btn.setFont(font)
        self.filtGo_btn.setDefault(True)
        self.filtGo_btn.setEnabled(False)
        r2.addSpacing(wspace5)
        r2.addWidget(self.filtAll_chk)
        r2.addWidget(self.filtGo_btn)
        lay1.addLayout(r1)
        lay1.addLayout(r2)
        self.filterLayout.addWidget(title1, stretch=0)
        self.filterLayout.addLayout(lay1, stretch=2)
        line_1 = pqi.vline('h')
        self.settingsLayout.addWidget(self.filterWidget)
        self.settingsLayout.addWidget(line_1)
        
        ### P-wave detection threshold ###
        self.thresWidget = QtWidgets.QWidget()
        self.thresWidget.setFixedHeight(pqi.px_h(160, self.HEIGHT))
        self.thresLayout = QtWidgets.QVBoxLayout(self.thresWidget)
        self.thresLayout.setContentsMargins(cmargins)
        self.thresLayout.setSpacing(hspace10)
        lay2 = QtWidgets.QVBoxLayout()
        lay2.setContentsMargins(0,0,0,0)
        lay2.setSpacing(hspace10)
        title2 = QtWidgets.QLabel('P-wave Threshold')
        title2.setAlignment(QtCore.Qt.AlignCenter)
        title2.setFixedHeight(titleHeight)
        title2.setFont(headerFont)
        r1 = QtWidgets.QVBoxLayout()  # P-wave threshold dropdown & spinbox
        r1.setSpacing(wspace1)
        r1.setContentsMargins(0,0,0,0)
        self.thresType = QtWidgets.QComboBox()
        self.thresType.setFont(font)
        self.thresType.addItems(['Raw value', 'Std. deviations', 'Percentile'])
        self.thres_val = QtWidgets.QDoubleSpinBox()
        self.thres_val.setFont(font)
        self.thres_val.setMaximum(1000)
        self.thres_val.setDecimals(1)
        self.thres_val.setSingleStep(0.1)
        r1.addWidget(self.thresType)
        r1.addWidget(self.thres_val)
        r2 = QtWidgets.QHBoxLayout()  # min. separation label & spinbox
        r2.setSpacing(wspace1)
        r2.setContentsMargins(0,0,0,0)
        dupWin_label = QtWidgets.QLabel('Min. separation')
        dupWin_label.setAlignment(QtCore.Qt.AlignCenter)
        dupWin_label.setFont(font)
        dupWin_label.setWordWrap(True)
        self.dupWin_val = QtWidgets.QDoubleSpinBox()
        self.dupWin_val.setFont(font)
        self.dupWin_val.setDecimals(0)
        self.dupWin_val.setSuffix (' ms')
        r2.addWidget(dupWin_label)
        r2.addWidget(self.dupWin_val)
        self.detectPwaves_btn = QtWidgets.QPushButton('Detect P-waves')  # run P-wave detection
        self.detectPwaves_btn.setFont(font)
        self.detectPwaves_btn.setDefault(True)
        self.detectPwaves_btn.setEnabled(False)
        lay2.addLayout(r1)
        lay2.addLayout(r2)
        lay2.addWidget(self.detectPwaves_btn)
        self.thresLayout.addWidget(title2, stretch=0)
        self.thresLayout.addLayout(lay2, stretch=2)
        line_2 = pqi.vline('h')
        self.settingsLayout.addWidget(self.thresWidget)
        self.settingsLayout.addWidget(line_2)
        
        
        ### Plot view ###
        self.viewWidget = QtWidgets.QWidget()
        self.viewLayout = QtWidgets.QVBoxLayout(self.viewWidget)
        self.viewLayout.setContentsMargins(cmargins)
        self.viewLayout.setSpacing(hspace10)
        layX = QtWidgets.QVBoxLayout()
        layX.setContentsMargins(0,0,0,0)
        layX.setSpacing(hspace15)
        titleX = QtWidgets.QLabel('Plot View')
        titleX.setAlignment(QtCore.Qt.AlignCenter)
        titleX.setFixedHeight(titleHeight)
        titleX.setFont(headerFont)
        
        gbox_h = pqi.px_h(55, self.HEIGHT)
        self.npData_box = QtWidgets.QGroupBox('Show LFPs or single units?')   # show Neuropixel LFPs vs single unit data
        self.npData_box.setFont(font)
        self.npData_box.setFixedHeight(gbox_h)
        r1 = QtWidgets.QHBoxLayout(self.npData_box)
        r1.setContentsMargins(wspace5, hspace10, wspace5, hspace10)
        r1.setSpacing(wspace5)
        self.showLFPs_btn = QtWidgets.QRadioButton('LFPs')
        self.showLFPs_btn.setChecked(True)
        self.showUnits_btn = QtWidgets.QRadioButton('Single units')
        self.showUnits_btn.setEnabled(False)
        r1.addWidget(self.showLFPs_btn)
        r1.addWidget(self.showUnits_btn)
        
        # create widget for LFP plot params
        self.lfpParams_widget = QtWidgets.QWidget()
        self.lfpParams_widget.setContentsMargins(0,0,0,0)
        self.lfpParams_lay = QtWidgets.QVBoxLayout(self.lfpParams_widget)
        self.lfpParams_lay.setContentsMargins(0,0,0,0)
        self.lfpParams_lay.setSpacing(hspace15)
        raw_filt_box = QtWidgets.QGroupBox('Raw or filtered LFPs?')  ### toggle raw vs filtered LFP signals 
        raw_filt_box.setFont(font)
        raw_filt_box.setFixedHeight(gbox_h)
        r2a = QtWidgets.QHBoxLayout(raw_filt_box)
        r2a.setSpacing(wspace5)
        self.showRaw_btn = QtWidgets.QRadioButton('Raw')
        self.showRaw_btn.setChecked(True)
        self.showFilt_btn = QtWidgets.QRadioButton('Filtered')
        r2a.addWidget(self.showRaw_btn)
        r2a.addWidget(self.showFilt_btn)
        sig_std_box = QtWidgets.QGroupBox('LFP signal or S.D?')      #### toggle LFP signal vs signal deviation
        sig_std_box.setFont(font)
        sig_std_box.setFixedHeight(gbox_h)
        r3a = QtWidgets.QHBoxLayout(sig_std_box)
        r3a.setSpacing(wspace5)
        self.showSig_btn = QtWidgets.QRadioButton('Signal')
        self.showSig_btn.setChecked(True)
        self.showSD_btn = QtWidgets.QRadioButton('S.D')
        r3a.addWidget(self.showSig_btn)
        r3a.addWidget(self.showSD_btn)
        self.lfpParams_lay.addWidget(raw_filt_box)
        self.lfpParams_lay.addWidget(sig_std_box)
        
        # create widget for single unit plot params
        self.unitParams_widget = QtWidgets.QWidget()
        self.unitParams_widget.setContentsMargins(0,0,0,0)
        self.unitParams_lay = QtWidgets.QVBoxLayout(self.unitParams_widget)
        self.unitParams_lay.setContentsMargins(0,0,0,0)
        self.unitParams_lay.setSpacing(hspace15)
        fr_spikes_box = QtWidgets.QGroupBox('Firing rates or spike trains?')  ### toggle unit firing rates vs spike trains
        fr_spikes_box.setFont(font)
        fr_spikes_box.setFixedHeight(gbox_h)
        r2b = QtWidgets.QHBoxLayout(fr_spikes_box)
        r2b.setSpacing(wspace5)
        self.showTrain_btn = QtWidgets.QRadioButton('Spikes')
        self.showTrain_btn.setEnabled(False)
        self.showFR_btn = QtWidgets.QRadioButton('FR')
        self.showFR_btn.setChecked(True)
        self.showFR_btn.setEnabled(False)
        r2b.addWidget(self.showTrain_btn)
        r2b.addWidget(self.showFR_btn)
        qbox = QtWidgets.QGroupBox('Unit classification?')      ### categorize mice with user notes or auto-classify?
        qbox.setFont(font)
        qbox.setFixedHeight(gbox_h)
        
        self.viewClassifications_btn = QtWidgets.QPushButton(qbox)
        self.viewClassifications_btn.setCheckable(True)
        self.viewClassifications_btn.setChecked(False)
        self.viewClassifications_btn.setStyleSheet('QPushButton'
                                                   '{'
                                                   'border : none;'
                                                   'image : url("icons/hide_icon.png") '
                                                   '}'
                                                   'QPushButton:checked'
                                                   '{image : url("icons/show_icon.png") }')
        self.viewClassifications_btn.setGeometry(int(self.settings_width - wspace50 - wspace10), -hspace1, wspace20, hspace20)
        
        
        r3b = QtWidgets.QHBoxLayout(qbox)
        r3b.setSpacing(wspace5)
        self.useManualClass_btn = QtWidgets.QRadioButton('Manual')
        self.useManualClass_btn.setChecked(True)
        self.useManualClass_btn.setEnabled(False)
        self.useAutoClass_btn = QtWidgets.QRadioButton('Auto')
        self.useAutoClass_btn.setEnabled(False)
        r3b.addWidget(self.useManualClass_btn)
        r3b.addWidget(self.useAutoClass_btn)
        self.unitParams_lay.addWidget(fr_spikes_box)
        self.unitParams_lay.addWidget(qbox)
        self.unitParams_widget.hide()
        
        # control which P-wave/noise data items are plotted
        self.data_view_box = QtWidgets.QGroupBox('Show or hide data?')
        self.data_view_box.setContentsMargins(0,0,0,0)
        self.data_view_box.setFont(font)
        grid = QtWidgets.QGridLayout(self.data_view_box)
        grid.setHorizontalSpacing(wspace5)
        grid.setVerticalSpacing(wspace10)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        colFont = QtGui.QFont()
        colFont.setPointSize(8)
        colFont.setUnderline(True)
        show_label = QtWidgets.QLabel('Show')
        show_label.setAlignment(QtCore.Qt.AlignCenter)
        show_label.setFont(colFont)
        hide_label = QtWidgets.QLabel('Hide')
        hide_label.setAlignment(QtCore.Qt.AlignCenter)
        hide_label.setFont(colFont)
        grid.addWidget(show_label, 0, 1)
        grid.addWidget(hide_label, 0, 2)
        
        # list of data items to show or hide
        self.dataView_btns = {}
        fxdict = {'P-waves'   : lambda state : self.graph_pwaves.pidx_item.setVisible(state),
                  'Threshold' : lambda state : self.graph_pwaves.pthres_item.setVisible(state),
                  'Noise'     : lambda state : self.graph_pwaves.noise_item.setVisible(state),
                  'P-times'   : lambda state : [graph.ptimes_item.setVisible(state) for graph in self.LFP_graphs]}
        
        for i,(name,fx) in enumerate(fxdict.items()): 
            # create data label
            label = QtWidgets.QLabel(name)
            label.setAlignment(QtCore.Qt.AlignCenter)
            # create show/hide checkboxes, initialize as hidden
            show = QtWidgets.QCheckBox()
            hide = QtWidgets.QCheckBox()
            hide.setChecked(True)
            # add checkboxes to button group, store in dictionary
            bgrp = QtWidgets.QButtonGroup(self.data_view_box)
            bgrp.addButton(show)
            bgrp.addButton(hide)
            show.stateChanged.connect(fx)
            if name == 'Threshold':
                fx2 = lambda state : self.graph_pwaves.pthres_item.label.setVisible(state)
                show.stateChanged.connect(fx2)
            self.dataView_btns[name] = bgrp
            # add label and checkboxes to groupbox
            grid.addWidget(label, i+1, 0)
            grid.addWidget(show, i+1, 1, alignment=QtCore.Qt.AlignCenter)
            grid.addWidget(hide, i+1, 2, alignment=QtCore.Qt.AlignCenter)
        
        self.session_data_box = QtWidgets.QGroupBox('Choose session plot')  # choose data item for session graph
        self.session_data_box.setFont(font)
        r4 = QtWidgets.QVBoxLayout(self.session_data_box)
        r4.setSpacing(wspace5)
        self.showEMGAmpl_btn = QtWidgets.QRadioButton('EMG amplitude')
        self.showEMGAmpl_btn.setChecked(True)
        self.showPfreq_btn = QtWidgets.QRadioButton('P-wave frequency')
        self.showNoise_btn = QtWidgets.QRadioButton('Annotated noise')
        self.session_data_bgrp = QtWidgets.QButtonGroup(self.session_data_box)
        self.session_data_bgrp.addButton(self.showEMGAmpl_btn)
        self.session_data_bgrp.addButton(self.showPfreq_btn)
        self.session_data_bgrp.addButton(self.showNoise_btn)
        r4.addWidget(self.showEMGAmpl_btn)
        r4.addWidget(self.showPfreq_btn)
        r4.addWidget(self.showNoise_btn)
        
        # add groupboxes to plot view widget
        layX.addWidget(self.npData_box)
        layX.addWidget(self.lfpParams_widget)
        layX.addWidget(self.unitParams_widget)
        layX.addWidget(self.data_view_box)
        layX.addWidget(self.session_data_box)
        
        self.viewLayout.addWidget(titleX, stretch=0)
        self.viewLayout.addLayout(layX, stretch=2)
        line_X = pqi.vline('h')
        self.settingsLayout.addWidget(self.viewWidget)
        self.settingsLayout.addWidget(line_X)
        
        # channel plots (work in progress)
        chPlots_lay = QtWidgets.QVBoxLayout()
        chPlots_lay.setSpacing(2)
        chPlots_label = QtWidgets.QLabel('Channel Plots')
        chPlots_label.setFont(font)
        self.chPlots_dropdown = QtWidgets.QComboBox()
        self.chPlots_dropdown.setFont(font)
        self.chPlots_dropdown.addItems(['P-wave frequency'])
        self.chPlots_dropdown.setCurrentIndex(-1)
        self.chPlots_dropdown.setPlaceholderText('--Choose plot--')
        chPlots_lay.addWidget(chPlots_label)
        chPlots_lay.addWidget(self.chPlots_dropdown)
        # self.settingsLayout.addLayout(chPlots_lay)
        # line_X2 = pqi.vline('h')
        # self.settingsLayout.addWidget(line_X2)
        
        ### Action buttons ###
        self.btnsWidget = QtWidgets.QWidget()
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Expanding)
        self.btnsWidget.setSizePolicy(policy)
        self.btnsLayout = QtWidgets.QVBoxLayout(self.btnsWidget)
        self.btnsLayout.setSpacing(hspace5)
        self.annotMode_btn = QtWidgets.QPushButton('Bulk Annotation')
        self.annotMode_btn.setFont(font)
        self.annotMode_btn.setCheckable(True)
        self.annotSave_btn = QtWidgets.QPushButton('Save annotation')
        self.annotSave_btn.setFont(font)
        self.dbg = QtWidgets.QPushButton('debug')
        self.dbg.setFont(font)
        self.btnsLayout.addWidget(self.annotMode_btn)
        self.btnsLayout.addWidget(self.annotSave_btn)
        self.btnsLayout.addWidget(self.dbg)
        self.settingsLayout.addWidget(self.btnsWidget)
        
        self.centralLayout.addWidget(self.pgraphWidget)
        self.centralLayout.addWidget(self.settingsWidget)
        
        self.setCentralWidget(self.centralWidget)
    
    
    def _createMenuBar(self):
        """
        ***Work in progress
        """
        menuBar = self.menuBar()
        
        viewMenu = QtWidgets.QMenu('View', self)
        
        analyzeMenu = QtWidgets.QMenu('Analyze', self)
        # classify single units
        self.classifyUnits_submenu = analyzeMenu.addMenu('Classify units')
        self.classifyUnits_brstate = QtWidgets.QAction('State-dependent activity', self)
        self.classifyUnits_pwaves = QtWidgets.QAction('P-wave-dependent activity', self)
        self.classifyUnits_submenu.addAction(self.classifyUnits_brstate)
        self.classifyUnits_submenu.addAction(self.classifyUnits_pwaves)
        # filter single units (e.g. by brain state FR, P-wave FR, peak FR, etc)
        self.filterUnits = QtWidgets.QAction('Filter units', self)
        # initially disable single unit options
        self.classifyUnits_submenu.setEnabled(False)
        self.filterUnits.setEnabled(False)
        analyzeMenu.addAction(self.filterUnits)
        
        menuBar.addMenu(viewMenu)
        menuBar.addMenu(analyzeMenu)
        
        
    
    
    def connect_buttons(self):
        """
        Connect widgets to data management/analysis functions
        """
        # connect LFP filtering buttons
        self.lfpFreqLo_val.valueChanged.connect(self.update_filter_params)
        self.lfpFreqHi_val.valueChanged.connect(self.update_filter_params)
        self.filtAll_chk.toggled.connect(self.select_all_lfps)
        self.filtGo_btn.clicked.connect(self.bp_filter_lfps)
        
        # connect LFP thresholding buttons
        self.thresType.currentTextChanged.connect(self.update_thres_params)
        self.thres_val.valueChanged.connect(self.update_thres_params)
        self.dupWin_val.valueChanged.connect(self.update_thres_params)
        self.detectPwaves_btn.clicked.connect(self.detect_pwaves)
        
        # connect plot view buttons
        self.showLFPs_btn.toggled.connect(self.switch_np_data)
        self.showSD_btn.toggled.connect(self.switch_lfp_plotmode)
        self.showFR_btn.toggled.connect(self.switch_np_plotmode)
        self.showRaw_btn.toggled.connect(self.switch_lfp_data)
        self.viewClassifications_btn.toggled.connect(self.view_unit_classification)
        self.useAutoClass_btn.toggled.connect(self.switch_classification_file)
        self.session_data_bgrp.buttonToggled.connect(self.switch_session_data)
        
        # connect menu bar actions (work in progress)
        self.classifyUnits_brstate.triggered.connect(self.state_classification_window)
        #self.classifyUnits_brstate.triggered.connect(self.classify_units_brstate)
        self.classifyUnits_pwaves.triggered.connect(self.classify_units_pwaves)
        self.filterUnits.triggered.connect(self.filt_units)
        
        #self.chPlots_dropdown.currentTextChanged.connect(self.plot_channel_data)
        
        # connect action buttons
        #self.annotMode_btn.toggled.connect(self.switch_annotation_mode)
        self.annotSave_btn.clicked.connect(self.save_annot)
        self.dbg.clicked.connect(self.debug)
    
    
    ##################         UPDATING FUNCTIONS         ##################
    
    def view_unit_classification(self):
        for graph in self.unit_graphs:
            graph.class_label.setVisible(self.viewClassifications_btn.isChecked())
    
    
    def index_range(self):
        """
        Return indices collected for bulk brain state annotation
        """
        if len(self.index_list) == 1:
            return self.index_list        
        a = self.index_list[0]
        b = self.index_list[-1]        
        if a<=b:
            return list(range(a,b+1))
        else:
            return list(range(b,a+1))
        
        
    def select_all_lfps(self):
        """
        Set all Neuropixel LFP graphs as "selected"
        """
        select = bool(self.filtAll_chk.isChecked())
        for graph in self.LFP_graphs:
            graph.isSelected = select
            graph.update()
        self.enable_filtering()
    
    
    def enable_filtering(self):
        """
        Enable band-pass filtering of LFPs ONLY IF:
            1) Highest freq in bandpass filter is greater than the lowest filter freq
            2) One or more LFPs is selected for filtering
        """
        x1 = self.f1 <= self.f0
        x2 = not any([graph.isSelected for graph in self.LFP_graphs])
        self.filtGo_btn.setDisabled(any([x1,x2]))
    
    
    def bp_filter_lfps(self):
        """
        Band-pass filter all selected LFPs
        """
        self.setWindowTitle('Filtering selected LFPs ...')
        # get currently selected graphs, calculate filter cutoffs (0 to 1)
        sel_graphs = [graph for graph in self.LFP_graphs if graph.isSelected]
        w0, w1 = [float(f / (self.sr_np/2)) for f in [self.f0, self.f1]]
        for graph in sel_graphs:
            # filter raw data, store in second row of LFP array
            dset = graph.LFP[graph.pointer]
            data_filt = sleepy.my_bpfilter(dset[0,:], w0=w0, w1=w1, N=4)
            dset[1,:] = data_filt
            # calculate downsampled standard deviation
            sd_dn = pqi.downsample_sd(data_filt, length=self.num_bins, nbin=self.fbin_np)
            graph.SD[graph.pointer][1,:] = sd_dn
            # update dataset attribute, graph frequency bounds
            dset.attrs['signals'] = ['raw', f'bp_{self.f0}_{self.f1}']
            graph.f0, graph.f1 = float(self.f0), float(self.f1)
            graph.updated_data()
        
        # plot filtered signal for all graphs
        if self.showFilt_btn.isChecked():
            self.switch_lfp_data(rawShow=False)
        else:
            self.showFilt_btn.setChecked(True)
        
        # deselect all graphs
        self.filtAll_chk.setChecked(False)
        self.select_all_lfps()
        # save filtering settings
        self.save_pdata()
            
        self.setWindowTitle('Filtering selected LFPs ... Done!')
        time.sleep(1)
        self.setWindowTitle(self.name)
            
    
    def update_filter_params(self):
        """
        Update LFP band-pass filtering paramseters from user input
        """
        # update lowest and highest frequencies to use for band-pass filtering
        self.f0 = float(self.lfpFreqLo_val.value())
        self.f1 = float(self.lfpFreqHi_val.value())
    
    
    def update_thres_params(self):
        """
        Update LFP thresholding parameters from user input
        """
        # update threshold value
        self.thres = float(self.thres_val.value())
        
        # update thresholding method
        i = self.thresType.currentIndex()
        d = [['raw',' uV',1000,0,10],['std',' s.t.d',50,1,0.1],['perc',' %',100,1,1]][i]
        self.thres_type, suffix, maxval, decimals, singlestep = d
        # adjust settings for spinbox
        self.thres_val.setSuffix(suffix)
        self.thres_val.setMaximum(maxval)
        self.thres_val.setDecimals(decimals)
        self.thres_val.setSingleStep(singlestep)
        
        # update duplicate window
        self.dup_win = float(self.dupWin_val.value())
    
    
    def dict_from_vars(self):
        """
        Collect all current parameter values and return as dictionary
        """
        ddict = {'f0' : float(self.f0),
                 'f1' : float(self.f1),
                 'thres' : float(self.thres),
                 'thres_type' : str(self.thres_type),
                 'dup_win' : float(self.dup_win)}
        return ddict
    
    
    def update_vars_from_dict(self, ddict={}):
        """
        Update all parameter variables from the inputted dictionary $ddict
        """
        try:
            self.f0 = ddict['f0']
            self.f1 = ddict['f1']
            self.thres = ddict['thres']
            self.thres_type = ddict['thres_type']
            self.dup_win = ddict['dup_win']
        except KeyError:
            print('### ERROR: One or more params missing from settings dictionary; unable to update variables ###')
    
    
    def update_gui_from_vars(self):
        """
        Update all GUI widgets from current parameter variable values
        """
        ddict = self.dict_from_vars()
        # set LFP filtering params
        self.lfpFreqLo_val.setValue(ddict['f0'])
        self.lfpFreqHi_val.setValue(ddict['f1'])
        
        # set LFP thresholding params
        i = ['raw','std','perc'].index(ddict['thres_type'])
        self.thresType.setCurrentIndex(i)
        self.thres_val.setValue(ddict['thres'])
        self.dupWin_val.setValue(ddict['dup_win'])
    
    
    def update_vars_from_gui(self):
        """
        Update all parameter variables from user input
        """
        self.update_filter_params()
        self.update_thres_params()
    
    
    def save_pdata(self):
        """
        Save LFP filtering/P-wave detection parameters in p_idx.mat
        """
        # save filtering params
        if self.sender() == self.filtGo_btn:
            self.pfile.attrs['f0'] = float(self.f0)
            self.pfile.attrs['f1'] = float(self.f1)
        # save P-wave detection params
        elif self.sender() == self.detectPwaves_btn:
            self.pfile.attrs['thres'] = float(self.thres)
            self.pfile.attrs['thres_type'] = str(self.thres_type)
            self.pfile.attrs['dup_win'] = float(self.dup_win)
            self.p_train.attrs['p_thr'] = float(self.p_thr)
    
    
    def save_annot(self):
        """
        Save brain state annotation
        """
        self.setWindowTitle('Annotation saved!')
        pqi.rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
        time.sleep(1)
        self.setWindowTitle(self.name)
    
    
    def edit_unit_note(self): 
        """
        Create dialog window to edit notes for a Neuropixel single unit
        """
        # identify unit to write note
        graph = self.sender().parent()
        u = int(graph.curUnit)
        
        # create dialog window
        editWin = QtWidgets.QDialog()
        editWin.setWindowTitle(str(u) + '_' + str(graph.name))
        lay = QtWidgets.QVBoxLayout(editWin)
        lay.setSpacing(pqi.px_h(25, self.HEIGHT))
        hbox_widgets = QtWidgets.QHBoxLayout()
        hbox_widgets.setSpacing(pqi.px_w(10, self.WIDTH))
        # dropdown for unit brain state activity
        vbox1 = QtWidgets.QVBoxLayout()
        vbox1.setSpacing(1)
        state_label = QtWidgets.QLabel('State')
        state_dropdown = QtWidgets.QComboBox()
        state_items = ['-','R-on','R-max','R-off','RW','W-max','W-min','N-max','X']
        state_dropdown.addItems(state_items)
        vbox1.addWidget(state_label)
        vbox1.addWidget(state_dropdown)
        # dropdown for unit P-wave-triggered activity
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.setSpacing(1)
        p_label = QtWidgets.QLabel('P-wave')
        p_dropdown = QtWidgets.QComboBox()
        p_items = ['-','P+','P-','P0']
        p_dropdown.addItems(p_items)
        vbox2.addWidget(p_label)
        vbox2.addWidget(p_dropdown)
        # text input for any other comments about unit
        vbox3 = QtWidgets.QVBoxLayout()
        vbox3.setSpacing(1)
        comment_label = QtWidgets.QLabel('Comments')
        comment_lineedit = QtWidgets.QLineEdit()
        vbox3.addWidget(comment_label)
        vbox3.addWidget(comment_lineedit)
        
        # initialize dialog window with current unit notes
        if u not in self.unotes.keys():
            self.unotes[u] = ['-','-','-']
        state_note = self.unotes[u][0]
        if state_note not in state_items:
            state_dropdown.addItem(state_note)
        state_dropdown.setCurrentText(state_note)
        p_note = self.unotes[u][1]
        if p_note not in p_items:
            p_dropdown.addItem(p_note)
        p_dropdown.setCurrentText(p_note)
        comment_note = self.unotes[u][2]
        comment_lineedit.setText(comment_note)
        hbox_widgets.addLayout(vbox1)
        hbox_widgets.addLayout(vbox2)
        hbox_widgets.addLayout(vbox3)
        # add buttons to save unit note or cancel edit
        hbox_btns = QtWidgets.QHBoxLayout()
        hbox_btns.setSpacing(5)
        save_btn = QtWidgets.QPushButton('Save')
        save_btn.setDefault(True)
        save_btn.clicked.connect(editWin.accept)
        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.clicked.connect(editWin.reject)
        hbox_btns.addWidget(save_btn)
        hbox_btns.addWidget(cancel_btn)
        lay.addLayout(hbox_widgets, stretch=2)
        lay.addLayout(hbox_btns, stretch=0)
        
        # execute dialog window, save user-edited note if 'Save' is pressed
        res = editWin.exec()
        if res:
            new_state_note = state_dropdown.currentText()
            new_p_note = p_dropdown.currentText()
            new_comment = comment_lineedit.text()
            self.unotes[u] = [new_state_note, new_p_note, new_comment]
            pqi.save_unit_notes(self.ppath, self.name, self.unotes)
            if self.useManualClass_btn.isChecked():
                graph.class_label.set_info(txt=f'{new_state_note} | {new_p_note}', color=graph.color, size='10pt')
        
    
    #################         DATA ANALYSIS FUNCTIONS         #################
    
    def get_threshold(self, signal, thres, thres_type):
        """
        Calculate P-wave detection threshold for LFP signal
        @Params
        signal - LFP channel containing P-waves
        thres - value (X) used to determine threshold level
        thres_type - method of threshold calculation
                     * 'raw' = X uV, 'std' = mean * X st. deviations, 'perc' = Xth percentile
        @Returns
        p_thr - calculated spike detection threshold
        """
        if thres_type == 'raw':
            p_thr = float(thres)
        elif thres_type == 'std':
            mn = np.nanmean(signal)
            std = np.nanstd(signal)
            p_thr = mn + thres*std
        elif thres_type == 'perc':
            p_thr = np.nanpercentile(signal, thres)
        return p_thr
    
    
    def validate_spikes(self, pi, signal, sr, dup_win):
        """
        Eliminate "duplicate" P-waves, or multiple threshold crossings belonging
        to a single waveform
        @Params
        pi - detected P-wave indices
        signal - LFP signal
        sr - sampling rate (Hz)
        dup_win - min. separation (ms) between distinct P-wave events
        @Returns
        pi - indices of validated P-waves
        """
        nsep = dup_win / 1000. * sr  # min. samples between separate waves
        idups = np.where(np.diff(pi) < nsep)[0].astype('int32')
        if len(idups) == 0:
            return pi
        
        # keep the largest wave from each group of duplicates, eliminate the rest
        ibreaks = np.r_[0, np.where(np.diff(idups) != 1)[0]+1, len(idups)].astype('int32')
        igrps = [pi[range(idups[i],idups[j-1]+2)] for i,j in zip(ibreaks[0:-1], ibreaks[1:])]
        ielim = np.concatenate([np.delete(igrp, np.argmin(signal[igrp])) for igrp in igrps], axis=0)
        pi = np.setdiff1d(pi, ielim)
        return pi
        
    
    def detect_pwaves(self):
        """
        Detect P-waves in selected LFP channel 
        """
        self.setWindowTitle('Detecting P-waves ...')
        # load LFP signal, convert all noisy values to NaNs
        psignal = self.graph_pwaves.LFP[0][0,:]
        psignal_nannoise = self.graph_pwaves.LFP[0][0,:]
        inoise = np.nonzero(self.noise_train[0,:])[0].astype('int32')
        psignal_nannoise[inoise] = np.nan
        
        # calculate detection threshold
        self.p_thr = self.get_threshold(psignal_nannoise, self.thres, self.thres_type)
            
        # get indices of all waveforms crossing the calculated threshold
        pi = pwaves.spike_threshold(psignal, self.p_thr)
        
        # get indices of all "duplicate" waves that closely precede the subsequent wave
        pi = self.validate_spikes(pi, psignal, self.sr_np, self.dup_win)
        
        # update p_train dataset (0=no P-wave, 1=P-wave, -1=P-wave during noise)
        self.p_train[0,:] = 0
        self.p_train[0,pi] = 1
        self.p_train[0,inoise] *= -1
        
        # update P-wave frequency, update plot
        self.update_pwave_data(updatePlot=True)
        
        # update detection threshold in plot
        txt = pqi.pthres_txt(self.p_thr, self.thres, self.thres_type)
        self.graph_pwaves.pthres_item.setValue(-self.p_thr)
        self.graph_pwaves.pthres_item.label.setFormat(txt)
        
        # save detection settings
        self.save_pdata()
        
        self.setWindowTitle('Detecting P-waves ... Done!')
        time.sleep(1)
        self.setWindowTitle(self.name)
    
    
    def get_units_PFR(self, region=None, istate=1, win=1, dn=10, ma_thr=20, ma_state=3, 
                      flatten_is=4, ignore_filter=True):
        """
        Get the mean P-wave-triggered firing rate for each Neuropixel unit
        @Params
        region - optional argument to analyze units only in a specific region
        istate - state(s) for which to collect P-waves
        win - time window (s) to collect pre and post-P-wave
        dn - downsample single unit firing rate (~2500 Hz) by X bins
        ma_thr, ma_state - max duration and brain state for microarousals
        flatten_is - brain state for transition sleep
        ignore_filter - if True, get data for all units regardless of the filter settings
        @Returns
        PFR_mx - matrix (units x time bins) of firing rates surrounding P-waves
        t - vector of timepoints (s), corresponding to columns in PFR_mx
        unit_list - unit IDs, corresponding to rows in PFR_mx
        region_list - list with regions corresponding to each unit 
        """
        if not istate:  # []/0/False/None/etc --> []
            istate = []
        elif isinstance(istate, int):  # 1 --> [1]
            istate = [istate]
        
        # get units from one or all regions
        graph_list = [g for g in self.unit_graphs if g.name == region]
        if len(graph_list) == 0:
            graph_list = list(self.unit_graphs)
        unit_list = np.concatenate([list(g.units) for g in graph_list])
        region_list = np.concatenate([[g.name]*len(g.units) for g in graph_list])
        if ignore_filter == False:
            hidden_unit_list = np.concatenate([list(g.hiddenUnits) for g in graph_list])
            a,b = zip(*[[u,r] for u,r in zip(unit_list,region_list) if u not in hidden_unit_list])
            unit_list,region_list = [list(a), list(b)]
        
        # get P-wave indices in Neuropixel time, downsample to Intan time
        p_idx = np.where(self.p_train[0,:] == 1)[0].astype('int32')
        M = AS.adjust_brainstate(self.M.flatten(), 2.5, ma_thr=ma_thr, ma_state=ma_state, flatten_is=flatten_is) 
        if istate:
            p_idx = np.array([i for i in p_idx if M[int(i/self.fbin_np)] in istate])
        nbin = self.dt * self.sr_np
        p_idx_dn = np.round(np.divide(p_idx, nbin)).astype('int32')
        iwin_intan = int(round(win * self.sr))  # number of spike train samples (1000 Hz) in $win
        iwin_np = int(round(win * self.sr_np))  # number of LFP samples (2500 Hz) in $win
        
        DATA = []
        
        for unit in unit_list:
            k = str(unit) + '_good'
            if k not in self.units_npz.keys():
                continue
            # get spikes x timepoints matrix
            udata = np.array(self.units_npz[k], dtype='int8')
            unit_mx = np.vstack([udata[pidn-iwin_intan : pidn+iwin_intan] for pidn in p_idx_dn \
                                 if pidn > iwin_intan and pidn < len(udata)-iwin_intan-1])
            # downsample to calculate frequency
            j = int(np.floor(unit_mx.shape[1] / dn) * dn)
            col_grps = np.hsplit(unit_mx[:,0:j], int(j/dn))
            unit_mx_freq = np.vstack([np.divide(x.sum(axis=1),x.shape[1]) for x in col_grps]).T  # get firing rate in downsampled bins
            if j < unit_mx.shape[1]:
                end = np.divide(unit_mx[:,j:].sum(axis=1), unit_mx.shape[1] - j)
                unit_mx_freq = np.hstack([unit_mx_freq, np.atleast_2d(end).T])
            # get mean unit activity, save to data list
            unit_freq = np.mean(unit_mx_freq, axis=0) * 1000
            DATA.append(unit_freq)
        
        # create matrix of units x time bins
        PFR_mx = np.vstack(DATA)
        t = np.linspace(-np.abs(win), np.abs(win), PFR_mx.shape[1])
        
        return PFR_mx, t, unit_list, region_list
    
    
    def state_classification_window(self):
        print('state dialog called')
        x = pqi.UnitClassificationWindow(parent=self, cat='state')
        x.show()
        x.exec()
    
    
    def classify_unit_brstate(self, data, M):
        # get firing rate in each state
        fr_rem, fr_wake, fr_nrem = [data[np.where(M.flatten()==s)[0]] for s in [1,2,3]]
        
        # compare unit firing rates between states
        RW_ratio = np.nanmean(fr_rem) / np.nanmean(fr_wake)
        RN_ratio = np.nanmean(fr_rem) / np.nanmean(fr_nrem)
        WN_ratio = np.nanmean(fr_wake) / np.nanmean(fr_nrem)
        
        # classify unit
        UNIT_CLASS = '-'
        if RW_ratio >= 2:
            if RN_ratio >= 2:
                # REM > wake ~= NREM
                if WN_ratio >= 0.5 and WN_ratio < 2:
                    UNIT_CLASS = 'R-on'
                # REM > wake > NREM
                elif WN_ratio >= 2:
                    UNIT_CLASS = 'R-max'
                # REM > NREM > wake
                elif WN_ratio < 0.5:
                    UNIT_CLASS = 'W-min'
            # REM ~= NREM > wake
            elif RN_ratio >= 0.5 and RN_ratio < 2:
                UNIT_CLASS = 'W-min'
            # NREM > REM > wake
            elif RN_ratio < 0.5:
                UNIT_CLASS = 'N-max'
        elif RW_ratio >= 0.5 and RW_ratio < 2:
            # REM ~= wake > NREM
            if RN_ratio >= 2 and WN_ratio >= 1.5:
                UNIT_CLASS = 'RW'
            # NREM > REM ~= wake
            elif RN_ratio < 0.5 and WN_ratio < 0.5:
                UNIT_CLASS = 'N-max'
            # NREM ~= REM ~= wake
            else:
                UNIT_CLASS = 'X'
        elif RW_ratio < 0.5:
            # NREM/wake > REM
            if RN_ratio < 0.5:
                UNIT_CLASS = 'R-off'
            # wake > REM/NREM
            else:
                UNIT_CLASS = 'W-max'
        # R_mean, W_mean, N_mean, RW_mean, RN_mean, WN_mean, R_peak, W_peak, N_peak, RW_peak, RN_peak, WN_peak
        data_dict = {'R_mean' : np.nanmean(fr_rem), 'W_mean' : np.nanmean(fr_wake), 'N_mean' : np.nanmean(fr_nrem),
                     'R_peak' : np.nanmax(fr_rem), 'W_peak' : np.nanmax(fr_wake), 'N_peak' : np.nanmax(fr_nrem),
                     'RW_mean' : RW_ratio, 'RN_mean' : RN_ratio, 'WN_mean' : WN_ratio}
        
        return UNIT_CLASS, data_dict
        
    
    def classify_units_brstate(self):
        """
        Classify Neuropixel single units based on firing rate across brain states
        """
        self.setWindowTitle('Classifying single units by brain state activity ...')
        #unit_FRs = []
        #unit_classifications = []
        #DATA = []
        self.unit_state_df = pd.DataFrame()
        for graph in self.unit_graphs:
            for i,unit in enumerate(graph.units):
                #UNIT_CLASS = '-'
                data = graph.fr_mx[i,:]
                
                
                UNIT_CLASS, ddict = self.classify_unit_brstate(data, self.M)
                # R_mean, W_mean, N_mean, RW_mean, RN_mean, WN_mean, R_peak, W_peak, N_peak, RW_peak, RN_peak, WN_peak
                try:
                    ddf = pd.DataFrame(ddict, index=[0])
                except:
                    pdb.set_trace()
                ddf.insert(0, 'state_class', UNIT_CLASS)
                ddf.insert(0, 'region', graph.name)
                ddf.insert(0, 'unit', unit)
                
                # save unit state FR data
                self.unit_state_df = pd.concat([self.unit_state_df, ddf], ignore_index=True)
                #DATA.append([unit, graph.name, UNIT_CLASS, RW_ratio, RN_ratio, WN_ratio])
                
                #R_mean, W_mean, N_mean = [ddict[k] for k in ['R_mean','W_mean','N_mean']]
                #RW_mean, RN_mean, WN_mean = [ddict[k] for k in ['RW_mean','RN_mean','WN_mean']]
                
                # make sure unit is in $auto_unotes, update brain state classification
                self.auto_unotes[unit] = self.auto_unotes.get(unit, ['-','-','-'])
                self.auto_unotes[unit][0] = UNIT_CLASS
                
                
            # if showing auto-classifications, update label of current unit in each graph
            if self.useAutoClass_btn.isChecked():
                notes = self.auto_unotes.get(graph.curUnit, ['-','-','-'])
                graph.class_label.set_info(txt=f'{notes[0]} | {notes[1]}', color=graph.color, size='10pt')
        # save updated file
        pqi.save_unit_notes(self.ppath, self.name, self.auto_unotes, auto=True)
        
        # save unit brainstate data
        #cols = ['unit', 'region', 'state_class', 'R','W','N','RW_ratio', 'RN_ratio', 'WN_ratio']
        #self.unit_state_df = pd.DataFrame(DATA, columns=cols)
        self.unit_state_df.to_csv(os.path.join(self.ppath, self.name, 'unit_state_df'), index=False)
                # unit_classifications.append([unit, graph.name, UNIT_CLASS, data.mean(), data[np.argsort(data)[-2]],
                #                              fr_rem.mean(), fr_wake.mean(), fr_nrem.mean(), RW_ratio, RN_ratio, WN_ratio])
        # df = pd.DataFrame(unit_classifications, columns=['unit','region','state_activity','mean_FR','peak_FR','REM_FR',
        #                                                  'wake_FR','NREM_FR','RW_ratio','RN_ratio','WN_ratio'])
        # csv_path = os.path.join(self.ppath, self.name, 'units_state_classification')
        # df.to_csv(csv_path, index=False)
        self.setWindowTitle('Classifying single units by brain state activity ... Done!')
        time.sleep(1)
        self.setWindowTitle(self.name)
        
    
    def classify_units_pwaves(self, istate=1, win=1, dn=10, ma_thr=20, ma_state=3,
                              flatten_is=4, pload=False, psave=False):
        """
        Classify Neuropixel single units based on firing rate in relation to P-waves
        @Params
        istate - state(s) in which to analyze P-waves
        win - time window (s) to collect pre and post-P-wave
        dn - downsample single unit firing rate (~2500 Hz) by X bins
        ma_thr, ma_state - max duration and brain state for microarousals
        flatten_is - brain state for transition sleep
        pload - optionally classify units using FRs loaded from a saved data file
        """
        self.setWindowTitle('Classifying single units by P-wave-related activity ...')
        if not istate:  # []/0/False/None/etc --> []
            istate = []
        elif isinstance(istate, int):  # 1 --> [1]
            istate = [istate]
        
        if pload:
            try:
                # load matrix of firing rates (units x time) from recording folder
                filename = pload.replace('.mat','') if isinstance(pload,str) else f'units_PFR_data'
                if istate:
                    filename += '_' + ''.join([str(s) for s in istate])
                data = so.loadmat(os.path.join(self.ppath, self.name, filename + '.mat'), squeeze_me=True)
                PFR_mx = data['PFR_mx']
                t = data['t']
                unit_list = data['units']
                region_list = data['regions']
            except:
                print('\nUnable to load .mat file - calculating new firing rates ...\n')
                pload = False
        
        if not pload:
            data = self.get_units_PFR(region=None, istate=istate, win=win, dn=dn, ma_thr=ma_thr, 
                                      ma_state=ma_state, flatten_is=flatten_is, ignore_filter=True)
            PFR_mx, t, unit_list, region_list = data
        
        DATA = []
        for i,unit in enumerate(unit_list):
            # get unit PFR time course, z-score for comparison with other units
            unit_freq = PFR_mx[i,:]
            zfreq = scipy.stats.zscore(unit_freq)
            
            # find peaks in mean firing rate (min. 3 SD above/below the mean)
            ipks_up, ddict_up = scipy.signal.find_peaks(zfreq, height=3, prominence=3)  # increased FR
            ipks_dn, ddict_dn = scipy.signal.find_peaks(-zfreq, height=3, prominence=3) # decreased FR
            
            # no substantial peaks/valleys in firing rate
            if len(ipks_up) == 0 and len(ipks_dn) == 0:
                UNIT_CLASS = 'P0'
            # FR peak is upwards (increased firing)
            elif len(ipks_up) > 0 and len(ipks_dn) == 0:
                ipks, ddict = ipks_up, ddict_up
                UNIT_CLASS = 'P+'
            # FR peak is downwards (decreased firing)
            elif len(ipks_up) == 0 and len(ipks_dn) > 0:
                ipks, ddict = ipks_dn, ddict_dn
                UNIT_CLASS = 'P-'
            else:
                if np.max(ddict_up['peak_heights']) >= np.max(ddict_dn['peak_heights']):
                    ipks, ddict = ipks_up, ddict_up
                    UNIT_CLASS = 'P+'
                else:
                    ipks, ddict = ipks_dn, ddict_dn
                    UNIT_CLASS = 'P-'
                
            # make sure unit is in $auto_unotes, update P-wave classification
            self.auto_unotes[unit] = self.auto_unotes.get(unit, ['-','-','-'])
            self.auto_unotes[unit][1] = UNIT_CLASS
            
            # collect values describing the shape/size of the PFR peak
            try:
                if UNIT_CLASS in ['P+','P-']:
                    ii = np.argmax(ddict['peak_heights']); ipk = ipks[ii]
                    zheight, prominence = [ddict[k][ii] for k in ['peak_heights','prominences']]
                    width = np.abs(t[ddict['left_bases'][ii]] - t[ddict['right_bases'][ii]])
                    peak_PFR, peak_t = [unit_freq[ipk], t[ipk]]
                    change_PFR = np.abs(unit_freq.mean() - peak_PFR)
                else:
                    zheight = prominence = width = peak_PFR = peak_t = change_PFR = np.nan
            except:
                pdb.set_trace()
            # save unit PFR data
            DATA.append([unit, region_list[i], UNIT_CLASS, peak_t, width, zheight, 
                         prominence, peak_PFR, change_PFR])
            
        # if showing auto-classifications, update label of current unit for each graph
        if self.useAutoClass_btn.isChecked():
            notes_list = [self.auto_unotes.get(g.curUnit, ['-','-','-']) for g in self.unit_graphs]
            _ = [g.class_label.set_info(txt=f'{n[0]} | {n[1]}', color=g.color, size='10pt') for n,g in zip(notes_list, self.unit_graphs)]
        
        # save updated unit classification file
        pqi.save_unit_notes(self.ppath, self.name, self.auto_unotes, auto=True)
        
        # save updated unit PFR data
        cols = ['unit', 'region', 'PFR_class', 'peak_t', 'peak_width', 'peak_zheight', 'peak_prominence', 'peak_PFR', 'change_PFR']
        self.unit_PFR_df = pd.DataFrame(DATA, columns=cols)
        self.unit_PFR_df.to_csv(os.path.join(self.ppath, self.name, 'unit_PFR_df'), index=False)
        
        if psave:
            filename = psave.replace('.mat','') if isinstance(psave,str) else f'units_PFR_data'
            if istate:
                filename += '_' + ''.join([str(s) for s in istate])
            
            ppath = os.path.join(self.ppath, self.name, f'{filename}.mat')
            so.savemat(ppath, {'PFR_mx' : np.vstack(unit_frequencies),
                               't' : t,
                               'units' : unit_list,
                               'regions' : region_list})
        
        self.setWindowTitle('Classifying single units by P-wave-related activity ... Done!')
        time.sleep(1)
        self.setWindowTitle(self.name)

    
    ##################           LIVE PLOTTING           ##################
        
        
    def plot_treck(self):     
        """
        Plot overview data, annotated states, and current position in recording
        """
        # clear plot, load annotation history and current index
        self.graph_treck.clear()
        self.graph_treck.plot(self.ftime, self.K*0.5, pen=(150,150,150))
        self.graph_treck.plot(self.ftime, np.zeros((self.ftime.shape[0],)), 
                              pen=pg.mkPen(color=(255,255,255,255), width=10))
        
        # plot currently annotated point
        self.graph_treck.plot([self.ftime[self.index] + 0.5*self.fdt], 
                              [0.0], pen=(0,0,0), symbolPen=(255,0,0), 
                              symbolBrush=(255, 0, 0), symbolSize=5)
        # plot laser overview
        if self.pplot_laser:
            self.graph_treck.plot(self.ftime, self.laser, pen=(0,0,255))
        
        # set axis params
        limits = {'xMin': self.ftime[0], 'xMax': self.ftime[-1]+self.fdt, 
                  'yMin': -1.1, 'yMax': 1.1}
        self.graph_treck.vb.setLimits(**limits)
        self.graph_treck.setXLink(self.graph_spectrum.vb)
    
    
    def plot_brainstate(self):
        """
        Plot color-coded brain states
        """
        # clear plot, load annotation vector
        self.graph_brainstate.clear()
        self.image_brainstate = pg.ImageItem() 
        self.graph_brainstate.addItem(self.image_brainstate)
        self.image_brainstate.setImage(self.M.T)
        # set time scale and color map
        tr = QtGui.QTransform()
        tr.scale(self.fdt, 1)
        self.image_brainstate.setTransform(tr)
        self.image_brainstate.setLookupTable(self.lut_brainstate)
        self.image_brainstate.setLevels([0, 7])
        
        # set axis params
        self.graph_brainstate.vb.setMouseEnabled(x=True, y=False)
        limits = {'xMin': self.ftime[0], 'xMax': self.ftime[-1]+self.fdt, 
                  'yMin': 0, 'yMax': 1}
        self.graph_brainstate.vb.setLimits(**limits)
        self.graph_brainstate.setXLink(self.graph_spectrum.vb)
    
    
    def plot_session(self):
        """
        Plot EEG spectrogram and other FFT data for the recording session
            e.g. EEG spectrogram, EMG amplitude, P-wave frequency
        """
        # set spectrogram for recording
        self.image_spectrum.setImage(self.eeg_spec[0:self.ifreq[-1],:].T)
        # set time scale and color map
        tr = QtGui.QTransform()
        tr.scale(self.fdt, self.fdx)
        self.image_spectrum.setTransform(tr)
        self.image_spectrum.setLookupTable(self.lut_spectrum)
        # set axis params
        limits = {'xMin': self.ftime[0], 'xMax': self.ftime[-1]+self.fdt, 
                  'yMin': 0, 'yMax': self.freq[self.ifreq[-1]]}
        self.graph_spectrum.vb.setLimits(**limits)
        
        self.session_item.setData(self.ftime, self.session_data, padding=None)
        limits = {'xMin': self.ftime[0], 'xMax': self.ftime[-1]+self.fdt}
        self.graph_session.vb.setLimits(**limits)
        self.graph_session.setXLink(self.graph_spectrum.vb)
        
        
    def plot_signals(self):
        """
        Plot local Neuropixel LFP signals/single unit data
        """
        self.tstart = self.timepoint - self.twin/2  # start time (s) of loaded data
        self.tend = self.timepoint + self.twin/2    # end time (s) of loaded data
        tmin = self.timepoint - self.twin_view/2    # start time (s) of viewing window
        tmax = self.timepoint + self.twin_view/2    # end time (s) of viewing window
        
        # update Intan indices
        n = int(np.round((self.twin*self.sr)/2))
        self.iseq_intan = np.arange(self.index_intan-n, self.index_intan+n,   # Intan indices to load
                                    dtype='int32')
        self.tseq_intan = np.linspace(self.tstart, self.tend,                 # timepoints of Intan indices
                                      len(self.iseq_intan), dtype='float32')
        
        # plot Intan signal, set x-limits and x-range
        data = self.Intan[0, self.iseq_intan]
        self.curIntan.setData(self.tseq_intan, data, padding=None)
        pqi.set_xlimits(self.graph_intan, self.tstart, self.tend)
        self.graph_intan.setXRange(tmin, tmax, padding=None)
        
        # update Neuropixel indices
        n2 = int(np.round((self.twin*self.sr_np)/2))
        self.iseq_np = np.arange(self.index_np-n2, self.index_np+n2, dtype='int32')  # Neuropixel indices to load
        self.tseq_np = np.linspace(self.tstart, self.tend, 
                                   len(self.iseq_np), dtype='float32')               # timepoints of Neuropixel indices
        
        # plot local Neuropixel LFPs, set x-limits and x-range
        if self.showLFPs_btn.isChecked() and self.lfp_plotmode == 0:
            for graph in self.LFP_graphs:
                data = graph.LFP[graph.pointer][graph.i,self.iseq_np]
                graph.data_item.setData(self.tseq_np, data, padding=None)
                pqi.set_xlimits(graph, self.tstart, self.tend)
                graph.setXRange(tmin, tmax, padding=None)
                graph.updated_data()
            
        # plot local spike trains, set x-limits and x-range
        elif self.showUnits_btn.isChecked() and self.np_plotmode == 0:
            for graph in self.unit_graphs:
                k = str(graph.curUnit) + '_good'
                if k in self.units_npz.keys():
                    data = self.units_npz[k][self.iseq_intan]
                else:
                    data = np.ones(len(self.tseq_intan)) * -1
                graph.data_item.setData(self.tseq_intan, data, padding=None)
                pqi.set_xlimits(graph, self.tstart, self.tend)
                graph.setXRange(tmin, tmax, padding=None)
        
        # update plot P-waves UNLESS plotting LFP standard deviation
        if not (self.lfp_plotmode == 1 and self.showLFPs_btn.isChecked()):
            if len(self.graph_pwaves.LFP) > 0:
                data = self.graph_pwaves.LFP[0][0,self.iseq_np]
                self.graph_pwaves.data_item.setData(self.tseq_np, data, padding=None)
                pqi.set_xlimits(self.graph_pwaves, self.tstart, self.tend)
                self.graph_pwaves.setXRange(tmin, tmax, padding=None)
                self.graph_pwaves.updated_data()
                
            # plot local indices of noise and P-waves
            self.update_plot_pwaves()
            self.update_plot_noise()
            
    
    def update_pwave_data(self, updatePlot=True):
        """
        Update P-wave-related data (e.g. P-wave frequency) after one of the following actions:
           1) Initial loading in of P-wave indices from saved p_idx.mat file
           2) Running P-wave detection
           3) Annotating a section of the P-wave LFP channel as "noisy" or "clean"
        """
        self.pfreq = np.zeros(self.num_bins, dtype='float32')
        
        # calculate P-wave frequency
        if self.p_train[0,:].any():
            # replace 1's in P-wave train with SR, downsample to FFT bins
            pvec = np.zeros(self.p_train.shape[1], dtype='float32')
            pi = np.where(self.p_train[0,:] == 1)[0].astype('int32')
            pvec[pi] = float(self.sr_np)
            pfreq = sleepy.downsample_vec(pvec, int(self.fbin_np)).astype('float32')
            self.pfreq = np.resize(pfreq, self.num_bins)
            # enable classification of single units by P-wave-triggered activity pattern
            self.classifyUnits_pwaves.setEnabled(True)
        else:
            self.classifyUnits_pwaves.setEnabled(False)  # no P-waves, no classification
        
        if updatePlot:
            self.update_plot_pwaves()
    
    
    def update_plot_pwaves(self):
        """
        Update plotted P-wave items after one of the follwing actions:
           1) Running P-wave detection
           2) Annotating a section of the P-wave LFP channel as "noisy" or "clean"
           3) Loading a new window of data into the local P-wave graph
        """
        # LFP standard deviation is the only config that doesn't require updating plot P-waves
        if not (self.lfp_plotmode == 1 and self.showLFPs_btn.isChecked()):
                
            pi_seq = np.array((), dtype='int32')
            if len(self.graph_pwaves.LFP) > 0:
                pi_seq = np.where(self.p_train[0,self.iseq_np] == 1)[0].astype('int32')
                
            if len(pi_seq) > 0:
                # plot P-wave indices in LFP processed
                x = self.tseq_np[pi_seq]
                y = self.graph_pwaves.LFP[0][0,self.iseq_np][pi_seq] - 30
                self.graph_pwaves.pidx_item.setData(x, y, padding=None)
                
                # plot P-wave times for other LFPs
                if self.showLFPs_btn.isChecked():
                    for graph in self.LFP_graphs:
                        graph.ptimes_item.setData(np.repeat(x,2),
                                                  np.tile(graph.vb.viewRange()[1], len(x)), 
                                                  padding=None)
                        graph.update()
            else:
                self.graph_pwaves.pidx_item.clear()
                for graph in self.LFP_graphs:
                    graph.ptimes_item.clear()
                    graph.update()
            
            self.graph_pwaves.update()
        
        # update P-wave frequency plot
        if self.showPfreq_btn.isChecked():
            self.session_data = self.pfreq
            self.session_item.setData(self.ftime, self.session_data, padding=None)
    
    
    def update_noise_data(self, updatePlot=True):
        """
        Update noise-related data (e.g. downsampled noise indices) after one of the follwing actions:
           1) Initial loading in of noise indices from saved p_idx.mat file
           2) Annotating a section of the P-wave LFP channel as "noisy" or "clean"
        """
        self.noise_train_dn = np.zeros(self.num_bins, dtype='int8')
        # get downsampled noise train
        if self.noise_train[0,:].any():
            ndn = sleepy.downsample_vec(self.noise_train[0,:], int(self.fbin_np))
            ndn[np.nonzero(ndn)[0]] = 1
            self.noise_train_dn = np.resize(ndn, self.num_bins).astype('int8')
            
        if updatePlot:
            self.update_plot_noise()
    
    
    def update_plot_noise(self):
        """
        Update plotted noise items after one of the follwing actions:
           1) Annotating a section of the P-wave LFP channel as "noisy" or "clean"
           2) Loading a new window of data into the local P-wave graph
        """
        inoise = np.array((), dtype='int32')
        
        if len(self.graph_pwaves.LFP) > 0:
            if (self.lfp_plotmode == 1) and self.showLFPs_btn.isChecked():
                # get downsampled indices of noise in session SD
                inoise = np.nonzero(self.noise_train_dn)[0].astype('int32')
                signal = self.graph_pwaves.SD[0][0,:]
                x = np.array(self.ftime, dtype='float32')
            
            else:  # get indices of local noise in LFP signal
                inoise = np.nonzero(self.noise_train[0,self.iseq_np])[0].astype('int32')
                signal = self.graph_pwaves.LFP[0][0,self.iseq_np]
                x = np.array(self.tseq_np, dtype='float32')
            
        # plot noise indices in $graph_pwaves
        if len(inoise) > 0:
            data = np.empty(len(signal), dtype='float32') * np.nan
            data[inoise] = signal[inoise]
            self.graph_pwaves.noise_item.setData(x, data, padding=None)
        else:
            self.graph_pwaves.noise_item.clear()
        self.graph_pwaves.update()
        
        ### update session plot
        if self.showNoise_btn.isChecked():
            self.session_data = self.noise_train_dn
            self.session_item.setData(self.ftime, self.session_data, padding=None)
            
        
    def update_index(self):
        """
        Update live plots when user moves to a different timepoint in the recording
        """
        # update current timepoint, Intan index, and Neuropixel index
        self.timepoint = self.ftime[self.index] + self.fdt
        #self.timepoint = self.ftime[self.index]
        self.index_intan = int(np.round(self.timepoint*self.sr))
        self.index_np = int(np.round(self.timepoint*self.sr_np))
        
        if self.annotation_mode == True:
            self.annotate_only()
            return
        
        # get time range of new viewing window
        tmin = self.timepoint - self.twin_view/2  # start time (s) of viewed data
        tmax = self.timepoint + self.twin_view/2  # end time (s) of viewed data
        
        # update graphs with new data and x-range, or update x-range only
        if tmin < self.tstart or tmax > self.tend:
            try:
                self.plot_signals()
            except:
                pdb.set_trace()
        else:
            self.graph_intan.setXRange(tmin, tmax, padding=None)
            if self.lfp_plotmode == 0:
                self.graph_pwaves.setXRange(tmin, tmax, padding=None)
                for graph in self.LFP_graphs:
                    graph.setXRange(tmin, tmax, padding=None)
                    
        # update index tracker
        self.plot_treck()
        if self.bulk_annot == False:
            if self.pcollect_index == True:
                self.index_list.append(self.index)
            else:
                self.index_list = [self.index]
            
    
    def plot_lfp_sd(self):
        """
        Plot standard deviation (S.D.) of Neuropixel LFP signals across the recording
        """
        if len(self.graph_pwaves.SD) > 0:
            data = self.graph_pwaves.SD[0][0,:]
            self.graph_pwaves.data_item.setData(self.ftime, data, padding=None)
            pqi.set_xlimits(self.graph_pwaves, self.ftime[0], self.ftime[-1], padding=0)
            self.graph_pwaves.updated_data()
            
        for graph in self.LFP_graphs:
            # plot standard deviation for recording session, set x-limits
            data = graph.SD[graph.pointer][graph.i,:]
            graph.data_item.setData(self.ftime, data, padding=None)
            pqi.set_xlimits(graph, self.ftime[0], self.ftime[-1], padding=0)
            graph.updated_data()
        self.update_plot_noise()
    
    
    def plot_unit_fr(self):
        """
        Plot firing rate of Neuropixel single units across the recording
        """
        for graph in self.unit_graphs:
            # plot firing rate for recording session, set x-limits
            data = graph.fr_mx[graph.i,:]
            graph.data_item.setData(self.ftime, data, padding=None)
            pqi.set_xlimits(graph, self.ftime[0], self.ftime[-1], padding=0)
    
    
    def switch_unit(self, bool, graph=None, new_unit=None):
        """
        Switch currently plotted unit for the Neuropixel graph of a given brain region
        """
        if graph is None:
            graph = self.sender().parent()
        if new_unit is None:
            new_unit = int(self.sender().text().strip())
        graph.curUnit = new_unit
        graph.i = graph.units.index(new_unit)
        
        # plot current spike train of new unit
        if self.np_plotmode == 0:
            k = str(new_unit) + '_good'
            if k in self.units_npz.keys():
                data = self.units_npz[k][self.iseq_intan]
            else:
                data = np.ones(len(self.tseq_intan)) * -1
            graph.data_item.setData(self.tseq_intan, data, padding=None)
        
        # plot overall firing rate of new unit, update y-range
        elif self.np_plotmode == 1:
            graph.data_item.setData(self.ftime, graph.fr_mx[graph.i,:], padding=None)
            graph.vb.enableAutoRange(axis='y')
            graph.vb.setAutoVisible(y=True)
        
        # update graph labels
        graph.label.set_info(txt=str(new_unit) + '_' + graph.name, color=graph.color)
        ddict = dict(self.unotes) if self.useManualClass_btn.isChecked() else dict(self.auto_unotes)
        notes = ddict.get(new_unit, ['-','-','-'])
        graph.class_label.set_info(txt=f'{notes[0]} | {notes[1]}', color=graph.color, size='10pt')
        
        
    def filt_units(self):
        """
        Show/hide Neuropixel units based on brain state and/or P-wave-triggered activity
        """
        # set classification filters for units in all regions
        if self.sender() == self.filterUnits:
            filt_dict = dict(self.ufilt)
        # set classification filters by region
        else:
            graph = self.sender().parent()
            filt_dict = self.region_ufilt[graph.name]
            
        # create popup window to change filtering params
        popup = pqi.UnitFiltWindow(filt_dict)
        res = popup.exec()
        if res:
            # update filtering param dictionary
            filt_dict = dict(popup.filt_dict)
            
            if self.useManualClass_btn.isChecked():
                notes_dicts = [dict(self.unotes), dict(self.auto_unotes)]
            else:
                notes_dicts = [dict(self.auto_unotes), dict(self.unotes)]
            # filter all units
            if self.sender() == self.filterUnits:
                self.ufilt = dict(filt_dict)
                for graph in self.unit_graphs:
                    self.region_ufilt[graph.name] = filt_dict
                    graph.hiddenUnits = pqi.find_hidden_units(graph.units, notes_dicts, graph.peak_fr, filt_dict)
                    # if currently plotted unit is filtered out, switch to included unit if possible
                    if graph.curUnit in graph.hiddenUnits and len(graph.units) > len(graph.hiddenUnits):
                        new_unit = [u for u in graph.units if u not in graph.hiddenUnits][0]
                        self.switch_unit(True, graph=graph, new_unit=int(new_unit))
            # filter units in specific region
            else:
                self.region_ufilt[graph.name] = filt_dict
                graph.hiddenUnits = pqi.find_hidden_units(graph.units, notes_dicts, graph.peak_fr, filt_dict)
                if graph.curUnit in graph.hiddenUnits and len(graph.units) > len(graph.hiddenUnits):
                    new_unit = [u for u in graph.units if u not in graph.hiddenUnits][0]
                    self.switch_unit(True, graph=graph, new_unit=int(new_unit))
    
    
    def switch_classification_file(self, auto):
        """
        Switch between classifying units manually ($unit_notes.txt) and automatically
        (Menu -> Analyze -> Classify -> State/P-wave-dependent activity)
        """
        # update unit classification labels
        ddict = dict(self.unotes) if self.useManualClass_btn.isChecked() else dict(self.auto_unotes)
        for graph in self.unit_graphs:
            notes = ddict.get(graph.curUnit, ['-','-','-'])
            graph.class_label.set_info(txt=f'{notes[0]} | {notes[1]}', color=graph.color, size='10pt')
            
    
    def switch_np_data(self, lfpShow):
        """
        Switch Neuropixel plots between LFPs and single units
        """
        # unlink P-wave graph from other items
        self.graph_pwaves.vb.linkView(self.graph_pwaves.vb.XAxis, None)
        
        if lfpShow:
            # show Neuropixel LFP graphs
            self.lfpView.qscroll.show()
            self.lfpParams_widget.show()
            self.unitView.qscroll.hide()
            self.unitParams_widget.hide()
            self.switch_lfp_plotmode(mode=int(self.lfp_plotmode))
        else:
            # show Neuropixel single units
            self.lfpView.qscroll.hide()
            self.lfpParams_widget.hide()
            self.unitView.qscroll.show()
            self.unitParams_widget.show()
            self.switch_np_plotmode(mode=int(self.np_plotmode))
        
        # enable/disable menu options for LFP and single unit analysis
        self.classifyUnits_submenu.setEnabled(not lfpShow)
        self.filterUnits.setEnabled(not lfpShow)
            
    
    def switch_lfp_plotmode(self, mode):
        """
        In Neuropixel LFP plots, show local LFP signal (mode=0) or session LFP standard deviation (mode=1)
        *** Called from GUI: mode = showSD button status (1 = checked, 0 = unchecked)
        ***Called by $switch_np_data: mode = current lfp_plotmode value
        """
        # unlink P-wave graph from other items
        self.graph_pwaves.vb.linkView(self.graph_pwaves.vb.XAxis, None)
        
        if not mode:
            self.lfp_plotmode = 0
            # LOCAL LFP SIGNALS --> link x-axes of LFP graphs, P-wave graph to EEG
            pqi.xlink_graphs(self.LFP_graphs, target_graph=self.graph_intan)
            _ = [graph.getAxis('left').setLabel('LFP (uV)', units='') for graph in self.LFP_graphs]
            self.graph_pwaves.setXLink(self.graph_intan)
            self.graph_pwaves.getAxis('left').setLabel('LFP (uV)', units='')
            self.plot_signals()
            
        if mode:
            self.lfp_plotmode = 1
            # SESSION LFP S.D. --> link x-axes of LFP graphs, P-wave graph to spectrogram
            pqi.xlink_graphs(self.LFP_graphs, target_graph=self.graph_spectrum)
            _ = [graph.getAxis('left').setLabel('S.D.', units='') for graph in self.LFP_graphs]
            self.graph_pwaves.setXLink(self.graph_spectrum)
            self.graph_pwaves.getAxis('left').setLabel('S.D.', units='')
            self.plot_lfp_sd()
            
    
    def switch_np_plotmode(self, mode):
        """
        In Neuropixel single-unit plots, show local unit spike trains (mode=0) or session unit firing rate (mode=1)
        *** Called from GUI: mode = showFR button status (1 = checked, 0 = unchecked)
        ***Called by $switch_np_data: mode = current np_plotmode value
        """
        # unlink P-wave graph from other items
        self.graph_pwaves.vb.linkView(self.graph_pwaves.vb.XAxis, None)
        
        if not mode:
            self.np_plotmode = 0
            # LOCAL SPIKE TRAINS --> link x-axes of unit graphs, P-wave graph to EEG
            pqi.xlink_graphs(self.unit_graphs, target_graph=self.graph_intan)
            for graph in self.unit_graphs:
                graph.getAxis('left').setLabel('Spks', units='')
                graph.setYRange(0, 1.3)
            self.graph_pwaves.setXLink(self.graph_intan)
            self.graph_pwaves.getAxis('left').setLabel('LFP (uV)', units='')
            self.plot_signals()
            
        if mode:
            self.np_plotmode = 1
            # SESSION FIRING RATES --> link x-axes of unit graphs to spectrogram, P-wave graph to EEG
            pqi.xlink_graphs(self.unit_graphs, target_graph=self.graph_spectrum)
            for graph in self.unit_graphs:
                graph.getAxis('left').setLabel('FR', units='')
                graph.vb.enableAutoRange(axis='y')
                graph.vb.setAutoVisible(y=True)
            self.graph_pwaves.setXLink(self.graph_intan)
            self.graph_pwaves.getAxis('left').setLabel('LFP (uV)', units='')
            self.plot_unit_fr()
    
    
    def switch_lfp_data(self, rawShow):
        """
        Switch Neuropixel LFP plots between raw and filtered signals
        """
        if rawShow:
            i = 0
        else:
            i = 1
        
        for graph in self.LFP_graphs:
            graph.i = i
            if self.lfp_plotmode == 0:
                x = np.array(self.tseq_np, dtype='float32')
                data = graph.LFP[graph.pointer][graph.i,self.iseq_np]
            elif self.lfp_plotmode == 1:
                x = np.array(self.ftime, dtype='float32')
                data = graph.SD[graph.pointer][graph.i,:]
            graph.data_item.setData(x, data, padding=None)
            graph.updated_data()
    
    
    def switch_session_data(self):
        """
        Switch session data plot between EMG amplitude, P-wave frequency, and annotated noise
        """
        # show EMG amplitude
        if self.showEMGAmpl_btn.isChecked():
            self.session_data = self.EMGAmpl_list[0]
            self.session_item.setPen((255,255,255),width=1)
            self.graph_session.getAxis(name='left').setLabel('EMG Amp.')
            
        # show P-wave frequency
        elif self.showPfreq_btn.isChecked():
            self.session_data = self.pfreq
            self.session_item.setPen((255,255,0),width=1)
            self.graph_session.getAxis(name='left').setLabel('Pfreq')
        
        # show downsampled noise
        elif self.showNoise_btn.isChecked():
            self.session_data = self.noise_train_dn
            self.session_item.setPen((255,20,147),width=1)
            self.graph_session.getAxis(name='left').setLabel('Noise')
        
        self.session_item.setData(self.ftime, self.session_data, padding=None)
        self.graph_session.vb.enableAutoRange(axis='y')
        self.graph_session.vb.setAutoVisible(y=True)
    
    
    def switch_annotation_mode(self):
        """
        Switch annotation mode ON (efficient brain state scoring) or OFF (view detailed data items)
        """
        # turn off annotation mode
        if self.annotation_mode == True:
            self.annotation_mode = False
            
            # show all LFPs
            self.lfpView.qscroll.setVisible(True)
            
            # enable LFP selection / enable filter button if 1+ LFPs selected
            self.filtAll_chk.setEnabled(True)
            self.enable_filtering()
            
            # enable P-wave detection if LFP_processed has been set
            if len(self.graph_pwaves.LFP) > 0:
                self.detectPwaves_btn.setEnabled(True)
            
            # enable viewing buttons
            self.showRaw_btn.setEnabled(True)
            self.showFilt_btn.setEnabled(True)
            self.showSig_btn.setEnabled(True)
            self.showSD_btn.setEnabled(True)
            
            self.plot_signals()
        
        # turn on annotation mode
        elif self.annotation_mode == False:
            self.annotation_mode = True
            
            self.setWindowTitle('Setting annotation mode ...')
            
            if self.lfp_plotmode == 1:
                self.switch_lfp_plotmode(self, 1)
            
            # hide all non-P-wave LFPs
            self.lfpView.qscroll.setVisible(False)
            
            # disable LFP filtering/P-wave detection buttons
            self.filtAll_chk.setEnabled(False)
            self.filtGo_btn.setEnabled(False)
            self.detectPwaves_btn.setEnabled(False)
            
            # disable viewing buttons
            self.showRaw_btn.setEnabled(False)
            self.showFilt_btn.setEnabled(False)
            self.showSig_btn.setEnabled(False)
            self.showSD_btn.setEnabled(False)
            
            # load data for entire recording
            # set EEG data item
            self.iseq_intan = np.arange(0, self.Intan.shape[1], dtype='int32')
            self.tseq_intan = np.linspace(self.ftime[0], self.ftime[-1], len(self.iseq_intan), dtype='float32')
            self.curIntan.setData(self.tseq_intan, self.Intan[0,:], padding=None)
            self.iseq_np = np.arange(0, len(self.graph_pwaves.LFP[0][0, :]), dtype='int32')
            self.tseq_np = np.linspace(self.ftime[0], self.ftime[-1], len(self.iseq_np), dtype='float32')
            self.graph_pwaves.data_item.setData(self.tseq_np, self.graph_pwaves.LFP[0][0, :], padding=None)
            
            # downsample P-wave indices and set P-wave data item
            if len(self.graph_pwaves.LFP) == 0:
                self.graph_pwaves.pidx_item.clear()
            else:
                pi_seq = np.where(self.p_train[0,:] == 1)[0].astype('int32')
                if len(pi_seq) > 0:
                    #pi_seq_dn = np.round(np.divide(pi_seq, nbin)).astype('int32')
                    self.graph_pwaves.pidx_item.setData(self.tseq_np[pi_seq], 
                                                        self.graph_pwaves.LFP[0][0, :][pi_seq] - 30, 
                                                        padding=None)
                else:
                    self.graph_pwaves.pidx_item.clear()
            
            if len(self.graph_pwaves.LFP) == 0:
                self.graph_pwaves.noise_item.clear()
            else:
                inoise = np.nonzero(self.noise_train[0,:])[0].astype('int32')
                if len(inoise) > 0:
                    #inoise_dn = np.round(np.divide(inoise, nbin)).astype('int32')
                    data = np.empty(len(self.tseq_np), dtype='float32') * np.nan
                    data[inoise] = self.graph_pwaves.LFP[0][0, :][inoise]
                    self.graph_pwaves.noise_item.setData(self.tseq_np, data, padding=None)
                else:
                    self.graph_pwaves.noise_item.clear()
            
            self.tstart = self.timepoint - self.twin/2  # set x-limits
            self.tend = self.timepoint + self.twin/2
            pqi.set_xlimits(self.graph_intan, self.tstart, self.tend)
            pqi.set_xlimits(self.graph_pwaves, self.tstart, self.tend)
            
            tmin = self.timepoint - self.twin_view/2    # set x-range
            tmax = self.timepoint + self.twin_view/2
            self.graph_intan.setXRange(tmin, tmax, padding=None)
            self.graph_pwaves.setXRange(tmin, tmax, padding=None)
            
            self.setWindowTitle('Setting annotation mode ... Done!')
            time.sleep(1)
            self.setWindowTitle(self.name)
    
    
    def annotate_only(self):
        """
        Called by updateIndex() when only EEG/P-wave signal updates are required
        """
        
        tmin = self.timepoint - self.twin_view/2
        tmax = self.timepoint + self.twin_view/2
        
        if tmin < self.tstart or tmax > self.tend:
            self.tstart = self.timepoint - self.twin/2 
            self.tend = self.timepoint + self.twin/2
            
            pqi.set_xlimits(self.graph_intan, self.tstart, self.tend)
            pqi.set_xlimits(self.graph_pwaves, self.tstart, self.tend)
        
        self.graph_intan.setXRange(tmin, tmax, padding=None)
        self.graph_pwaves.setXRange(tmin, tmax, padding=None)
        
        # update index tracker
        self.plot_treck()
        if self.pcollect_index == 1:
            self.index_list.append(self.index)
        else:
            self.index_list = [self.index]
            
    
    def choose_pwave_channel(self):
        """
        Set user-selected Neuropixel LFP signal as the P-wave channel
        """
        self.setWindowTitle('Updating P-wave channel ...')
        # get user-selected graph
        menu = self.sender().associatedWidgets()[0]
        name = menu.objectName().split('_menu')[0]  # e.g. 'LFP_40_PRN
        graph = [g for g in self.LFP_graphs if g.name == name][0]
        
        # if a previous LFP_processed file exists, close it
        pwave_path = os.path.join(self.ppath, self.name, 'LFP_processed.mat')
        if os.path.exists(pwave_path):
            for f in self.open_fid:
                if f.filename == pwave_path:
                    f.close()
                    self.open_fid.remove(f)
        
        # create new 'LFP_processed.mat' h5py file
        with h5py.File(pwave_path, 'w') as f:
            dset = f.create_dataset('LFP_processed', 
                                    shape=(1, graph.LFP[graph.pointer].shape[1]), 
                                    dtype='float32')
            dset[0,:] = graph.LFP[graph.pointer][graph.i,:]
            # save raw channel name and bp filter info as dataset attributes 
            dset.attrs['ch_name'] = name
            dset.attrs['f0'] = -1 if graph.i == 0 else float(graph.f0)
            dset.attrs['f1'] = -1 if graph.i == 0 else float(graph.f1)
            
            
        # open LFP_processed file, save dataset to P-wave graph
        fid = h5py.File(pwave_path, 'r+')
        self.open_fid.append(fid)
        dset = fid['LFP_processed']
        self.graph_pwaves.ch_name = dset.attrs['ch_name']
        self.graph_pwaves.f0 = dset.attrs['f0']
        self.graph_pwaves.f1 = dset.attrs['f1']
        self.graph_pwaves.LFP = [dset]
        mx = np.zeros((1, self.num_bins), dtype='float32')
        sd_dn = pqi.downsample_sd(dset[0,:], length=self.num_bins, nbin=self.fbin_np)
        mx[0,:] = sd_dn
        self.graph_pwaves.SD = [mx]
        
        # plot LFP signal
        if self.lfp_plotmode == 0:
            self.plot_signals()
        # plot LFP standard deviation
        elif self.lfp_plotmode == 1:
            self.plot_lfp_sd()
        
        self.graph_pwaves.show()
        self.graph_intan.hideAxis('bottom')
        self.detectPwaves_btn.setEnabled(True)
        
        self.setWindowTitle('Updating P-wave channel ... Done!')
        graph.isSelected = False
        graph.update()
        self.enable_filtering()
        time.sleep(1)
        self.setWindowTitle(self.name)
        
    
    def keyPressEvent(self, event):
        """
        Allow user to show/hide/shift/annotate data with keyboard inputs
        """
        # E - show [E]EG/switch EEG channel
        if event.key() == QtCore.Qt.Key_E:
            self.pointers['EMG'] = -1
            num_eeg = len(self.EEG_list)
            if self.pointers['EEG'] < num_eeg-1:
                self.pointers['EEG'] += 1   
            else:
                self.pointers['EEG'] = 0
            self.Intan = self.EEG_list[self.pointers['EEG']]
            self.curIntan.setData(self.tseq_intan, self.Intan[0,self.iseq_intan], padding=None)
            self.graph_intan.getAxis(name='left').setLabel(f'EEG{self.pointers["EEG"]+1}', units='V')
            
        # M - show E[M]G/switch EMG channel
        elif event.key() == QtCore.Qt.Key_M:
            self.pointers['EEG'] = -1
            num_emg = len(self.EMG_list)
            if self.pointers['EMG'] < num_emg-1:
                self.pointers['EMG'] += 1   
            else:
                self.pointers['EMG'] = 0
            self.Intan = self.EMG_list[self.pointers['EMG']]
            self.curIntan.setData(self.tseq_intan, self.Intan[0,self.iseq_intan], padding=None)
            self.graph_intan.getAxis(name='left').setLabel(f'EMG{self.pointers["EMG"]+1}', units='V')
        
        # R - [R]EM sleep
        elif event.key() == QtCore.Qt.Key_R:            
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 1
            self.index_list = [self.index]
            self.pcollect_index = False
            self.bulk_annot = False
            self.bulk_index1 = None
            self.annotMode_btn.setText('Bulk annot OFF')
            self.plot_brainstate()
        
        # W - [W]ake
        elif event.key() == QtCore.Qt.Key_W:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 2
            self.index_list = [self.index]
            self.pcollect_index = False
            self.bulk_annot = False
            self.bulk_index1 = None
            self.annotMode_btn.setText('Bulk annot OFF')
            self.plot_brainstate()
        
        # N - [N]REM sleep
        elif event.key() == QtCore.Qt.Key_N:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 3
            self.index_list = [self.index]
            self.pcollect_index = False
            self.bulk_annot = False
            self.bulk_index1 = None
            self.annotMode_btn.setText('Bulk annot OFF')
            self.plot_brainstate()
        
        # I - [i]ntermediate/transition sleep
        elif event.key() == QtCore.Qt.Key_I:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 4
            self.index_list = [self.index]
            self.pcollect_index = False
            self.bulk_annot = False
            self.bulk_index1 = None
            self.annotMode_btn.setText('Bulk annot OFF')
            self.plot_brainstate()
            
        # J - failed transition sleep
        elif event.key() == QtCore.Qt.Key_J:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 5
            self.index_list = [self.index]
            self.pcollect_index = False
            self.bulk_annot = False
            self.bulk_index1 = None
            self.annotMode_btn.setText('Bulk annot OFF')
            self.plot_brainstate()
            
        # cursor right - shift current index one FFT bin right
        elif event.key() == QtCore.Qt.Key_Right:
            if self.index < self.num_bins-self.fft_load/2:
                self.index += 1
                self.K[self.index] = 1
                self.update_index()
                
        # cursor left - shift current index one FFT bin left
        elif event.key() == QtCore.Qt.Key_Left:
            if self.index >= self.fft_load/2:
                self.index -= 1
                self.K[self.index] = 1
                self.update_index()
        
        # space - starting from current bin, collect indices visited with cursor
        elif event.key() == QtCore.Qt.Key_Space:
            self.pcollect_index = True
            self.index_list = [self.index]
        
        # return - annotate all indices between the current bin and the next mouse click
        elif event.key() == QtCore.Qt.Key_Return:
            if self.bulk_annot == False:
                self.bulk_annot = True
                self.bulk_index1 = int(self.index)
                self.annotMode_btn.setText('Bulk annot ON')
            elif self.bulk_annot == True:
                self.bulk_annot = False
                self.bulk_index1 = None
                self.index_list = [self.index]
                self.annotMode_btn.setText('Bulk annot OFF')
            self.index_list = [self.index]
        
        # cursor down - brighten spectrogram
        elif event.key() == QtCore.Qt.Key_Down:
            self.color_max -= self.color_max/10
            self.image_spectrum.setLevels((0, self.color_max))
        
        # cursor up - darken spectrogram
        elif event.key() == QtCore.Qt.Key_Up:
            self.color_max += self.color_max/10
            self.image_spectrum.setLevels((0, self.color_max))
        
        # X/C - annotate selected signal as noise/clean
        elif event.key() == QtCore.Qt.Key_X or event.key() == QtCore.Qt.Key_C:
            iselect = self.graph_pwaves.iselect
            if len(iselect) > 0:
                if event.key() == QtCore.Qt.Key_X:
                    # multiply noisy values in P-wave train by their negatives
                    # 1 (P-wave) -> -1 (noisy P-wave) // -1 -> -1 // 0 -> 0
                    self.p_train[0,iselect] *= -self.p_train[0,iselect]
                    self.noise_train[0,iselect] = 1
                elif event.key() == QtCore.Qt.Key_C:
                    # multiply cleaned values in P-wave train by themselves
                    # -1 (noisy P-wave) -> 1 (P-wave) // 1 -> 1 // 0 -> 0
                    self.p_train[0,iselect] *= self.p_train[0,iselect]
                    self.noise_train[0,iselect] = 0
                
                # reset user selection in $graph_pwaves, update P-wave and noise data
                self.graph_pwaves.reset_selection()
                self.update_pwave_data(updatePlot=True)
                self.update_noise_data(updatePlot=True)
        
        event.accept()
    
    
    def mousePressEvent(self, QMouseEvent):
        """
        Allow user to visit any recording index by double-clicking on the timepoint
        """
        pos = QMouseEvent.pos()
        if QMouseEvent.type() == QtCore.QEvent.MouseButtonDblClick:
            # clicked point must be within bounds of spectrogram/brainstate/treck plots
            if self.graph_spectrum.sceneBoundingRect().contains(pos)   \
            or self.graph_brainstate.sceneBoundingRect().contains(pos) \
            or self.graph_treck.sceneBoundingRect().contains(pos):
                mousePoint = self.graph_spectrum.vb.mapSceneToView(pos)
                
                # convert clicked point to recording index
                self.index = int(mousePoint.x()/self.fdt)
                if self.pcollect_index == True:
                    self.index_list.append(self.index)
                else:
                    self.index_list = [self.index]
                if self.bulk_annot == True:
                    # press Return - if bulk_annot is off, set current index as bulk_index1, next mouse click sets index list
                    #                if bulk_annot is on, clear everything
                    if self.bulk_index1 is not None:
                        if self.index > self.bulk_index1:
                            self.index_list = list(np.arange(self.bulk_index1, self.index).astype('int'))
                        elif self.bulk_index1 > self.index:
                            self.index_list = list(np.arange(self.index, self.bulk_index1).astype('int'))
                    else:
                        print('unexpected')
                        
                        
                self.update_index()
    
    
    
    ################        DATA FIGURE PLOTTING FUNCTIONS        ###############
    
    def pplot_unit_raster(self):
        """
        Plot the P-wave-triggered firing of the selected unit
        """
        # identify selected unit, get spike train data
        graph = self.sender().parent()
        k = str(graph.curUnit) + '_good'
        if k not in self.units_npz.keys():
            return
        data = np.array(self.units_npz[k], dtype='int8')
        
        # get P-wave indices, downsample to 1000 Hz
        p_idx = np.where(self.p_train[0,:] == 1)[0].astype('int32')
        nbin = self.dt * self.sr_np
        p_idx_dn = np.round(np.divide(p_idx, nbin)).astype('int32')
        # plot FR for P-waves during REM sleep only
        istate = 1
        if istate:
            p_idx_dn = [i for i in p_idx_dn if self.M.flatten()[int(i/self.fbin)] == istate]
        
        win = 1  # time window (s) to collect pre and post-P-wave
        iwin = int(round(win * self.sr))  # number of spike train samples (1000 Hz) in $win
        dn = 10  # downsample single unit firing rate by X bins
        
        # get array of spike trains surrounding P-waves
        unit_mx = np.vstack([data[idn-iwin : idn+iwin] for idn in p_idx_dn if idn > iwin and idn < len(data)-iwin-1])
        
        # get nodes, connections of raster plot
        height = np.arange(1,unit_mx.shape[0]+1)[::-1]
        t = np.linspace(-win,win,unit_mx.shape[1])
        rows, cols = np.nonzero(unit_mx)
        x = np.array(list(zip(t[cols],t[cols]))).flatten()
        y = np.array(list(zip(height[rows],height[rows]-1))).flatten()
        nodes = np.array([x,y]).T
        connections = np.arange(0,nodes.shape[0]).reshape(int(nodes.shape[0]/2),2)
        
        # sum the number of spikes in each consecutive group of samples
        j = int(np.floor(unit_mx.shape[1] / dn) * dn)
        col_grps = np.hsplit(unit_mx[:,0:j], int(j/dn))
        unit_mx_freq = np.vstack([np.divide(x.sum(axis=1),x.shape[1]) for x in col_grps]).T  # get firing rate in downsampled bins
        if j < unit_mx.shape[1]:
            end = np.divide(unit_mx[:,j:].sum(axis=1), unit_mx.shape[1] - j)
            unit_mx_freq = np.hstack([unit_mx_freq, np.atleast_2d(end).T])
        # get mean firing frequency and SEM
        unit_freq = np.mean(unit_mx_freq, axis=0)
        yerr = np.divide(np.std(unit_mx_freq, axis=0), np.sqrt(unit_mx_freq.shape[0]))
        
        t = np.linspace(-win,win,len(unit_freq))
        
        ### PLOT
        self.w = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(self.w)
        vbox.setSpacing(0)
        ywidth = int(pqi.px_w(50, self.WIDTH))
        
        # create widget and scroll area for raster plot
        glw1 = pg.GraphicsLayoutWidget(self.w)
        glw1.wheelEvent = lambda event: None
        q = QtWidgets.QScrollArea(self.w)
        q.setWidgetResizable(True)
        q.setWidget(glw1)
        q.setFixedHeight(int(pqi.px_h(500, self.HEIGHT)))
        glw1.setFixedHeight(unit_mx.shape[0]*5)
        
        # create raster plot
        rplot = glw1.addPlot()
        raster_item = pg.GraphItem()
        rplot.addItem(raster_item)
        raster_item.setData(pos=nodes, adj=connections, pen=graph.color, size=0)
        # set axis params
        rplot.getAxis(name='left').setWidth(ywidth)
        rplot.getAxis(name='left').setTicks([[]])
        pqi.set_xlimits(rplot, t[0], t[-1], padding=0.01)
        rplot.setLimits(**{'yMin':0, 'yMax':max(height)})
        
        ### plot mean firing frequency
        glw2 = pg.GraphicsLayoutWidget(self.w)
        glw2.setBackground('w')
        fplot = glw2.addPlot()
        d = pg.PlotCurveItem(t, unit_freq, pen=(0,0,0,255))
        yerr1 = pg.PlotCurveItem(t, unit_freq-yerr, pen=None)
        yerr2 = pg.PlotCurveItem(t, unit_freq+yerr, pen=None)
        yfill = pg.FillBetweenItem(yerr1, yerr2, brush=(0,0,0,100))
        fplot.addItem(yerr1)
        fplot.addItem(yerr2)
        fplot.addItem(yfill)
        fplot.addItem(d)
        # set axis params
        fplot.getAxis(name='left').setWidth(ywidth)
        fplot.getAxis(name='left').setLabel('Firing frequency (0-1)')
        pqi.set_xlimits(fplot, t[0], t[-1], padding=0.01)
        fplot.setXLink(rplot.vb)
        
        vbox.addWidget(q)
        vbox.addWidget(glw2)
        
        # set geometry, show plot
        w, h = [int(pqi.px_w(1000, self.WIDTH)), int(pqi.px_h(1000, self.HEIGHT))]
        x, y = [int(self.WIDTH/2 - w/2), int(self.HEIGHT/2 - h/2)]
        self.w.setGeometry(QtCore.QRect(x, y, w, h))
        self.w.show()
    
    
    def pplot_region_units_PFR(self):
        """
        Plot the P-wave-triggered firing rates of all units in a given brain region
        """
        graph = self.sender().parent()
        unit_list = [u for u in graph.units if u not in graph.hiddenUnits]
        
        # get new gradient of colors
        c1 = mcolors.to_hex(tuple([c/255 for c in graph.color]))
        c2 = mcolors.to_hex((1,1,1))
        colors = [np.multiply(clr.get_rgb(), 255) for clr in list(Color(c1).range_to(Color(c2), len(unit_list)))]
        
        # set plot dimensions
        graph_height = int(pqi.px_h(50, self.HEIGHT))
        graph_width = int(pqi.px_w(1000, self.WIDTH))
        graph_spacing = int(pqi.px_h(10, self.HEIGHT))
        ywidth = int(pqi.px_w(50, self.WIDTH))
        nplots = len(unit_list)+1
        
        # create plot view widget
        self.w = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(self.w)
        vbox.setSpacing(0)
        
        # get P-wave indices during REM sleep in Neuropixel time, downsample to Intan time
        p_idx = np.where(self.p_train[0,:] == 1)[0].astype('int32')
        istate = 1
        if istate:
            p_idx = np.array([i for i in p_idx if self.M.flatten()[int(i/self.fbin_np)] == istate])
        nbin = self.dt * self.sr_np
        p_idx_dn = np.round(np.divide(p_idx, nbin)).astype('int32')
        
        win = 1  # time window (s) to collect pre and post-P-wave
        dn = 10  # downsample single unit firing rate by X bins
        iwin_intan = int(round(win * self.sr))  # number of spike train samples (1000 Hz) in $win
        iwin_np = int(round(win * self.sr_np))  # number of LFP samples (2500 Hz) in $win
        
        # collect LFP waveforms in trial x timepoint matrix, get mean and SEM
        pdata = self.graph_pwaves.LFP[0][0,:]
        p_mx = np.vstack([pdata[pi-iwin_np : pi+iwin_np] for pi in p_idx if pi > iwin_np and pi < len(pdata)-iwin_np-1])
        pwaveform = np.mean(p_mx, axis=0)
        yerr = np.divide(np.std(p_mx, axis=0), np.sqrt(p_mx.shape[0]))
        t1 = np.linspace(-win, win, len(pwaveform))
        
        # show average P-waveform in first row
        glw1 = pg.GraphicsLayoutWidget(self.w)
        pgraph = glw1.addPlot()
        pgraph.resize(graph_width, graph_height)
        d = pg.PlotCurveItem(t1, pwaveform, pen=(255,255,255,255))
        yerr1 = pg.PlotCurveItem(t1, pwaveform-yerr, pen=None)
        yerr2 = pg.PlotCurveItem(t1, pwaveform+yerr, pen=None)
        yfill = pg.FillBetweenItem(yerr1, yerr2, brush=(255,255,255,150))
        pgraph.addItem(yerr1)
        pgraph.addItem(yerr2)
        pgraph.addItem(yfill)
        pgraph.addItem(d)
        # set axis params
        pgraph.getAxis(name='left').setWidth(ywidth)
        pgraph.getAxis(name='left').setLabel('LFP (uV)')
        pqi.set_xlimits(pgraph, t1[0], t1[-1], padding=0.02)
        
        # create widget and scroll area for individual unit plots
        glw2 = pg.GraphicsLayoutWidget()
        glw2.wheelEvent = lambda event: None
        q = QtWidgets.QScrollArea(self.w)
        q.setWidgetResizable(True)
        q.setWidget(glw2)
        q.setFixedHeight(int(pqi.px_h(870, self.HEIGHT)))
        glw2.setFixedHeight(nplots*graph_height + graph_spacing*(nplots+1))
        
        plts = []
        for i,unit in enumerate(unit_list):
            if unit in graph.hiddenUnits:
                continue
            k = str(unit) + '_good'
            if k not in self.units_npz.keys():
                continue
            
            # get spikes x timepoints matrix
            udata = np.array(self.units_npz[k], dtype='int8')
            unit_mx = np.vstack([udata[pidn-iwin_intan : pidn+iwin_intan] for pidn in p_idx_dn \
                                 if pidn > iwin_intan and pidn < len(udata)-iwin_intan-1])
            # downsample to calculate frequency
            j = int(np.floor(unit_mx.shape[1] / dn) * dn)
            col_grps = np.hsplit(unit_mx[:,0:j], int(j/dn))
            unit_mx_freq = np.vstack([np.divide(x.sum(axis=1),x.shape[1]) for x in col_grps]).T  # get firing rate in downsampled bins
            if j < unit_mx.shape[1]:
                end = np.divide(unit_mx[:,j:].sum(axis=1), unit_mx.shape[1] - j)
                unit_mx_freq = np.hstack([unit_mx_freq, np.atleast_2d(end).T])
            # get mean unit activity and SEM
            unit_freq = np.mean(unit_mx_freq, axis=0)
            yerr = np.divide(np.std(unit_mx_freq, axis=0), np.sqrt(unit_mx_freq.shape[0]))
            t2 = np.linspace(-win,win,len(unit_freq))
            
            ### PLOT MEAN FIRING FREQUENCY
            ugraph = glw2.addPlot()
            ugraph.resize(graph_width, graph_height)
            r,g,b = colors[i]
            d = pg.PlotCurveItem(t2, unit_freq, pen=(r,g,b,255))
            yerr1 = pg.PlotCurveItem(t2, unit_freq-yerr, pen=None)
            yerr2 = pg.PlotCurveItem(t2, unit_freq+yerr, pen=None)
            yfill = pg.FillBetweenItem(yerr1, yerr2, brush=(r,g,b,100))
            ugraph.addItem(yerr1)
            ugraph.addItem(yerr2)
            ugraph.addItem(yfill)
            ugraph.addItem(d)
            # set axis params
            if i == 0:
                ugraph.showAxis('top')
                ugraph.getAxis(name='top').setTicks([[]])
                ugraph.getAxis(name='top').setPen(None)
                ugraph.getAxis(name='top').setLabel(str(graph.name), **{'color':colors[i], 
                                                                        'font-size':'16pt', 
                                                                        'font-weight':'700'})
            if i < len(unit_list)-1:
                ugraph.hideAxis('bottom')
            ugraph.getAxis(name='left').setWidth(ywidth)
            ugraph.getAxis(name='left').setLabel('FR')
            pqi.set_xlimits(ugraph, t2[0], t2[-1], padding=0.02)
            # set unit label
            label = pg.LabelItem(text=str(unit), **{'color':colors[i], 'size':'14pt'})
            label.setParentItem(ugraph)
            label.anchor(itemPos=(1,1), parentPos=(1,1), offset=(10,-10))
            plts.append(ugraph)
            glw2.nextRow()
        # link x-axes of all graphs
        pqi.xlink_graphs(plts, target_graph=pgraph)
        vbox.addWidget(glw1)
        vbox.addWidget(q)
        
        # set geometry, show plot
        w, h = [int(pqi.px_w(1600, self.WIDTH)), int(pqi.px_h(1000, self.HEIGHT))]
        x, y = [int(self.WIDTH/2 - w/2), int(self.HEIGHT/2 - h/2)]
        self.w.setGeometry(QtCore.QRect(x, y, w, h))
        self.w.show()
                
    
    def pplot_region_units_FR(self):
        """
        Plot the firing rates of all units in a given brain region across the recording
        """
        graph = self.sender().parent()
        unit_list = [u for u in graph.units if u not in graph.hiddenUnits]
        
        # get new gradient of colors
        c1 = mcolors.to_hex(tuple([c/255 for c in graph.color]))
        c2 = mcolors.to_hex((1,1,1))
        colors = [clr.hex for clr in list(Color(c1).range_to(Color(c2), len(unit_list)))]
        
        # set plot dimensions
        graph_height = int(pqi.px_h(50, self.HEIGHT))
        graph_width = int(pqi.px_w(1000, self.WIDTH))
        graph_spacing = int(pqi.px_h(10, self.HEIGHT))
        ywidth = int(pqi.px_w(50, self.WIDTH))
        nplots = len(unit_list)+1
        
        # create plot view widget
        self.w = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(self.w)
        vbox.setSpacing(0)
        
        # show brainstate annotation in first row
        glw1 = pg.GraphicsLayoutWidget(self.w)
        brstate = glw1.addPlot()
        brstate.resize(graph_width, graph_height)
        img = pg.ImageItem() 
        brstate.addItem(img)
        img.setImage(self.M.T)
        # set time scale and color map
        tr = QtGui.QTransform()
        tr.scale(self.fdt, 1)
        img.setTransform(tr)
        img.setLookupTable(self.lut_brainstate)
        img.setLevels([0, 7])
        # set axis params
        brstate.hideAxis('bottom')
        brstate.getAxis(name='left').setTicks([[]])
        brstate.getAxis(name='left').setWidth(ywidth)
        brstate.getAxis(name='left').setLabel('State')
        pqi.set_xlimits(brstate, tmin=self.ftime[0], tmax=self.ftime[-1]+self.fdt)
        
        # create widget and scroll area for individual unit plots
        glw2 = pg.GraphicsLayoutWidget()
        glw2.wheelEvent = lambda event: None
        q = QtWidgets.QScrollArea(self.w)
        q.setWidgetResizable(True)
        q.setWidget(glw2)
        q.setFixedHeight(int(pqi.px_h(870, self.HEIGHT)))
        glw2.setFixedHeight(nplots*graph_height + graph_spacing*(nplots+1))
        
        plts = []
        for i,unit in enumerate(unit_list):
            # plot session FR for each unit
            ugraph = glw2.addPlot()
            ugraph.resize(graph_width, graph_height)
            ugraph.plot(self.ftime, graph.fr_mx[i,:], pen=colors[i], padding=None)
            # set axis params
            if i == 0:
                ugraph.showAxis('top')
                ugraph.getAxis(name='top').setTicks([[]])
                ugraph.getAxis(name='top').setPen(None)
                ugraph.getAxis(name='top').setLabel(str(graph.name), **{'color':colors[i], 
                                                                        'font-size':'16pt', 
                                                                        'font-weight':'700'})
            if i < len(unit_list)-1:
                ugraph.hideAxis('bottom')
            ugraph.getAxis(name='left').setWidth(ywidth)
            ugraph.getAxis(name='left').setLabel('FR')
            pqi.set_xlimits(ugraph, tmin=self.ftime[0], tmax=self.ftime[-1]+self.fdt)
            # set unit label
            label = pg.LabelItem(text=str(unit), **{'color':colors[i], 'size':'14pt'})
            label.setParentItem(ugraph)
            label.anchor(itemPos=(1,1), parentPos=(1,1), offset=(10,-10))
            plts.append(ugraph)
            glw2.nextRow()
        # link x-axes of all graphs
        pqi.xlink_graphs(plts, target_graph=brstate)
        
        vbox.addWidget(glw1)
        vbox.addWidget(q)
        
        # set geometry, show plot
        w, h = [int(pqi.px_w(1600, self.WIDTH)), int(pqi.px_h(1000, self.HEIGHT))]
        x, y = [int(self.WIDTH/2 - w/2), int(self.HEIGHT/2 - h/2)]
        self.w.setGeometry(QtCore.QRect(x, y, w, h))
        self.w.show()
        
    
    def plot_channel_data(self):
        """
        Plot P-wave frequency for each Neuropixel LFP channel
        ***work in progress
        """
        fig, axs = plt.subplots(figsize=(5,10), 
                                nrows=len(self.LFP_graphs)+1, ncols=1, 
                                sharex=True, tight_layout=True)
        # plot brain state annotation
        my_map, vmin, vmax = AS.hypno_colormap()
        axs[0].pcolorfast(self.ftime, [0, 1], self.M, vmin=vmin, vmax=vmax, cmap=my_map)
        axs[0].axes.get_yaxis().set_visible(False)
        axs[0].axes.get_xaxis().set_visible(False)
        # plot frequency of P-waves in each LFP channel
        for ax,graph in zip(axs[1:], self.LFP_graphs):
            print('Detecting P-waves for ' + graph.name + ' ...')
            # detect P-waves in filtered signal
            psignal = graph.LFP[0][1,:]
            
            # calculate detection threshold, find and validate spikes
            p_thr = self.get_threshold(psignal, self.thres, self.thres_type)
            pi = pwaves.spike_threshold(psignal, p_thr)
            pi = self.validate_spikes(pi, psignal, self.sr_np, self.dup_win)
            
            pvec = np.zeros(len(psignal), dtype='float32')
            pvec[pi] = float(self.sr_np)
            pfreq = sleepy.downsample_vec(pvec, int(self.fbin_np)).astype('float32')
            pfreq = np.resize(pfreq, self.num_bins)
            
            c = np.divide(graph.color, 255)
            ax.plot(self.ftime, pfreq, color=c)
            ax.set_ylabel('P-waves/s')
            ax.set_title(graph.name, fontdict=dict(color=c), loc='right')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if ax != axs[-1]:
                ax.set_xticklabels([])
            
        y = [min([ax.get_ylim()[0] for ax in axs[1:]]), max([ax.get_ylim()[1] for ax in axs[1:]])]
        _ = [ax.set_ylim(y) for ax in axs[1:]]
        axs[-1].set_xlabel('Time (s)')
        
        dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/all_channels_pfreq.svg'
        plt.savefig(dpath, format="svg")
    
    
    ################          DATA LOADING FUNCTIONS           ################
    
    
    def openFileNameDialog(self):
        """
        Allow user to choose recording folder in computer
        """
        fileDialog = QtWidgets.QFileDialog(self)
        fileDialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        name = fileDialog.getExistingDirectory(self, "Choose Recording Directory")
        (self.ppath, self.name) = os.path.split(name)        
        print("Setting base folder %s and recording %s" % (self.ppath, self.name))
        
        
    def load_recording(self):
        """
        Load recording data, including raw EEG/EMG/LFP signals, EEG spectrogram, 
        brain state annotation, laser stimulation train, and other 
        experiment info (sampling rate, light/dark cycles, etc)
        """
        # user selects recording folder in computer
        if self.name == '':
            self.openFileNameDialog()
        # set title for window
        self.setWindowTitle(self.name)
        
        # load EEG and EMG signals, initialize pointers
        self.EEG_list, self.EMG_list, openfid = pqi.load_eeg_emg(self.ppath, self.name)
        self.open_fid += openfid
        self.pointers = {'EEG':0, 'EMG':-1}
        
        # load EEG and EMG spectrogram information
        SPEC, MSPEC = pqi.load_sp_msp(self.ppath, self.name)
        
        self.ftime = np.squeeze(SPEC['t'])        # vector of SP timepoints
        self.num_bins = len(self.ftime)           # no. SP time bins
        self.fdt = float(np.squeeze(SPEC['dt']))  # SP time resolution (e.g. 2.5)
        self.twin = self.fft_load * self.fdt      # duration of data (s) loaded into memory
        self.twin_view = self.fft_view * self.fdt # duration of data (s) viewable in window
        self.timepoint = self.ftime[self.index]   # timepoint (s) of current index
        
        self.freq = np.squeeze(SPEC['freq'])      # vector of SP frequencies
        self.fdx = self.freq[1] - self.freq[0]    # SP frequency resolution (e.g. 0.5)
        self.ifreq = np.where(self.freq<=25)[0]   # freqs to show in EEG spectrogram
        self.mfreq = np.where((self.freq>=10) &   # freqs to use in EMG amplitude calculation
                              (self.freq<=500))[0]
        # collect EEG spectrograms
        self.eeg_spec_list = [SPEC['SP']]
        if 'SP2' in SPEC:
            self.eeg_spec_list.append(SPEC['SP2'])
        self.eeg_spec = self.eeg_spec_list[0]  # plot SP1
        self.color_max = np.max(self.eeg_spec)
        
        # collect EMG spectrograms, calculate EMG amplitude
        self.EMGAmpl_list = []
        EMGAmpl1 = np.sqrt(MSPEC['mSP'][self.mfreq,:].sum(axis=0))
        self.EMGAmpl_list.append(EMGAmpl1)
        if 'mSP2' in MSPEC:
            EMGAmpl2 = np.sqrt(MSPEC['mSP2'][self.mfreq,:].sum(axis=0))
            self.EMGAmpl_list.append(EMGAmpl2)
        self.session_data = self.EMGAmpl_list[0]
        
        # set time bins, sampling rates etc.        
        self.sr = sleepy.get_snr(self.ppath, self.name)  # Intan sampling rate (e.g. 1000 Hz)
        self.dt = 1/self.sr                              # s per Intan sample (e.g. 0.001)
        self.fbin = np.round(self.sr*self.fdt)           # no. Intan samples per FFT bin (e.g. 2500)
        if self.fbin % 2 == 1:
            self.fbin += 1
        # current Intan index
        self.index_intan = int(np.round(self.timepoint*self.sr))
        # fix Intan vector sizes
        nsamples_intan = int(self.fbin * self.num_bins)
        if nsamples_intan != self.EEG_list[0].shape[1]:
            _ = [dset.resize(nsamples_intan, axis=1) for dset in self.EEG_list]
            _ = [dset.resize(nsamples_intan, axis=1) for dset in self.EMG_list]
            
        self.Intan = self.EEG_list[0]  # plot EEG1
        # median of Intan signal to scale the laser signal
        self.intan_amp = np.nanmedian(np.abs(self.Intan))
        
        
        self.sr_np = pqi.get_snr_np(self.ppath, self.name)  # Neuropixel sampling rate (e.g. 2500 Hz)
        self.dt_np = 1/self.sr_np                           # s per Neuropixel sample (e.g. 0.0004)
        self.fbin_np = np.round(self.sr_np*self.fdt)        # no. Neuropixel samples per FFT bin (e.g. 6250)
        if self.fbin_np % 2 == 1:
            self.fbin_np += 1
        # current Neuropixel index
        self.index_np = int(np.round(self.timepoint*self.sr_np))
        
        self.graph_pwaves.get_sampling_rate()
        
        # load brain state annotation
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
            # predict brain state from EEG/EMG data
            M,S = sleepy.sleep_state(self.ppath, self.name, pwrite=1, pplot=0)
        (A,self.K) = sleepy.load_stateidx(self.ppath, self.name)
        # create 1 x nbin matrix for display
        self.M = np.zeros((1,self.num_bins))
        self.M[0,:] = A
        self.M_old = self.M.copy()
                        
        # load/plot laser
        self.laser = np.zeros((self.num_bins,))  # laser signal in SP time
        self.pplot_laser = False
        lfile = os.path.join(self.ppath, self.name, 'laser_' + self.name + '.mat')
        if os.path.isfile(lfile):
            lsr = sleepy.load_laser(self.ppath, self.name)
            (start_idx, end_idx) = sleepy.laser_start_end(lsr, self.sr)
            # laser signal in Intan time
            self.laser_raw = lsr
            self.pplot_laser = True
            if len(start_idx) > 0:
                for (i,j) in zip(start_idx, end_idx) :
                    i = int(np.round(i/self.fbin))
                    j = int(np.round(j/self.fbin))
                    self.laser[i:j+1] = 1
        else:
            self.laser_raw = np.zeros(self.Intan.shape)
                
        ### LOAD LFP NEUROPIXEL DATA
        lfps, lnames, lgrps, openfid = pqi.load_lfp_files(self.ppath, self.name, 
                                                           keymatch=False, sort_grps=True)
        self.open_fid += openfid
        nsamples_np = int(self.fbin_np * self.num_bins)
        if nsamples_np != lfps[0].shape[1]:
            _ = [dset.resize(nsamples_np, axis=1) for dset in lfps]
        
        # populate graph for P-wave channel
        if 'LFP_processed' in lnames:
            # remove processed signal from LFP dictionary
            i = lnames.index('LFP_processed')
            dset, _, _ = [lfps.pop(i), lnames.pop(i), lgrps.pop(i)]
            # get channel name and bp filtering info from attributes
            self.graph_pwaves.ch_name = dset.attrs.get('ch_name', default='')
            self.graph_pwaves.f0 = dset.attrs.get('f0', default=-1)
            self.graph_pwaves.f1 = dset.attrs.get('f1', default=-1)
            # assign signal and downsampled LFP to $graph_pwaves
            self.graph_pwaves.LFP = [dset]
            mx = np.zeros((1, self.num_bins), dtype='float32')
            sd_dn = pqi.downsample_sd(dset[0,:], length=self.num_bins, nbin=self.fbin_np)
            mx[0,:] = sd_dn
            self.graph_pwaves.SD = [mx]
            # show P-wave graph, enable P-wave detection
            self.graph_pwaves.show()
            self.graph_intan.hideAxis('bottom')
            self.detectPwaves_btn.setEnabled(True)
        self.load_pdata(nsamples=nsamples_np)
            
        # create graph for each LFP signal
        self.LFP_graphs = self.lfpView.create_lfp_graphs(lnames, graph_type='LFP')
        
        # assign each LFP array to its graph
        for igraph,(dset,name,grp) in enumerate(zip(lfps,lnames,lgrps)):
            graph = self.LFP_graphs[igraph]
            graph.LFP = [dset]
            graph.grp = grp
            # calculate downsampled standard deviation for each LFP signal/row
            mx = np.zeros((dset.shape[0], self.num_bins), dtype='float32')
            for i in range(dset.shape[0]):
                sd_dn = pqi.downsample_sd(dset[i,:], length=self.num_bins, nbin=self.fbin_np)
                mx[i,:] = sd_dn
            graph.SD = [mx]
            # 'signals' attribute (if it exists) has descriptive string for each signal/row
            llist = dset.attrs.get('signals', default=[])
            if len(llist) == 2 and llist[1].startswith('bp_'):
                graph.f0, graph.f1 = llist[1].split('_')[1:]
        
        
        ### LOAD SINGLE-UNIT NEUROPIXEL DATA
        self.unit_graphs = []
        
        # load dataframe with single unit channel IDs and brain regions
        if os.path.exists(os.path.join(self.ppath, self.name, 'channel_locations.json')):
            self.ch_locations = pd.read_json(os.path.join(self.ppath, self.name, 'channel_locations.json')).T.iloc[0:-1]
            self.ch_locations['ch'] = self.ch_locations.index.str.split('_').str[-1].astype('int64')
        else:
            self.ch_locations = []
        
        # get path to single unit firing rates
        if os.path.exists(os.path.join(self.ppath, self.name, 'traind_cleaned.csv')):
            FRpath = os.path.join(self.ppath, self.name, 'traind_cleaned.csv')
        elif os.path.exists(os.path.join(self.ppath, self.name, 'traind.csv')):
            FRpath = os.path.join(self.ppath, self.name, 'traind.csv')
        else:
            FRpath = ''
        # get path to NPZ spike train file
        if os.path.exists(os.path.join(self.ppath, self.name, '1k_train.npz')):
            NPZpath = os.path.join(self.ppath, self.name, '1k_train.npz')
        else:
            NPZpath = ''
        
        if (FRpath or NPZpath) and len(self.ch_locations) > 0:
            self.showUnits_btn.setEnabled(True)    # can show some single-unit data
            self.useManualClass_btn.setEnabled(True)  # can classify units by state (FRpath) and/or P-wave (NPZpath) FRs
            # load dataframe of channel firing rates, isolate "good" units
            if FRpath:
                self.showFR_btn.setEnabled(True)
                fr_df = pd.read_csv(FRpath)
                ucols,uids = zip(*[(icol,int(uname.split('_')[0])) for icol,uname in enumerate(fr_df.columns) if uname.split('_')[1] == 'good'])
                good_units = fr_df.iloc[:, np.array(ucols)].reset_index(drop=True)
                good_units.columns = list(uids)
                
                if len(fr_df) != self.num_bins:
                    print('uh oh, mismatching lengths of firing rate vectors and brain state vector')
            
            # load read-only NPZ file of channel spike trains (1000 Hz)
            if NPZpath:
                self.showTrain_btn.setEnabled(True)
                self.units_npz = np.load(NPZpath, allow_pickle=True, mmap_mode='r')
                if not FRpath:
                    uids = [int(uname.split('_')[0]) for uname in list(self.units_npz.keys()) if uname.split('_')[1] == 'good']
                    # if firing rates not found, automatically show spike trains
                    self.np_plotmode = 0
                    self.showFR_btn.blockSignals(True)
                    self.showTrain_btn.blockSignals(True)
                    self.showTrain_btn.setChecked(True)
                    self.showFR_btn.blockSignals(False)
                    self.showTrain_btn.blockSignals(False)
                    
            # match each unit with corresponding brain state in regions df
            region_dict = {}
            icol = list(self.ch_locations.columns).index('brain_region')
            for uid in uids:
                irow = np.where(self.ch_locations['ch']==uid)[0]
                if len(irow) > 0:
                    reg = self.ch_locations.iloc[irow[0], icol]
                else:
                    reg = '___'
                if reg in region_dict:
                    region_dict[reg].append(uid)
                else:
                    region_dict[reg] = [uid]
                #self.unit_lookup[uid] = reg
            
            # load/create unit notes file
            if not os.path.exists(os.path.join(self.ppath, self.name, 'unit_notes.txt')):
                pqi.save_unit_notes(self.ppath, self.name, list(uids))
            self.unotes = pqi.load_unit_notes(self.ppath, self.name)
            # load file with automatic classifications, if it exists
            self.auto_unotes = pqi.load_unit_notes(self.ppath, self.name, auto=True)
            if self.auto_unotes:
                self.useAutoClass_btn.setEnabled(True)  # can load previously saved classifications
                
                # look for saved dataframes from auto-classifications
                if os.path.exists(os.path.join(self.ppath, self.name, 'unit_state_df')):
                    self.unit_state_df = pd.read_csv(os.path.join(self.ppath, self.name, 'unit_state_df'))
                if os.path.exists(os.path.join(self.ppath, self.name, 'unit_PFR_df')):
                    self.unit_PFR_df = pd.read_csv(os.path.join(self.ppath, self.name, 'unit_PFR_df'))
            
            # create graph for each unique brain region
            self.unit_graphs = self.unitView.create_lfp_graphs(list(region_dict.keys()), graph_type='unit')
            # create dictionary to track which units are shown/hidden from analysis for each region
            state_cats = ['R-on','R-max','R-off','RW','W-max','W-min','N-max','X','Unclassified state']
            pwave_cats = ['P+','P-','P0','Unclassified pwaves']
            pk_fr_bins = np.arange(0, 201, 40)
            pk_fr_cats = [f'{a} - {b}' for a,b in zip(pk_fr_bins[0:-1], pk_fr_bins[1:])] + [f'{pk_fr_bins[-1]}+']
            classmatch_cats = ['Same brainstate group', 
                               'Different brainstate groups', 
                               'Missing brainstate classification(s)',
                               'Same P-wave group',
                               'Different P-wave groups',
                               'Missing P-wave classification(s)']
            cats = state_cats + pwave_cats + pk_fr_cats + classmatch_cats
                    
            self.ufilt = {cat : 1 for cat in cats}
            self.region_ufilt = {}
            
            for igraph,(region,units) in enumerate(region_dict.items()):
                graph = self.unit_graphs[igraph]
                # assign corresponding units, set current unit in the middle of the region
                graph.units = list(units)
                graph.curUnit = int(units[int(len(units)/2)])
                graph.i = int(graph.units.index(graph.curUnit))
                graph.label.set_info(txt=str(graph.curUnit) + '_' + str(graph.name), color=graph.color)
                notes = self.unotes.get(graph.curUnit, ['-','-','-'])
                graph.class_label.set_info(txt=f'{notes[0]} | {notes[1]}', color=graph.color, size='10pt')
                # initialize unit filtering params for the region (1's = all units shown)
                self.region_ufilt[region] = {cat : 1 for cat in cats}
                
                if FRpath:
                    graph.fr_mx = np.array(good_units.loc[:,units].T, dtype='float32')
                    graph.peak_fr = np.max(graph.fr_mx, axis=1)
            
        elif (FRpath or NPZpath) and len(self.ch_locations) == 0:
            if FRpath and NPZpath:
                print('Error: single unit firing rates and NPZ spike trains found, but missing channel_locations.json file')
            elif FRpath:
                print('Error: single unit firing rates found, but missing channel_locations.json file')
            elif NPZpath:
                print('Error: NPZ spike trains found, but missing channel_locations.json file')
                
        # link x-axes of LFP (or unit) graphs together, link to Intan or spectrogram x-axis
        if len(self.LFP_graphs) > 0:
            if self.lfp_plotmode == 0:
                pqi.xlink_graphs(self.LFP_graphs, target_graph=self.graph_intan)
            else:
                pqi.xlink_graphs(self.LFP_graphs, target_graph=self.graph_spectrum)
        if len(self.unit_graphs) > 0:
            if self.np_plotmode == 0:
                pqi.xlink_graphs(self.unit_graphs, target_graph=self.graph_intan)
            else:
                pqi.xlink_graphs(self.unit_graphs, target_graph=self.graph_spectrum)
                
        # link x-axis of P-wave graph to spectrogram if plotting LFP SD, otherwise link to Intan
        if self.lfp_plotmode == 1 and self.showLFPs_btn.isChecked():
            self.graph_pwaves.setXLink(self.graph_spectrum.vb)
        else:
            self.graph_pwaves.setXLink(self.graph_intan.vb)
    
    
    def load_pdata(self, nsamples=None, pfile='p_idx.mat'):
        """
        Load P-wave data and detection settings from h5py file
        """
        # path to P-wave info file
        fpath = os.path.join(self.ppath, self.name, pfile)
        
        # load/create h5py file
        self.pfile = h5py.File(fpath, 'a')
        self.open_fid.append(self.pfile)
        
        # get default param values
        ddict = dict(self.defaults)
        for key,val in ddict.items():
            if self.pfile.attrs.__contains__(key):
                # load param value from h5py file attributes
                ddict[key] = self.pfile.attrs[key]
            else:
                # save param value in file
                self.pfile.attrs[key] = val
            
        if 'p_train' in self.pfile.keys():
            self.p_train = self.pfile['p_train']
            self.p_thr = self.p_train.attrs.get('p_thr', default=0.0)
        else:
            if nsamples is not None:
                self.p_train = self.pfile.create_dataset('p_train', shape=(1,nsamples), dtype='int8')
                self.p_train[0,:] = 0
            else:
                self.p_train = self.pfile.create_dataset('p_train', dtype='int8')
            self.p_train.attrs['p_thr'] = 0.0
            self.p_thr = 0.0
        
        if 'noise_train' in self.pfile.keys():
            self.noise_train = self.pfile['noise_train']
        else:
            if nsamples is not None:
                self.noise_train = self.pfile.create_dataset('noise_train', shape=(1,nsamples), dtype='int8')
                self.noise_train[0,:] = 0
            else:
                self.noise_train = self.pfile.create_dataset('noise_train', dtype='int8')

        self.update_pwave_data(updatePlot=False)  # calculate P-wave frequency
        self.update_noise_data(updatePlot=False)  # downsample noise train
        
        # update graph threshold and label
        if self.p_thr != 0:
            txt = pqi.pthres_txt(self.p_thr, ddict['thres'], ddict['thres_type'])
            self.graph_pwaves.pthres_item.setValue(-self.p_thr)
            self.graph_pwaves.pthres_item.label.setFormat(txt)
        
        # update variables/GUI with loaded values
        self.update_vars_from_dict(ddict)
        self.update_gui_from_vars()
    
    
    def closeEvent(self, event):
        if len(self.open_fid) > 0:
            _ = [f.close() for f in self.open_fid]
            self.open_fid = []
        
        
    def debug(self):
        pdb.set_trace()

    
ppath = '/media/fearthekraken/Mandy_HardDrive1/neuropixel/Processed_Sleep_Recordings'
#name = 'DL158_012623n1'  # no LFP raw data in continuous file, but has 1k spike train .npz
#name = 'DL157_012623n1'  # has raw LFPs and downsampled unit firing rates, but no 1k spike train
name = 'DL161_exp2_012623n1'  # has 1k spike train .npz, INCREDIBLE P-waves in raw LFPs

app = QtWidgets.QApplication([])
app.setStyle('Fusion')
w = MainWindow(ppath, name)
w.show()
w.raise_()
sys.exit(app.exec())