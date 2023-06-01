#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 18:21:18 2023

@author: fearthekraken
"""

import sys
import os
import re
import h5py
import scipy.io as so
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import colorsys
import pyautogui
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pdb
# custom modules
import sleepy
from sleepy import calculate_spectrum
import AS
from pqt_items import *


class MainWindowInfo(QtWidgets.QMainWindow):
    
    def __init__(self, ppath, name):
        """
        Instantiate skeleton of main window with loaded recording information
        """
        QtWidgets.QMainWindow.__init__(self)
        
        self.WIDTH, self.HEIGHT = pyautogui.size()
        self.coors = pd.Series(data=[20, 20, self.WIDTH-80, self.HEIGHT-80], index=['x','y','w','h'])
        self.settings_width = px_w(200, self.WIDTH)  # pqi
        self.graph_width = self.coors.w - self.settings_width - 40
        
        self.ppath = ppath
        self.name  = name
        self.open_fid = []  # open h5py files
        
        ###   GENERAL PARAMS   ###
        pos = np.linspace(0, 1, 8)
        color = np.array([[0, 0, 0, 200], [0, 255, 255, 200], 
                          [150, 0, 255, 200], [150, 150, 150, 200], 
                          [66,86,219,200], [255,20,20,200], 
                          [0,255,43,200], [255,255,0,200]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.lut_brainstate = cmap.getLookupTable(0.0, 1.0, 8)
        
        ###   LOAD RECORDING INFORMATION   ###
        
        # load EEG and EMG signals, initialize pointers
        #self.EEG_list, self.EMG_list, openfid = load_eeg_emg(self.ppath, self.name)  # pqi
        #self.open_fid += openfid
        #self.pointers = {'EEG':0, 'EMG':-1}
        
        # load EEG and EMG spectrogram information
        SPEC, MSPEC = load_sp_msp(self.ppath, self.name)  # pqi
        self.ftime = np.squeeze(SPEC['t'])        # vector of SP timepoints
        self.num_bins = len(self.ftime)           # no. SP time bins
        self.fdt = float(np.squeeze(SPEC['dt']))  # SP time resolution (e.g. 2.5)
        
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
        
        self.sr_np = get_snr_np(self.ppath, self.name)  # Neuropixel sampling rate (e.g. 2500 Hz)  # pqi
        self.dt_np = 1/self.sr_np                           # s per Neuropixel sample (e.g. 0.0004)
        self.fbin_np = np.round(self.sr_np*self.fdt)        # no. Neuropixel samples per FFT bin (e.g. 6250)
        if self.fbin_np % 2 == 1:
            self.fbin_np += 1
        
        # load brain state annotation
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
            # predict brain state from EEG/EMG data
            M,S = sleepy.sleep_state(self.ppath, self.name, pwrite=1, pplot=0)
        (A,self.K) = sleepy.load_stateidx(self.ppath, self.name)
        # create 1 x nbin matrix for display
        self.M = np.zeros((1,self.num_bins))
        self.M[0,:] = A
        self.M_old = self.M.copy()
        
        
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
            # load dataframe of channel firing rates, isolate "good" units
            if FRpath:
                fr_df = pd.read_csv(FRpath)
                ucols,uids = zip(*[(icol,int(uname.split('_')[0])) for icol,uname in enumerate(fr_df.columns) if uname.split('_')[1] == 'good'])
                good_units = fr_df.iloc[:, np.array(ucols)].reset_index(drop=True)
                good_units.columns = list(uids)
            
            # load read-only NPZ file of channel spike trains (1000 Hz)
            if NPZpath:
                self.units_npz = np.load(NPZpath, allow_pickle=True, mmap_mode='r')
                if not FRpath:
                    uids = [int(uname.split('_')[0]) for uname in list(self.units_npz.keys()) if uname.split('_')[1] == 'good']
                    
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
            
            # load/create unit notes file
            if not os.path.exists(os.path.join(self.ppath, self.name, 'unit_notes.txt')):
                save_unit_notes(self.ppath, self.name, list(uids))  # pqi
            self.unotes = load_unit_notes(self.ppath, self.name)  # pqi
            # load file with automatic classifications, if it exists
            self.auto_unotes = load_unit_notes(self.ppath, self.name, auto=True)  # pqi
            if self.auto_unotes:
                # look for saved dataframes from auto-classifications
                if os.path.exists(os.path.join(self.ppath, self.name, 'unit_state_df')):
                    self.unit_state_df = pd.read_csv(os.path.join(self.ppath, self.name, 'unit_state_df'))
                else:
                    self.unit_state_df = None
                if os.path.exists(os.path.join(self.ppath, self.name, 'unit_PFR_df')):
                    self.unit_PFR_df = pd.read_csv(os.path.join(self.ppath, self.name, 'unit_PFR_df'))
                else:
                    self.unit_PFR_df = None
            
            # create graph for each unique brain region
            self.unitView = LFPView(parent=self)  # pqi
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


class UnitComparisonWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        """
        """
        super(UnitComparisonWindow, self).__init__()
        
        if parent is None:
            self.mainWin = MainWindowInfo(ppath, name)
            self.TEST_MODE = True
            self.setWindowTitle('TEST_MODE')
        else:
            self.mainWin = parent
            self.TEST_MODE = False
        
        ###   IMPORTED VARIABLES   ###
        self.ppath = str(self.mainWin.ppath)
        self.name = str(self.mainWin.name)
        self.coors = self.mainWin.coors.copy()
        ###
        
        ###
        self.gen_layout()
    
    def gen_layout(self):
        wpx, hpx = pyautogui.size()
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20, wspace50, wspace100, wspace200 = [px_w(w, wpx) for w in [5,10,15,20,50,100,200]]
        hspace5, hspace10, hspace15, hspace20, hspace50, hspace100, hspace200 = [px_h(h, hpx) for h in [5,10,15,20,50,100,200]]
        
        # set contents margins, central layout
        self.setFixedWidth(self.coors.w)
        self.settings_width = px_w(350, wpx)
        self.graph_width = int(hspace200*3)
        self.graph_height = int(hspace200*3)
        
        # top, right, bottom, left
        self.setContentsMargins(wspace5,hspace5,wspace5,hspace5)
        self.centralLayout = QtWidgets.QHBoxLayout(self)
        # set GUI fonts
        font = QtGui.QFont()
        font.setPointSize(10)
        labelFont = QtGui.QFont()
        labelFont.setPointSize(11)
        labelFont.setBold(True)
        headerFont = QtGui.QFont()
        headerFont.setPointSize(12)
        headerFont.setBold(True)
        headerFont.setUnderline(True)
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.setFont(font)
        
        ###   PLOT LAYOUT
        self.plotWidget = QtWidgets.QWidget(self)
        self.plotLayout = QtWidgets.QVBoxLayout(self.plotWidget)
        
        #self.graphView = pg.GraphicsLayoutWidget(self.plotWidget)
        self.scatterWidget = pg.ScatterPlotWidget(self.plotWidget)
        self.scatterWidget.setFixedSize(1000,1000)
        #self.scatterWidget.setFixedSize(self.graph_width, self.graph_height)
        
        test_data = np.random.randint(0,10,(10,3))  # 10 x 3 array of random numbers
        test_data[:,1] *= 5
        test_data[:,2] *= -2
        test_colnames = ['x','y','z']
        test_rownames = [f'unit_{i+1}' for i in range(10)]
        
        # self.test_recarray = np.recarray(shape=(10,3), names=test_colnames,
        #                                  dtype=[('x','float32'), ('y','float32'), ('z','float32')])
        self.test_recarray = np.core.records.array(test_data, 
                                                   dtype=[('x','float32'), ('y','float32'), ('z','float32')], 
                                                   shape=(10,3), names=test_colnames)
        #pos = np.linspace(test_data[:,0].min(), test_data[:,0].max(), test_data.shape[0])
        pos = np.linspace(test_data.min(), test_data.max(), 20)
        color = [clr.get_rgb() for clr in list(Color('red').range_to(Color('white'), 20))]
        cmap = pg.ColorMap(pos, color)
        self.test_fields = [('x', {'mode':'range', 'units':'mV', 'defaults':{'colormap':cmap}}),
                            ('y', {'mode':'range', 'units':'mV', 'defaults':{'colormap':cmap}}),
                            ('z', {'mode':'range', 'units':'mV', 'defaults':{'colormap':cmap}})]
        
        
        
        # Make up some tabular data with structure
        data = np.empty(1000, dtype=[('x_pos', float), ('y_pos', float), 
                                     ('count', int), ('amplitude', float), 
                                     ('decay', float), ('type', 'S10')])
        strings = ['Type-A', 'Type-B', 'Type-C', 'Type-D', 'Type-E']
        typeInds = np.random.randint(5, size=1000)
        data['type'] = np.array(strings)[typeInds]
        data['x_pos'] = np.random.normal(size=1000)
        data['x_pos'][data['type'] == 'Type-A'] -= 1
        data['x_pos'][data['type'] == 'Type-B'] -= 1
        data['x_pos'][data['type'] == 'Type-C'] += 2
        data['x_pos'][data['type'] == 'Type-D'] += 2
        data['x_pos'][data['type'] == 'Type-E'] += 2
        data['y_pos'] = np.random.normal(size=1000) + data['x_pos']*0.1
        data['y_pos'][data['type'] == 'Type-A'] += 3
        data['y_pos'][data['type'] == 'Type-B'] += 3
        data['amplitude'] = data['x_pos'] * 1.4 + data['y_pos'] + np.random.normal(size=1000, scale=0.4)
        data['count'] = (np.random.exponential(size=1000, scale=100) * data['x_pos']).astype(int)
        data['decay'] = np.random.normal(size=1000, scale=1e-3) + data['amplitude'] * 1e-4
        data['decay'][data['type'] == 'Type-A'] /= 2
        data['decay'][data['type'] == 'Type-E'] *= 3
 
 
        self.scatterWidget.setFields([
            ('x_pos', {'units': 'm'}),
            ('y_pos', {'units': 'm'}),
            ('count', {}),
            ('amplitude', {'units': 'V'}),
            ('decay', {'units': 's'}),    
            ('type', {'mode': 'enum', 'values': strings}),
            ])
             
        self.scatterWidget.setData(data)
        self.scatterWidget.show()
        
        #self.scatterWidget.setData(self.test_recarray)
        #self.scatterWidget.setFields(self.test_fields)
        self.plotLayout.addWidget(self.scatterWidget)
        
        self.centralLayout.addWidget(self.plotWidget)
    


class UnitClassificationWindow(QtWidgets.QDialog):
    
    def __init__(self, parent, cat):
        """
        Instantiate QDialog with GraphicsLayoutWidget to hold graphs
        @Params
        parent - main Neuropixel window
        cat - show unit classifications by 'state' or 'pwaves'
        """
        super(UnitClassificationWindow, self).__init__()
        
        if parent is None:
            self.mainWin = MainWindowInfo(ppath, name)
            self.TEST_MODE = True
            self.setWindowTitle('TEST_MODE')
        else:
            self.mainWin = parent
            self.TEST_MODE = False
        self.cat = cat
        
        ###   IMPORTED VARIABLES   ###
        self.ppath = str(self.mainWin.ppath)
        self.name = str(self.mainWin.name)
        self.coors = self.mainWin.coors.copy()
        self.ftime = np.array(self.mainWin.ftime)
        self.fdt = float(self.mainWin.fdt)
        self.lut_brainstate = np.array(self.mainWin.lut_brainstate)
        self.M = np.array(self.mainWin.M)
        self.unotes = dict(self.mainWin.unotes)
        self.auto_unotes = dict(self.mainWin.auto_unotes)
        self.ufilt = {cat : 1 for cat in self.mainWin.ufilt.keys()}
        self.unit_graphs = list(self.mainWin.unit_graphs)
        self.unit_state_df = self.mainWin.unit_state_df
        self.unit_PFR_df = self.mainWin.unit_PFR_df
        ###
        
        self.all_units = []     # all loaded units
        self.hidden_units = []  # units belonging to one or more filtered categories
        self.unit_plots = []    # current unit graphs in the scroll area
        self.selected_ugraph = None  # currently selected (i.e. data shown with matplotlib) unit graph 
        self.current_unotes = {}  # notes dictionary currently being used for classification/filtering
        
        self.fig = plt.figure(figsize=(20, 8.5), constrained_layout=True)
        #self.fig = plt.figure(constrained_layout=True)
        
        ########
        self.gen_layout()
        self.connect_buttons()
    
    
    def gen_layout(self):
        wpx, hpx = pyautogui.size()
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20, wspace50, wspace100, wspace200 = [px_w(w, wpx) for w in [5,10,15,20,50,100,200]]
        hspace5, hspace10, hspace15, hspace20, hspace50, hspace100, hspace200 = [px_h(h, hpx) for h in [5,10,15,20,50,100,200]]
        
        # set contents margins, central layout
        self.setFixedWidth(self.coors.w)
        self.settings_width = px_w(350, wpx)
        self.graph_width = int(self.coors.w - self.settings_width - wspace50)
        # top, right, bottom, left
        self.setContentsMargins(wspace5,hspace5,wspace5,hspace5)
        self.centralLayout = QtWidgets.QHBoxLayout(self)
        # set GUI fonts
        font = QtGui.QFont()
        font.setPointSize(10)
        labelFont = QtGui.QFont()
        labelFont.setPointSize(11)
        labelFont.setBold(True)
        headerFont = QtGui.QFont()
        headerFont.setPointSize(12)
        headerFont.setBold(True)
        headerFont.setUnderline(True)
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.setFont(font)
        
        
        ###   PLOT LAYOUT
        self.plotWidget = QtWidgets.QWidget(self)
        self.plotLayout = QtWidgets.QVBoxLayout(self.plotWidget)
        
        self.graph_height = int(hspace100)   # unit graph height
        self.graph_spacing = int(hspace10)     # spacing between unit graphs
        self.graph_cm = 2                      # unit graph content margins
        self.ywidth = int(wspace50)            # width of y-axis for brainstate/P-wave/unit graphs
        self.xheight = int(hspace50)           # height of x-axis for bottom unit graph
        self.qscroll_maxHeight = int(hspace200 * 4)  # scrolling window heights when SHOWING/HIDING data plots
        self.qscroll_minHeight = int(self.graph_height + self.xheight + self.graph_spacing*2)
        
        # create widget for brainstate plot
        self.brainstateView = pg.GraphicsLayoutWidget(self.plotWidget)
        self.brainstateView.setFixedHeight(self.graph_height)
        self.graph_brainstate = self.brainstateView.addPlot()
        self.graph_brainstate.resize(self.graph_width, self.graph_height)
        img = pg.ImageItem()
        self.graph_brainstate.addItem(img)
        img.setImage(self.M.T)
        # set time scale and color map
        tr = QtGui.QTransform()
        tr.scale(self.fdt, 1)
        img.setTransform(tr)
        img.setLookupTable(self.lut_brainstate)
        img.setLevels([0, 7])
        # set axis params
        self.graph_brainstate.vb.setMouseEnabled(x=True, y=False)
        self.graph_brainstate.hideAxis('bottom')
        self.graph_brainstate.getAxis(name='left').setTicks([[]])
        self.graph_brainstate.getAxis(name='left').setWidth(self.ywidth)
        self.graph_brainstate.getAxis(name='left').setLabel('State', **self.labelStyle)
        set_xlimits(self.graph_brainstate, tmin=self.ftime[0], tmax=self.ftime[-1]+self.fdt, padding=None)
        
        # create widget and scroll area for individual unit plots
        self.graphView = pg.GraphicsLayoutWidget()
        self.graphView.ci.layout.setSpacing(self.graph_spacing)
        self.graphView.wheelEvent = lambda event: None
        self.qscroll = QtWidgets.QScrollArea(self.plotWidget)
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.graphView)
        self.qscroll.setFixedHeight(self.qscroll_maxHeight)
        
        # create matplotlib widget for plotting unit data
        self.canvas = FigureCanvas(self.fig)
        self.canvas.resize(self.graph_width, hspace200*3)
        self.canvas.hide()
        # matplotlib figure settings
        #plt.rcdefaults()
        plt.rc('font', size=15)
        plt.rc('figure.subplot', left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.rc('figure.constrained_layout', w_pad=0.4, h_pad=0.8)
        plt.rc('axes', titlesize=15, titleweight=600, titlepad=20, ymargin=0.1)
        # plt.rc('xtick', labelsize=20)
        # plt.rc('xtick.major', size=0, pad=20)
        
        ###
        self.plotLayout.addWidget(self.brainstateView)
        self.plotLayout.addWidget(self.qscroll)
        self.plotLayout.addWidget(self.canvas)
        
        
        ###   SETTINGS LAYOUT
        self.settingsWidget = QtWidgets.QWidget(self)
        self.settingsWidget.setFixedWidth(self.settings_width)
        self.settingsLayout = QtWidgets.QVBoxLayout(self.settingsWidget)
        self.settingsLayout.setSpacing(hspace10)
        titleHeight = int(hspace15)
        buttonHeight = int(hspace50-hspace10)
        
        # look at all the units from one region
        self.regionWidget = QtWidgets.QWidget()
        self.regionWidget.setFixedHeight(hspace200*3)
        self.regionWidget.setContentsMargins(0,0,0,0)
        self.regionLayout = QtWidgets.QHBoxLayout(self.regionWidget)
        self.regionLayout.setContentsMargins(0,0,0,0)
        self.regionLayout.setSpacing(wspace5)
        #title = QtWidgets.QLabel('pass')
        #title.setAlignment(QtCore.Qt.AlignCenter)
        #title.setFixedHeight(titleHeight)
        #title.setFont(headerFont)
        #lay = QtWidgets.QHBoxLayout()
        
        # create region buttons
        c1 = QtWidgets.QVBoxLayout()
        c1.setSpacing(hspace20)
        hlay1 = QtWidgets.QHBoxLayout()
        label1 = QtWidgets.QLabel('Regions')
        label1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        label1.setFont(labelFont)
        hlay1.addWidget(label1)
        vlay1 = QtWidgets.QVBoxLayout()
        vlay1.setSpacing(hspace10)
        self.region_btns = []
        for i,graph in enumerate(self.unit_graphs):
            btn = RegionButton(graph.name, graph.color, width=int(buttonHeight*2.5), height=int(buttonHeight))
            btn.units = list(graph.units)
            btn.peak_fr = np.array(graph.peak_fr)
            vlay1.addWidget(btn, alignment=QtCore.Qt.AlignCenter)
            self.region_btns.append(btn)
            self.all_units += list(graph.units)
        vlay11 = QtWidgets.QVBoxLayout()
        vlay11.setSpacing(hspace5)
        # create button to filter for units meeting specific criteria
        btnFont = QtGui.QFont()
        btnFont.setPointSize(11)
        btnFont.setWeight(75)
        self.filterUnits_btn = FancyButton2('Filter Units', (160,160,175), width=int(buttonHeight*2.5+wspace10), 
                                            height=int(buttonHeight+hspace5), border_style='outset', border_width=4,
                                            border_cparams=[0.2,0], border_cparams_pressed=[0.5,1], color_pressed=(70,70,70),
                                            text_color_pressed=(200,200,200), font_size=10, font_weight=600)
        self.filterUnits_btn.setCheckable(False)
        self.filterUnits_btn.setFont(btnFont)
        vlay11.addWidget(self.filterUnits_btn, alignment=QtCore.Qt.AlignCenter)
        c1.addLayout(hlay1, stretch=0)
        c1.addLayout(vlay1, stretch=2)
        line = vline('h')
        line.setLineWidth(2)
        c1.addWidget(line, stretch=0)
        c1.addLayout(vlay11, stretch=1)
        
        # create units list
        c2 = QtWidgets.QVBoxLayout()
        hlay2 = QtWidgets.QHBoxLayout()
        label2 = QtWidgets.QLabel('Units')
        label2.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        label2.setFont(labelFont)
        self.plotUnits_btn = QtWidgets.QPushButton('Plot unit(s)')
        self.plotUnits_btn.setCheckable(False)
        self.plotUnits_btn.setEnabled(False)
        hlay2.addWidget(label2)
        hlay2.addWidget(self.plotUnits_btn)
        vlay2 = QtWidgets.QVBoxLayout()
        self.unit_qlist = UnitQList(self)  # QListWidget displaying units
        vlay2.addWidget(self.unit_qlist)
        hlay21 = QtWidgets.QHBoxLayout()
        self.sortQList_btn = QtWidgets.QPushButton('Sort')
        self.sortQList_btn.setCheckable(False)
        self.clearQList_btn = QtWidgets.QPushButton('Clear')
        self.clearQList_btn.setCheckable(False)
        hlay21.addWidget(self.sortQList_btn)
        hlay21.addWidget(self.clearQList_btn)
        c2.addLayout(hlay2, stretch=0)
        c2.addLayout(vlay2, stretch=3)
        c2.addLayout(hlay21, stretch=1)
        
        
        self.regionLayout.addLayout(c1)
        self.regionLayout.addLayout(c2)
        #lay.addLayout(c1)
        #lay.addLayout(c2)
        
        #self.regionLayout.addWidget(title, stretch=0)
        #self.regionLayout.addLayout(lay, stretch=2)
        line = vline('h')
        self.settingsLayout.addWidget(self.regionWidget)
        self.settingsLayout.addWidget(line)
        
        ### other stuff
        lay = QtWidgets.QHBoxLayout()
        c1 = QtWidgets.QVBoxLayout()
        self.showManualClass_btn = QtWidgets.QRadioButton('Show manual classifications')
        self.showAutoClass_btn = QtWidgets.QRadioButton('Show auto-classifications')
        if self.auto_unotes != {}:
            self.showAutoClass_btn.setChecked(True)
            self.current_unotes = dict(self.auto_unotes)
        else:
            self.showManualClass_btn.setChecked(True)
            self.showAutoClass_btn.setEnabled(False)
            self.current_unotes = dict(self.unotes)
        c1.addWidget(self.showManualClass_btn)
        c1.addWidget(self.showAutoClass_btn)
        lay.addLayout(c1)
        self.settingsLayout.addLayout(lay)
        line = vline('h')
        self.settingsLayout.addWidget(line)
        
        ### action buttons
        self.btnsWidget = QtWidgets.QWidget()
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Expanding)
        self.btnsWidget.setSizePolicy(policy)
        self.btnsLayout = QtWidgets.QVBoxLayout(self.btnsWidget)
        self.btnsLayout.setSpacing(hspace5)
        debug_btn = QtWidgets.QPushButton('debug')
        debug_btn.clicked.connect(self.debug)
        self.btnsLayout.addWidget(debug_btn)
        
        self.settingsLayout.addWidget(self.btnsWidget)
        
        
        
        
        self.centralLayout.addWidget(self.plotWidget, stretch=2)
        self.centralLayout.addWidget(self.settingsWidget, stretch=0)
    
    
    def connect_buttons(self):
        # update/sort/clear QListWidget
        for btn in self.region_btns:
            btn.signal.connect(self.unit_qlist.update_qlist_from_btns)
        self.sortQList_btn.clicked.connect(lambda : self.unit_qlist.sortItems())
        self.clearQList_btn.clicked.connect(self.clear_qlist)
        # plot selected units
        self.plotUnits_btn.clicked.connect(self.plot_selected_units)
        # adjust unit filtering params
        self.filterUnits_btn.clicked.connect(self.change_filter_params)
        
        
        self.showAutoClass_btn.toggled.connect(self.switch_classification_file)
    
    
    def update_widget_units(self, update_classifications=False):
        """
        1) User changed the unit filtering params in the filter window --> find new hidden units, add/remove list and graph items
        2) User switched between manual and auto classification --> find new hidden units, add/remove list and graph items, update list icons and graph classlabels
        3) 
        """
        # get new set of hidden units based on new filter params/classification file
        all_peak_fr = np.concatenate([btn.peak_fr for btn in self.region_btns])
        other_unotes = dict(self.auto_unotes) if (self.current_unotes == self.unotes) else dict(self.unotes)
        notes_dicts = [self.current_unotes, other_unotes]
        self.hidden_units = find_hidden_units(self.all_units, notes_dicts, all_peak_fr, self.ufilt)
        
        # clear filtered units from QListWidget, add newly qualifying units
        self.unit_qlist.update_qlist_from_filt()
        
        # remove graphs of filtered units from graphView and unit_plots list
        rm_plots = [ugraph for ugraph in self.unit_plots if ugraph.unit in self.hidden_units]
        _ = [self.graphView.removeItem(ugraph) for ugraph in rm_plots[::-1]]
        _ = [self.unit_plots.remove(ugraph) for ugraph in rm_plots[::-1]]
        
        # set height of GraphicsLayoutWidget based on number of remaining plots
        nplots = len(self.unit_plots)
        self.graphView_height = int(nplots*self.graph_height + self.graph_spacing*(nplots+1) + self.xheight)
        
        # no selected graph
        if self.selected_ugraph is None:
            self.hide_classification_data()
        # graph was already selected/matplotlib data shown
        elif self.selected_ugraph is not None:
            # selected graph was removed
            if self.selected_ugraph.unit in self.hidden_units:
                self.selected_ugraph = None
                self.hide_classification_data()
            # selected graph remains in graphView
            else:
                self.graphView.setFixedHeight(self.graphView_height)
                val = int(self.selected_ugraph.geometry().y() - self.graph_spacing - 1)
                self.qscroll.verticalScrollBar().setSliderPosition(val)
                
        if update_classifications:
            # update all icons in QListWidget
            items = [self.unit_qlist.item(i) for i in range(self.unit_qlist.count())]
            for item in items:
                u = int(item.text())
                notes = self.current_unotes.get(u, ['-','-','-'])
                notes[2] = '-'
                icon = get_unit_icon(notes, self.unit_qlist.icon_widths)
                item.setIcon(icon)
            # update labels in unit plots
            for ugraph in self.unit_plots:
                class_label = [item for item in ugraph.childItems() if item.objectName().split('_')[-1] == 'classlabel'][0]
                #u = int(class_label.objectName().split('_')[0])
                notes = self.current_unotes.get(ugraph.unit, ['-','-','-'])
                class_label.setText(f'{notes[0]} | {notes[1]}')
            
        
    
    def switch_classification_file(self, auto):
        # update active notes dictionary
        if auto:
            self.current_unotes = self.auto_unotes
        else:
            self.current_unotes = self.unotes
        # filter units based on new classifications, update widgets
        self.update_widget_units(update_classifications=True)
    
    
    def change_filter_params(self):
        # create popup window to change filtering params
        popup = UnitFiltWindow(self.ufilt)
        res = popup.exec()
        if res:
            # update filtering param dictionary
            self.ufilt = dict(popup.filt_dict)
            # filter units based on new params, update widgets
            self.update_widget_units(update_classifications=False)
            
            # # if selected graph was deleted, hide matplotlib data and reset selection
            # if self.selected_ugraph is not None and self.selected_ugraph not in self.unit_plots:
            #     self.selected_ugraph = None
            #     self.hide_classification_data()
    def clear_qlist(self):
        # clear all items from QList, make sure all region buttons are toggled off
        self.unit_qlist.clear()
        for btn in self.region_btns:
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
    
    
    def plot_selected_units(self):
        self.canvas.hide()
        self.graphView.clear()
        self.unit_plots = []
        self.selected_ugraph = None
        unit_list = list(self.unit_qlist.selected_units)
        
        # set heights of GraphicsLayoutWidget and scrolling window
        nplots = len(unit_list)
        self.graphView_height = int(nplots*self.graph_height + self.graph_spacing*(nplots+1) + self.xheight)
        self.graphView.setFixedHeight(max([self.graphView_height, self.qscroll_maxHeight]))
        self.qscroll.setFixedHeight(self.qscroll_maxHeight)
        
        
        
        for i,u in enumerate(unit_list):
            # find corresponding graph in main window, get unit firing rate
            graph = [g for g in self.unit_graphs if u in g.units][0]
            row = list(graph.units).index(u)
            
            # add plot for unit
            ugraph = GraphUnitClassificationWindow(unit=u, region=graph.name, color=graph.color)
            ugraph.showSignal.connect(self.show_classification_data)
            ugraph.hideSignal.connect(self.hide_classification_data)
            self.graphView.addItem(ugraph)
            
            ugraph.plot(self.ftime, graph.fr_mx[row,:], pen=graph.color, padding=None)
            
            # set axis params
            if i < len(unit_list)-1:
                ugraph.setFixedHeight(self.graph_height)
                ugraph.resize(self.graph_width, self.graph_height)
                ugraph.hideAxis('bottom')
            else:
                ugraph.setFixedHeight(self.graph_height + self.xheight)
                ugraph.resize(self.graph_width, self.graph_height + self.xheight)
                ugraph.showAxis('bottom')
            ugraph.getAxis(name='bottom').setHeight(self.xheight)
            ugraph.getAxis(name='bottom').setLabel('Time (s)', **self.labelStyle)
            ugraph.getAxis(name='left').setWidth(self.ywidth)
            ugraph.getAxis(name='left').setLabel('FR', **self.labelStyle)
            
            # add labels for unit/region and classification
            ugraph.label = LFPLabel(parent=self.graphView)
            ugraph.label.setObjectName(f'{u}_label')
            ugraph.label.set_info(txt=f'{u}_{graph.name}', color=graph.color)
            ugraph.label.setParentItem(ugraph)
            ugraph.label.anchor(itemPos=(1,0), parentPos=(1,0))
            ugraph.class_label = LFPLabel(parent=self.graphView)
            ugraph.class_label.setObjectName(f'{u}_classlabel')
            notes = self.current_unotes.get(u, ['-','-','-'])
            ugraph.class_label.set_info(txt=f'{notes[0]} | {notes[1]}', color=graph.color, size='10pt')
            ugraph.class_label.setParentItem(ugraph)
            ugraph.class_label.anchor(itemPos=(0,0), parentPos=(0,0), offset=(self.ywidth+5, 0))
            set_xlimits(ugraph, tmin=self.ftime[0], tmax=self.ftime[-1]+self.fdt, padding=None)
            
            self.unit_plots.append(ugraph)
            self.graphView.nextRow()
        # link x-axes of all graphs
        xlink_graphs(self.unit_plots, target_graph=self.graph_brainstate)
        
        # clear QListWidget selection
        #_ = [self.unit_qlist.item(i).setSelected(False) for i in range(self.unit_qlist.count())]
    
    
    def hide_classification_data(self):
        self.canvas.hide()
        # scrolling window height is maximized, graphView height is the same OR larger (stacked plots)
        self.graphView.setFixedHeight(max([self.graphView_height, self.qscroll_maxHeight]))
        self.qscroll.setFixedHeight(self.qscroll_maxHeight)
        
        if len(self.unit_plots) > 0:
            # resize all plots above the bottom
            _ = [ug.setFixedHeight(self.graph_height) for ug in self.unit_plots][0:-1]
            _ = [ug.resize(self.graph_width, self.graph_height) for ug in self.unit_plots][0:-1]
            _ = [ug.hideAxis('bottom') for ug in self.unit_plots][0:-1]
            # resize bottom plot
            self.unit_plots[-1].setFixedHeight(self.graph_height + self.xheight)
            self.unit_plots[-1].resize(self.graph_width, self.graph_height + self.xheight)
            self.unit_plots[-1].showAxis('bottom')
        
        # automatically scroll to the deselected graph
        if self.selected_ugraph is not None:
            val = int(self.selected_ugraph.geometry().y() - self.graph_spacing - 1)
            self.qscroll.verticalScrollBar().setSliderPosition(val)
        # deselected graph has been removed, scroll to the top
        else:
            self.qscroll.verticalScrollBar().setSliderPosition(0)
        self.selected_ugraph = None
                
            
    
    
    def show_classification_data(self):
        self.canvas.show()
        drawCanvas = bool(self.selected_ugraph is not None)
        # scrolling window height is reduced, graphView height is the number of stacked plots
        self.graphView.setFixedHeight(self.graphView_height)
        self.qscroll.setFixedHeight(self.qscroll_minHeight)
        
        # enlarge selected graph & show x axis
        ugraph = self.sender()
        ugraph.showAxis('bottom')
        ugraph.setFixedHeight(self.graph_height + self.xheight)
        # reset size, hide x axis, and deselect all other graphs
        for ug in self.unit_plots:
            if ug == ugraph:
                continue
            ug.hideAxis('bottom')
            ug.setFixedHeight(self.graph_height)
            ug.isSelected = False
            ug.update()
        # automatically scroll to the selected graph, show matplotlib canvas
        val = int(ugraph.geometry().y() - self.graph_spacing - 1)
        self.qscroll.verticalScrollBar().setSliderPosition(val)
        
        # get selected unit info
        self.selected_ugraph = ugraph
        unit = ugraph.unit
        region = ugraph.region
        color = ugraph.color
        
        self.fig.clear()
        #self.fig.set_constrained_layout_pads(w_pad=0.4, h_pad=1.)
        if self.cat == 'state':
            #grid = GridSpec(1, 3, width_ratios=[1,1,2], figure=self.fig)
            axs = self.fig.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,2]})
            ax_FRmean, ax_FRpeak, ax_FRratios = axs
            
            if self.mainWin.unit_state_df is not None:
                ddf = self.unit_state_df.iloc[np.where(self.unit_state_df['unit'] == unit)[0], :].reset_index(drop=True)
                
                # mean FR bar graph
                mean_FRs = np.array(ddf.loc[0, ['R_mean', 'W_mean', 'N_mean']], dtype='float32')
                _ = ax_FRmean.bar(['R','W','N'], mean_FRs, color=['cyan','purple','gray'], edgecolor='black', linewidth=2)
                ax_FRmean.set_ylabel('FR mean')
                ax_FRmean.set_title('Mean Firing Rate')
                sns.despine()
                
                # peak FR bar graph
                peak_FRs = np.array(ddf.loc[0, ['R_peak', 'W_peak', 'N_peak']], dtype='float32')
                _ = ax_FRpeak.bar(['R','W','N'], peak_FRs, color=['cyan','purple','gray'], edgecolor='black', linewidth=2)
                ax_FRpeak.set_ylabel('FR max')
                ax_FRpeak.set_title('Peak Firing Rate')
                sns.despine()
                
                # # firing rate ratios
                ratio_states = ['RW','RN','WN']
                color_tups = [('cyan','purple'), ('cyan','gray'), ('purple','gray')]
                FR_ratios = np.array([float(ddf[f'{rs}_mean']) for rs in ratio_states])
                ratio_colors = [ratio2rgb(r, c[0], c[1], scale=1) for r,c in zip(FR_ratios, color_tups)]
                pal = {k:v for k,v in zip(ratio_states, ratio_colors)}
                ax_FRratios = sns.barplot(x=ratio_states, y=np.round(FR_ratios-1, 1), bottom=1, log=True, 
                                palette=pal, edgecolor='black', linewidth=2, ax=ax_FRratios)
                _ = ax_FRratios.bar_label(ax_FRratios.containers[0], padding=10)
                _ = ax_FRratios.plot(ax_FRratios.get_xlim(), [1,1], color='black', linewidth=3)
                sns.despine()
                ax_FRratios.spines['bottom'].set_visible(False)
                # set x-axis params
                xticks = ax_FRratios.get_xticks()
                ax_FRratios.set_xticks(xticks, labels=[f'{rs[0]} : {rs[1]}' for rs in ratio_states], **{'fontweight':600, 
                                                                                                        'fontsize':15})
                ax_FRratios.tick_params('x', top=False, labeltop=True, bottom=False, labelbottom=False)
                # set y-axis params
                ax_FRratios.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax_FRratios.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
                ax_FRratios.yaxis.get_major_formatter().set_scientific(False)
                ax_FRratios.yaxis.get_minor_formatter().set_scientific(False)
                
                # set y-axis limits so 1 is in the middle
                if ax_FRratios.get_ylim()[1] >= 1/ax_FRratios.get_ylim()[0]:
                    ax_FRratios.set_ylim(bottom=max([0.1, np.round(1/ax_FRratios.get_ylim()[1],1)]), 
                                          top=np.ceil(ax_FRratios.get_ylim()[1]))
                else:
                    ax_FRratios.set_ylim(bottom=max([0.1, np.round(ax_FRratios.get_ylim()[0],1)]), 
                                          top=np.ceil(1/ax_FRratios.get_ylim()[0]))
                ax_FRratios.set_ylabel('log(FR ratio)')
                ax_FRratios.set_title('Firing Rate Ratio (X:Y)')
        
        if drawCanvas:
            self.canvas.draw()
        # try:
        #     self.canvas.draw()
        # except:
        #     print('singular matrix')
        # #self.canvas.draw()
                
                
        
    def closeEvent(self, event):
        plt.close()
        if self.TEST_MODE:
            if len(self.mainWin.open_fid) > 0:
                _ = [f.close() for f in self.mainWin.open_fid]
    
    def debug(self):
        pdb.set_trace()

ppath = '/media/fearthekraken/Mandy_HardDrive1/neuropixel/Processed_Sleep_Recordings'
name = 'DL161_exp2_012623n1'

app = QtWidgets.QApplication([])
x = UnitComparisonWindow(parent=None)
#x = UnitClassificationWindow(parent=None, cat='state')
#x = MainWindowInfo(ppath, name)
x.show()
x.raise_()
sys.exit(app.exec())
