#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom dialog window to create and adjust figure plots

@author: fearthekraken
"""
import os
import re
import scipy
import scipy.io as so
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import pyautogui
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtGui, QtCore
import warnings
import pdb
# custom modules
import sleepy
import AS
import pwaves
import pqt_items as pqi


class FigureWindow(QtWidgets.QDialog):
    def __init__(self, plotType, parent=None):
        """
        Instantiate figure window, set initial variable values
        @Params
        plotType - string specifying the type of figure to plot
        parent - main annotation window
        """
        super(FigureWindow, self).__init__(parent)
        self.WIDTH, self.HEIGHT = pyautogui.size()
        
        self.stateMap = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS-R', 5:'IS-W', 6:'MA'}
        self.mainWin = parent
        self.plotType = plotType
        
        # recording path and name
        self.ppath = str(self.mainWin.ppath)
        self.name = str(self.mainWin.name)
        self.recordings = []
        
        # create dictionary with each plot type, the required widgets, the associated plotting function, and misc info
        self.plotTypeWidgets = {
                                'P-wave frequency'          : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget'], 
                                                               'fx':self.plot_state_freq, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win', 'ci']},
                                'P-waveforms'               : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget'], 
                                                               'fx':self.plot_pwaveforms, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win', 'mouse_avg', 'ci', 'signal']},
                                'P-wave spectrogram'        : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget', 'spWidget'], 
                                                               'fx':self.plot_avg_sp, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win', 'mouse_avg']},
                                'Single P-wave spectrogram' : {'widgets':['dataWidget', 'spWidget'], 
                                                               'fx':self.plot_single_sp, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win']},
                                'P-wave spectrum'           : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget', 'spWidget'],
                                                               'fx':self.plot_pwave_spectrum, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win', 'excl_win', 'ci']},
                                'P-wave EMG'                : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget', 'emgWidget'], 
                                                               'fx':self.plot_avg_pwave_emg, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win', 'mouse_avg', 'ci']},
                                'Single P-wave EMG'         : {'widgets':['dataWidget', 'emgWidget'],
                                                               'fx':self.plot_single_pwave_emg, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win']},
                                'Laser P-wave stats'        : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget', 'laserWidget'], 
                                                               'fx':self.plot_lsr_pwave_stats, 'req_pwaves':True, 'time_res':'state',
                                                               'data_params':['ci']},
                                'P-wave transitions'\
                                    '(time-normalized)'     : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget', 'spWidget'],
                                                               'fx':self.plot_stateseq, 'req_pwaves':True, 'time_res':'state',
                                                               'data_params':['mouse_avg', 'ci', 'sf']},
                                'P-wave \u0394F/F'          : {'widgets':['dataWidget', 'brstateWidget', 'pwaveWidget', 'dffWidget'],
                                                               'fx':self.plot_dff_pwaves, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win','mouse_avg','sf','ci']},
                                'Single P-wave \u0394F/F'   : {'widgets':['dataWidget', 'dffWidget'],
                                                               'fx':self.plot_single_dff_pwave, 'req_pwaves':True, 'time_res':'event',
                                                               'data_params':['win','sf']},
                                '\u0394F/F activity'        : {'widgets':['dataWidget', 'brstateWidget'],
                                                               'fx':self.plot_dff_activity, 'req_pwaves':False, 'time_res':'state',
                                                               'data_params':['mouse_avg','ci', 'zscore']},
                                'Sleep timecourse'          : {'widgets':['sleepWidget', 'brstateWidget', 'pwaveWidget'], 
                                                               'fx':self.plot_sleep_timecourse, 'req_pwaves':False, 'time_res':'state',
                                                               'data_params':['mouse_avg', 'ci']},
                                'Sleep spectrum'            : {'widgets':['dataWidget', 'brstateWidget', 'spWidget'],
                                                               'fx':self.plot_sleep_spectrum, 'req_pwaves':False, 'time_res':'state',
                                                               'data_params':['mouse_avg', 'ci']},
                                'Opto brain state'          : {'widgets':['dataWidget', 'brstateWidget'],
                                                               'fx':self.plot_lsr_brainstate, 'req_pwaves':False, 'time_res':'state',
                                                               'data_params':['pre', 'post', 'mouse_avg', 'ci', 'sf']},
                                'EMG twitches'              : {'widgets':['dataWidget', 'brstateWidget', 'emgWidget', 'twitchWidget'],
                                                               'fx':self.plot_emg_twitches, 'req_pwaves':False, 'time_res':'state',
                                                               'data_params':['mouse_avg', 'twitch_avg', 'ci']}
                                }
        # does recording include P-wave-triggering laser pulses?
        if self.mainWin.lsrTrigPwaves:
            self.plotTypeWidgets['P-waveforms']['widgets'].append('laserWidget')
            self.plotTypeWidgets['P-wave spectrogram']['widgets'].append('laserWidget')
            self.plotTypeWidgets['P-wave EMG']['widgets'].append('laserWidget')
        # does recording include behavioral optogenetic stimulation?
        if self.mainWin.optoMode:
            self.plotTypeWidgets['Sleep spectrum']['widgets'].append('laserSpectrumWidget')
        
        # load plot settings from main window
        self.plotSettings = dict(self.mainWin.plotFigure_settings)
        # load EMG amplitude from main window, re-collect P-wave surrounds
        self.EMG_amp, self.mnbin, self.mdt = list(self.mainWin.EMG_amp_data)
        self.lsr_pwaves, self.spon_pwaves, self.success_lsr, self.fail_lsr, self.p_signal = [None,None,None,None,None]
        
        ### Initialize plotting params ###
        res = self.update_vars_from_dict(ddict=self.plotSettings)
        if res == 0:
            # sleep timecourse params
            self.stat = 'perc'  # type of sleep data to plot
            self.tbin = 18000   # no. seconds per timecourse bin
            self.num_bins = 1   # no. timecourse bins to plot
            
            # data collection params
            if 'P-wave spectrogram' in self.plotType:
                self.win = [-3,3]
            else:
                self.win = [-0.5,0.5]  # time window (s) surrounding event
            self.lsr_win = [0, 1]      # time window (s) surrounding laser pulse
            self.excl_win = [-0.5,0.5] # time window (s) surrounding event to exclude in control data
            self.pre = 120             # time (s) preceding event
            self.post = 240            # time (s) following event
            self.mouse_avg = 'trial'   # average data by 'trial', ['rec']ording, or 'mouse'
            self.twitch_avg = 'all'    # average EMG twitch freq by 'all' REM or 'each' REM period
            self.ci = 'sem'            # set confidence interval
            self.sf = 0                # set data smoothing factor
            self.signal_type = 'LFP'   # set signal type (LFP/EMG/EEG/EEG2)
            self.pzscore = 0           # set z-scoring method
            
            # brain state params
            self.ma_thr = 20     # max duration (s) for microarousals
            self.ma_state = 3    # brain state to assign microarousals
            self.flatten_is = 4  # handle IS-R and IS-W states
            self.tstart = 0      # starting time (s)for analysis
            self.tend = -1       # ending time (s) for analysis
            self.istate = [1,2,3,4]    # brain state(s) to analyze
            self.exclude_noise = True  # exclude or ignore LFP noise
            # state transition params
            self.sequence = [3,4,1,2]     # ordered list of brain states in transition
            self.nstates = [20,20,20,20]  # no. bins for each brain state
            self.state_thres = [0,0,0,0]  # min or max duration (s) of each state
            self.sign = ['>','>','>','>'] # indicator of min ('>') or max ('<') duration
                
            # P-wave params
            self.p_iso = 0     # min inter-P-wave interval for single P-waves
            self.pcluster = 0  # max inter-P-wave interval for cluster P-waves
            self.clus_event = 'waves'  # event to return for P-wave clusters
            self.clus_iso = 0  # eliminate clusters with P-wave/laser within X s
            
            # EEG spectrogram params
            self.nsr_seg = 2             # FFT bin size (s)
            self.perc_overlap = 0.95     # FFT bin overlap (%)
            self.recalc_highres = False  # recalculate high-res spectrogram?
            if self.plotTypeWidgets[self.plotType]['time_res']=='state' \
            or 'spectrum' in self.plotType:
                self.pnorm = 0  # normalization method (0=none, 1=by recording)
            elif self.plotTypeWidgets[self.plotType]['time_res']=='event':
                self.pnorm = 2  # 2=by time window; option for event-centered SP only
            self.psmooth = []   # SP smoothing factors for [x,y] axes
            self.vm = []        # SP saturation
            self.fmax = 25      # max SP freq
            
            # Opto laser params
            self.pmode = int(len(self.mainWin.optoMode)>0) # compare laser-off vs laser-on spectrums?
            self.exclusive_mode = 0    # handle brain states partially overlapping laser
            self.harmcs = 10           # base freq to interpolate harmonics
            self.harmcs_mode = 'iplt'  # interpolate all spectrograms or EMG only?
            self.iplt_level = 1        # interpolate using 1 or 2 neighboring freqs?
            
            # P-wave laser params
            self.plaser = bool(self.mainWin.lsrTrigPwaves)  # compare laser-triggered vs spontaneous P-waves?
            self.post_stim = 0.1      # max latency (s) of laser-triggered P-wave from laser onset
            self.lsr_mode = 'pwaves'  # plot signals surrounding 'pwaves' or 'lsr' pulses
            self.lsr_iso = 0          # eliminate laser pulses preceded by P-waves within X s
            
            # EMG params
            self.emg_source = 'raw'  # use 'raw' EMG or 'msp' for EMG amplitude
            self.nsr_seg_msp = 0.2   # FFT bin size (s) for mSP
            self.perc_overlap_msp = 0.5  # FFT bin overlap (%) for mSP
            self.recalc_amp = False  # recalculate EMG amplitude?
            self.pzscore_emg = 2 # z-score EMG
            self.r_mu = [10,500] # [min,max] freqs in mSP
            self.w0_emg = 0.02   # high-pass raw EMG above X Hz * sr/2
            self.w1_emg = -1     # low-pass raw EMG below X Hz * sr/2
            self.emg_dn = 50     # downsample raw EMG by X bins
            self.emg_sm = 100    # smoothing factor for raw EMG
            
            # EMG twitch params
            self.twitch_thres = 99  # threshold for EMG twitch detection
            self.twitch_thres_type = 'perc'  # threshold method (raw value, st. deviations, or %iles)
            self.twitch_thres_mode = 1   # set threshold for all REM sleep (1) or each REM period (0)
            self.twitch_thres_first = 0  # if >0, use first X s to calculate threshold
            self.recalc_twitches = False # re-detect EMG twitches?
            self.min_twitchdur = 0.1  # min duration (s) of EMG twitch
            self.min_twitchsep = 0.2  # min separation (s) of distinct twitches
            self.min_twitchREM = 10   # min REM duration (s) to look for twitches
            self.twitch_REMcutoff = 2  # no. bins to ignore at end of REM period
            self.twitchFile = []  # name of saved twitch info file
            
            # DF/F params
            self.dff_dn = 500  # downsample DFF signal by X bins
            self.dff_z = 2     # z-scoring method for DFF signal
            self.dff_zwin = [-0.5,0.5]  # z-score by different time window than plot
            self.dff_psm = []  # smoothing factors for [x,y] axes of trial heatmap
            self.dff_vm = []   # saturation of trial heatmap
            
            # replace default EMG params with values from saved file, if possible
            if self.plotType == 'EMG twitches':
                # look for saved twitch settings files
                sfiles = [f for f in os.listdir(os.path.join(self.ppath, self.name)) if f.endswith('.pkl')]
                if len(sfiles) == 1:   # if 1 file found, automatically load settings
                    self.load_twitch_settings(os.path.join(self.ppath, self.name, sfiles[0]))
                elif len(sfiles) > 1:  # if 2+ files found, user selects settings file
                    self.load_twitch_settings()
                elif len(sfiles) == 0:
                    print('No existing EMG settings file found; window initialized with default values')

            self.plotSettings = {}
        
        # adjust initial brainstate and data collection window for specific plots
        REMplots = ['P-wave spectrogram', 'P-wave spectrum', 'P-wave \u0394F/F', 'EMG twitches']
        self.istate = [1] if self.plotType in REMplots else [1,2,3,4]
        if self.plotType == 'Single P-wave \u0394F/F':
            self.dff_psm = []
            self.dff_vm = []
        
        # create plot/settings window, connect buttons
        self.fig = plt.figure(constrained_layout=True)
        self.setup_gui()
        self.connect_buttons()
        self.update_gui_from_vars()
        
        # automatically single-event graphs with default settings
        if 'Single' in self.plotType:
            self.plotFig_btn.click()
            
    
    def setup_gui(self):
        """
        Layout for figure window
        """
        try:
            # set contents margins
            cm = pqi.px_w(11, self.WIDTH)
            self.setContentsMargins(cm,cm,cm,cm)
            # set fonts
            headerFont = QtGui.QFont()
            headerFont.setPointSize(12)
            headerFont.setBold(True)
            headerFont.setWeight(85)
            subheaderFont = QtGui.QFont()
            subheaderFont.setPointSize(10)
            subheaderFont.setBold(True)
            subheaderFont.setUnderline(True)
            font = QtGui.QFont()
            font.setPointSize(9)
            # get set of pixel widths and heights, standardized by monitor dimensions
            titleHeight = pqi.px_h(30, self.HEIGHT)
            wspace1, wspace5, wspace10, wspace15, wspace20 = [pqi.px_w(w, self.WIDTH) for w in [1,5,10,15,20]]
            hspace1, hspace5, hspace10, hspace15, hspace20 = [pqi.px_h(h, self.HEIGHT) for h in [1,5,10,15,20]]
            
            self.centralLayout = QtWidgets.QHBoxLayout(self)
            self.centralLayout.setSpacing(hspace20)
            
            ### PLOT LAYOUT ###
            self.plotWidget = QtWidgets.QWidget()
            self.plotWidget.setMinimumWidth(pqi.px_w(500, self.WIDTH))
            self.plotLayout = QtWidgets.QVBoxLayout(self.plotWidget)
            self.plotLayout.setContentsMargins(0,0,0,0)
            self.plotLayout.setSpacing(0)
            # matplotlib toolbar & canvas to display current figure
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            ### Action buttons 1 ###
            self.btnsWidget1 = QtWidgets.QFrame()
            self.btnsWidget1.setFrameShape(QtWidgets.QFrame.Panel)
            self.btnsWidget1.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.btnsWidget1.setLineWidth(5)
            self.btnsWidget1.setMidLineWidth(3)
            self.btnsWidget1.setFixedHeight(pqi.px_h(50, self.HEIGHT))
            self.btnsLayout1 = QtWidgets.QHBoxLayout(self.btnsWidget1)
            self.btnsLayout1.setContentsMargins(cm,0,cm,0)
            self.btnsLayout1.setSpacing(wspace15)
            self.plotRec_btn = QtWidgets.QPushButton('Plot Recording')
            self.plotRec_btn.setFont(font)
            self.plotRec_btn.setDisabled('Single' in self.plotType)
            self.plotExp_btn = QtWidgets.QPushButton('Plot Experiment')
            self.plotExp_btn.setFont(font)
            self.plotExp_btn.setDisabled('Single' in self.plotType)
            debug_btn = QtWidgets.QPushButton('DEBUG')
            debug_btn.setFont(font)
            debug_btn.clicked.connect(self.debug)
            self.btnsLayout1.addWidget(self.plotRec_btn)
            self.btnsLayout1.addWidget(self.plotExp_btn)
            self.btnsLayout1.addWidget(debug_btn)
            self.plotLayout.addWidget(self.toolbar)
            self.plotLayout.addWidget(self.canvas)
            self.plotLayout.addWidget(self.btnsWidget1)
            self.centralLayout.addWidget(self.plotWidget)
            
            ### SETTINGS LAYOUT ###
            self.settingsWidget = QtWidgets.QFrame()
            self.settingsWidget.setFrameShape(QtWidgets.QFrame.Box)
            self.settingsWidget.setFrameShadow(QtWidgets.QFrame.Raised)
            self.settingsWidget.setLineWidth(4)
            self.settingsWidget.setMidLineWidth(2)
            self.settingsLayout = QtWidgets.QVBoxLayout(self.settingsWidget)
            # title
            self.titleWidget = QtWidgets.QWidget()
            self.titleWidget.setFixedHeight(pqi.px_h(50, self.HEIGHT))
            self.titleLayout = QtWidgets.QVBoxLayout(self.titleWidget)
            self.titleLayout.setSpacing(hspace10)
            settingsTitle = QtWidgets.QLabel('FIGURE PARAMETERS')
            settingsTitle.setAlignment(QtCore.Qt.AlignCenter)
            settingsTitle.setFont(headerFont)
            line_0 = pqi.vline(orientation='h')
            line_0.setFrameShadow(QtWidgets.QFrame.Plain)
            self.titleLayout.addWidget(settingsTitle)
            self.titleLayout.addWidget(line_0)
            self.settingsLayout.addWidget(self.titleWidget)
            
            ### Sleep timecourse params ###
            self.sleepWidget = QtWidgets.QWidget()
            self.sleepWidget.setFixedHeight(pqi.px_h(110, self.HEIGHT))
            self.sleepLayout = QtWidgets.QVBoxLayout(self.sleepWidget)
            self.sleepLayout.setContentsMargins(cm,0,cm,cm)
            self.sleepLayout.setSpacing(hspace10)
            lay1 = QtWidgets.QHBoxLayout()
            lay1.setSpacing(wspace20)
            title1 = QtWidgets.QLabel('Sleep Behavior Parameters')
            title1.setAlignment(QtCore.Qt.AlignCenter)
            title1.setFixedHeight(titleHeight)
            title1.setFont(subheaderFont)
            c1 = QtWidgets.QVBoxLayout()
            c1.setSpacing(hspace1)
            # plot statistic label + dropdown
            plotStat_label = QtWidgets.QLabel('Statistic')
            plotStat_label.setAlignment(QtCore.Qt.AlignCenter)
            plotStat_label.setFont(font)
            self.plotStat_type = QtWidgets.QComboBox()
            self.plotStat_type.setFont(font)
            self.plotStat_type.addItems(['% time in state', 'State frequency', 
                                         'State duration', 'IS-->REM prob.', 
                                         'P-wave frequency'])
            c1.addWidget(plotStat_label)
            c1.addWidget(self.plotStat_type)
            c2 = QtWidgets.QVBoxLayout()
            c2.setSpacing(hspace1)
            # time bin size (s) / no. time bins
            self.binSize_btn = QtWidgets.QCheckBox('Bin size')
            self.binSize_btn.setFont(font)
            self.binSize_val = QtWidgets.QDoubleSpinBox()
            self.binSize_val.setFont(font)
            self.binSize_val.setDecimals(0)
            self.binSize_val.setMaximum(int(len(self.mainWin.EEG) / self.mainWin.sr))
            self.binSize_val.setSuffix(' s')
            c2.addWidget(self.binSize_btn)
            c2.addWidget(self.binSize_val)
            c3 = QtWidgets.QVBoxLayout()
            c3.setSpacing(hspace1)
            self.binNum_label = QtWidgets.QLabel('# of bins')
            self.binNum_label.setFont(font)
            self.binNum_val = QtWidgets.QDoubleSpinBox()
            self.binNum_val.setFont(font)
            self.binNum_val.setDecimals(0)
            c3.addWidget(self.binNum_label)
            c3.addWidget(self.binNum_val)
            lay1.addLayout(c1)
            lay1.addSpacing(wspace5)
            lay1.addLayout(c2)
            lay1.addLayout(c3)
            line_1 = pqi.vline(orientation='h')
            self.sleepLayout.addWidget(title1)
            self.sleepLayout.addLayout(lay1)
            self.sleepLayout.addWidget(line_1)
            if 'sleepWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.sleepWidget)
            
            ### Data collection windows ###
            self.dataWidget = QtWidgets.QWidget()
            self.dataWidget.setFixedHeight(pqi.px_h(105, self.HEIGHT))
            self.dataLayout = QtWidgets.QVBoxLayout(self.dataWidget)
            self.dataLayout.setContentsMargins(0,0,0,cm)
            self.dataLayout.setSpacing(hspace10)
            lay2 = QtWidgets.QHBoxLayout()
            lay2.setContentsMargins(0,0,0,0)
            title2 = QtWidgets.QLabel('Data Collection Parameters')
            title2.setAlignment(QtCore.Qt.AlignCenter)
            title2.setFixedHeight(titleHeight)
            title2.setFont(subheaderFont)
            # time window (s) to collect relative to P-waves
            c1_w = QtWidgets.QWidget()
            c1_w.setMinimumWidth(pqi.px_w(110, self.WIDTH))
            c1 = QtWidgets.QVBoxLayout(c1_w)
            c1.setContentsMargins(0,0,0,0)
            c1.setSpacing(hspace5)
            self.win_label = QtWidgets.QLabel('P-wave Window')
            self.win_label.setFont(font)
            self.win_label.setAlignment(QtCore.Qt.AlignCenter)
            c1r2 = QtWidgets.QHBoxLayout()
            c1r2.setSpacing(0)
            self.preWin_val = QtWidgets.QDoubleSpinBox()
            self.preWin_val.setFont(font)
            self.preWin_val.setMinimum(-500)
            self.preWin_val.setMaximum(0)
            self.preWin_val.setDecimals(1)
            self.preWin_val.setSingleStep(0.1)
            self.preWin_val.setSuffix (' s')
            win_dash = QtWidgets.QLabel(' - ')
            win_dash.setFont(font)
            win_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.postWin_val = QtWidgets.QDoubleSpinBox()
            self.postWin_val.setFont(font)
            self.postWin_val.setMinimum(0)
            self.postWin_val.setMaximum(500)
            self.postWin_val.setDecimals(1)
            self.postWin_val.setSingleStep(0.1)
            self.postWin_val.setSuffix (' s')
            c1r2.addWidget(self.preWin_val, stretch=2)
            c1r2.addWidget(win_dash, stretch=0)
            c1r2.addWidget(self.postWin_val, stretch=2)
            c1.addWidget(self.win_label)
            c1.addLayout(c1r2)
            # time window (s) to collect relative to laser pulses
            c2_w = QtWidgets.QWidget()
            c2_w.setMinimumWidth(pqi.px_w(110, self.WIDTH))
            c2 = QtWidgets.QVBoxLayout(c2_w)
            c2.setContentsMargins(0,0,0,0)
            c2.setSpacing(hspace5)
            self.winLsr_label = QtWidgets.QLabel('Laser Window')
            self.winLsr_label.setFont(font)
            self.winLsr_label.setAlignment(QtCore.Qt.AlignCenter)
            c2r2 = QtWidgets.QHBoxLayout()
            c2r2.setSpacing(0)
            self.preWinLsr_val = QtWidgets.QDoubleSpinBox()
            self.preWinLsr_val.setFont(font)
            self.preWinLsr_val.setMinimum(-500)
            self.preWinLsr_val.setMaximum(0)
            self.preWinLsr_val.setDecimals(1)
            self.preWinLsr_val.setSingleStep(0.1)
            self.preWinLsr_val.setSuffix (' s')
            self.winLsr_dash = QtWidgets.QLabel(' - ')
            self.winLsr_dash.setFont(font)
            self.winLsr_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.postWinLsr_val = QtWidgets.QDoubleSpinBox()
            self.postWinLsr_val.setFont(font)
            self.postWinLsr_val.setMinimum(0)
            self.postWinLsr_val.setMaximum(500)
            self.postWinLsr_val.setDecimals(1)
            self.postWinLsr_val.setSingleStep(0.1)
            self.postWinLsr_val.setSuffix (' s')
            c2r2.addWidget(self.preWinLsr_val, stretch=2)
            c2r2.addWidget(self.winLsr_dash, stretch=0)
            c2r2.addWidget(self.postWinLsr_val, stretch=2)
            c2.addWidget(self.winLsr_label)
            c2.addLayout(c2r2)
            # time window (s) to exclude when collecting control (non-event) data
            c3_w = QtWidgets.QWidget()
            c3_w.setMinimumWidth(pqi.px_w(110, self.WIDTH))
            c3 = QtWidgets.QVBoxLayout(c3_w)
            c3.setContentsMargins(0,0,0,0)
            c3.setSpacing(hspace5)
            self.winExcl_label = QtWidgets.QLabel('Exclusion Window')
            self.winExcl_label.setFont(font)
            self.winExcl_label.setAlignment(QtCore.Qt.AlignCenter)
            c3r2 = QtWidgets.QHBoxLayout()
            c3r2.setSpacing(0)
            self.preWinExcl_val = QtWidgets.QDoubleSpinBox()
            self.preWinExcl_val.setFont(font)
            self.preWinExcl_val.setMinimum(-500)
            self.preWinExcl_val.setMaximum(0)
            self.preWinExcl_val.setDecimals(1)
            self.preWinExcl_val.setSingleStep(0.1)
            self.preWinExcl_val.setSuffix (' s')
            self.winExcl_dash = QtWidgets.QLabel(' - ')
            self.winExcl_dash.setFont(font)
            self.winExcl_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.postWinExcl_val = QtWidgets.QDoubleSpinBox()
            self.postWinExcl_val.setFont(font)
            self.postWinExcl_val.setMinimum(0)
            self.postWinExcl_val.setMaximum(500)
            self.postWinExcl_val.setDecimals(1)
            self.postWinExcl_val.setSingleStep(0.1)
            self.postWinExcl_val.setSuffix (' s')
            c3r2.addWidget(self.preWinExcl_val, stretch=2)
            c3r2.addWidget(self.winExcl_dash, stretch=0)
            c3r2.addWidget(self.postWinExcl_val, stretch=2)
            c3.addWidget(self.winExcl_label)
            c3.addLayout(c3r2)
            # data averaging method dropdown
            c4_w = QtWidgets.QWidget()
            c4_w.setMinimumWidth(pqi.px_w(83, self.WIDTH))
            c4 = QtWidgets.QVBoxLayout(c4_w)
            c4.setContentsMargins(0,0,0,0)
            c4.setSpacing(hspace5)
            dataAvg_label = QtWidgets.QLabel('Avg. by')
            dataAvg_label.setFont(font)
            dataAvg_label.setAlignment(QtCore.Qt.AlignCenter)
            self.dataAvg_type = QtWidgets.QComboBox()
            self.dataAvg_type.setFont(font)
            self.dataAvg_type.addItems(['Trial', 'Recording', 'Mouse'])
            c4.addWidget(dataAvg_label)
            c4.addWidget(self.dataAvg_type)
            # EMG twitch averaging method dropdown
            c5_w = QtWidgets.QWidget()
            c5 = QtWidgets.QVBoxLayout(c5_w)
            c5.setContentsMargins(0,0,0,0)
            c5.setSpacing(hspace5)
            twitchAvg_label = QtWidgets.QLabel('Average across')
            twitchAvg_label.setFont(font)
            twitchAvg_label.setAlignment(QtCore.Qt.AlignCenter)
            self.twitchAvg_type = QtWidgets.QComboBox()
            self.twitchAvg_type.setFont(font)
            self.twitchAvg_type.addItems(['All REM sleep', 'Each REM period'])
            c5.addWidget(twitchAvg_label)
            c5.addWidget(self.twitchAvg_type)
            # error bar type dropdown
            c6_w = QtWidgets.QWidget()
            c6_w.setMinimumWidth(pqi.px_w(70, self.WIDTH))
            c6 = QtWidgets.QVBoxLayout(c6_w)
            c6.setContentsMargins(0,0,0,0)
            c6.setSpacing(hspace5)
            error_label = QtWidgets.QLabel('Error')
            error_label.setAlignment(QtCore.Qt.AlignCenter)
            error_label.setFont(font)
            self.error_type = QtWidgets.QComboBox()
            self.error_type.setFont(font)
            self.error_type.addItems(['S.E.M.', 'S.D.', '95% CI'])
            c6.addWidget(error_label)
            c6.addWidget(self.error_type)
            # data smoothing factor
            c7_w = QtWidgets.QWidget()
            c7_w.setMinimumWidth(pqi.px_w(70, self.WIDTH))
            c7 = QtWidgets.QVBoxLayout(c7_w)
            c7.setContentsMargins(0,0,0,0)
            c7.setSpacing(hspace5)
            smoothing_label = QtWidgets.QLabel('Smooth')
            smoothing_label.setFont(font)
            smoothing_label.setAlignment(QtCore.Qt.AlignCenter)
            self.smooth_val = QtWidgets.QDoubleSpinBox()
            self.smooth_val.setFont(font)
            self.smooth_val.setMaximum(9999)
            self.smooth_val.setDecimals(0)
            c7.addWidget(smoothing_label)
            c7.addWidget(self.smooth_val)
            # signal type dropdown
            c8_w = QtWidgets.QWidget()
            c8_w.setMinimumWidth(pqi.px_w(55, self.WIDTH))
            c8 = QtWidgets.QVBoxLayout(c8_w)
            c8.setContentsMargins(0,0,0,0)
            c8.setSpacing(hspace5)
            signal_label = QtWidgets.QLabel('Signal')
            signal_label.setFont(font)
            signal_label.setAlignment(QtCore.Qt.AlignCenter)
            self.signalType = QtWidgets.QComboBox()
            self.signalType.setFont(font)
            self.signalType.addItems(['LFP', 'EMG', 'EEG', 'EEG2'])
            c8.addWidget(signal_label)
            c8.addWidget(self.signalType)
            # z-scoring method dropdown
            c9_w = QtWidgets.QWidget()
            c9_w.setMinimumWidth(pqi.px_w(90, self.WIDTH))
            c9 = QtWidgets.QVBoxLayout(c9_w)
            c9.setContentsMargins(0,0,0,0)
            c9.setSpacing(hspace5)
            z_label = QtWidgets.QLabel('Z-score')
            z_label.setFont(font)
            z_label.setAlignment(QtCore.Qt.AlignCenter)
            self.z_type = QtWidgets.QComboBox()
            self.z_type.setFont(font)
            self.z_type.addItems(['None', 'Recording'])
            c9.addWidget(z_label)
            c9.addWidget(self.z_type)
            
            # add widgets to layout
            lay2.addWidget(c1_w)
            lay2.addWidget(c2_w)
            lay2.addWidget(c3_w)
            lay2.addWidget(c4_w)
            lay2.addWidget(c5_w)
            lay2.addWidget(c6_w)
            lay2.addWidget(c7_w)
            lay2.addWidget(c8_w)
            lay2.addWidget(c9_w)
            # hide params not used in plot
            if 'win' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c1_w.hide()
            if 'lsr_win' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c2_w.hide()
            if 'excl_win' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c3_w.hide()
            if 'mouse_avg' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c4_w.hide()
            if 'twitch_avg' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c5_w.hide()
            if 'ci' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c6_w.hide()
            if 'sf' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c7_w.hide()
            if 'signal' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c8_w.hide()
            if 'zscore' not in self.plotTypeWidgets[self.plotType]['data_params']:
                c9_w.hide()
            lay2.setSpacing(wspace10)
            line_2 = pqi.vline(orientation='h')
            self.dataLayout.addWidget(title2)
            self.dataLayout.addLayout(lay2)
            self.dataLayout.addWidget(line_2)
            if 'dataWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.dataWidget)
            
            ### Brain state parameters ###
            self.brstateWidget = QtWidgets.QWidget()
            self.brstateWidget.setFixedHeight(pqi.px_h(240, self.HEIGHT))
            self.brstateLayout = QtWidgets.QVBoxLayout(self.brstateWidget)
            self.brstateLayout.setContentsMargins(wspace5,0,wspace5,cm)
            self.brstateLayout.setSpacing(hspace15)
            lay3 = QtWidgets.QVBoxLayout()
            lay3.setSpacing(hspace10)
            title3 = QtWidgets.QLabel('Brain State Parameters')
            title3.setAlignment(QtCore.Qt.AlignCenter)
            title3.setFixedHeight(titleHeight)
            title3.setFont(subheaderFont)
            # state(s) to use in figure ($istate param)
            istateLayout = QtWidgets.QHBoxLayout()
            istateLayout.setSpacing(wspace10)
            r1_b1w = QtWidgets.QFrame()
            r1_b1w.setFixedHeight(pqi.px_h(50, self.HEIGHT))
            r1_b1w.setFrameShape(QtWidgets.QFrame.Panel)
            r1_b1w.setFrameShadow(QtWidgets.QFrame.Sunken)
            r1_b1 = QtWidgets.QGridLayout(r1_b1w)
            r1_b1.setContentsMargins(cm,hspace1*2,cm,hspace1*2)
            r1_b1.setVerticalSpacing(hspace1)
            rows=[0,0,0,1,1,1]; cols=[0,1,2,0,1,2]
            self.state_btn_grp1 = QtWidgets.QButtonGroup(self.brstateWidget)
            self.plotStates = {}
            for i,state in self.stateMap.items():
                # create checkbox for each brain state
                btn = QtWidgets.QCheckBox(state, self.brstateWidget)
                btn.setFont(font)
                self.state_btn_grp1.addButton(btn)
                r1_b1.addWidget(btn, rows[i-1], cols[i-1])
                self.plotStates[i] = btn
                if state != 'REM':
                    btn.setDisabled(self.plotType == 'EMG twitches')
            self.state_btn_grp1.setExclusive(self.plotType in ['P-wave spectrogram',
                                                               'P-wave spectrum',
                                                               'P-wave \u0394F/F', 
                                                               'EMG twitches'])
            r1_b2w = QtWidgets.QFrame()
            r1_b2w.setFixedHeight(pqi.px_h(50, self.HEIGHT))
            r1_b2w.setFrameShape(QtWidgets.QFrame.Panel)
            r1_b2w.setFrameShadow(QtWidgets.QFrame.Sunken)
            r1_b2 = QtWidgets.QVBoxLayout(r1_b2w)
            r1_b2.setContentsMargins(cm,hspace1*2,cm,hspace1*2)
            r1_b2.setSpacing(hspace1)
            # plot "each" state separately or "all" states combined
            state_btn_grp2 = QtWidgets.QButtonGroup(self.brstateWidget)
            self.plotEachState_btn = QtWidgets.QRadioButton('Each', self.brstateWidget)
            self.plotEachState_btn.setFont(font)
            self.plotAllStates_btn = QtWidgets.QRadioButton('All', self.brstateWidget)
            self.plotAllStates_btn.setFont(font)
            state_btn_grp2.addButton(self.plotEachState_btn)
            state_btn_grp2.addButton(self.plotAllStates_btn)
            r1_b2.addWidget(self.plotEachState_btn)
            r1_b2.addWidget(self.plotAllStates_btn)
            tmp = ['P-wave spectrogram', 'P-wave spectrum', 'Sleep timecourse', 'EMG twitches']
            self.plotEachState_btn.setDisabled(self.plotType in tmp)
            self.plotAllStates_btn.setDisabled(self.plotType in tmp)
            self.plotEachState_btn.setChecked(self.plotType not in tmp)
            istateLayout.addWidget(r1_b1w, stretch=2)
            istateLayout.addWidget(r1_b2w, stretch=0)
            # sequence of states in brain state transitions ($sequence param)
            stateseqLayout = QtWidgets.QGridLayout()
            stateseqLayout.setContentsMargins(0,0,0,0)
            stateseqLayout.setVerticalSpacing(0)
            stateseqLayout.setHorizontalSpacing(wspace5 - wspace1)
            wspc,hspc = [pqi.px_w(100, self.WIDTH), pqi.px_h(50, self.HEIGHT)]
            qspacer = QtWidgets.QSpacerItem(wspc,hspc,QtWidgets.QSizePolicy.Maximum)
            stateseqLayout.addItem(qspacer, 0, 0, 3, 1)
            maxStates = 4 if 'time-normalized' in self.plotType else 2 
            data = [d[0:maxStates] for d in [self.sequence, self.nstates, self.sign, self.state_thres]]
            self.sequence, self.nstates, self.sign, self.state_thres = data
            self.stateseqWidgets = []
            for i in range(maxStates):
                # widget containing sub-widgets with state info
                widget = QtWidgets.QWidget(parent=self)
                widget.setFixedWidth(pqi.px_w(70, self.WIDTH))
                layout = QtWidgets.QVBoxLayout(widget)
                layout.setSpacing(0)
                layout.setContentsMargins(0,0,0,0)
                # brain state dropdown
                state = QtWidgets.QComboBox(parent=widget)
                state.setFont(font)
                state.setObjectName('state')
                state.addItems(list(self.stateMap.values()))
                state.setCurrentIndex(0)
                # no. absolute or normalized time bins in state
                bins = QtWidgets.QDoubleSpinBox(parent=widget)
                bins.setFont(font)
                bins.setObjectName('bins')
                bins.setDecimals(0 if 'time-normalized' in self.plotType else 1)
                bins.setSuffix(' bins' if 'time-normalized' in self.plotType else 's')
                bins.setValue(0)
                hbox = QtWidgets.QHBoxLayout()
                hbox.setSpacing(0)
                hbox.setContentsMargins(0,0,0,0)
                # sign of state duration threshold (max or min value)
                sign = QtWidgets.QPushButton(parent=widget)
                sign.setFont(font)
                sign.setObjectName('sign')
                sign.setFixedWidth(wspace20)
                sign.setFocusPolicy(QtCore.Qt.NoFocus)
                sign.setText('>')
                # state duration threshold (s)
                thres = QtWidgets.QDoubleSpinBox(parent=widget)
                thres.setFont(font)
                thres.setObjectName('thres')
                thres.setDecimals(0)
                thres.setSuffix (' s')
                thres.setValue(0)
                hbox.addWidget(sign)
                hbox.addWidget(thres)
                layout.addWidget(state)
                layout.addWidget(bins)
                layout.addLayout(hbox)
                widget.hide()
                stateseqLayout.addWidget(widget, 0, i*2+1, 3, 1)
                if 'transitions' in self.plotType:
                    self.stateseqWidgets.append(widget)
                # arrow label between states
                if i < maxStates-1:
                    arrow = QtWidgets.QPushButton(parent=self)
                    arrow.setObjectName('arrow' + str(i+1))
                    arrow.setStyleSheet('QPushButton'
                                        '{ background-color : transparent;'
                                        'border : none;'
                                        'image : url("icons/arrowR_icon.png");'
                                        'image-position : center }')
                    arrow.setFixedWidth(wspace10)
                    stateseqLayout.addWidget(arrow, 0, i*2+2, 1, 1, 
                                             alignment=QtCore.Qt.AlignCenter)
                    arrow.hide()
            # buttons to add or subtract state
            self.addState_btn = QtWidgets.QPushButton()
            self.addState_btn.setStyleSheet('QPushButton'
                                            '{ background-color : rgba(240,240,240,255);'
                                            'border : 2px outset gray;'
                                            'image : url("icons/plusEnabled_icon.png");'
                                            'image-position : center;'
                                            'margin-left : 0px;'
                                            f'max-width : {pqi.px_w(12, self.WIDTH)}px;'
                                            'padding : 2px }'
                                            'QPushButton:pressed'
                                            '{ background-color : rgba(200,200,200,255);'
                                            'image : url("icons/plusEnabled_icon.png") }'
                                            'QPushButton:disabled'
                                            '{ background-color : rgba(220,220,220,255);'
                                            'image : url("icons/plusDisabled_icon.png") }')
            self.addState_btn.setDisabled(len(self.sequence)==maxStates)
            self.removeState_btn = QtWidgets.QPushButton()
            self.removeState_btn.setStyleSheet('QPushButton'
                                               '{ background-color : rgba(240,240,240,255);'
                                                'border : 2px outset gray;'
                                                'image : url("icons/minusEnabled_icon.png");'
                                                'image-position : center;'
                                                'margin-left : 0px;'
                                                f'max-width : {pqi.px_w(12, self.WIDTH)}px;'
                                                'padding : 2px }'
                                                'QPushButton:pressed'
                                                '{ background-color : rgba(200,200,200,255);'
                                                'image : url("icons/minusEnabled_icon.png") }'
                                                'QPushButton:disabled'
                                                '{ background-color : rgba(220,220,220,255);'
                                                'image : url("icons/minusDisabled_icon.png") }')
            self.removeState_btn.setDisabled(len(self.sequence)==2)
            qspacer = QtWidgets.QSpacerItem(wspc,hspc,QtWidgets.QSizePolicy.Maximum)
            if maxStates > 2:
                stateseqLayout.addWidget(self.addState_btn, 0, 8, 1, 1, 
                                         alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
                stateseqLayout.addWidget(self.removeState_btn, 1, 8, 1, 1, 
                                         alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
                stateseqLayout.addItem(qspacer, 0, 9, 3, 1)
            else:
                stateseqLayout.addItem(qspacer, 0, 8, 3, 1)
            
            # adjust brainstate annotation to handle MAs/transition states/noise
            r2 = QtWidgets.QGridLayout()
            r2.setVerticalSpacing(hspace5)
            r2.setHorizontalSpacing(wspace15)
            b1 = QtWidgets.QVBoxLayout()
            b1.setSpacing(hspace1)
            maState_label = QtWidgets.QLabel('MA brain state')
            maState_label.setFont(font)
            maState_label.setAlignment(QtCore.Qt.AlignCenter)
            self.maState_type = QtWidgets.QComboBox()
            self.maState_type.setFont(font)
            baw = pqi.px_w(95, self.WIDTH)
            self.maState_type.setFixedWidth(baw)
            self.maState_type.addItems(['NREM','Wake','MA'])
            b1.addWidget(maState_label)
            b1.addWidget(self.maState_type)
            b2 = QtWidgets.QVBoxLayout()
            b2.setSpacing(hspace1)
            maThr_label = QtWidgets.QLabel('MA threshold')
            maThr_label.setFont(font)
            maThr_label.setAlignment(QtCore.Qt.AlignCenter)
            self.maThr_val = QtWidgets.QDoubleSpinBox()
            self.maThr_val.setFont(font)
            self.maThr_val.setFixedWidth(baw)
            self.maThr_val.setDecimals(1)
            self.maThr_val.setSuffix(' s')
            b2.addWidget(maThr_label)
            b2.addWidget(self.maThr_val)
            b3 = QtWidgets.QVBoxLayout()
            b3.setSpacing(hspace1)
            isState_label = QtWidgets.QLabel('IS brain state')
            isState_label.setFont(font)
            isState_label.setAlignment(QtCore.Qt.AlignCenter)
            self.isState_type = QtWidgets.QComboBox()
            self.isState_type.setFont(font)
            self.isState_type.setFixedWidth(baw)
            self.isState_type.addItems(['IS','IS-R & IS-W','NREM','REM'])
            b3.addWidget(isState_label)
            b3.addWidget(self.isState_type)
            b4 = QtWidgets.QVBoxLayout()
            b4.setSpacing(hspace1)
            noiseHandle_label = QtWidgets.QLabel('Handle noise')
            noiseHandle_label.setFont(font)
            noiseHandle_label.setAlignment(QtCore.Qt.AlignCenter)
            self.noiseHandle_type = QtWidgets.QComboBox()
            self.noiseHandle_type.setFont(font)
            self.noiseHandle_type.setFixedWidth(baw)
            self.noiseHandle_type.addItems(['Exclude','Ignore'])
            b4.addWidget(noiseHandle_label)
            b4.addWidget(self.noiseHandle_type)
            b5 = QtWidgets.QVBoxLayout()
            b5.setSpacing(hspace1)
            tstart_label = QtWidgets.QLabel('Start time')
            tstart_label.setFont(font)
            tstart_label.setAlignment(QtCore.Qt.AlignCenter)
            self.tstart_val = QtWidgets.QDoubleSpinBox()
            self.tstart_val.setFont(font)
            baw2 = pqi.px_w(75, self.WIDTH)
            self.tstart_val.setFixedWidth(baw2)
            self.tstart_val.setDecimals(0)
            self.tstart_val.setMaximum(int(len(self.mainWin.EEG) / self.mainWin.sr))
            self.tstart_val.setSuffix(' s')
            b5.addWidget(tstart_label)
            b5.addWidget(self.tstart_val)
            b6 = QtWidgets.QVBoxLayout()
            b6.setSpacing(hspace1)
            tend_label = QtWidgets.QLabel('End time')
            tend_label.setFont(font)
            tend_label.setAlignment(QtCore.Qt.AlignCenter)
            self.tend_val = QtWidgets.QDoubleSpinBox()
            self.tend_val.setFont(font)
            self.tend_val.setFixedWidth(baw2)
            self.tend_val.setDecimals(0)
            self.tend_val.setMinimum(-1)
            self.tend_val.setMaximum(int(len(self.mainWin.EEG) / self.mainWin.sr))
            self.tend_val.setSuffix(' s')
            b6.addWidget(tend_label)
            b6.addWidget(self.tend_val)
            r2.addLayout(b1, 0, 0)
            r2.addLayout(b2, 0, 1)
            r2.addLayout(b3, 1, 0)
            r2.addLayout(b4, 1, 1)
            r2.addLayout(b5, 0, 2)
            r2.addLayout(b6, 1, 2)
            if 'transitions' in self.plotType:
                lay3.addLayout(stateseqLayout)
            else:
                lay3.addLayout(istateLayout)
            lay3.addLayout(r2)
            line_3 = pqi.vline(orientation='h')
            self.brstateLayout.addWidget(title3)
            self.brstateLayout.addLayout(lay3)
            self.brstateLayout.addWidget(line_3)
            if 'brstateWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.brstateWidget)
            
            ### P-wave parameters ###
            self.pwaveWidget = QtWidgets.QWidget()
            self.pwaveWidget.setFixedHeight(pqi.px_h(185, self.HEIGHT))
            self.pwaveLayout = QtWidgets.QVBoxLayout(self.pwaveWidget)
            self.pwaveLayout.setContentsMargins(cm,0,cm,cm)
            self.pwaveLayout.setSpacing(hspace15)
            lay4 = QtWidgets.QHBoxLayout()
            lay4.setSpacing(wspace20)
            title4 = QtWidgets.QLabel('P-wave Parameters')
            title4.setAlignment(QtCore.Qt.AlignCenter)
            title4.setFixedHeight(titleHeight)
            title4.setFont(subheaderFont)
            # use single, clustered, or all P-waves
            c1_w = QtWidgets.QFrame()
            c1_w.setFixedWidth(pqi.px_w(85, self.WIDTH))
            c1_w.setFrameShape(QtWidgets.QFrame.Panel)
            c1_w.setFrameShadow(QtWidgets.QFrame.Sunken)
            c1 = QtWidgets.QVBoxLayout(c1_w)
            self.useAllPwaves_btn = QtWidgets.QRadioButton('All')
            self.useAllPwaves_btn.setFont(font)
            self.useSinglePwaves_btn = QtWidgets.QRadioButton('Single')
            self.useSinglePwaves_btn.setFont(font)
            self.useClusterPwaves_btn = QtWidgets.QRadioButton('Cluster')
            self.useClusterPwaves_btn.setFont(font)
            c1.addWidget(self.useAllPwaves_btn)
            c1.addWidget(self.useSinglePwaves_btn)
            c1.addWidget(self.useClusterPwaves_btn)
            # thresholds for classifying single & clustered P-waves
            c2 = QtWidgets.QVBoxLayout()
            c2.setSpacing(hspace5)
            c2b1 = QtWidgets.QVBoxLayout()
            c2b1.setSpacing(hspace1)
            self.singlePwaveWin_label = QtWidgets.QLabel('Single Wave\nThreshold')
            self.singlePwaveWin_label.setFont(font)
            self.singlePwaveWin_label.setAlignment(QtCore.Qt.AlignCenter)
            self.singlePwaveWin_val = QtWidgets.QDoubleSpinBox()
            self.singlePwaveWin_val.setFont(font)
            self.singlePwaveWin_val.setSuffix(' s')
            self.singlePwaveWin_val.setMaximum(10)
            self.singlePwaveWin_val.setDecimals(2)
            self.singlePwaveWin_val.setSingleStep(0.1)
            c2b1.addWidget(self.singlePwaveWin_label)
            c2b1.addWidget(self.singlePwaveWin_val)
            c2b2 = QtWidgets.QVBoxLayout()
            c2b2.setSpacing(hspace1)
            self.clusterPwaveWin_label = QtWidgets.QLabel('Cluster\nThreshold')
            self.clusterPwaveWin_label.setFont(font)
            self.clusterPwaveWin_label.setAlignment(QtCore.Qt.AlignCenter)
            self.clusterPwaveWin_val = QtWidgets.QDoubleSpinBox()
            self.clusterPwaveWin_val.setFont(font)
            self.clusterPwaveWin_val.setSuffix(' s')
            self.clusterPwaveWin_val.setMaximum(10)
            self.clusterPwaveWin_val.setDecimals(2)
            self.clusterPwaveWin_val.setSingleStep(0.1)
            c2b2.addWidget(self.clusterPwaveWin_label)
            c2b2.addWidget(self.clusterPwaveWin_val)
            c2.addLayout(c2b1)
            c2.addLayout(c2b2)
            # cluster analysis params
            c3 = QtWidgets.QVBoxLayout()
            c3.setSpacing(hspace5)
            c3b1 = QtWidgets.QVBoxLayout()
            c3b1.setSpacing(hspace1)
            self.clusterAnalysis_label = QtWidgets.QLabel('Cluster Analysis')
            self.clusterAnalysis_label.setFont(font)
            self.clusterAnalysis_label.setAlignment(QtCore.Qt.AlignCenter)
            self.clusterAnalysis_type = QtWidgets.QComboBox()
            self.clusterAnalysis_type.setFont(font)
            self.clusterAnalysis_type.addItems(['Each wave', 'Cluster start', 
                                                'Cluster center', 'Cluster end'])
            c3b1.addWidget(self.clusterAnalysis_label)
            c3b1.addWidget(self.clusterAnalysis_type)
            c3b2 = QtWidgets.QVBoxLayout()
            c3b2.setSpacing(hspace1)
            self.clusterIso_label = QtWidgets.QLabel('Isolate cluster')
            self.clusterIso_label.setFont(font)
            self.clusterIso_label.setAlignment(QtCore.Qt.AlignCenter)
            self.clusterIso_val = QtWidgets.QDoubleSpinBox()
            self.clusterIso_val.setFont(font)
            self.clusterIso_val.setSuffix(' ms')
            c3b2.addWidget(self.clusterIso_label)
            c3b2.addWidget(self.clusterIso_val)
            c3.addLayout(c3b1)
            c3.addLayout(c3b2)
            lay4.addWidget(c1_w)
            lay4.addLayout(c2)
            lay4.addLayout(c3)
            line_4 = pqi.vline(orientation='h')
            self.pwaveLayout.addWidget(title4)
            self.pwaveLayout.addLayout(lay4)
            self.pwaveLayout.addWidget(line_4)
            if 'pwaveWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.pwaveWidget)
            
            ### EEG spectrogram parameters ###
            self.spWidget = QtWidgets.QWidget()
            self.spWidget.setFixedHeight(pqi.px_h(195, self.HEIGHT))
            self.spLayout = QtWidgets.QVBoxLayout(self.spWidget)
            self.spLayout.setContentsMargins(cm,0,cm,cm)
            self.spLayout.setSpacing(hspace15)
            lay5 = QtWidgets.QVBoxLayout()
            lay5.setSpacing(hspace15)
            title5 = QtWidgets.QLabel('Spectrogram Parameters')
            title5.setAlignment(QtCore.Qt.AlignCenter)
            title5.setFixedHeight(titleHeight)
            title5.setFont(subheaderFont)
            r1 = QtWidgets.QHBoxLayout()
            r1.setSpacing(pqi.px_w(12, self.WIDTH))
            # FFT window size (s)
            r1b1 = QtWidgets.QVBoxLayout()
            r1b1.setSpacing(hspace1)
            spWin_label = QtWidgets.QLabel('FFT size')
            spWin_label.setFont(font)
            spWin_label.setAlignment(QtCore.Qt.AlignCenter)
            self.spWin_val = QtWidgets.QDoubleSpinBox()
            self.spWin_val.setFont(font)
            self.spWin_val.setMinimum(0.1)
            self.spWin_val.setMaximum(10)
            self.spWin_val.setSingleStep(0.1)
            self.spWin_val.setSuffix(' s')
            self.spWin_val.setFixedWidth(pqi.px_w(60, self.WIDTH))
            r1b1.addWidget(spWin_label)
            r1b1.addWidget(self.spWin_val, alignment=QtCore.Qt.AlignCenter)
            # FFT window overlap (%)
            r1b2 = QtWidgets.QVBoxLayout()
            r1b2.setSpacing(hspace1)
            spOverlap_label = QtWidgets.QLabel('% overlap')
            spOverlap_label.setFont(font)
            spOverlap_label.setAlignment(QtCore.Qt.AlignCenter)
            self.spOverlap_val = QtWidgets.QDoubleSpinBox()
            self.spOverlap_val.setFont(font)
            self.spOverlap_val.setMinimum(0)
            self.spOverlap_val.setMaximum(100)
            self.spOverlap_val.setDecimals(0)
            self.spOverlap_val.setSuffix(' %')
            self.spOverlap_val.setFixedWidth(pqi.px_w(60, self.WIDTH))
            r1b2.addWidget(spOverlap_label)
            r1b2.addWidget(self.spOverlap_val, alignment=QtCore.Qt.AlignCenter)
            # max frequency (Hz)
            r1b3 = QtWidgets.QVBoxLayout()
            r1b3.setSpacing(hspace1)
            spFmax_label = QtWidgets.QLabel('f-max')
            spFmax_label.setFont(font)
            spFmax_label.setAlignment(QtCore.Qt.AlignCenter)
            self.spFmax_val = QtWidgets.QDoubleSpinBox()
            self.spFmax_val.setFont(font)
            self.spFmax_val.setMaximum(500)
            self.spFmax_val.setDecimals(0)
            self.spFmax_val.setSuffix(' Hz')
            self.spFmax_val.setFixedWidth(pqi.px_w(65, self.WIDTH))
            r1b3.addWidget(spFmax_label)
            r1b3.addWidget(self.spFmax_val, alignment=QtCore.Qt.AlignCenter)
            # SP normalization method dropdown
            r1b4 = QtWidgets.QVBoxLayout()
            r1b4.setSpacing(hspace1)
            spNorm_label = QtWidgets.QLabel('Normalization')
            spNorm_label.setFont(font)
            spNorm_label.setAlignment(QtCore.Qt.AlignCenter)
            self.spNorm_type = QtWidgets.QComboBox()
            self.spNorm_type.setFont(font)
            self.spNorm_type.addItems(['None','Recording'])
            if self.plotTypeWidgets[self.plotType]['time_res']=='event' \
            and 'spectrum' not in self.plotType:
                self.spNorm_type.addItems(['Time window'])
            r1b4.addWidget(spNorm_label)
            r1b4.addWidget(self.spNorm_type, alignment=QtCore.Qt.AlignCenter)
            r1.addLayout(r1b1)
            r1.addLayout(r1b2)
            r1.addLayout(r1b3)
            r1.addLayout(r1b4)
            r2 = QtWidgets.QHBoxLayout()
            r2.setSpacing(pqi.px_w(18, self.WIDTH))
            # load or recalculate SP
            r2b1 = QtWidgets.QVBoxLayout()
            r2b1.setSpacing(hspace1)
            spRecalc_label = QtWidgets.QLabel('Recalculate?')
            spRecalc_label.setFont(font)
            spRecalc_label.setAlignment(QtCore.Qt.AlignHCenter)
            btn_grp1 = QtWidgets.QButtonGroup(self.spWidget)
            h1 = QtWidgets.QHBoxLayout()
            h1.setContentsMargins(0,0,0,0)
            h2 = QtWidgets.QHBoxLayout()
            h2.setContentsMargins(0,0,0,0)
            self.spCalc_btn = QtWidgets.QRadioButton('Yes', self.spWidget)
            self.spCalc_btn.setFont(font)
            self.spLoad_btn = QtWidgets.QRadioButton('No', self.spWidget)
            self.spLoad_btn.setFont(font)
            btn_grp1.addButton(self.spCalc_btn)
            btn_grp1.addButton(self.spLoad_btn)
            h1.addSpacing(wspace5)
            h1.addWidget(self.spCalc_btn)
            h2.addSpacing(wspace5)
            h2.addWidget(self.spLoad_btn)
            r2b1.addWidget(spRecalc_label)
            r2b1.addLayout(h1)
            r2b1.addLayout(h2)
            # SP smoothing factors
            r2b2 = QtWidgets.QVBoxLayout()
            r2b2.setSpacing(hspace1)
            spSmooth_label = QtWidgets.QLabel('Smoothing')
            spSmooth_label.setFont(font)
            spSmooth_label.setAlignment(QtCore.Qt.AlignHCenter)
            r2b2_mid = QtWidgets.QHBoxLayout()
            r2b2_mid.setSpacing(0)
            self.spSmoothX_chk =  QtWidgets.QCheckBox('X :')
            self.spSmoothX_chk.setFont(font)
            self.spSmoothX_val = QtWidgets.QDoubleSpinBox()
            self.spSmoothX_val.setFont(font)
            self.spSmoothX_val.setMinimum(1)
            self.spSmoothX_val.setDecimals(0)
            smw = pqi.px_w(40, self.WIDTH)
            self.spSmoothX_val.setFixedWidth(smw)
            r2b2_mid.addWidget(self.spSmoothX_chk)
            r2b2_mid.addWidget(self.spSmoothX_val)
            r2b2_bot = QtWidgets.QHBoxLayout()
            r2b2_bot.setSpacing(0)
            self.spSmoothY_chk =  QtWidgets.QCheckBox('Y :')
            self.spSmoothY_chk.setFont(font)
            self.spSmoothY_val = QtWidgets.QDoubleSpinBox()
            self.spSmoothY_val.setFont(font)
            self.spSmoothY_val.setMinimum(1)
            self.spSmoothY_val.setDecimals(0)
            self.spSmoothY_val.setFixedWidth(smw)
            r2b2_bot.addWidget(self.spSmoothY_chk)
            r2b2_bot.addWidget(self.spSmoothY_val)
            r2b2.addWidget(spSmooth_label)
            r2b2.addLayout(r2b2_mid)
            r2b2.addLayout(r2b2_bot)
            # SP saturation value
            r2b3 = QtWidgets.QVBoxLayout()
            r2b3.setSpacing(hspace1)
            spVm_label = QtWidgets.QLabel('Saturation')
            spVm_label.setFont(font)
            spVm_label.setAlignment(QtCore.Qt.AlignHCenter)
            r2b3_mid = QtWidgets.QHBoxLayout()
            r2b3_mid.setSpacing(wspace5)
            btn_grp2 = QtWidgets.QButtonGroup(self.spWidget)
            self.spVmAuto_btn = QtWidgets.QRadioButton('Auto', self.spWidget)
            self.spVmAuto_btn.setFont(font)
            self.spVmCustom_btn = QtWidgets.QRadioButton('Custom', self.spWidget)
            self.spVmCustom_btn.setFont(font)
            btn_grp2.addButton(self.spVmAuto_btn)
            btn_grp2.addButton(self.spVmCustom_btn)
            r2b3_mid.addWidget(self.spVmAuto_btn)
            r2b3_mid.addWidget(self.spVmCustom_btn)
            r2b3_bot = QtWidgets.QHBoxLayout()
            r2b3_bot.setSpacing(0)
            spVm_dash = QtWidgets.QLabel(' - ')
            spVm_dash.setFont(font)
            spVm_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.spVmMin_val = QtWidgets.QDoubleSpinBox()
            self.spVmMin_val.setFont(font)
            vmw = pqi.px_w(53, self.WIDTH)
            self.spVmMin_val.setFixedWidth(vmw)
            self.spVmMax_val = QtWidgets.QDoubleSpinBox()
            self.spVmMax_val.setFont(font)
            self.spVmMax_val.setFixedWidth(vmw)
            r2b3_bot.addWidget(self.spVmMin_val, stretch=2)
            r2b3_bot.addWidget(spVm_dash, stretch=0)
            r2b3_bot.addWidget(self.spVmMax_val, stretch=2)
            r2b3.addWidget(spVm_label)
            r2b3.addLayout(r2b3_mid)
            r2b3.addLayout(r2b3_bot)
            r2.addLayout(r2b1)
            r2.addLayout(r2b2)
            r2.addSpacing(wspace10)
            r2.addLayout(r2b3)
            # disable widgets for high-resolution SP calculation
            if self.plotTypeWidgets[self.plotType]['time_res']=='state':
                self.spWin_val.setEnabled(False)
                self.spWin_val.lineEdit().setVisible(False)
                spWin_label.setStyleSheet('color : gray')
                self.spOverlap_val.setEnabled(False)
                self.spOverlap_val.lineEdit().setVisible(False)
                spOverlap_label.setStyleSheet('color : gray')
                self.spCalc_btn.setCheckable(False)
                self.spLoad_btn.setCheckable(False)
                self.spCalc_btn.setStyleSheet('color : gray')
                self.spLoad_btn.setStyleSheet('color : gray')
                spRecalc_label.setStyleSheet('color : gray')
            # disable widgets for SP smoothing and color saturation
            if 'spectrum' in self.plotType:
                self.spSmoothX_chk.setCheckable(False)
                self.spSmoothX_chk.setStyleSheet('color : gray')
                self.spSmoothX_val.setEnabled(False)
                self.spSmoothX_val.lineEdit().setVisible(False)
                self.spSmoothY_chk.setCheckable(False)
                self.spSmoothY_chk.setStyleSheet('color : gray')
                self.spSmoothY_val.setEnabled(False)
                self.spSmoothY_val.lineEdit().setVisible(False)
                spSmooth_label.setStyleSheet('color : gray')
                self.spVmAuto_btn.setCheckable(False)
                self.spVmCustom_btn.setCheckable(False)
                self.spVmAuto_btn.setStyleSheet('color : gray')
                self.spVmCustom_btn.setStyleSheet('color : gray')
                self.spVmMin_val.setEnabled(False)
                self.spVmMin_val.lineEdit().setVisible(False)
                self.spVmMax_val.setEnabled(False)
                self.spVmMax_val.lineEdit().setVisible(False)
                spVm_label.setStyleSheet('color : gray')
            lay5.addLayout(r1)
            lay5.addLayout(r2)
            line_5 = pqi.vline(orientation='h')
            self.spLayout.addWidget(title5)
            self.spLayout.addLayout(lay5)
            self.spLayout.addWidget(line_5)
            if 'spWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.spWidget)
            
            ### Laser spectrum parameters ###
            self.laserSpectrumWidget = QtWidgets.QWidget()
            self.laserSpectrumWidget.setFixedHeight(pqi.px_h(180, self.HEIGHT))
            self.laserSpectrumLayout = QtWidgets.QVBoxLayout(self.laserSpectrumWidget)
            self.laserSpectrumLayout.setContentsMargins(cm,0,cm,cm)
            self.laserSpectrumLayout.setSpacing(hspace15)
            lay6 = QtWidgets.QHBoxLayout()
            lay6.setSpacing(wspace10)
            title6 = QtWidgets.QLabel('Laser Spectrum Parameters')
            title6.setAlignment(QtCore.Qt.AlignCenter)
            title6.setFixedHeight(titleHeight)
            title6.setFont(subheaderFont)
            # separate or combine laser-on vs laser-off spectrums
            c1_w = QtWidgets.QFrame()
            c1_w.setContentsMargins(0,0,0,0)
            c1_w.setFrameShape(QtWidgets.QFrame.Panel)
            c1_w.setFrameShadow(QtWidgets.QFrame.Sunken)
            c1 = QtWidgets.QVBoxLayout(c1_w)
            c1.setContentsMargins(wspace5,hspace10,wspace5,hspace10)
            c1.setSpacing(hspace10)
            pmode_label = QtWidgets.QLabel('Plot laser?')
            pmode_label.setFont(font)
            pmode_label.setAlignment(QtCore.Qt.AlignCenter)
            self.pmode1_btn = QtWidgets.QRadioButton('Yes')
            self.pmode1_btn.setFont(font)
            self.pmode0_btn = QtWidgets.QRadioButton('No')
            self.pmode0_btn.setFont(font)
            c1.addWidget(pmode_label)
            c1.addWidget(self.pmode1_btn)
            c1.addWidget(self.pmode0_btn)
            c2 = QtWidgets.QVBoxLayout()
            c2.setSpacing(1)
            h = QtWidgets.QHBoxLayout()
            h.setSpacing(wspace10)
            # harmonic interpolation frequency (Hz)
            hc1 = QtWidgets.QVBoxLayout()
            hc1.setSpacing(hspace5)
            self.harmcs_btn = QtWidgets.QCheckBox('Interpolate\nharmonics')
            self.harmcs_btn.setFont(font)
            self.harmcs_val = QtWidgets.QDoubleSpinBox()
            self.harmcs_val.setFont(font)
            self.harmcs_val.setDecimals(0)
            self.harmcs_val.setSuffix(' Hz')
            self.harmcs_val.setFixedWidth(pqi.px_w(70, self.WIDTH))
            hc1.addWidget(self.harmcs_btn)
            hc1.addWidget(self.harmcs_val, alignment=QtCore.Qt.AlignCenter)
            # harmonic interpolation method dropdown
            hc2 = QtWidgets.QVBoxLayout()
            hc2.setSpacing(hspace5)
            self.harmcsMode_type = QtWidgets.QComboBox()
            self.harmcsMode_type.setFont(font)
            self.harmcsMode_type.addItems(['EEG & EMG spectrums', 'EMG spectrum only'])
            box = QtWidgets.QHBoxLayout()
            box.setContentsMargins(0,0,0,0)
            box.setSpacing(wspace5)
            # interpolate X adjacent frequencies
            self.iplt_label1 = QtWidgets.QLabel('Use')
            self.iplt_label1.setFont(font)
            self.iplt_label1.setAlignment(QtCore.Qt.AlignCenter)
            self.iplt_val = QtWidgets.QDoubleSpinBox()
            self.iplt_val.setFont(font)
            self.iplt_val.setMinimum(1)
            self.iplt_val.setMaximum(2)
            self.iplt_val.setDecimals(0)
            self.iplt_val.setFixedWidth(wspace20+wspace10)
            self.iplt_label2 = QtWidgets.QLabel('adjacent freq(s)')
            self.iplt_label2.setFont(font)
            self.iplt_label2.setAlignment(QtCore.Qt.AlignCenter)
            box.addWidget(self.iplt_label1)
            box.addWidget(self.iplt_val)
            box.addWidget(self.iplt_label2)
            hc2.addWidget(self.harmcsMode_type)
            hc2.addLayout(box)
            h.addLayout(hc1)
            h.addLayout(hc2)
            # policy for analyzing states that partially overlap with laser
            self.excluMode_label = QtWidgets.QLabel('Handle partial laser overlaps')
            self.excluMode_label.setFont(font)
            self.excluMode_label.setAlignment(QtCore.Qt.AlignCenter)
            self.excluMode_type = QtWidgets.QComboBox()
            self.excluMode_type.setFont(font)
            self.excluMode_type.addItems(['Split into laser ON and laser OFF bins',
                                          'Exclude laser OFF bins',
                                          'Classify all bins as laser ON'])
            c2.addLayout(h)
            c2.addSpacing(hspace15)
            c2.addWidget(self.excluMode_label)
            c2.addWidget(self.excluMode_type, alignment=QtCore.Qt.AlignCenter)
            lay6.addWidget(c1_w)
            lay6.addLayout(c2)
            line_6 = pqi.vline(orientation='h')
            self.laserSpectrumLayout.addWidget(title6)
            self.laserSpectrumLayout.addLayout(lay6)
            self.laserSpectrumLayout.addWidget(line_6)
            if 'laserSpectrumWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.laserSpectrumWidget)
            
            ### EMG parameters ###
            self.emgWidget = QtWidgets.QWidget()
            self.emgWidget.setFixedHeight(pqi.px_h(200, self.HEIGHT))
            self.emgLayout = QtWidgets.QVBoxLayout(self.emgWidget)
            self.emgLayout.setContentsMargins(cm,0,cm,cm)
            self.emgLayout.setSpacing(hspace15)
            lay7 = QtWidgets.QVBoxLayout()
            lay7.setSpacing(0)
            title7 = QtWidgets.QLabel('EMG Parameters')
            title7.setAlignment(QtCore.Qt.AlignCenter)
            title7.setFixedHeight(titleHeight)
            title7.setFont(subheaderFont)
            r1 = QtWidgets.QHBoxLayout()
            # EMG amplitude source
            emgSource_label = QtWidgets.QLabel('EMG Source')
            emgSource_label.setFont(font)
            emgSource_label.setAlignment(QtCore.Qt.AlignCenter)
            btn_grp3 = QtWidgets.QButtonGroup(self.emgWidget)
            self.useEMGraw_btn = QtWidgets.QRadioButton('Raw EMG', self.emgWidget)
            self.useEMGraw_btn.setFont(font)
            self.useMSP_btn = QtWidgets.QRadioButton('EMG spectrogram', self.emgWidget)
            self.useMSP_btn.setFont(font)
            btn_grp3.addButton(self.useEMGraw_btn)
            btn_grp3.addButton(self.useMSP_btn)
            r1.addWidget(self.useEMGraw_btn, alignment=QtCore.Qt.AlignCenter)
            r1.addWidget(self.useMSP_btn, alignment=QtCore.Qt.AlignCenter)
            r2 = QtWidgets.QHBoxLayout()
            r2.setSpacing(0)
            c1_w = QtWidgets.QFrame()
            c1_w.setContentsMargins(0,pqi.px_h(8, self.HEIGHT),0,pqi.px_h(8, self.HEIGHT))
            c1_w.setFrameShape(QtWidgets.QFrame.Panel)
            c1_w.setFrameShadow(QtWidgets.QFrame.Sunken)
            c1 = QtWidgets.QVBoxLayout(c1_w)
            c1.setContentsMargins(wspace5,hspace10,wspace5,hspace10)
            c1.setSpacing(hspace10)
            # load or recalculate EMG amplitude
            emgAmpRecalc_label = QtWidgets.QLabel('Recalculate?')
            emgAmpRecalc_label.setFont(font)
            emgAmpRecalc_label.setAlignment(QtCore.Qt.AlignCenter)
            btn_grp4 = QtWidgets.QButtonGroup(self.emgWidget)
            h1 = QtWidgets.QHBoxLayout()
            h1.setContentsMargins(0,0,0,0)
            h2 = QtWidgets.QHBoxLayout()
            h2.setContentsMargins(0,0,0,0)
            self.emgAmpCalc_btn = QtWidgets.QRadioButton('Yes', self.emgWidget)
            self.emgAmpCalc_btn.setFont(font)
            self.emgAmpLoad_btn = QtWidgets.QRadioButton('No', self.emgWidget)
            self.emgAmpLoad_btn.setFont(font)
            btn_grp4.addButton(self.emgAmpCalc_btn)
            btn_grp4.addButton(self.emgAmpLoad_btn)
            h1.addSpacing(pqi.px_w(8, self.WIDTH))
            h1.addWidget(self.emgAmpCalc_btn)
            h2.addSpacing(pqi.px_w(8, self.WIDTH))
            h2.addWidget(self.emgAmpLoad_btn)
            c1.addWidget(emgAmpRecalc_label, alignment=QtCore.Qt.AlignTop)
            c1.addLayout(h1)
            c1.addLayout(h2)
            c2_w = QtWidgets.QWidget()
            c2_w.setFixedWidth(pqi.px_w(180, self.WIDTH))
            c2 = QtWidgets.QVBoxLayout(c2_w)
            c2.setSpacing(hspace5)
            c2_b1 = QtWidgets.QVBoxLayout()
            c2_b1.setSpacing(hspace1)
            # raw EMG filtering params
            self.emgfilt_type = QtWidgets.QComboBox()
            self.emgfilt_type.setFont(font)
            self.emgfilt_type.addItems(['No filter', 'Low-pass filter', 
                                        'High-pass filter', 'Band-pass filter'])
            self.mspfilt_label = QtWidgets.QLabel('mSP Frequency Band')
            self.mspfilt_label.setFont(font)
            self.mspfilt_label.setAlignment(QtCore.Qt.AlignCenter)
            emgfiltFreq_lay = QtWidgets.QHBoxLayout()
            emgfiltFreq_lay.setContentsMargins(0,0,0,0)
            emgfiltFreq_lay.setSpacing(0)
            self.emgfiltLo_val = QtWidgets.QDoubleSpinBox()
            self.emgfiltLo_val.setFont(font)
            self.emgfiltLo_val.setFixedWidth(pqi.px_w(60, self.WIDTH))
            self.emgfiltLo_val.setMinimum(0)
            self.emgfiltLo_val.setMaximum(self.mainWin.sr/2)
            self.emgfiltLo_val.setDecimals(0)
            self.emgfiltLo_val.setSuffix('Hz')
            self.emgfiltHi_val = QtWidgets.QDoubleSpinBox()
            self.emgfiltHi_val.setFont(font)
            self.emgfiltHi_val.setFixedWidth(pqi.px_w(60, self.WIDTH))
            self.emgfiltHi_val.setMinimum(1)
            self.emgfiltHi_val.setMaximum(self.mainWin.sr/2)
            self.emgfiltHi_val.setDecimals(0)
            self.emgfiltHi_val.setSuffix('Hz')
            self.emgfilt_label1 = QtWidgets.QLabel(' <')
            self.emgfilt_label1.setAlignment(QtCore.Qt.AlignCenter)
            self.emgfilt_label2 = QtWidgets.QLabel('f')
            self.emgfilt_label2.setAlignment(QtCore.Qt.AlignCenter)
            self.emgfilt_label3 = QtWidgets.QLabel('< ')
            self.emgfilt_label3.setAlignment(QtCore.Qt.AlignCenter)
            font_math = QtGui.QFont('Cambria', 9, 1, True)
            self.emgfilt_label1.setFont(font_math)
            self.emgfilt_label2.setFont(font_math)
            self.emgfilt_label3.setFont(font_math)
            emgfiltFreq_lay.addWidget(self.emgfiltLo_val, stretch=2)
            emgfiltFreq_lay.addWidget(self.emgfilt_label1, stretch=0)
            emgfiltFreq_lay.addWidget(self.emgfilt_label2, stretch=0)
            emgfiltFreq_lay.addWidget(self.emgfilt_label3, stretch=0)
            emgfiltFreq_lay.addWidget(self.emgfiltHi_val, stretch=2)
            c2_b1.addWidget(self.emgfilt_type)
            c2_b1.addWidget(self.mspfilt_label, alignment=QtCore.Qt.AlignTop)
            c2_b1.addLayout(emgfiltFreq_lay)
            # EMG z-scoring method dropdown
            c2_b2 = QtWidgets.QVBoxLayout()
            c2_b2.setSpacing(hspace1)
            emgZ_label = QtWidgets.QLabel('Z-score')
            emgZ_label.setFont(font)
            emgZ_label.setAlignment(QtCore.Qt.AlignCenter)
            self.emgZ_type = QtWidgets.QComboBox()
            self.emgZ_type.setFont(font)
            self.emgZ_type.addItems(['None','Recording', 'Time window'])
            c2_b2.addWidget(emgZ_label)
            c2_b2.addWidget(self.emgZ_type)
            c2.addLayout(c2_b1)
            c2.addLayout(c2_b2)
            # raw EMG downsampling & smoothing params
            self.rawCol = QtWidgets.QWidget()
            self.rawCol.setFixedWidth(pqi.px_w(80, self.WIDTH))
            self.rawCol_lay = QtWidgets.QVBoxLayout(self.rawCol)
            self.rawCol_lay.setSpacing(hspace10)
            rawCol_b1 = QtWidgets.QVBoxLayout()
            rawCol_b1.setSpacing(hspace1)
            emgDn_label = QtWidgets.QLabel('Bin size')
            emgDn_label.setFont(font)
            emgDn_label.setAlignment(QtCore.Qt.AlignCenter)
            self.emgDn_val = QtWidgets.QDoubleSpinBox()
            self.emgDn_val.setFont(font)
            self.emgDn_val.setMinimum(1)
            self.emgDn_val.setMaximum(5000)
            self.emgDn_val.setDecimals(0)
            rawCol_b1.addWidget(emgDn_label)
            rawCol_b1.addWidget(self.emgDn_val)
            rawCol_b2 = QtWidgets.QVBoxLayout()
            rawCol_b2.setSpacing(hspace1)
            emgSm_label = QtWidgets.QLabel('Smooth')
            emgSm_label.setFont(font)
            emgSm_label.setAlignment(QtCore.Qt.AlignCenter)
            self.emgSm_val = QtWidgets.QDoubleSpinBox()
            self.emgSm_val.setFont(font)
            self.emgSm_val.setMinimum(0)
            self.emgSm_val.setMaximum(5000)
            self.emgSm_val.setDecimals(0)
            rawCol_b2.addWidget(emgSm_label)
            rawCol_b2.addWidget(self.emgSm_val)
            self.rawCol_lay.addLayout(rawCol_b1)
            self.rawCol_lay.addLayout(rawCol_b2)
            # mSP calculation params
            self.mspCol = QtWidgets.QWidget()
            self.mspCol.setFixedWidth(pqi.px_w(80, self.WIDTH))
            self.mspCol_lay = QtWidgets.QVBoxLayout(self.mspCol)
            self.mspCol_lay.setSpacing(hspace10)
            mspCol_b1 = QtWidgets.QVBoxLayout()
            mspCol_b1.setSpacing(hspace1)
            mspWin_label = QtWidgets.QLabel('FFT size')
            mspWin_label.setFont(font)
            mspWin_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mspWin_val = QtWidgets.QDoubleSpinBox()
            self.mspWin_val.setFont(font)
            self.mspWin_val.setMinimum(0.1)
            self.mspWin_val.setMaximum(10)
            self.mspWin_val.setSingleStep(0.1)
            self.mspWin_val.setSuffix(' s')
            mspCol_b1.addWidget(mspWin_label)
            mspCol_b1.addWidget(self.mspWin_val)
            mspCol_b2 = QtWidgets.QVBoxLayout()
            mspCol_b2.setSpacing(hspace1)
            mspOverlap_label = QtWidgets.QLabel('% overlap')
            mspOverlap_label.setFont(font)
            mspOverlap_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mspOverlap_val = QtWidgets.QDoubleSpinBox()
            self.mspOverlap_val.setFont(font)
            self.mspOverlap_val.setMinimum(0)
            self.mspOverlap_val.setMaximum(100)
            self.mspOverlap_val.setDecimals(0)
            self.mspOverlap_val.setSuffix(' %')
            mspCol_b2.addWidget(mspOverlap_label, alignment=QtCore.Qt.AlignCenter)
            mspCol_b2.addWidget(self.mspOverlap_val)
            self.mspCol_lay.addLayout(mspCol_b1)
            self.mspCol_lay.addLayout(mspCol_b2)
            r2.addWidget(c1_w, stretch=0)
            r2.addWidget(c2_w, stretch=2)
            r2.addSpacing(wspace15)
            r2.addWidget(self.rawCol, stretch=2)
            r2.addWidget(self.mspCol, stretch=2)
            lay7.addLayout(r1)
            lay7.addLayout(r2)
            line_7 = pqi.vline(orientation='h')
            self.emgLayout.addWidget(title7)
            self.emgLayout.addLayout(lay7)
            self.emgLayout.addWidget(line_7)
            if 'emgWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.emgWidget)
            
            ### EMG twitch parameters ###
            self.twitchWidget = QtWidgets.QWidget()
            self.twitchWidget.setFixedHeight(pqi.px_h(285, self.HEIGHT))
            self.twitchLayout = QtWidgets.QVBoxLayout(self.twitchWidget)
            self.twitchLayout.setContentsMargins(cm,0,cm,cm)
            self.twitchLayout.setSpacing(hspace15)
            lay8 = QtWidgets.QVBoxLayout()
            lay8.setSpacing(hspace10)
            title8 = QtWidgets.QLabel('Twitch Detection Parameters')
            title8.setAlignment(QtCore.Qt.AlignCenter)
            title8.setFixedHeight(titleHeight)
            title8.setFont(subheaderFont)
            r1 = QtWidgets.QHBoxLayout()
            r1.setSpacing(wspace10)
            # twitch detection threshold
            r1c1 = QtWidgets.QVBoxLayout()
            r1c1.setSpacing(hspace1)
            twitchThres_label = QtWidgets.QLabel('Detection Thres.')
            twitchThres_label.setFont(font)
            twitchThres_label.setAlignment(QtCore.Qt.AlignCenter)
            self.twitchThres_type = QtWidgets.QComboBox()
            self.twitchThres_type.setFont(font)
            self.twitchThres_type.setFixedWidth(pqi.px_w(105, self.WIDTH))
            self.twitchThres_type.addItems(['Raw value', 'Std. deviations', 'Percentile'])
            self.twitchThres_val = QtWidgets.QDoubleSpinBox()
            self.twitchThres_val.setFont(font)
            self.twitchThres_val.setFixedWidth(pqi.px_w(105, self.WIDTH))
            self.twitchThres_val.setMaximum(1000)
            r1c1.addWidget(twitchThres_label)
            r1c1.addWidget(self.twitchThres_type)
            r1c1.addWidget(self.twitchThres_val)
            # calculate threshold for all REM sleep or individually for each REM period
            r1c2 = QtWidgets.QVBoxLayout()
            r1c2.setContentsMargins(0,pqi.px_h(7,self.HEIGHT),0,pqi.px_h(7,self.HEIGHT))
            r1c2.setSpacing(hspace10)
            btn_grp5 = QtWidgets.QButtonGroup(self.twitchWidget)
            self.twitchThresAllREM_btn = QtWidgets.QRadioButton('All REM sleep', 
                                                                self.twitchWidget)
            self.twitchThresAllREM_btn.setFont(font)
            self.twitchThresEachREM_btn = QtWidgets.QRadioButton('Each REM bout', 
                                                                 self.twitchWidget)
            self.twitchThresEachREM_btn.setFont(font)
            btn_grp5.addButton(self.twitchThresAllREM_btn)
            btn_grp5.addButton(self.twitchThresEachREM_btn)
            r1c2.addWidget(self.twitchThresAllREM_btn)
            r1c2.addWidget(self.twitchThresEachREM_btn)
            # threshold using 1st X seconds of REM
            r1c3 = QtWidgets.QGridLayout()
            r1c3.setContentsMargins(0,cm,0,pqi.px_h(7,self.HEIGHT))
            r1c3.setHorizontalSpacing(wspace1*3)
            r1c3.setVerticalSpacing(hspace1)
            self.twitchThresFirst_btn = QtWidgets.QCheckBox('Use first')
            self.twitchThresFirst_btn.setFont(font)
            self.twitchThresFirst_label = QtWidgets.QLabel('  second(s) of REM')
            self.twitchThresFirst_label.setFont(font)
            self.twitchThresFirst_val = QtWidgets.QDoubleSpinBox()
            self.twitchThresFirst_val.setFont(font)
            self.twitchThresFirst_val.setFixedWidth(pqi.px_w(40, self.WIDTH))
            self.twitchThresFirst_val.setMinimum(0)
            self.twitchThresFirst_val.setMaximum(60)
            self.twitchThresFirst_val.setDecimals(0)
            r1c3.addWidget(self.twitchThresFirst_btn, 0, 0, 1, 1)
            r1c3.addWidget(self.twitchThresFirst_val, 0, 1, 1, 1)
            r1c3.addWidget(self.twitchThresFirst_label, 1, 0, 1, 2)
            r1.addLayout(r1c1)
            r1.addLayout(r1c2)
            r1.addLayout(r1c3)
            r2 = QtWidgets.QHBoxLayout()
            r2.setSpacing(pqi.px_w(12, self.WIDTH))
            # load or re-detect EMG twitches
            r2c1_w = QtWidgets.QFrame()
            r2c1_w.setFixedWidth(pqi.px_w(100, self.WIDTH))
            r2c1_w.setContentsMargins(0,0,0,0)
            r2c1_w.setFrameShape(QtWidgets.QFrame.Panel)
            r2c1_w.setFrameShadow(QtWidgets.QFrame.Sunken)
            r2c1 = QtWidgets.QVBoxLayout(r2c1_w)
            r2c1.setContentsMargins(wspace1*3,hspace10,wspace1*3,hspace10)
            r2c1.setSpacing(hspace10)
            emgTwitchRecalc_label = QtWidgets.QLabel('Re-detect?')
            emgTwitchRecalc_label.setFont(font)
            emgTwitchRecalc_label.setAlignment(QtCore.Qt.AlignCenter)
            btn_grp6 = QtWidgets.QButtonGroup(self.twitchWidget)
            h1 = QtWidgets.QHBoxLayout()
            h1.setContentsMargins(0,0,0,0)
            h2 = QtWidgets.QHBoxLayout()
            h2.setContentsMargins(0,0,0,0)
            self.twitchCalc_btn = QtWidgets.QRadioButton('Yes', self.twitchWidget)
            self.twitchCalc_btn.setFont(font)
            self.twitchLoad_btn = QtWidgets.QRadioButton('No', self.twitchWidget)
            self.twitchLoad_btn.setFont(font)
            btn_grp6.addButton(self.twitchCalc_btn)
            btn_grp6.addButton(self.twitchLoad_btn)
            h1.addSpacing(wspace10)
            h1.addWidget(self.twitchCalc_btn)
            h2.addSpacing(wspace10)
            h2.addWidget(self.twitchLoad_btn)
            r2c1.addWidget(emgTwitchRecalc_label, alignment=QtCore.Qt.AlignTop)
            r2c1.addLayout(h1)
            r2c1.addLayout(h2)
            r2c2 = QtWidgets.QVBoxLayout()
            r2c2.setSpacing(hspace10)
            # min. twitch duration (s)
            r2c2_b1 = QtWidgets.QVBoxLayout()
            r2c2_b1.setContentsMargins(0,0,0,0)
            r2c2_b1.setSpacing(hspace1)
            minTwitchDur_label = QtWidgets.QLabel('Min. Twitch Dur.')
            minTwitchDur_label.setFont(font)
            minTwitchDur_label.setAlignment(QtCore.Qt.AlignCenter)
            self.minTwitchDur_val = QtWidgets.QDoubleSpinBox()
            self.minTwitchDur_val.setFont(font)
            self.minTwitchDur_val.setFixedWidth(pqi.px_w(100, self.WIDTH))
            self.minTwitchDur_val.setMinimum(0)
            self.minTwitchDur_val.setMaximum(5000)
            self.minTwitchDur_val.setDecimals(2)
            self.minTwitchDur_val.setSingleStep(0.1)
            self.minTwitchDur_val.setSuffix(' s')
            r2c2_b1.addWidget(minTwitchDur_label)
            r2c2_b1.addWidget(self.minTwitchDur_val)
            # min. twitch separation (s)
            r2c2_b2 = QtWidgets.QVBoxLayout()
            r2c2_b2.setContentsMargins(0,0,0,0)
            r2c2_b2.setSpacing(hspace1)
            minTwitchSep_label = QtWidgets.QLabel('Min. Twitch Sep.')
            minTwitchSep_label.setFont(font)
            minTwitchSep_label.setAlignment(QtCore.Qt.AlignCenter)
            self.minTwitchSep_val = QtWidgets.QDoubleSpinBox()
            self.minTwitchSep_val.setFont(font)
            self.minTwitchSep_val.setFixedWidth(pqi.px_w(100, self.WIDTH))
            self.minTwitchSep_val.setMinimum(0)
            self.minTwitchSep_val.setMaximum(5000)
            self.minTwitchSep_val.setDecimals(2)
            self.minTwitchSep_val.setSingleStep(0.1)
            self.minTwitchSep_val.setSuffix(' s')
            r2c2_b2.addWidget(minTwitchSep_label)
            r2c2_b2.addWidget(self.minTwitchSep_val)
            r2c2.addLayout(r2c2_b1)
            r2c2.addLayout(r2c2_b2)
            r2c3 = QtWidgets.QVBoxLayout()
            r2c3.setSpacing(hspace10)
            # min. REM duration (s)
            r2c3_b1 = QtWidgets.QVBoxLayout()
            r2c3_b1.setContentsMargins(0,0,0,0)
            r2c3_b1.setSpacing(hspace1)
            minREMDur_label = QtWidgets.QLabel('Min. REM Dur.')
            minREMDur_label.setFont(font)
            minREMDur_label.setAlignment(QtCore.Qt.AlignCenter)
            self.minREMDur_val = QtWidgets.QDoubleSpinBox()
            self.minREMDur_val.setFont(font)
            self.minREMDur_val.setFixedWidth(pqi.px_w(100, self.WIDTH))
            self.minREMDur_val.setMinimum(0)
            self.minREMDur_val.setMaximum(60)
            self.minREMDur_val.setDecimals(0)
            self.minREMDur_val.setSuffix(' s')
            r2c3_b1.addWidget(minREMDur_label)
            r2c3_b1.addWidget(self.minREMDur_val)
            # time (s) to cut off end of each REM bout to avoid waking EMG twitch
            r2c3_b2 = QtWidgets.QVBoxLayout()
            r2c3_b2.setContentsMargins(0,0,0,0)
            r2c3_b2.setSpacing(hspace1)
            REMcutoff_label = QtWidgets.QLabel('End REM cutoff')
            REMcutoff_label.setFont(font)
            REMcutoff_label.setAlignment(QtCore.Qt.AlignCenter)
            self.REMcutoff_val = QtWidgets.QDoubleSpinBox()
            self.REMcutoff_val.setFont(font)
            self.REMcutoff_val.setFixedWidth(pqi.px_w(100, self.WIDTH))
            self.REMcutoff_val.setMinimum(0)
            self.REMcutoff_val.setMaximum(10)
            self.REMcutoff_val.setDecimals(1)
            self.REMcutoff_val.setSingleStep(2.5)
            self.REMcutoff_val.setSuffix(' s')
            r2c3_b2.addWidget(REMcutoff_label)
            r2c3_b2.addWidget(self.REMcutoff_val)
            r2c3.addLayout(r2c3_b1)
            r2c3.addLayout(r2c3_b2)
            r2.addWidget(r2c1_w)
            r2.addLayout(r2c2)
            r2.addLayout(r2c3)
            # load twitch settings/open live annotation window for EMG twitches
            r3 = QtWidgets.QHBoxLayout()
            r3.setContentsMargins(cm,0,cm,0)
            r3.setSpacing(wspace20)
            self.editTwitchSettings_btn = QtWidgets.QPushButton('Live Edit')
            self.editTwitchSettings_btn.setFont(font)
            self.editTwitchSettings_btn.setFixedHeight(pqi.px_h(30, self.HEIGHT))
            self.loadTwitchSettings_btn = QtWidgets.QPushButton('Load Settings')
            self.loadTwitchSettings_btn.setFont(font)
            self.loadTwitchSettings_btn.setObjectName('loadTwitchSettings')
            self.loadTwitchSettings_btn.setFixedHeight(pqi.px_h(30, self.HEIGHT))
            r3.addWidget(self.editTwitchSettings_btn)
            r3.addWidget(self.loadTwitchSettings_btn)
            lay8.addLayout(r1)
            lay8.addSpacing(hspace10)
            lay8.addLayout(r2)
            lay8.addLayout(r3)
            line_8 = pqi.vline(orientation='h')
            self.twitchLayout.addWidget(title8)
            self.twitchLayout.addLayout(lay8)
            self.twitchLayout.addWidget(line_8)
            if 'twitchWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.twitchWidget)
            
            ### Laser parameters ###
            self.laserWidget = QtWidgets.QWidget()
            self.laserWidget.setFixedHeight(pqi.px_h(135, self.HEIGHT))
            self.laserLayout = QtWidgets.QVBoxLayout(self.laserWidget)
            self.laserLayout.setContentsMargins(cm,0,cm,cm)
            self.laserLayout.setSpacing(hspace15)
            lay9 = QtWidgets.QHBoxLayout()
            lay9.setSpacing(wspace15)
            title9 = QtWidgets.QLabel('Laser Parameters')
            title9.setAlignment(QtCore.Qt.AlignCenter)
            title9.setFixedHeight(titleHeight)
            title9.setFont(subheaderFont)
            # plot all P-waves or separate spontaneous vs. laser-triggered waves
            c1 = QtWidgets.QVBoxLayout()
            c1.setSpacing(hspace5)
            laser_label = QtWidgets.QLabel('Plot laser?')
            laser_label.setFont(font)
            laser_label.setAlignment(QtCore.Qt.AlignCenter)
            btn_grp7 = QtWidgets.QButtonGroup(self.laserWidget)
            self.laserYes_btn = QtWidgets.QRadioButton('Yes', self.laserWidget)
            self.laserYes_btn.setFont(font)
            self.laserNo_btn = QtWidgets.QRadioButton('No', self.laserWidget)
            self.laserNo_btn.setFont(font)
            btn_grp7.addButton(self.laserYes_btn)
            btn_grp7.addButton(self.laserNo_btn)
            c1.addWidget(laser_label)
            c1.addWidget(self.laserYes_btn)
            c1.addWidget(self.laserNo_btn)
            # post-stimulation window (s) for detecting laser-triggered P-waves
            c2 = QtWidgets.QVBoxLayout()
            c2.setSpacing(hspace5)
            self.postStim_label = QtWidgets.QLabel('Post-laser\nwindow')
            self.postStim_label.setFont(font)
            self.postStim_label.setAlignment(QtCore.Qt.AlignHCenter)
            self.postStim_val = QtWidgets.QDoubleSpinBox()
            self.postStim_val.setFont(font)
            self.postStim_val.setMaximum(1.0)
            self.postStim_val.setDecimals(2)
            self.postStim_val.setSingleStep(0.1)
            self.postStim_val.setSuffix(' s')
            self.postStim_val.setFixedWidth(pqi.px_w(62, self.WIDTH))
            c2.addWidget(self.postStim_label)
            c2.addWidget(self.postStim_val)
            c2.addSpacing(hspace1*2)
            # plot spontaneous vs. laser-triggered P-waves or successful vs. failed laser pulses
            c3 = QtWidgets.QVBoxLayout()
            c3.setSpacing(hspace5)
            self.laserPlotMode_label = QtWidgets.QLabel('Plot Mode')
            self.laserPlotMode_label.setFont(font)
            self.laserPlotMode_label.setAlignment(QtCore.Qt.AlignCenter)
            btn_grp8 = QtWidgets.QButtonGroup(self.laserWidget)
            self.laserPwaveMode_btn = QtWidgets.QRadioButton('P-waves', 
                                                             self.laserWidget)
            self.laserPwaveMode_btn.setFont(font)
            self.laserLaserMode_btn = QtWidgets.QRadioButton('Laser pulses', 
                                                             self.laserWidget)
            self.laserLaserMode_btn.setFont(font)
            btn_grp8.addButton(self.laserPwaveMode_btn)
            btn_grp8.addButton(self.laserLaserMode_btn)
            c3.addWidget(self.laserPlotMode_label)
            c3.addWidget(self.laserPwaveMode_btn)
            c3.addWidget(self.laserLaserMode_btn)
            # plot laser pulses without P-waves during preceding X s
            c4 = QtWidgets.QVBoxLayout()
            c4.setSpacing(hspace5)
            self.laserIso_label = QtWidgets.QLabel('Isolate\nlaser')
            self.laserIso_label.setFont(font)
            self.laserIso_label.setAlignment(QtCore.Qt.AlignHCenter)
            self.laserIso_val = QtWidgets.QDoubleSpinBox()
            self.laserIso_val.setFont(font)
            self.laserIso_val.setMaximum(10)
            self.laserIso_val.setDecimals(2)
            self.laserIso_val.setSingleStep(0.1)
            self.laserIso_val.setSuffix(' s')
            self.laserIso_val.setFixedWidth(pqi.px_w(62, self.WIDTH))
            c4.addWidget(self.laserIso_label)
            c4.addWidget(self.laserIso_val)
            c4.addSpacing(hspace1*2)
            lay9.addLayout(c1)
            lay9.addLayout(c2)
            lay9.addLayout(c3)
            lay9.addLayout(c4)
            line_9 = pqi.vline(orientation='h')
            self.laserLayout.addWidget(title9)
            self.laserLayout.addLayout(lay9)
            self.laserLayout.addWidget(line_9)
            if 'laserWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.laserWidget)
                
            ### DF/F parameters ###
            self.dffWidget = QtWidgets.QWidget()
            self.dffWidget.setFixedHeight(pqi.px_h(170, self.HEIGHT))
            self.dffLayout = QtWidgets.QVBoxLayout(self.dffWidget)
            self.dffLayout.setContentsMargins(wspace5,0,int(wspace5/2),cm)
            self.dffLayout.setSpacing(hspace10)
            lay10 = QtWidgets.QHBoxLayout()
            lay10.setSpacing(wspace15)
            title10 = QtWidgets.QLabel('\u0394F/F Parameters')
            title10.setAlignment(QtCore.Qt.AlignCenter)
            title10.setFixedHeight(titleHeight)
            title10.setFont(subheaderFont) 
            c1 = QtWidgets.QVBoxLayout()
            c1.setSpacing(hspace5)
            c1r1 = QtWidgets.QVBoxLayout()
            c1r1.setSpacing(hspace1)
            # downsample DFF signal
            dffdn_label = QtWidgets.QLabel('Downsample')
            dffdn_label.setFont(font)
            dffdn_label.setAlignment(QtCore.Qt.AlignCenter)
            self.dffdn_val = QtWidgets.QDoubleSpinBox()
            self.dffdn_val.setFont(font)
            self.dffdn_val.setFixedWidth(pqi.px_w(110, self.WIDTH))
            self.dffdn_val.setMaximum(10000)
            self.dffdn_val.setDecimals(0)
            self.dffdn_val.setSingleStep(50)
            c1r1.addWidget(dffdn_label)
            c1r1.addWidget(self.dffdn_val, alignment=QtCore.Qt.AlignCenter)
            c1r2 = QtWidgets.QVBoxLayout()
            c1r2.setSpacing(hspace1)
            # z-score DFF signal
            dffz_label = QtWidgets.QLabel('Z-score')
            dffz_label.setFont(font)
            dffz_label.setAlignment(QtCore.Qt.AlignCenter)
            self.dffz_type = QtWidgets.QComboBox()
            self.dffz_type.setFont(font)
            self.dffz_type.setFixedWidth(pqi.px_w(125, self.WIDTH))
            self.dffz_type.addItems(['None', 'Recording', 'Time window', 'Custom window'])
            dffzWin_hbox = QtWidgets.QHBoxLayout()
            dffzWin_hbox.setSpacing(0)
            dffzWin_hbox.setContentsMargins(0,0,0,0)
            self.preZWin_val = QtWidgets.QDoubleSpinBox()
            self.preZWin_val.setFont(font)
            self.preZWin_val.setMinimum(-500)
            self.preZWin_val.setMaximum(0)
            self.preZWin_val.setDecimals(1)
            self.preZWin_val.setSingleStep(0.1)
            self.preZWin_val.setSuffix (' s')
            self.zwin_dash = QtWidgets.QLabel(' - ')
            self.zwin_dash.setFont(font)
            self.zwin_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.postZWin_val = QtWidgets.QDoubleSpinBox()
            self.postZWin_val.setFont(font)
            self.postZWin_val.setMinimum(0)
            self.postZWin_val.setMaximum(500)
            self.postZWin_val.setDecimals(1)
            self.postZWin_val.setSingleStep(0.1)
            self.postZWin_val.setSuffix (' s')
            dffzWin_hbox.addWidget(self.preZWin_val, stretch=2)
            dffzWin_hbox.addWidget(self.zwin_dash, stretch=0)
            dffzWin_hbox.addWidget(self.postZWin_val, stretch=2)
            c1r2.addWidget(dffz_label)
            c1r2.addWidget(self.dffz_type, alignment=QtCore.Qt.AlignCenter)
            c1r2.addLayout(dffzWin_hbox)
            c1.addLayout(c1r1)
            c1.addLayout(c1r2)
            # DFF trial map smoothing
            c2 = QtWidgets.QVBoxLayout()
            c2.setSpacing(hspace5)
            dffSm_label = QtWidgets.QLabel('Heat map\nsmoothing')
            dffSm_label.setFont(font)
            dffSm_label.setAlignment(QtCore.Qt.AlignCenter)
            c2r2 = QtWidgets.QHBoxLayout()
            c2r2.setSpacing(0)
            self.dffSmX_chk =  QtWidgets.QCheckBox('X :')
            self.dffSmX_chk.setFont(font)
            self.dffSmX_chk.setDisabled(self.plotType == 'Single P-wave \u0394F/F')
            self.dffSmX_val = QtWidgets.QDoubleSpinBox()
            self.dffSmX_val.setFont(font)
            self.dffSmX_val.setMinimum(1)
            self.dffSmX_val.setDecimals(0)
            c2r2.addWidget(self.dffSmX_chk)
            c2r2.addWidget(self.dffSmX_val)
            c2r3 = QtWidgets.QHBoxLayout()
            c2r3.setSpacing(0)
            self.dffSmY_chk =  QtWidgets.QCheckBox('Y :')
            self.dffSmY_chk.setFont(font)
            self.dffSmY_chk.setDisabled(self.plotType == 'Single P-wave \u0394F/F')
            self.dffSmY_val = QtWidgets.QDoubleSpinBox()
            self.dffSmY_val.setFont(font)
            self.dffSmY_val.setMinimum(1)
            self.dffSmY_val.setDecimals(0)
            c2r3.addWidget(self.dffSmY_chk)
            c2r3.addWidget(self.dffSmY_val)
            c2.addWidget(dffSm_label)
            c2.addLayout(c2r2)
            c2.addLayout(c2r3)
            c2.addSpacing(hspace10)
            # DFF trial map saturation
            c3 = QtWidgets.QVBoxLayout()
            c3.setSpacing(hspace5)
            dffVm_label = QtWidgets.QLabel('Heat map\ncolor saturation')
            dffVm_label.setFont(font)
            dffVm_label.setAlignment(QtCore.Qt.AlignCenter)
            c3r2 = QtWidgets.QHBoxLayout()
            c3r2.setSpacing(int(hspace5/2))
            self.dffVmAuto_btn = QtWidgets.QRadioButton('Auto')
            self.dffVmAuto_btn.setFont(font)
            self.dffVmAuto_btn.setDisabled(self.plotType == 'Single P-wave \u0394F/F')
            self.dffVmCustom_btn = QtWidgets.QRadioButton('Custom')
            self.dffVmCustom_btn.setFont(font)
            self.dffVmCustom_btn.setDisabled(self.plotType == 'Single P-wave \u0394F/F')
            c3r2.addWidget(self.dffVmAuto_btn)
            c3r2.addWidget(self.dffVmCustom_btn)
            c3r3 = QtWidgets.QHBoxLayout()
            c3r3.setSpacing(0)
            dffVm_dash = QtWidgets.QLabel(' - ')
            dffVm_dash.setFont(font)
            dffVm_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.dffVmMin_val = QtWidgets.QDoubleSpinBox()
            self.dffVmMin_val.setFont(font)
            self.dffVmMin_val.setDecimals(1)
            self.dffVmMax_val = QtWidgets.QDoubleSpinBox()
            self.dffVmMax_val.setFont(font)
            self.dffVmMax_val.setDecimals(1)
            c3r3.addWidget(self.dffVmMin_val, stretch=2)
            c3r3.addWidget(spVm_dash, stretch=0)
            c3r3.addWidget(self.dffVmMax_val, stretch=2)
            c3.addWidget(dffVm_label)
            c3.addLayout(c3r2)
            c3.addLayout(c3r3)
            c3.addSpacing(hspace10)
            lay10.addLayout(c1)
            lay10.addLayout(c2)
            lay10.addLayout(c3)
            line_10 = pqi.vline(orientation='h')
            self.dffLayout.addWidget(title10)
            self.dffLayout.addLayout(lay10)
            self.dffLayout.addWidget(line_10)
            if 'dffWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
                self.settingsLayout.addWidget(self.dffWidget)
            
            ### Action buttons 2 ###
            self.btnsWidget2 = QtWidgets.QWidget()
            self.btnsWidget2.setFixedHeight(pqi.px_h(40, self.HEIGHT))
            self.btnsLayout2 = QtWidgets.QVBoxLayout(self.btnsWidget2)
            self.btnsLayout2.setContentsMargins(cm,0,cm,0)
            self.plotFig_btn = QtWidgets.QPushButton('PLOT')
            self.plotFig_btn.setFont(font)
            self.btnsLayout2.addWidget(self.plotFig_btn)
            self.settingsLayout.addWidget(self.btnsWidget2)
            
            # settings layout spacing
            self.settingsLayout.setContentsMargins(cm,cm,cm,0)
            self.settingsLayout.setSpacing(0)
            self.settingsWidget.setFixedWidth(pqi.px_w(415, self.WIDTH))
            self.centralLayout.addWidget(self.settingsWidget)
            
        except Exception as e:
            print('whoopsie')
            pdb.set_trace()
            print(e)
            #import sys; sys.exit()
    
    
    def connect_buttons(self):
        """
        Connect widgets to data plotting functions
        """
        # update sleep params
        self.plotStat_type.currentTextChanged.connect(self.update_sleep_params)
        self.binSize_btn.toggled.connect(self.update_sleep_params)
        self.binSize_val.valueChanged.connect(self.update_sleep_params)
        self.binNum_val.valueChanged.connect(self.update_sleep_params)
        
        # update data collection windows
        self.preWin_val.valueChanged.connect(self.update_win_params)
        self.postWin_val.valueChanged.connect(self.update_win_params)
        self.preWinLsr_val.valueChanged.connect(self.update_win_params)
        self.postWinLsr_val.valueChanged.connect(self.update_win_params)
        self.preWinExcl_val.valueChanged.connect(self.update_win_params)
        self.postWinExcl_val.valueChanged.connect(self.update_win_params)
        if self.plotType == 'P-wave spectrum':
            self.preWin_val.valueChanged.connect(self.sync_win_vals)
            self.postWin_val.valueChanged.connect(self.sync_win_vals)
            self.preWinExcl_val.valueChanged.connect(self.sync_win_vals)
            self.postWinExcl_val.valueChanged.connect(self.sync_win_vals)
        self.dataAvg_type.currentTextChanged.connect(self.update_win_params)
        self.twitchAvg_type.currentTextChanged.connect(self.update_win_params)
        self.error_type.currentTextChanged.connect(self.update_win_params)
        self.smooth_val.valueChanged.connect(self.update_win_params)
        self.signalType.currentTextChanged.connect(self.update_win_params)
        
        # update brainstate parameters
        for btn in self.plotStates.values():
            btn.toggled.connect(self.update_brainstate_params)
        self.plotEachState_btn.toggled.connect(self.update_brainstate_params2)
        self.maState_type.currentTextChanged.connect(self.update_brainstate_params)
        self.maThr_val.valueChanged.connect(self.update_brainstate_params)
        self.isState_type.currentTextChanged.connect(self.update_brainstate_params)
        self.noiseHandle_type.currentTextChanged.connect(self.update_brainstate_params)
        self.tstart_val.valueChanged.connect(self.update_brainstate_params)
        self.tend_val.valueChanged.connect(self.update_brainstate_params)
        for w in self.stateseqWidgets:
            w.findChild(QtWidgets.QComboBox, 'state').currentTextChanged.connect(self.update_stateseq_params)
            w.findChild(QtWidgets.QDoubleSpinBox, 'bins').valueChanged.connect(self.update_stateseq_params)
            w.findChild(QtWidgets.QPushButton, 'sign').clicked.connect(self.update_stateseq_params)
            w.findChild(QtWidgets.QDoubleSpinBox, 'thres').valueChanged.connect(self.update_stateseq_params)
        self.addState_btn.clicked.connect(self.change_stateseq)
        self.removeState_btn.clicked.connect(self.change_stateseq)
        
        # update P-wave parameters
        self.useAllPwaves_btn.toggled.connect(self.update_pwave_params)
        self.useSinglePwaves_btn.toggled.connect(self.update_pwave_params)
        self.singlePwaveWin_val.valueChanged.connect(self.update_pwave_params)
        self.clusterPwaveWin_val.valueChanged.connect(self.update_pwave_params)
        self.clusterAnalysis_type.currentTextChanged.connect(self.update_pwave_params)
        self.clusterIso_val.valueChanged.connect(self.update_pwave_params)
        
        # update high-res spectrogram parameters
        self.spWin_val.valueChanged.connect(self.update_sp_params)
        self.spOverlap_val.valueChanged.connect(self.update_sp_params)
        self.spFmax_val.valueChanged.connect(self.update_sp_params)
        self.spNorm_type.currentTextChanged.connect(self.update_sp_params)
        self.spCalc_btn.toggled.connect(self.update_sp_params)
        self.spSmoothX_chk.toggled.connect(self.update_sp_params)
        self.spSmoothX_val.valueChanged.connect(self.update_sp_params)
        self.spSmoothY_chk.toggled.connect(self.update_sp_params)
        self.spSmoothY_val.valueChanged.connect(self.update_sp_params)
        self.spVmAuto_btn.toggled.connect(self.update_sp_params)
        self.spVmMin_val.valueChanged.connect(self.update_sp_params)
        self.spVmMax_val.valueChanged.connect(self.update_sp_params)
        
        # update laser spectrum parameters
        self.pmode1_btn.toggled.connect(self.update_laser_spectrum_params)
        self.pmode0_btn.toggled.connect(self.update_laser_spectrum_params)
        self.harmcs_btn.toggled.connect(self.update_laser_spectrum_params)
        self.harmcs_val.valueChanged.connect(self.update_laser_spectrum_params)
        self.harmcsMode_type.currentTextChanged.connect(self.update_laser_spectrum_params)
        self.iplt_val.valueChanged.connect(self.update_laser_spectrum_params)
        self.excluMode_type.currentTextChanged.connect(self.update_laser_spectrum_params)
        
        # update EMG parameters
        self.useEMGraw_btn.toggled.connect(self.switch_emg_source)
        self.emgfilt_type.currentTextChanged.connect(self.update_emg_params)
        self.emgfiltLo_val.valueChanged.connect(self.update_emg_params)
        self.emgfiltHi_val.valueChanged.connect(self.update_emg_params)
        self.emgDn_val.valueChanged.connect(self.update_emg_params)
        self.emgSm_val.valueChanged.connect(self.update_emg_params)
        self.mspWin_val.valueChanged.connect(self.update_emg_params)
        self.mspOverlap_val.valueChanged.connect(self.update_emg_params)
        self.emgAmpCalc_btn.toggled.connect(self.update_emg_params)
        self.emgZ_type.currentTextChanged.connect(self.update_emg_params)
        
        # update twitch detection parameters
        self.twitchThres_type.currentTextChanged.connect(self.update_twitch_params)
        self.twitchThres_val.valueChanged.connect(self.update_twitch_params)
        self.twitchThresAllREM_btn.toggled.connect(self.update_twitch_params)
        self.twitchThresFirst_btn.toggled.connect(self.update_twitch_params)
        self.twitchThresFirst_val.valueChanged.connect(self.update_twitch_params)
        self.twitchCalc_btn.toggled.connect(self.update_twitch_params)
        self.minTwitchDur_val.valueChanged.connect(self.update_twitch_params)
        self.minTwitchSep_val.valueChanged.connect(self.update_twitch_params)
        self.minREMDur_val.valueChanged.connect(self.update_twitch_params)
        self.REMcutoff_val.valueChanged.connect(self.update_twitch_params)
        self.editTwitchSettings_btn.clicked.connect(self.mainWin.twitch_annot_window)
        self.loadTwitchSettings_btn.clicked.connect(self.load_twitch_settings)
        
        # update laser parameters
        self.laserYes_btn.toggled.connect(self.update_laser_params)
        self.postStim_val.valueChanged.connect(self.update_laser_params)
        self.laserPwaveMode_btn.toggled.connect(self.update_laser_params)
        self.laserIso_val.valueChanged.connect(self.update_laser_params)
        
        # update DF/F parameters
        self.dffdn_val.valueChanged.connect(self.update_dff_params)
        self.dffz_type.currentTextChanged.connect(self.update_dff_params)
        self.preZWin_val.valueChanged.connect(self.update_dff_params)
        self.postZWin_val.valueChanged.connect(self.update_dff_params)
        self.dffSmX_chk.toggled.connect(self.update_dff_params)
        self.dffSmX_val.valueChanged.connect(self.update_dff_params)
        self.dffSmY_chk.toggled.connect(self.update_dff_params)
        self.dffSmY_val.valueChanged.connect(self.update_dff_params)
        self.dffVmAuto_btn.toggled.connect(self.update_dff_params)
        self.dffVmMin_val.valueChanged.connect(self.update_dff_params)
        self.dffVmMax_val.valueChanged.connect(self.update_dff_params)
        
        # connect plot buttons
        if self.plotType in self.plotTypeWidgets.keys():
            self.plotFig_btn.clicked.connect(self.plotTypeWidgets[self.plotType]['fx'])
        else:
            print('Unable to connect plotting function')
        self.plotExp_btn.clicked.connect(self.plot_experiment)
        self.plotRec_btn.clicked.connect(self.plot_recording)
    
    
    def resizeMe(self):
        """
        Adjust window height when showing/hiding widgets
        """
        self.adjustSize()
        self.resize(self.minimumSizeHint())
        
        
    def disable_laser_widgets(self, disable):
        """
        If plot type has no option to plot laser, disable laser widgets
        """
        
        if self.plotType in ['P-wave frequency', 'Single P-wave spectrogram', 'Single P-wave EMG']:
            disable = True

        c = 'gray' if disable == True else 'black'
        # disable data collection widgets
        self.winLsr_label.setStyleSheet(f'color : {c}')
        self.winLsr_dash.setStyleSheet(f'color : {c}')
        self.preWinLsr_val.setDisabled(disable)
        self.preWinLsr_val.lineEdit().setVisible(not disable)
        self.postWinLsr_val.setDisabled(disable)
        self.postWinLsr_val.lineEdit().setVisible(not disable)
        # disable laser-triggered event widgets
        self.postStim_label.setStyleSheet(f'color : {c}')
        self.laserPlotMode_label.setStyleSheet(f'color : {c}')
        self.laserIso_label.setStyleSheet(f'color : {c}')
        self.postStim_val.setDisabled(disable)
        self.postStim_val.lineEdit().setVisible(not disable)
        self.laserPwaveMode_btn.setDisabled(disable)
        self.laserLaserMode_btn.setDisabled(disable)
        self.laserIso_val.setDisabled(disable)
        self.laserIso_val.lineEdit().setVisible(not disable)
        # disable laser spectrum widgets
        self.harmcs_btn.setStyleSheet(f'color : {c}')
        self.harmcs_btn.setDisabled(disable)
        self.harmcs_val.setDisabled(disable)
        self.harmcs_val.lineEdit().setVisible(not disable)
        self.harmcsMode_type.setDisabled(disable)
        self.iplt_label1.setStyleSheet(f'color : {c}')
        self.iplt_label2.setStyleSheet(f'color : {c}')
        self.iplt_val.setDisabled(disable)
        self.iplt_val.lineEdit().setVisible(not disable)
        self.excluMode_label.setStyleSheet(f'color : {c}')
        self.excluMode_type.setDisabled(disable)

        
    def update_sleep_params(self):
        """
        Update sleep timecourse params from user input
        """
        # update plot statistic
        previous_stat = str(self.stat)
        self.stat = ['perc','freq','dur','is prob','pwave freq'][self.plotStat_type.currentIndex()]
        # show/hide P-wave widgets 
        self.plotTypeWidgets['Sleep timecourse']['req_pwaves'] = self.stat=='pwave freq'
        self.pwaveWidget.setVisible(self.stat == 'pwave freq')
        QtCore.QTimer.singleShot(0, self.resizeMe)
            
        if self.binSize_btn.isChecked():
            # partition recording into time bins
            self.tbin = int(self.binSize_val.value())
            self.num_bins = int(self.binNum_val.value())
        else:
            # plot all recording data in one bin
            self.tbin = int(len(self.mainWin.EEG) / self.mainWin.sr)
            self.num_bins = 1
        # enable/disable bin value box on GUI
        self.binSize_val.lineEdit().setVisible(self.binSize_btn.isChecked())
        self.binSize_val.setEnabled(self.binSize_btn.isChecked())
        self.binNum_val.lineEdit().setVisible(self.binSize_btn.isChecked())
        self.binNum_val.setEnabled(self.binSize_btn.isChecked())
        
        if self.sender() == self.plotStat_type:
            # if IS-->REM analysis is selected/deselected, set auto params 
            for widget in list(self.plotStates.values()) + [self.isState_type]:
                widget.blockSignals(True)
            
            # user switched plot type from a different analysis to IS-->REM
            if self.stat == 'is prob':
                # treat IS-R and IS-W as distinct, disable options
                self.isState_type.setCurrentIndex(1)
                self.isState_type.setEnabled(False)
                self.isState_type.setStyleSheet('QComboBox:disabled'
                                                '{ background-color : rgba(250,250,175,75) }')
                # check REM button, uncheck and disable all other state buttons
                self.plotStates[1].setChecked(True)
                for k in [2,3,4,5,6]:
                    self.plotStates[k].setChecked(False)
                    self.plotStates[k].setEnabled(False)
            
            # user switched plot type from IS-->REM to a different analysis
            if previous_stat == 'is prob':
                # enable all options for classifying IS
                self.isState_type.setEnabled(True)
                # enable NREM, wake, and both IS-R & IS-W
                for k in [2,3,4,5]:
                    self.plotStates[k].setEnabled(True)
                    self.isState_type.setStyleSheet('')
                # enable MA button if considering MAs a distinct state
                self.plotStates[6].setEnabled(self.maState_type.currentText() == 'MA')
            
            # restore widget signaling, update brain state params
            for widget in list(self.plotStates.values()) + [self.isState_type]:
                widget.blockSignals(False)
            self.update_brainstate_params()
                
    
    def update_win_params(self):
        """
        Update general data collection and processing params from user input
        """
        # update data collection windows
        self.win = [round(self.preWin_val.value(),2), round(self.postWin_val.value(),2)]
        self.lsr_win = [float(self.preWinLsr_val.value()), float(self.postWinLsr_val.value())]
        self.excl_win = [float(self.preWinExcl_val.value()), float(self.postWinExcl_val.value())]
        # update data averaging methods
        self.mouse_avg = self.dataAvg_type.currentText().lower()
        self.twitch_avg = self.twitchAvg_type.currentText().lower().split(' ')[0]
        self.ci = self.error_type.currentText().lower().replace('.','').split('%')[0]
        # update smoothing/normalizing params
        self.sf = int(self.smooth_val.value())
        self.pzscore = self.z_type.currentIndex()
        self.signal_type = self.signalType.currentText()
    
    
    def sync_win_vals(self):
        """
        Allow only one unique value for a data collection time window
         e.g. win = 5 --> analyze 10 seconds (5 pre, 5 post) surrounding event
        """
        # connect spinbox values for P-wave collection window
        if self.sender() == self.preWin_val:
            self.postWin_val.setValue(np.abs(self.preWin_val.value()))
        elif self.sender() == self.postWin_val:
            self.preWin_val.setValue(-np.abs(self.postWin_val.value()))
        
        # connect spinbox values for control data exclusion window
        elif self.sender() == self.preWinExcl_val:
            self.postWinExcl_val.setValue(np.abs(self.preWinExcl_val.value()))
        elif self.sender() == self.postWinExcl_val:
            self.preWinExcl_val.setValue(-np.abs(self.postWinExcl_val.value()))
    
        
    def update_brainstate_params(self):
        """
        Update brain state params from user input
        """
        # update states on plot
        self.istate = [int(k) for k,btn in self.plotStates.items() if btn.isChecked()]
        # update MA state and threshold
        self.ma_thr = float(self.maThr_val.value())
        self.ma_state = int([k for k,val in self.stateMap.items() if val==self.maState_type.currentText()][0])
        # for EMG twitches and IS-->REM transitions, REM is the only button enabled
        REMonly = ['EMG twitches']
        if self.stat == 'is prob':
            REMonly.append('Sleep timecourse')
        if self.maState_type.currentText() == 'MA':  # enable MA checkbox unless plot is REM only
            self.plotStates[6].setDisabled(self.plotType in REMonly)
        else:
            self.plotStates[6].setDisabled(True)     # disable MA checkbox
            self.plotStates[6].setChecked(False)
        # treat successful & failed IS as two separate states
        if self.isState_type.currentText() == 'IS-R & IS-W':
            self.flatten_is = False
            self.plotStates[4].setDisabled(self.plotType in REMonly)
            self.plotStates[4].setText('IS-R')
            self.plotStates[5].setDisabled(self.plotType in REMonly)
            self.stateMap[4] = 'IS-R'
        # treat all transition sleep the same
        elif self.isState_type.currentText() == 'IS':
            self.flatten_is = 4
            self.plotStates[4].setDisabled(self.plotType in REMonly)
            self.plotStates[4].setText('IS')
            self.plotStates[5].setDisabled(True)
            self.plotStates[5].setChecked(False)
            self.stateMap[4] = 'IS'
        # treat transition sleep as NREM or REM
        else:
            self.flatten_is = int([k for k,val in self.stateMap.items() if val==self.isState_type.currentText()][0])
            self.plotStates[4].setDisabled(True)
            self.plotStates[4].setChecked(False)
            self.plotStates[4].setText('IS-R')
            self.plotStates[5].setDisabled(True)
            self.plotStates[5].setChecked(False)
        
        # update dropdown menus for states in transition sequences
        if 'transitions' in self.plotType:
            self.update_stateseq_states()
        # exclude or ignore noise indices
        if self.noiseHandle_type.currentText() == 'Exclude':
            self.exclude_noise = True
        if self.noiseHandle_type.currentText() == 'Ignore':
            self.exclude_noise = False
        # update start and end times for analysis
        self.tstart = int(self.tstart_val.value())
        self.tend = int(self.tend_val.value())
        
    
    def update_brainstate_params2(self):
        """
        Adjust brain state params specifically for DF/F plots
        """
        if self.plotType == 'P-wave \u0394F/F':
            # if plotting single brain state, set state buttons to exclusive mode & uncheck all but REM
            # if plotting all states, disable exclusive mode & check all buttons
            self.state_btn_grp1.setExclusive(False)
            
            for k in self.plotStates.keys():
                btn = self.plotStates[k]
                # temporarily block signals to prevent errors updating from GUI
                btn.blockSignals(True)
                if self.plotAllStates_btn.isChecked():
                    btn.setDisabled(True)  # disable all state buttons
                    if k==6:               # check MA box if MAs considered a distinct state
                        btn.setChecked(self.maState_type.currentText() == 'MA')
                    elif k==5:             # check IS-W box if failed transitions considered a distinct state
                        btn.setChecked(self.isState_type.currentText() == 'IS-R & IS-W')
                    elif k==4:             # check IS box if transition sleep considered a distinct state
                        btn.setChecked('IS' in self.isState_type.currentText())
                    else:
                        btn.setChecked(True)
                elif self.plotEachState_btn.isChecked():
                    btn.setChecked(k==1)   # check REM button only
                    if k==6:               # enable MA box if MAs considered a distinct state
                        btn.setEnabled(self.maState_type.currentText() == 'MA')
                    elif k==5:             # enable IS-W box if failed transitions considered a distinct state
                        btn.setEnabled(self.isState_type.currentText() == 'IS-R & IS-W')
                    elif k==4:             # enable IS box if transition sleep considered a distinct state
                        btn.setEnabled('IS' in self.isState_type.currentText())
                    else:
                        btn.setEnabled(True)
                # restore signals
                btn.blockSignals(False)
            self.state_btn_grp1.setExclusive(self.plotEachState_btn.isChecked())
    
    
    def update_stateseq_states(self):
        """
        Update available brain states in transition plot based on MA and IS
        classifications
        """
        for i,w in enumerate(self.stateseqWidgets):
            cb = w.findChild(QtWidgets.QComboBox, 'state')
            # enable or disable MA state
            cb.model().item(5).setEnabled(self.ma_state==6)
            # treat IS as 2 distinct states, 1 distinct state, or subset of another state
            txt = 'IS' if self.flatten_is == 4 else 'IS-R'
            cb.model().item(3).setText(txt)
            cb.model().item(3).setEnabled(self.flatten_is in [False,4])
            cb.model().item(4).setEnabled(self.flatten_is == False)
            if (cb.currentIndex()==3 and self.flatten_is not in [False,4]) or \
                (cb.currentIndex()==4 and self.flatten_is != False):
                i = cb.findText(self.stateMap[self.flatten_is])
                cb.setCurrentIndex(i)
    
    
    def update_stateseq_params(self):
        """
        Update brain state transition params from user input
        """
        u = self.sender()
        if u.objectName() == 'sign':
            # update sign for state threshold
            txt = [y for x,y in zip(['>','<','x'], ['<','x','>']) if x==u.text()][0]
            u.setText(txt)
        # update state sequence params from non-hidden widgets
        states = [w.findChild(QtWidgets.QComboBox, 'state').currentText() for w in self.stateseqWidgets if w.isVisibleTo(self)]
        self.sequence = [list(self.stateMap.values()).index(s)+1 for s in states]
        self.nstates = [w.findChild(QtWidgets.QDoubleSpinBox, 'bins').value() for w in self.stateseqWidgets if w.isVisibleTo(self)]
        self.sign = [w.findChild(QtWidgets.QPushButton, 'sign').text() for w in self.stateseqWidgets if w.isVisibleTo(self)]
        self.state_thres = [w.findChild(QtWidgets.QDoubleSpinBox, 'thres').value() for w in self.stateseqWidgets if w.isVisibleTo(self)]
        for w in self.stateseqWidgets:
            sg = w.findChild(QtWidgets.QPushButton, 'sign').text()  # hide state threshold if sign = 'x'
            sb = w.findChild(QtWidgets.QDoubleSpinBox, 'thres')
            sb.setEnabled(sg!='x')
            sb.lineEdit().setVisible(sg!='x')
    
    
    def change_stateseq(self):
        """
        Add or delete brain state in transition sequence from user input
        """
        # get index of widget at the end of transition sequence
        i = len([w for w in self.stateseqWidgets if w.isVisibleTo(self)]) - 1
        if self.sender() == self.addState_btn:
            self.stateseqWidgets[i+1].show()  # add an additional state widget (max of 4 states)
            self.brstateWidget.findChild(QtWidgets.QPushButton, f'arrow{i+1}').show()
        else:
            self.stateseqWidgets[i].hide()    # remove the last state widget (min of 2 states)
            self.brstateWidget.findChild(QtWidgets.QPushButton, f'arrow{i}').hide()
        # get number of visible widgets
        j = len([w for w in self.stateseqWidgets if w.isVisibleTo(self)])
        self.addState_btn.setDisabled(j == len(self.stateseqWidgets))
        self.removeState_btn.setDisabled(j == 2)
        # update sequence variables
        self.update_stateseq_params()
            
        
    def update_pwave_params(self):
        """
        Update P-wave analysis params from user input
        """
        # update P-wave type to analyze in plot
        if self.useAllPwaves_btn.isChecked():
            self.p_iso = 0
            self.singlePwaveWin_val.setDisabled(True)             # disable single wave settings
            self.singlePwaveWin_val.lineEdit().setVisible(False)
            self.singlePwaveWin_label.setStyleSheet('color:gray')
            self.pcluster = 0                                     # disable cluster wave settings
            self.clusterPwaveWin_val.setDisabled(True)
            self.clusterPwaveWin_val.lineEdit().setVisible(False)
            self.clusterPwaveWin_label.setStyleSheet('color:gray')
            self.clusterAnalysis_type.setDisabled(True)
            self.clusterAnalysis_label.setStyleSheet('color:gray')
            self.clusterIso_val.setDisabled(True)
            self.clusterIso_val.lineEdit().setVisible(False)
            self.clusterIso_label.setStyleSheet('color:gray')
        elif self.useSinglePwaves_btn.isChecked():
            self.p_iso = float(self.singlePwaveWin_val.value())
            self.singlePwaveWin_val.setDisabled(False)             # enable single wave settings
            self.singlePwaveWin_val.lineEdit().setVisible(True)
            self.singlePwaveWin_label.setStyleSheet('color:black')
            self.pcluster = 0
            self.clusterPwaveWin_val.setDisabled(True)             # disable cluster wave settings
            self.clusterPwaveWin_val.lineEdit().setVisible(False)
            self.clusterPwaveWin_label.setStyleSheet('color:gray')
            self.clusterAnalysis_type.setDisabled(True)
            self.clusterAnalysis_label.setStyleSheet('color:gray')
            self.clusterIso_val.setDisabled(True)
            self.clusterIso_val.lineEdit().setVisible(False)
            self.clusterIso_label.setStyleSheet('color:gray')
        elif self.useClusterPwaves_btn.isChecked():
            self.p_iso = 0
            self.singlePwaveWin_val.setDisabled(True)              # disable single wave settings
            self.singlePwaveWin_val.lineEdit().setVisible(False)
            self.singlePwaveWin_label.setStyleSheet('color:gray')
            self.pcluster = float(self.clusterPwaveWin_val.value())
            self.clusterPwaveWin_val.setDisabled(False)             # enable cluster wave settings
            self.clusterPwaveWin_val.lineEdit().setVisible(True)
            self.clusterPwaveWin_label.setStyleSheet('color:black')
            self.clusterAnalysis_type.setDisabled(False)
            self.clusterAnalysis_label.setStyleSheet('color:black')
            self.clusterIso_val.setDisabled(False)
            self.clusterIso_val.lineEdit().setVisible(True)
            self.clusterIso_label.setStyleSheet('color:black')
        # update type of cluster analysis
        self.clus_event = ['waves','cluster_start','cluster_mid','cluster_end'][self.clusterAnalysis_type.currentIndex()]
        # update cluster isolation param
        self.clus_iso = float(self.clusterIso_val.value())
    
    
    def update_sp_params(self):
        """
        Update EEG spectrogram params from user input
        """
        # update FFT window size, overlap, and max freq. for SP calculation
        self.nsr_seg = float(self.spWin_val.value())
        self.perc_overlap = float(self.spOverlap_val.value()/100)
        self.fmax = float(self.spFmax_val.value())
        
        # update SP normalization method
        self.pnorm = int(self.spNorm_type.currentIndex())
        
        # update SP recalculation variable
        self.recalc_highres = bool(self.spCalc_btn.isChecked())
        
        # update SP smoothing variables
        self.spSmoothX_val.setEnabled(self.spSmoothX_chk.isChecked())
        self.spSmoothX_val.lineEdit().setVisible(self.spSmoothX_chk.isChecked())
        self.spSmoothY_val.setEnabled(self.spSmoothY_chk.isChecked())
        self.spSmoothY_val.lineEdit().setVisible(self.spSmoothY_chk.isChecked())
        if not self.spSmoothX_chk.isChecked() and not self.spSmoothY_chk.isChecked(): # no smoothing
            self.psmooth = []
        elif self.spSmoothX_chk.isChecked() and self.spSmoothY_chk.isChecked():       # smooth across rows/freqs AND columns/time
            self.psmooth = [int(self.spSmoothY_val.value()), int(self.spSmoothX_val.value())]
        elif self.spSmoothX_chk.isChecked():                                          # smooth across columns/time only
            self.psmooth = [1, int(self.spSmoothX_val.value())]
        elif self.spSmoothY_chk.isChecked():                                          # smooth across rows/freqs only
            self.psmooth = [int(self.spSmoothY_val.value()), 1]
        
        # update saturation for SP plot
        self.spVmMin_val.setEnabled(self.spVmCustom_btn.isChecked())
        self.spVmMax_val.setEnabled(self.spVmCustom_btn.isChecked())
        self.spVmMin_val.lineEdit().setVisible(self.spVmCustom_btn.isChecked())
        self.spVmMax_val.lineEdit().setVisible(self.spVmCustom_btn.isChecked())
        if self.spVmAuto_btn.isChecked():
            self.vm = []
        elif self.spVmCustom_btn.isChecked():
            self.vm = [float(self.spVmMin_val.value()), float(self.spVmMax_val.value())]
            d = [[10, -5000, 5000, 0], [0.1, -50, 50, 1]][bool(self.pnorm)]
            singlestep, minval, maxval, decimals = d
            self.spVmMin_val.setSingleStep(singlestep)
            self.spVmMin_val.setMinimum(minval)
            self.spVmMin_val.setMaximum(maxval)
            self.spVmMin_val.setDecimals(decimals)
            self.spVmMax_val.setSingleStep(singlestep)
            self.spVmMax_val.setMinimum(minval)
            self.spVmMax_val.setMaximum(maxval)
            self.spVmMax_val.setDecimals(decimals)
            
            
    def switch_emg_source(self):
        """
        Update EMG amplitude filtering params when user switches the EMG source
        """
        # temporarily block signals to prevent errors updating from GUI
        for widget in [self.emgfilt_type, self.emgfiltLo_val, self.emgfiltHi_val]:
            widget.blockSignals(True)
        
        # set filter frequencies using variables from selected source
        if self.useEMGraw_btn.isChecked():
            self.emgfiltLo_val.setValue(self.w0_emg * (self.mainWin.sr/2))
            self.emgfiltHi_val.setValue(self.w1_emg * (self.mainWin.sr/2))
            # set filter combobox to correct option
            ff = [self.w0_emg==-1, self.w1_emg==-1]
            i = [[True,True],[True,False],[False,True],[False,False]].index(ff)
            self.emgfilt_type.setCurrentIndex(i)
        elif self.useMSP_btn.isChecked():
            self.emgfiltLo_val.setValue(self.r_mu[0])
            self.emgfiltHi_val.setValue(self.r_mu[1])
        
        # restore widget signaling
        for widget in [self.emgfilt_type, self.emgfiltLo_val, self.emgfiltHi_val]:
            widget.blockSignals(False)
        
        self.update_emg_params()
    
    
    def load_twitch_settings(self, sfile=[]):
        """
        Load EMG twitch indices and param values from file
        """
        # load EMG twitch variables from saved dictionary
        if not sfile:
            sfile = QtWidgets.QFileDialog().getOpenFileName(self, "Load your settings", 
                                                            os.path.join(self.ppath, self.name), 
                                                           "Dictionaries (*.pkl)")[0]
        if sfile:
            with open(sfile, mode='rb') as f:
                # load settings dict, set local variables, update GUI
                ddict = pickle.load(f)
                # set local EMG and twitch variables (res=0 if loaded dictionary is missing info)
                self.update_emg_params(ddict=ddict)
                self.update_twitch_params(ddict=ddict)
                if self.sender().objectName() == 'loadTwitchSettings':
                    print('Twitch settings loaded!')
                    self.update_gui_from_vars()
            self.twitchFile = str(sfile)
        
        
    @QtCore.pyqtSlot()
    def update_emg_params(self, ddict={}):
        """
        Update EMG amplitude params from user input OR dictionary loaded from saved file
        """
        if ddict:
            try:
                # dictionary is inputted in following scenarios:
                # 1) initial loading of twitch settings (automatically when plot window is created)
                # 2) manual loading of twitch settings (when user presses "Load Settings" button)
                # 3) updating of twitch settings (when live EMG annotation window is closed)
                self.emg_source = ddict['ampsrc']
                self.w0_emg = ddict['w0_raw']
                self.w1_emg = ddict['w1_raw']
                self.nsr_seg_msp = ddict['nsr_seg_msp']
                self.perc_overlap_msp = ddict['perc_overlap_msp']
                self.r_mu = ddict['r_mu']
                self.emg_dn = ddict['dn_raw']
                self.emg_sm = ddict['sm_raw']
            except:
                # if input dictionary is incomplete, stick with current/default params
                print('ERROR: Unable to load EMG settings; one or more keys missing from input dictionary')
                return
        
        if not ddict:
            # update EMG source
            if self.useEMGraw_btn.isChecked():
                self.emg_source = 'raw'
                # update raw EMG filtering freq variables
                self.w0_emg = float(self.emgfiltLo_val.value()  / (self.mainWin.sr/2))
                self.w1_emg = float(self.emgfiltHi_val.value()  / (self.mainWin.sr/2))
                if self.emgfilt_type.currentText() == 'No filter':
                    self.w0_emg = -1
                    self.w1_emg = -1
                elif self.emgfilt_type.currentText() == 'Low-pass filter':
                    self.w0_emg = -1
                elif self.emgfilt_type.currentText() == 'High-pass filter':
                    self.w1_emg = -1
                # disable/enable freq value boxes
                lo_bool, lo_col = [True,'black'] if self.w0_emg != -1 else [False, 'gray']
                hi_bool, hi_col = [True,'black'] if self.w1_emg != -1 else [False, 'gray']
                self.emgfiltLo_val.setEnabled(lo_bool)
                self.emgfiltLo_val.lineEdit().setVisible(lo_bool)
                self.emgfilt_label1.setStyleSheet(f'color : {lo_col}')
                self.emgfiltHi_val.setEnabled(hi_bool)
                self.emgfiltHi_val.lineEdit().setVisible(hi_bool)
                self.emgfilt_label3.setStyleSheet(f'color : {hi_col}')
            elif self.useMSP_btn.isChecked():
                self.emg_source = 'msp'
                # update mSP freq band variables
                self.r_mu = [float(self.emgfiltLo_val.value()), float(self.emgfiltHi_val.value())]
                # enable freq value boxes
                self.emgfiltLo_val.setEnabled(True)
                self.emgfiltLo_val.lineEdit().setVisible(True)
                self.emgfilt_label1.setStyleSheet('color : black')
                self.emgfiltHi_val.setEnabled(True)
                self.emgfiltHi_val.lineEdit().setVisible(True)
                self.emgfilt_label3.setStyleSheet('color : black')
                
            # update raw EMG downsampling/filtering params and mSP calculation params
            self.emg_dn = float(self.emgDn_val.value())
            self.emg_sm = float(self.emgSm_val.value())
            self.nsr_seg_msp = float(self.mspWin_val.value())
            self.perc_overlap_msp = float(self.mspOverlap_val.value()/100)
        
            # update EMG amp recalculation variable/normalization method from GUI
            self.recalc_amp = bool(self.emgAmpCalc_btn.isChecked())
            self.pzscore_emg = int(self.emgZ_type.currentIndex())
        
            # show/hide params for raw EMG vs. mSP
            self.emgfilt_type.setVisible(self.emg_source == 'raw')
            self.rawCol.setVisible(self.emg_source == 'raw')
            self.mspfilt_label.setVisible(self.emg_source == 'msp')
            self.mspCol.setVisible(self.emg_source == 'msp')
    
    
    @QtCore.pyqtSlot()
    def update_twitch_params(self, ddict={}):
        """
        Update EMG twitch params from user input OR dictionary loaded from saved file
        """
        if ddict:
            try:
                self.twitch_thres = ddict['thres']
                self.twitch_thres_type = ddict['thres_type']
                self.twitch_thres_mode = ddict['thres_mode']
                self.twitch_thres_first = ddict['thres_first']
                self.min_twitchdur = ddict['min_twitchdur']
                self.min_twitchsep = ddict['min_twitchsep']
                self.min_twitchREM = ddict['min_dur']
                self.twitch_REMcutoff = ddict['rem_cutoff']
            except:
                # if input dictionary is incomplete, stick with current/default params
                print('ERROR: Unable to load twitch settings; one or more keys missing from input dictionary')
                return
                
        if not ddict:
            # update twitch threshold value
            self.twitch_thres = float(self.twitchThres_val.value())
            
            # update twitch threshold type
            i = self.twitchThres_type.currentIndex()
            d = [['raw', ' uV', 50, 1, 0.1], 
                 ['std', ' s.t.d', 10, 1, 0.1], 
                 ['perc', ' %', 100, 1, 1]][i]
            self.twitch_thres_type, suffix, maxval, decimals, singlestep = d
            self.twitchThres_val.setSuffix(suffix)
            self.twitchThres_val.setMaximum(maxval)
            self.twitchThres_val.setDecimals(decimals)
            self.twitchThres_val.setSingleStep(singlestep)

            # update twitch threshold mode
            if self.twitchThresAllREM_btn.isChecked():
                self.twitch_thres_mode = 1
            elif self.twitchThresEachREM_btn.isChecked():
                self.twitch_thres_mode = 2
                
            # threshold EMG amplitude by first X seconds of REM sleep
            if self.twitchThresFirst_btn.isChecked():
                self.twitch_thres_first = float(self.twitchThresFirst_btn.value())
                self.twitchThresFirst_val.setDisabled(False)
            else:
                self.twitch_thres_first = float(0)
                self.twitchThresFirst_val.setDisabled(True)
                self.twitchThresFirst_val.setValue(0)
                
            # load or calculate twitches
            self.recalc_twitches = bool(self.twitchCalc_btn.isChecked())
            
            # update twitch validation params
            self.min_twitchdur = float(self.minTwitchDur_val.value())
            self.min_twitchsep = float(self.minTwitchSep_val.value())
            
            # update REM sleep params
            self.min_twitchREM = float(self.minREMDur_val.value())
            self.twitch_REMcutoff = float(self.REMcutoff_val.value())
        
        
    def update_laser_params(self):
        """
        Update general laser params from user input
        """
        if 'laserWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
            # update laser plotting variable
            if self.laserYes_btn.isChecked():
                self.plaser = True
            elif self.laserNo_btn.isChecked():
                self.plaser = False
            self.disable_laser_widgets(disable=not self.plaser)
            
            # update post-laser stimulation window (+ post-stim variable in main window)
            self.post_stim = float(self.postStim_val.value())
            self.mainWin.post_stim = float(self.post_stim)
            
            # update laser plot mode
            if self.laserPwaveMode_btn.isChecked():
                self.lsr_mode = 'pwaves'
            elif self.laserLaserMode_btn.isChecked():
                self.lsr_mode = 'lsr'
                
            # update laser iso variable
            self.lsr_iso = float(self.laserIso_val.value())
    
    
    def update_laser_spectrum_params(self):
        """
        Update laser params for power spectrum plot from user input
        """
        if 'laserSpectrumWidget' in self.plotTypeWidgets[self.plotType]['widgets']:
            # update laser plotting variable
            if self.pmode1_btn.isChecked():
                self.pmode = 1
            elif self.pmode0_btn.isChecked():
                self.pmode = 0
            self.disable_laser_widgets(disable=not bool(self.pmode))
            
            # update harmonic interpolation variables
            self.harmcs = float(self.harmcs_val.value()) if self.harmcs_btn.isChecked() else 0.
            self.harmcs_mode = ['iplt','emg'][self.harmcsMode_type.currentIndex()]
            self.iplt_level = int(self.iplt_val.value())
            
            # update policy for brainstates partially overlapping laser
            self.exclusive_mode = int(self.excluMode_type.currentIndex())
            if self.pmode==1:
                self.harmcs_val.setEnabled(self.harmcs_btn.isChecked())
                self.harmcsMode_type.setEnabled(self.harmcs_btn.isChecked())
                self.iplt_val.setEnabled(self.harmcs_btn.isChecked())
                c = 'black' if self.harmcs_btn.isChecked() else 'gray'
                self.iplt_label1.setStyleSheet(f'color : {c}')
                self.iplt_label2.setStyleSheet(f'color : {c}')
    
    
    def update_dff_params(self):
        """
        Update fiber photometry params from user input
        """
        # update DFF downsampling/z-scoring
        self.dff_dn = int(self.dffdn_val.value())
        self.dff_z = self.dffz_type.currentIndex()
        self.dff_zwin = [round(self.preZWin_val.value(),2), round(self.postZWin_val.value(),2)]
        self.preZWin_val.setVisible(self.dff_z == 3)
        self.postZWin_val.setVisible(self.dff_z == 3) 
        self.zwin_dash.setVisible(self.dff_z == 3)
    
        # update DFF heatmap smoothing variables
        self.dffSmX_val.setEnabled(self.dffSmX_chk.isChecked())
        self.dffSmX_val.lineEdit().setVisible(self.dffSmX_chk.isChecked())
        self.dffSmY_val.setEnabled(self.dffSmY_chk.isChecked())
        self.dffSmY_val.lineEdit().setVisible(self.dffSmY_chk.isChecked())
        if not self.dffSmX_chk.isChecked() and not self.dffSmY_chk.isChecked(): # no smoothing
            self.dff_psm = []
        elif self.dffSmX_chk.isChecked() and self.dffSmY_chk.isChecked():       # smooth across rows/trials AND columns/time
            self.dff_psm = [int(self.dffSmY_val.value()), int(self.dffSmX_val.value())]
        elif self.dffSmX_chk.isChecked():                                       # smooth across columns/time only
            self.dff_psm = [1, int(self.dffSmX_val.value())]
        elif self.dffSmY_chk.isChecked():                                       # smooth across rows/trials only
            self.dff_psm = [int(self.dffSmY_val.value()), 1]
        
        # update saturation for DFF heatmap
        self.dffVmMin_val.setEnabled(self.dffVmCustom_btn.isChecked())
        self.dffVmMax_val.setEnabled(self.dffVmCustom_btn.isChecked())
        self.dffVmMin_val.lineEdit().setVisible(self.dffVmCustom_btn.isChecked())
        self.dffVmMax_val.lineEdit().setVisible(self.dffVmCustom_btn.isChecked())
        if self.dffVmAuto_btn.isChecked():
            self.vm = []
        elif self.dffVmCustom_btn.isChecked():
            self.vm = [float(self.dffVmMin_val.value()), float(self.dffVmMax_val.value())]
        
        
    def dict_from_vars(self):
        """
        Collect all current parameter values and return as dictionary
        """
        ddict = {'stat' : str(self.stat),
                 'tbin' : int(self.tbin),
                 'num_bins' : int(self.num_bins),
                 'win' : list(self.win),
                 'lsr_win' : list(self.lsr_win),
                 'excl_win' : list(self.excl_win),
                 'pre' : float(self.pre),
                 'post' : float(self.post),
                 'mouse_avg' : str(self.mouse_avg),
                 'twitch_avg' : str(self.twitch_avg),
                 'ci' : str(self.ci),
                 'sf' : int(self.sf),
                 'signal_type' : str(self.signal_type),
                 'pzscore' : int(self.pzscore),
                 'istate' : list(self.istate),
                 'sequence' : list(self.sequence),
                 'nstates' : list(self.nstates),
                 'state_thres' : list(self.state_thres),
                 'sign' : list(self.sign),
                 'ma_thr' : float(self.ma_thr),
                 'ma_state' : int(self.ma_state),
                 'flatten_is' : 0 if not self.flatten_is else int(self.flatten_is),
                 'exclude_noise' : bool(self.exclude_noise),
                 'tstart' : int(self.tstart),
                 'tend' : int(self.tend),
                 'p_iso' : float(self.p_iso),
                 'pcluster' : float(self.pcluster),
                 'clus_event' : str(self.clus_event),
                 'clus_iso' : float(self.clus_iso),
                 'nsr_seg' : float(self.nsr_seg),
                 'perc_overlap' : float(self.perc_overlap),
                 'recalc_highres' : bool(self.recalc_highres),
                 'fmax' : float(self.fmax),
                 'pnorm' : int(self.pnorm),
                 'psmooth' : list(self.psmooth),
                 'vm' : list(self.vm),
                 'pmode' : int(self.pmode),
                 'exclusive_mode' : int(self.exclusive_mode),
                 'harmcs' : float(self.harmcs),
                 'harmcs_mode' : str(self.harmcs_mode),
                 'iplt_level' : int(self.iplt_level),
                 'emg_source' : str(self.emg_source),
                 'nsr_seg_msp' : float(self.nsr_seg_msp),
                 'perc_overlap_msp' :  float(self.perc_overlap_msp),
                 'recalc_amp' : bool(self.recalc_amp),
                 'r_mu' : list(self.r_mu),
                 'w0_emg' : float(self.w0_emg),
                 'w1_emg' : float(self.w1_emg),
                 'emg_dn' : float(self.emg_dn),
                 'emg_sm' : float(self.emg_sm),
                 'pzscore_emg' : int(self.pzscore_emg),
                 'twitch_thres' : float(self.twitch_thres),
                 'twitch_thres_type' : str(self.twitch_thres_type),
                 'twitch_thres_mode' : int(self.twitch_thres_mode),
                 'twitch_thres_first' : float(self.twitch_thres_first),
                 'recalc_twitches' : bool(self.recalc_twitches),
                 'min_twitchdur' : float(self.min_twitchdur),
                 'min_twitchsep' : float(self.min_twitchsep),
                 'min_twitchREM' : float(self.min_twitchREM),
                 'twitch_REMcutoff' : float(self.twitch_REMcutoff),
                 'twitchFile' : str(self.twitchFile),
                 'plaser' : bool(self.plaser),
                 'post_stim' : float(self.post_stim),
                 'lsr_mode' : str(self.lsr_mode),
                 'lsr_iso' : float(self.lsr_iso),
                 'dff_dn' : int(self.dff_dn),
                 'dff_z' : int(self.dff_z),
                 'dff_zwin' : list(self.dff_zwin),
                 'dff_psm' : list(self.dff_psm),
                 'dff_vm' : list(self.dff_vm),
                 'rec' : [self.name] if len(self.recordings)==0 else self.recordings[0]}
        return ddict
    
    
    def update_gui_from_vars(self):
        """
        Update all GUI widgets from current parameter variable values
        """
        ddict = self.dict_from_vars()
        
        # set sleep timecourse params
        i = ['perc','freq','dur','is prob', 'pwave freq'].index(ddict['stat'])
        self.plotStat_type.setCurrentIndex(i)
        self.binSize_val.setValue(ddict['tbin'])
        self.binNum_val.setValue(ddict['num_bins'])
        if ddict['tbin'] >= int(len(self.mainWin.EEG) / self.mainWin.sr):
            self.binSize_btn.setChecked(False)
        else:
            self.binSize_btn.setChecked(True)
        
        # set data collection params
        self.preWin_val.setValue(ddict['win'][0])
        self.postWin_val.setValue(ddict['win'][1])
        self.preWinLsr_val.setValue(ddict['lsr_win'][0])
        self.postWinLsr_val.setValue(ddict['lsr_win'][1])
        self.preWinExcl_val.setValue(ddict['excl_win'][0])
        self.postWinExcl_val.setValue(ddict['excl_win'][1])
        self.dataAvg_type.setCurrentText(ddict['mouse_avg'].capitalize())
        i = ['all', 'each'].index(ddict['twitch_avg'])
        self.dataAvg_type.setCurrentIndex(i)
        j = ['sem', 'sd', '95'].index(ddict['ci'])
        self.error_type.setCurrentIndex(j)
        self.smooth_val.setValue(ddict['sf'])
        self.signalType.setCurrentText(ddict['signal_type'])
        self.z_type.setCurrentIndex(ddict['pzscore'])
        
        # set brainstate params
        for k,btn in self.plotStates.items():
            btn.setChecked(k in ddict['istate'])
        # set MA threshold and state
        self.maThr_val.setValue(ddict['ma_thr'])
        i = ['NREM','Wake','MA'].index(self.stateMap[ddict['ma_state']])
        self.maState_type.setCurrentIndex(i)
        # set IS state
        if not ddict['flatten_is']:
            self.isState_type.setCurrentIndex(1)
        elif ddict['flatten_is'] == 4:
            self.isState_type.setCurrentIndex(0)
        elif ddict['flatten_is'] == 3:
            self.isState_type.setCurrentIndex(2)
        elif ddict['flatten_is'] == 1:
            self.isState_type.setCurrentIndex(3)
        # set noise policy
        i = [True, False].index(ddict['exclude_noise'])
        self.noiseHandle_type.setCurrentIndex(i)
        # set analysis start and end time params
        self.tstart_val.setValue(ddict['tstart'])
        self.tend_val.setValue(ddict['tend'])
        
        # set state sequence params
        for i,w in enumerate(self.stateseqWidgets):
            if i < len(ddict['sequence']) and 'transitions' in self.plotType:
                # set params for widgets in current sequence
                w.show()
                w.findChild(QtWidgets.QComboBox, 'state').setCurrentText(self.stateMap[ddict['sequence'][i]])
                w.findChild(QtWidgets.QDoubleSpinBox, 'bins').setValue(ddict['nstates'][i])
                w.findChild(QtWidgets.QPushButton, 'sign').setText(ddict['sign'][i])
                w.findChild(QtWidgets.QDoubleSpinBox, 'thres').setValue(ddict['state_thres'][i])
                if i > 0:
                    self.brstateWidget.findChild(QtWidgets.QPushButton, f'arrow{i}').show()
            else:
                # hide unused widgets
                w.hide()
        
        # set P-wave params
        if ddict['p_iso'] == 0 and ddict['pcluster'] == 0:
            self.useAllPwaves_btn.setChecked(True)
        elif ddict['p_iso'] != 0:
            self.useSinglePwaves_btn.setChecked(True)
        elif ddict['pcluster'] != 0:
            self.useClusterPwaves_btn.setChecked(True)
        self.singlePwaveWin_val.setValue(ddict['p_iso'])
        self.clusterPwaveWin_val.setValue(ddict['pcluster'])
        i = ['waves','cluster_start','cluster_mid','cluster_end'].index(ddict['clus_event'])
        self.clusterAnalysis_type.setCurrentIndex(i)
        self.clusterIso_val.setValue(ddict['clus_iso'])
        
        # set EEG spectrogram params
        self.spWin_val.setValue(ddict['nsr_seg'])
        self.spOverlap_val.setValue(ddict['perc_overlap']*100)
        self.spFmax_val.setValue(ddict['fmax'])
        if ddict['recalc_highres']:
            self.spCalc_btn.setChecked(True)
        else:
            self.spLoad_btn.setChecked(True)
        self.spNorm_type.setCurrentIndex(ddict['pnorm'])
        if ddict['psmooth']:
            self.spSmoothX_chk.setChecked(False if ddict['psmooth'][1]==1 else True)
            self.spSmoothX_val.setValue(ddict['psmooth'][1])
            self.spSmoothY_chk.setChecked(False if ddict['psmooth'][0]==1 else True)
            self.spSmoothY_val.setValue(ddict['psmooth'][0])
        else:
            self.spSmoothX_chk.setChecked(False)
            self.spSmoothX_val.setValue(1)
            self.spSmoothY_chk.setChecked(False)
            self.spSmoothY_val.setValue(1)
        if ddict['vm']:
            self.spVmCustom_btn.setChecked(True)
            self.spVmMin_val.setValue(ddict['vm'][0])
            self.spVmMax_val.setValue(ddict['vm'][1])
        else:
            self.spVmAuto_btn.setChecked(True)
        
        # set laser spectrum params
        if ddict['pmode'] == 1:
            self.pmode1_btn.setChecked(True)
        elif ddict['pmode'] == 0:
            self.pmode0_btn.setChecked(True)
        self.harmcs_btn.setChecked(ddict['harmcs']>0)
        self.harmcs_val.setValue(ddict['harmcs'])
        i = ['iplt','emg'].index(ddict['harmcs_mode'])
        self.harmcsMode_type.setCurrentIndex(i)
        self.iplt_val.setValue(ddict['iplt_level'])
        self.excluMode_type.setCurrentIndex(ddict['exclusive_mode'])
        
        # set EMG params
        if ddict['emg_source'] == 'raw':
            self.useEMGraw_btn.setChecked(True)
        elif ddict['emg_source'] == 'msp':
            self.useMSP_btn.setChecked(True)
        self.switch_emg_source()
        self.emgDn_val.setValue(ddict['emg_dn'])
        self.emgSm_val.setValue(ddict['emg_sm'])
        self.mspWin_val.setValue(ddict['nsr_seg_msp'])
        self.mspOverlap_val.setValue(ddict['perc_overlap_msp']*100)
        if ddict['recalc_amp']:
            self.emgAmpCalc_btn.setChecked(True)
        else:
            self.emgAmpLoad_btn.setChecked(True)
        self.emgZ_type.setCurrentIndex(ddict['pzscore_emg'])
        
        # set EMG twitch params
        i = ['raw','std','perc'].index(ddict['twitch_thres_type'])
        self.twitchThres_type.setCurrentIndex(i)
        self.twitchThres_val.setValue(ddict['twitch_thres'])
        if ddict['twitch_thres_mode'] == 1:
            self.twitchThresAllREM_btn.setChecked(True)
        elif ddict['twitch_thres_mode'] == 2:
            self.twitchThresEachREM_btn.setChecked(True)
        self.twitchThresFirst_btn.setChecked(ddict['twitch_thres_first']>0)
        self.twitchThresFirst_val.setValue(ddict['twitch_thres_first'])
        if ddict['recalc_twitches']:
            self.twitchCalc_btn.setChecked(True)
        else:
            self.twitchLoad_btn.setChecked(True)
        self.minTwitchDur_val.setValue(ddict['min_twitchdur'])
        self.minTwitchSep_val.setValue(ddict['min_twitchsep'])
        self.minREMDur_val.setValue(ddict['min_twitchREM'])
        self.REMcutoff_val.setValue(ddict['twitch_REMcutoff'])
        
        # set laser params
        if ddict['plaser']:
            self.laserYes_btn.setChecked(True)
        else:
            self.laserNo_btn.setChecked(True)
        self.postStim_val.setValue(ddict['post_stim'])
        if ddict['lsr_mode'] == 'pwaves':
            self.laserPwaveMode_btn.setChecked(True)
        elif ddict['lsr_mode'] == 'lsr':
            self.laserLaserMode_btn.setChecked(True)
        self.laserIso_val.setValue(ddict['lsr_iso'])
        
        # set DF/F params
        self.dffdn_val.setValue(ddict['dff_dn'])
        self.dffz_type.setCurrentIndex(ddict['dff_z'])
        self.preZWin_val.setValue(ddict['dff_zwin'][0])
        self.postZWin_val.setValue(ddict['dff_zwin'][1])
        if ddict['dff_psm']:
            self.dffSmX_chk.setChecked(False if ddict['dff_psm'][1]==1 else True)
            self.dffSmX_val.setValue(ddict['dff_psm'][1])
            self.dffSmY_chk.setChecked(False if ddict['dff_psm'][0]==1 else True)
            self.dffSmY_val.setValue(ddict['dff_psm'][0])
        else:
            self.dffSmX_chk.setChecked(False)
            self.dffSmX_val.setValue(1)
            self.dffSmY_chk.setChecked(False)
            self.dffSmY_val.setValue(1)
        if ddict['dff_vm']:
            self.dffVmCustom_btn.setChecked(True)
            self.dffVmMin_val.setValue(ddict['dff_vm'][0])
            self.dffVmMax_val.setValue(ddict['dff_vm'][1])
        else:
            self.dffVmAuto_btn.setChecked(True)
            
    
    def update_vars_from_dict(self, ddict={}):
        """
        Update all parameter variables from the inputted dictionary $ddict
        """
        try:
            self.stat = ddict['stat']
            self.tbin = ddict['tbin']
            self.num_bins = ddict['num_bins']
            self.win = ddict['win']
            self.lsr_win = ddict['lsr_win']
            self.excl_win = ddict['excl_win']
            self.pre = ddict['pre']
            self.post = ddict['post']
            self.mouse_avg = ddict['mouse_avg']
            self.twitch_avg = ddict['twitch_avg']
            self.ci = ddict['ci']
            self.sf = ddict['sf']
            self.signal_type = ddict['signal_type']
            self.pzscore = ddict['pzscore']
            self.istate = ddict['istate']
            self.sequence = ddict['sequence']
            self.nstates = ddict['nstates']
            self.state_thres = ddict['state_thres']
            self.sign = ddict['sign']
            self.ma_thr = ddict['ma_thr']
            self.ma_state = ddict['ma_state']
            self.flatten_is = ddict['flatten_is']
            self.exclude_noise = ddict['exclude_noise']
            self.tstart = ddict['tstart']
            self.tend = ddict['tend']
            self.p_iso = ddict['p_iso']
            self.pcluster = ddict['pcluster']
            self.clus_event = ddict['clus_event']
            self.clus_iso = ddict['clus_iso']
            self.nsr_seg = ddict['nsr_seg']
            self.perc_overlap = ddict['perc_overlap']
            self.recalc_highres = ddict['recalc_highres']
            self.fmax = ddict['fmax']
            self.pnorm = ddict['pnorm']
            self.psmooth = ddict['psmooth']
            self.vm = ddict['vm']
            self.pmode = ddict['pmode']
            self.exclusive_mode = ddict['exclusive_mode']
            self.harmcs = ddict['harmcs']
            self.harmcs_mode = ddict['harmcs_mode']
            self.iplt_level = ddict['iplt_level']
            self.emg_source = ddict['emg_source']
            self.nsr_seg_msp = ddict['nsr_seg_msp']
            self.recalc_amp = ddict['recalc_amp']
            self.perc_overlap_msp = ddict['perc_overlap_msp']
            self.r_mu = ddict['r_mu']
            self.w0_emg = ddict['w0_emg']
            self.w1_emg = ddict['w1_emg']
            self.emg_dn = ddict['emg_dn']
            self.emg_sm = ddict['emg_sm']
            self.pzscore_emg = ddict['pzscore_emg']
            self.twitch_thres = ddict['twitch_thres']
            self.twitch_thres_type = ddict['twitch_thres_type']
            self.twitch_thres_mode = ddict['twitch_thres_mode']
            self.twitch_thres_first = ddict['twitch_thres_first']
            self.recalc_twitches = ddict['recalc_twitches']
            self.min_twitchdur = ddict['min_twitchdur']
            self.min_twitchsep = ddict['min_twitchsep']
            self.min_twitchREM = ddict['min_twitchREM']
            self.twitch_REMcutoff = ddict['twitch_REMcutoff']
            self.twitchFile = ddict['twitchFile']
            self.plaser = ddict['plaser']
            self.post_stim = ddict['post_stim']
            self.lsr_mode = ddict['lsr_mode']
            self.lsr_iso = ddict['lsr_iso']
            self.dff_dn = ddict['dff_dn']
            self.dff_z = ddict['dff_z']
            self.dff_zwin = ddict['dff_zwin']
            self.dff_psm = ddict['dff_psm']
            self.dff_vm = ddict['dff_vm']
            return 1
        except KeyError:
            return 0
    
    
    def get_signal(self, sig):
        """
        Load or calculate signal surrounding P-waves/laser events
        @Params
        sig - string indicating the type of signal to be collected/plotted
              'EEG', 'EMG', 'LFP', 'SP', 'SP_NORM'
        """
        # re-acquire data if any data collection variables change
        calcKeys = ['istate', 'win', 'tstart', 'tend', 'ma_thr', 'ma_state', 'flatten_is', 
                    'exclude_noise', 'p_iso', 'pcluster', 'plaser', 'post_stim', 'lsr_iso']
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        
        collectData = False
        if self.plotSettings == {}:
            collectData = True     # first plot, need initial data
        elif rec != self.plotSettings['rec']:
            collectData = True     # change in recording(s) on plot
        elif not self.plaser and self.p_signal is None:
            collectData = True     # settings loaded but P-waves not collected yet
        elif self.plaser and all(x is None for x in [self.lsr_pwaves, self.spon_pwaves, 
                                                     self.success_lsr, self.fail_lsr]):
            collectData = True
        elif self.pcluster > 0 and self.plotSettings['clus_event'] != self.clus_event:
            collectData = True     # if collecting clustered P-waves, run again if type of cluster event is changed
        elif sig=='SP' and self.recalc_highres:
            collectData = True     # recalculate high-res SP and run again
        elif sig=='SP' and set([self.pnorm, self.plotSettings['pnorm']]) == ({0,1} or {1,2}):
            collectData = True    # currently plotted SP has the wrong normalization
        elif sig!='SP' and self.plotSettings['signal_type'] != self.signal_type:
            collectData = True    # Intan signal type (LFP,EMG,etc.) changed from current plot
        else:
            ddict = self.dict_from_vars()
            for k in calcKeys:
                if self.plotSettings[k] != ddict[k]:
                    collectData = True  # run again if any data collection variables changed
                    break
        # specify SP signal normalization, get recording name(s)
        if sig == 'SP':
            sig = 'SP_NORM' if self.pnorm==1 else 'SP'
        # collect signal data according to new params
        if collectData:
            self.setWindowTitle(f'Calculating {sig} data ... ')
            if self.plaser:
                data = pwaves.get_lsr_surround(self.ppath, rec, istate=self.istate, win=self.win, 
                                               signal_type=sig, recalc_highres=self.recalc_highres, 
                                               tstart=self.tstart, tend=self.tend, ma_thr=self.ma_thr, 
                                               ma_state=self.ma_state, flatten_is=self.flatten_is, 
                                               exclude_noise=self.exclude_noise, nsr_seg=self.nsr_seg, 
                                               perc_overlap=self.perc_overlap, post_stim=self.post_stim, 
                                               lsr_iso=self.lsr_iso, null=False, p_iso=self.p_iso, 
                                               pcluster=self.pcluster, clus_event=self.clus_event, psave=False)
                self.lsr_pwaves, self.spon_pwaves, self.success_lsr, self.fail_lsr = data[0:4]
            elif not self.plaser:
                data = pwaves.get_surround(self.ppath, rec, istate=self.istate, win=self.win, 
                                           signal_type=sig, recalc_highres=self.recalc_highres, 
                                           tstart=self.tstart, tend=self.tend, ma_thr=self.ma_thr, 
                                           ma_state=self.ma_state, flatten_is=self.flatten_is,
                                           exclude_noise=self.exclude_noise, nsr_seg=self.nsr_seg, 
                                           perc_overlap=self.perc_overlap, null=False, p_iso=self.p_iso, 
                                           pcluster=self.pcluster, clus_event=self.clus_event, psave=False)
                self.p_signal = data[0]
            self.setWindowTitle('Done!')
    
    
    def get_emg_amp(self):
        """
        Load or calculate EMG amplitude
        """
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        
        calcMSP = False
        loadEMGAmp = False  # reload/recalculate EMG amplitude for current recording?
        if self.recalc_amp == True:
            loadEMGAmp = True; calcMSP=True  # manual instruction to recalculate EMG amplitude
        elif self.plotSettings == {}:
            loadEMGAmp = True     # first plot, need initial data
        elif all(x is None for x in [self.EMG_amp, self.mnbin, self.mdt]):
            loadEMGAmp = True     # EMG amplitude not yet loaded/calculated
        elif self.emg_source != self.plotSettings['emg_source']:
            loadEMGAmp = True     # EMG source changed from last calculation
        else:
            if self.emg_source == 'raw':  # EMG source is raw and raw params changed
                x1 = self.w0_emg != self.plotSettings['w0_emg']
                x2 = self.w1_emg != self.plotSettings['w1_emg']
                x3 = self.emg_dn != self.plotSettings['emg_dn']
                x4 = self.emg_sm != self.plotSettings['emg_sm']
            elif self.emg_source == 'msp':  # EMG source is spectrogram and mSP params changed
                x1 = self.nsr_seg_msp != self.plotSettings['nsr_seg_msp']
                x2 = self.perc_overlap_msp != self.plotSettings['perc_overlap_msp']
                x3 = self.r_mu != self.plotSettings['r_mu']
                x4 = False
            if any([x1,x2,x3,x4]):
                loadEMGAmp = True; self.recalc_amp = True
                if self.emg_source == 'msp' and (x1 or x2):
                    calcMSP = True
        if loadEMGAmp:
            self.setWindowTitle('Calculating EMG amplitude')
            # update EMG amp vector of current recording based on new params
            tmp = AS.emg_amplitude(self.ppath, self.name, self.emg_source, 
                                   recalc_amp=self.recalc_amp, nsr_seg=self.nsr_seg_msp, 
                                   perc_overlap=self.perc_overlap_msp, r_mu=self.r_mu, 
                                   recalc_highres=calcMSP, w0=self.w0_emg, 
                                   w1=self.w1_emg, dn=self.emg_dn, smooth=self.emg_sm, 
                                   exclude_noise=self.exclude_noise, pemg2=False)
            self.EMG_amp, self.mnbin, self.mdt = tmp
            self.setWindowTitle('Done!')
        
        calcKeys = ['istate', 'win', 'tstart', 'tend', 'ma_thr', 'ma_state', 'flatten_is', 
                    'exclude_noise', 'p_iso', 'pcluster', 'plaser', 'post_stim', 'lsr_iso']
        collectData = False  # recollect EMG amp surrounding P-waves?
        if loadEMGAmp:
            collectData = True
        elif rec != self.plotSettings['rec']:
            collectData = True     # change in recording(s) on plot
        elif not self.plaser and self.p_signal is None:
            collectData = True     # settings loaded but P-waves not collected yet
        elif self.plaser and all(x is None for x in [self.lsr_pwaves, self.spon_pwaves, 
                                                     self.success_lsr, self.fail_lsr]):
            collectData = True
        elif self.pcluster > 0 and self.plotSettings['clus_event'] != self.clus_event:
            collectData = True    # if collecting clustered P-waves, run again if type of cluster event is changed
        else:
            ddict = self.dict_from_vars()
            for k in calcKeys:
                if self.plotSettings[k] != ddict[k]:
                    collectData = True  # run again if any data collection variables changed
                    break
        if collectData:
            self.setWindowTitle('Collecting EMG amplitude data ...')
            # use calculated vector to get avg. EMG amp surrounding P-waves for current recording
            emg_calculated = [self.EMG_amp, self.mnbin, self.mdt] if rec==[self.name] else None
            data = pwaves.pwave_emg(self.ppath, rec, self.emg_source, self.win, istate=self.istate, 
                                    rem_cutoff=True, recalc_amp=self.recalc_amp, nsr_seg=self.nsr_seg_msp, 
                                    perc_overlap=self.perc_overlap_msp, recalc_highres=calcMSP, 
                                    r_mu=self.r_mu, w0=self.w0_emg, w1=self.w1_emg, dn=self.emg_dn, 
                                    smooth=self.emg_sm, pzscore=self.pzscore_emg, tstart=self.tstart, 
                                    tend=self.tend, ma_thr=self.ma_thr, ma_state=self.ma_state, 
                                    flatten_is=self.flatten_is, exclude_noise=self.exclude_noise, 
                                    pemg2=False, p_iso=self.p_iso, pcluster=self.pcluster, 
                                    clus_event=self.clus_event, plaser=self.plaser, post_stim=self.post_stim, 
                                    lsr_iso=self.lsr_iso, lsr_mode=self.lsr_mode, pplot=False, 
                                    emg_calculated=emg_calculated)[0]
            if self.plaser:
                self.lsr_pwaves, self.spon_pwaves, self.success_lsr, self.fail_lsr = data
            else:
                self.p_signal = data[0]
            self.setWindowTitle('Done!')
    
    
    def plot_state_freq(self):
        """
        Plot mean P-wave frequency and averaged waveform in each brain state
        """
        self.fig.clear()
        
        # get average P-wave frequency in each state
        self.setWindowTitle('Calculating P-wave frequency ... ')
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        data = pwaves.state_freq(self.ppath, rec, self.istate, tstart=self.tstart, 
                                 tend=self.tend, wf_win=self.win, ma_thr=self.ma_thr, 
                                 ma_state=self.ma_state, flatten_is=self.flatten_is, 
                                 exclude_noise=self.exclude_noise, p_iso=self.p_iso, 
                                 pcluster=self.pcluster, clus_event=self.clus_event, 
                                 pplot=False, print_stats=False)
        mice, brstates, freq_mx, wform_mx = data
        self.setWindowTitle('Done!')

        # plot average P-wave frequency in each state
        self.fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
        grid = GridSpec(len(self.istate), 2, figure=self.fig)
        ax1 = self.fig.add_subplot(grid[:,0])
        data = np.nanmean(freq_mx, axis=0)
        yerr = np.nanstd(freq_mx, axis=0)
        if self.ci=='sem':
            yerr /= np.sqrt(freq_mx.shape[0])
        ax1.bar(brstates, data, yerr=yerr, edgecolor='black',
                color=[self.mainWin.lut_brainstate[s]/255 for s in self.istate])
        for i,m in enumerate(mice):
            ax1.plot(brstates, freq_mx[i,:], linewidth=3, label=m,
                     color=list(np.random.choice(range(255), size=3)/255))
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_ylabel('P-waves/s')
        ax1.set_title('State Frequency')
        ax1.legend()

        # plot average waveform in each state
        t = np.arange(-wform_mx.shape[1]/2, wform_mx.shape[1]/2)
        for i in range(len(self.istate)):
            ax = self.fig.add_subplot(grid[i,1])
            data = np.nanmean(wform_mx[:,:,i], axis=0)/1000
            yerr = np.nanstd(wform_mx[:,:,i], axis=0)/1000
            ax.plot(t, data, color=self.mainWin.lut_brainstate[self.istate[i]]/255, linewidth=3)
            ax.fill_between(t, data-yerr, data+yerr, alpha=0.5,
                            color=self.mainWin.lut_brainstate[self.istate[i]]/255)
            ax.set_ylim((-0.3,0.1))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(brstates[i])
            ax.set_ylabel('Amp. (mV)')
            if i == len(self.istate)-1:
                ax.set_xlabel('Time (ms)')
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_pwaveforms(self):
        """
        Plot averaged LFP/EEG/EMG signal surrounding P-waves or laser pulses
        """
        self.fig.clear()
        # re-collect LFP waveforms, if plot settings changed
        self.get_signal(self.signal_type)
        
        # get average waveforms
        if self.plaser:
            if self.lsr_mode == 'pwaves':
                # waveforms surrounding laser-triggered vs. spontaneous P-waves
                ddict, ddict2 = [dict(self.lsr_pwaves), dict(self.spon_pwaves)]
                title, title2 = ['Laser-triggered P-waves', 'Spontaneous P-waves']
                c, c2 = ['blue','green']
            elif self.lsr_mode == 'lsr':
                # waveforms surrounding successful vs. failed laser pulses
                ddict, ddict2 = [dict(self.success_lsr), dict(self.fail_lsr)]
                title, title2 = ['Successful laser pulses', 'Failed laser pulses']
                c, c2 = ['blue','red']
            # get mouse or trial-averaged matrix
            mx = [pwaves.mx2d(ddict[s], mouse_avg=self.mouse_avg)[0] for s in self.istate]
            mx2 = [pwaves.mx2d(ddict2[s], mouse_avg=self.mouse_avg)[0] for s in self.istate]
            
            # check for instances without any P-waves/laser pulses
            nrows, ncols = [1,2]
            for arr in mx + mx2:
                if arr.size > 0:
                    nrows, ncols = arr.shape
                    break
            # replace missing data with placeholder matrices filled with NaNs
            for i in range(len(self.istate)):
                if mx[i].size == 0:
                    mx[i] = np.empty((nrows,ncols))
                    mx[i][:] = np.nan
                if mx2[i].size == 0:
                    mx2[i] = np.empty((nrows,ncols))
                    mx2[i][:] = np.nan
            if self.plotAllStates_btn.isChecked():
                # combine waveforms from all selected states
                mx = [np.vstack(mx)] if self.mouse_avg=='trial' else [np.nanmean(mx, axis=0)]
                mx2 = [np.vstack(mx2)] if self.mouse_avg=='trial' else [np.nanmean(mx2, axis=0)]
                
        elif not self.plaser:
            # construct single event matrices
            mx = [pwaves.mx2d(self.p_signal[s], mouse_avg=self.mouse_avg)[0] for s in self.istate]
            nrows, ncols = [max([arr.shape[i] if arr.size > 0 else 0 for arr in mx]) for i in [0,1]]
            nrows, ncols = [max([nrows,1]), max([ncols,2])]
            for i in range(len(self.istate)):
                if mx[i].size == 0:
                    mx[i] = np.empty((nrows,ncols))
                    mx[i][:] = np.nan
            title = 'All P-waves'
            c='black'
            if self.plotAllStates_btn.isChecked():
                mx = [np.vstack(mx)] if self.mouse_avg=='trial' else [np.nanmean(mx, axis=0)]
        x = np.linspace(-np.abs(self.win[0]), self.win[1], ncols)
        all_states = ' + '.join([self.stateMap[s] for s in self.istate])
        
        # plot graphs
        self.fig.set_constrained_layout_pads(w_pad=0.3/(int(self.plaser)+1), 
                                             h_pad=3.0/(len(mx)**2))
        grid = GridSpec(len(mx), int(self.plaser)+1, figure=self.fig)
        # ignore runtime warnings about NaNs - these are intentional placeholders
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            iplt = 0
            for i in range(len(mx)):
                # plot all P-waves OR laser P-waves/successful laser
                ax = self.fig.add_subplot(grid[iplt])
                data = np.nanmean(mx[i], axis=0)
                yerr = np.nanstd(mx[i], axis=0)
                if self.ci=='sem':
                    yerr /= np.sqrt(mx[i].shape[0])
                ax.plot(x, data, color=c, linewidth=3)
                ax.fill_between(x, data-yerr, data+yerr, color=c, alpha=0.5)
                ax.set_ylabel(self.signal_type + ' Amplitude (uV)')
                if self.plotAllStates_btn.isChecked():
                    tmp = '\n(' + all_states + ')'
                else:
                    tmp = ' (' + self.stateMap[self.istate[i]] + ')'
                ax.set_title(title + tmp)
                if i == len(mx)-1:
                    ax.set_xlabel('Time (s)')
                iplt += 1
                # plot spontaneous P-waves/failed laser
                if self.plaser:
                    ax2 = self.fig.add_subplot(grid[iplt])
                    data2 = np.nanmean(mx2[i], axis=0)
                    yerr2 = np.nanstd(mx2[i], axis=0)
                    if self.ci=='sem':
                        yerr2 /= np.sqrt(mx2[i].shape[0])
                    ax2.plot(x, data2, color=c2, linewidth=3)
                    ax2.fill_between(x, data2-yerr2, data2+yerr2, color=c2, alpha=0.5)
                    ax2.set_title(title2 + tmp)
                    if i == len(mx2)-1:
                        ax2.set_xlabel('Time (s)')
                    # make y axes equivalent
                    y = [min(ax.get_ylim()[0], ax2.get_ylim()[0]), max(ax.get_ylim()[1], ax2.get_ylim()[1])]
                    ax.set_ylim(y)
                    ax2.set_ylim(y)
                    iplt += 1
        self.canvas.draw()
        self.cleanup()
        
    
    def plot_avg_sp(self):
        """
        Plot averaged EEG spectrogram surrounding P-waves or laser pulses
        """
        self.fig.clear()
        # re-collect spectrograms, if plot settings changed
        self.get_signal('SP')
        
        ifreq = np.arange(0, int(self.fmax*2+1))  # frequency indices
        freq = np.arange(0, self.fmax+0.5, 0.5)   # frequencies
        norm = True if self.pnorm==2 else False   # norm=True to normalize by time window
        state = self.istate[0]
        
        # get average spectrograms
        if self.plaser:
            if self.lsr_mode == 'pwaves':
                # spectrograms surrounding laser-triggered vs. spontaneous P-waves
                mx = pwaves.mx3d(self.lsr_pwaves[state], mouse_avg=self.mouse_avg)[0]
                mx2 = pwaves.mx3d(self.spon_pwaves[state], mouse_avg=self.mouse_avg)[0]
                title, title2 = ['Laser-triggered P-waves', 'Spontaneous P-waves']
            elif self.lsr_mode == 'lsr':
                # spectrograms surrounding successful vs. failed laser pulses
                mx = pwaves.mx3d(self.success_lsr[state], mouse_avg=self.mouse_avg)[0]
                mx2 = pwaves.mx3d(self.fail_lsr[state], mouse_avg=self.mouse_avg)[0]
                title, title2 = ['Successful laser pulses', 'Failed laser pulses']
        elif not self.plaser:
            mx = pwaves.mx3d(self.p_signal[state], mouse_avg=self.mouse_avg)[0]
            title = 'All P-waves'
        
        # plot figure
        self.fig.set_constrained_layout_pads(w_pad=0.45, 
                                             h_pad=0.45 if self.plaser else 3)
        x = np.linspace(-np.abs(-self.win[0]), self.win[1], mx.shape[1])
        ax = self.fig.add_subplot(int(self.plaser)+1, 1, 1)
        mx_plot = AS.adjust_spectrogram(np.nanmean(mx, axis=2)[ifreq, :], 
                                        norm, self.psmooth)
        im = ax.pcolorfast(x, freq, mx_plot, cmap='jet')
        if len(self.vm)==2:
            im.set_clim(self.vm)
        self.fig.colorbar(im, ax=ax, pad=0.05)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(f'{title} ({self.stateMap[state]})')
        if self.plaser:
            ax2 = self.fig.add_subplot(int(self.plaser)+1, 1, 2)
            mx2_plot = AS.adjust_spectrogram(np.nanmean(mx2, axis=2)[ifreq, :], 
                                             norm, self.psmooth)
            im2 = ax2.pcolorfast(x, freq, mx2_plot, cmap='jet')
            if len(self.vm)==2:
                im2.set_clim(self.vm)
            self.fig.colorbar(im2, ax=ax2, pad=0.05)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Freq (Hz)')
            ax2.set_title(f'{title2} ({self.stateMap[state]})')
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_single_sp(self):
        """
        Plot EEG spectrogram surrounding the P-wave or laser pulse currently
        selected in the main annotation window
        """
        self.fig.clear()
        # get P-wave index and surrounding data window
        pi = int(self.mainWin.curIdx)
        iwin1, iwin2 = pwaves.get_iwins(self.win, self.mainWin.sr)
        
        # load EEG, adjust Intan idx to properly translate to SP idx
        EEG = so.loadmat(os.path.join(self.ppath, self.name, 'EEG.mat'), squeeze_me=True)['EEG']
        spi_adjust = np.linspace(-self.mainWin.sr, self.mainWin.sr, len(EEG))
        # load high-res spectrogram, normalize by recording if $self.pnorm==1
        SPEC = AS.highres_spectrogram(self.ppath, self.name, nsr_seg=self.nsr_seg, 
                                      perc_overlap=self.perc_overlap, 
                                      recalc_highres=self.recalc_highres, mode='EEG')
        SP, f, t, sp_nbin, sp_dt = SPEC[0:5]
        if self.pnorm == 1:  # normalize entire spectrogram
            SP_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.repeat([SP_mean], SP.shape[1], axis=0).T)
        # translate Intan --> Fourier idx, get spectrogram surrounding P-wave
        sp_win1 = int(round(iwin1/sp_nbin))
        sp_win2 = int(round(iwin2/sp_nbin))
        spi = int(round((pi + spi_adjust[pi])/sp_nbin))
        SP_pwave = SP[:, spi-sp_win1 : spi+sp_win2]
        
        ifreq = np.arange(0, int(self.fmax*2+1))  # frequency idxs
        freq = np.arange(0, self.fmax+0.5, 0.5)   # frequencies
        norm = True if self.pnorm==2 else False   # norm=True to normalize by time window
        
        # normalize/smooth SP for plotting
        SP_plot = AS.adjust_spectrogram(SP_pwave[ifreq, :], norm, self.psmooth)
        x = np.linspace(-np.abs(self.win[0]), self.win[1], SP_plot.shape[1])
        ax = self.fig.add_subplot(111)
        im = ax.pcolorfast(x, freq, SP_plot, cmap='jet')
        if len(self.vm) == 2:
            im.set_clim(self.vm)
        self.fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.35)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(f'SP surrounding event at index {pi} ({round(pi/self.mainWin.sr, 1)} s)')
        self.fig.colorbar(im, ax=ax, pad=0.05)
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_lsr_pwave_stats(self):
        """
        Plot summary statistics for laser-triggered P-wave experiment
        """
        self.fig.clear()
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        
        # get summary stats for laser pulses and P-waves
        df = pwaves.lsr_pwaves_sumstats(self.ppath, rec, self.istate, tstart=self.tstart, 
                                        tend=self.tend, ma_thr=self.ma_thr, ma_state=self.ma_state, 
                                        flatten_is=self.flatten_is, exclude_noise=self.exclude_noise, 
                                        post_stim=self.post_stim, p_iso=self.p_iso, pcluster=self.pcluster, 
                                        clus_event=self.clus_event, lsr_iso=self.lsr_iso)
        if self.lsr_mode == 'pwaves':
            colnames = ['% lsr p-waves', 'lsr p-waves', 'spon p-waves']
            ylabel, title1, title2 = ['P-waves', 'Percent of P-waves triggered by laser', 'Total P-waves']
            patchlabels = ['Laser-triggered', 'Spontaneous']
        elif self.lsr_mode == 'lsr':
            colnames = ['% success lsr', 'success lsr', 'fail lsr']
            ylabel, title1, title2 = ['laser pulses', 'Percent successful laser pulses', 'Total laser pulses']
            patchlabels = ['Successful', 'Failed']
        
        # create subject x brainstate x 3 matrix for data averaging
        mice = np.unique(df.mouse)
        mx = np.zeros((len(mice), len(self.istate), 3))
        for i,s in enumerate(self.istate):
            midx = [np.intersect1d(np.where(df.state==s)[0], np.where(df.mouse==m)[0]) for m in mice]
            mx[:,i,0] = [np.nanmean(df[colnames[0]][mi]) for mi in midx]  # percent lsr p-waves/successful lsr pulses
            mx[:,i,1] = [np.nansum(df[colnames[1]][mi]) for mi in midx]   # total p-waves/lsr pulses
            mx[:,i,2] = [np.nansum(df[colnames[2]][mi]) for mi in midx]
        
        # plot bar graphs
        self.fig.set_constrained_layout_pads(w_pad=0.45, h_pad=0.35)
        brstates = [self.stateMap[s] for s in self.istate]
        brcolor = [self.mainWin.lut_brainstate[s]/255 for s in self.istate]
        x = np.arange(len(self.istate))
        width=0.4
        ax = self.fig.add_subplot(211)
        data = np.nanmean(mx[:,:,0], axis=0)
        yerr = np.nanstd(mx[:,:,0], axis=0)
        if self.ci=='sem':
            yerr /= np.sqrt(mx.shape[0])
        ax.bar(brstates, data, yerr=yerr, width=width*2, color=brcolor, edgecolor='black')
        ax.set_ylabel('% ' + ylabel)
        ax.set_title(title1)
        ax2 = self.fig.add_subplot(212)
        data2 = np.nansum(mx[:,:,1], axis=0)
        data3 = np.nansum(mx[:,:,2], axis=0)
        ax2.bar(x-(width/2), data2, width=width, color=brcolor, alpha=0.5, 
                edgecolor='gray', linewidth=3)
        ax2.bar(x+(width/2), data3, width=width, color='whitesmoke', 
                edgecolor=brcolor, linewidth=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(brstates)
        ax2.set_ylabel('# ' + ylabel)
        ax2.set_title(title2)
        for i,m in enumerate(mice):
            c = list(np.random.choice(range(255), size=3)/255)
            ax.plot(brstates, mx[i,:,0], color=c, linewidth=3, label=m)
            ax2.plot(x-(width/2), mx[i,:,1], color=c, linewidth=0, marker='o', 
                     markeredgecolor='black')
            ax2.plot(x+(width/2), mx[i,:,2], color=c, linewidth=0, marker='o', 
                     markeredgecolor='black')
        ax.legend()
        AS.label_bars(ax, above=1)
        AS.label_bars(ax2, above=5)
        patch1 = matplotlib.patches.Patch(facecolor='lightgray', edgecolor='lightgray', linewidth=3)
        patch2 = matplotlib.patches.Patch(facecolor='whitesmoke', edgecolor='black', linewidth=3)
        ax2.legend(handles=[patch1,patch2], labels=patchlabels)
        self.canvas.draw()
        self.cleanup()
        
    
    def plot_avg_pwave_emg(self):
        """
        Plot averaged EMG amplitude surrounding P-waves or laser pulses
        """
        self.fig.clear()
        # re-collect EMG amplitude, if plot settings changed
        self.get_emg_amp()
        
        # get average EMG amplitude
        if self.plaser:
            if self.lsr_mode == 'pwaves':
                # signal surrounding laser-triggered vs. spontaneous P-waves
                ddict, ddict2 = [dict(self.lsr_pwaves), dict(self.spon_pwaves)]
                title, title2 = ['Laser-triggered P-waves', 'Spontaneous P-waves']
                c, c2 = ['blue','green']
            elif self.lsr_mode == 'lsr':
                # signal surrounding successful vs. failed laser pulses
                ddict, ddict2 = [dict(self.success_lsr), dict(self.fail_lsr)]
                title, title2 = ['Successful laser pulses', 'Failed laser pulses']
                c, c2 = ['blue','red']
            # get mouse or trial-averaged matrix
            mx, mx2 = [[None]*len(self.istate), [None]*len(self.istate)]
            for i,s in enumerate(self.istate):
                if any([len(v)>0 for v in ddict[s].values()]):
                    mx[i] = pwaves.mx2d(ddict[s], mouse_avg=self.mouse_avg)[0]
                if any([len(v)>0 for v in ddict2[s].values()]):
                    mx2[i] = pwaves.mx2d(ddict2[s], mouse_avg=self.mouse_avg)[0]
                
            if self.plotAllStates_btn.isChecked():
                # combine waveforms from all selected states
                if self.mouse_avg == 'trial':
                    mx = [np.vstack([m for m in mx if m is not None])]
                    mx2 = [np.vstack([m for m in mx2 if m is not None])]
                else:
                    mx = [np.nanmean([m for m in mx if m is not None], axis=0)]
                    mx2 = [np.nanmean([m for m in mx2 if m is not None], axis=0)]
                    
        elif not self.plaser:
            mx = [None]*len(self.istate)
            for i,s in enumerate(self.istate):
                if any([len(v)>0 for v in self.p_signal[s].values()]):
                    mx[i] = pwaves.mx2d(self.p_signal[s], mouse_avg=self.mouse_avg)[0]
            title = 'All P-waves'
            c='black'
            if self.plotAllStates_btn.isChecked():
                if self.mouse_avg == 'trial':
                    mx = [np.vstack([m for m in mx if m is not None])]
                else:
                    mx = [np.nanmean([m for m in mx if m is not None], axis=0)]
        x = np.linspace(-np.abs(self.win[0]), self.win[1], mx[0].shape[1])
        all_states = ' + '.join([self.stateMap[s] for s in self.istate])
        
        # plot graphs
        self.fig.set_constrained_layout_pads(w_pad=0.3/(int(self.plaser)+1), 
                                             h_pad=3.0/(len(mx)**2))
        grid = GridSpec(len(mx), int(self.plaser)+1, figure=self.fig)
        iplt = 0
        for i in range(len(mx)):
            # plot all P-waves OR laser P-waves/successful laser
            ax = self.fig.add_subplot(grid[iplt])
            if mx[i] is not None:
                data = np.nanmean(mx[i], axis=0)
                yerr = np.nanstd(mx[i], axis=0)
                if self.ci=='sem':
                    yerr /= np.sqrt(mx[i].shape[1])
                ax.plot(x, data, color=c, linewidth=3)
                ax.fill_between(x, data-yerr, data+yerr, color=c, alpha=0.5)
                if i == len(mx)-1:
                    ax.set_xlabel('Time (s)')
            ax.set_ylabel('EMG Amp. (uV)')
            if self.plotAllStates_btn.isChecked():
                tmp = '\n(' + all_states + ')'
            else:
                tmp = ' (' + self.stateMap[self.istate[i]] + ')'
            ax.set_title(title + tmp)
            iplt += 1
            # plot spontaneous P-waves/failed laser
            if self.plaser:
                ax2 = self.fig.add_subplot(grid[iplt])
                if mx2[i] is not None:
                    data2 = np.nanmean(mx2[i], axis=0)
                    yerr2 = np.nanstd(mx2[i], axis=0) 
                    if self.ci=='sem':
                        yerr2 /= np.sqrt(mx2[i].shape[1])
                    ax2.plot(x, data2, color=c2, linewidth=3)
                    ax2.fill_between(x, data2-yerr2, data2+yerr2, color=c2, alpha=0.5)
                    if i == len(mx2)-1:
                        ax2.set_xlabel('Time (s)')
                ax2.set_title(title2 + tmp)
                # make y axes equivalent
                y = [min(ax.get_ylim()[0], ax2.get_ylim()[0]), 
                     max(ax.get_ylim()[1], ax2.get_ylim()[1])]
                ax.set_ylim(y)
                ax2.set_ylim(y)
                iplt += 1
        self.canvas.draw()
        self.cleanup()
        
        
    def plot_single_pwave_emg(self):
        """
        Plot EMG amplitude surrounding the P-wave or laser pulse currently
        selected in the main annotation window
        """
        self.fig.clear()
        # update EMG amplitude
        self.get_emg_amp()
        
        # get P-wave index and surrounding data window
        pi = int(self.mainWin.curIdx)
        iwin1 = int(round(np.abs(self.win[0]) / self.mdt))
        iwin2 = int(round(np.abs(self.win[1]) / self.mdt))
        emg_data = self.EMG_amp[int(pi/self.mnbin)-iwin1 : int(pi/self.mnbin)+iwin2+1]
        emg_data = scipy.stats.zscore(emg_data)
        
        # plot graph
        self.fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.35)
        x = np.linspace(-np.abs(self.win[0]), self.win[1], len(emg_data))
        ax = self.fig.add_subplot(111)
        ax.plot(x, emg_data, color='black', linewidth=5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('EMG Amp.')
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_stateseq(self):
        """
        Plot average time-normalized P-wave frequency or DF/F signal across 
        brain state transitions
        """
        self.fig.clear()
        # get list(s) of recordings
        doses = ['0']; C = ['gray']
        rec_list = []
        if len(self.recordings)==0:
            rec_list.append([self.name])
        else:
            rec_list.append(self.recordings[0])
            for dose in list(self.recordings[1].keys()):
                rec_list.append(self.recordings[1][dose])
                doses.append(dose)
                C.append('blue' if dose=='0.25' else 'red' if dose=='5' else 'green')
        mode = 'pwaves' if 'P-wave' in self.plotType else 'dff'
        data_list = []  # [mice, timecourse mx, SP mx] for each dose
        self.setWindowTitle('Calculating time-normalized state transitions ...')
        for rec, dose in zip(rec_list, doses):
            # get average P-wave frequency or DFF signal for each recording list
            data = pwaves.stateseq(self.ppath, rec, self.sequence, 
                                   nstates=[int(n) for n in self.nstates], 
                                   state_thres=self.state_thres, sign=self.sign, 
                                   mode=mode, tstart=self.tstart, tend=self.tend, 
                                   ma_thr=self.ma_thr, ma_state=self.ma_state, 
                                   flatten_is=self.flatten_is, p_iso=self.p_iso, 
                                   pcluster=self.pcluster, clus_event=self.clus_event, 
                                   fmax=self.fmax, pnorm=self.pnorm, sf=self.sf, 
                                   mouse_avg=self.mouse_avg, pplot=False, print_stats=False)
            data_list.append(data)
        self.setWindowTitle('Done!')
        
        # plot averaged spectrogram during transition
        self.fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1)
        grid = GridSpec(2, 1, height_ratios=[1,3], figure=self.fig)
        ax1 = self.fig.add_subplot(grid[0])
        SP = np.nanmean(np.dstack([np.nanmean(d[2], axis=0) for d in data_list]), axis=2)
        SP = AS.adjust_spectrogram(SP, pnorm=False, psmooth=self.psmooth)
        t = np.arange(0, SP.shape[1])
        im = ax1.pcolorfast(t, np.arange(0, self.fmax+0.5, 0.5), SP, cmap='jet')
        if len(self.vm)==2:
            im.set_clim(self.vm)
        self.fig.colorbar(im, ax=ax1, pad=0.0)
        x = np.concatenate(([0], np.cumsum(self.nstates)))
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
        ax1.set_ylabel('Freq. (Hz)')
        title = ' > '.join([self.stateMap[s] for s in self.sequence])
        ax1.set_title(title)
        # plot activity timecourse(s)
        ax2 = self.fig.add_subplot(grid[1])
        for d,c,dose in zip(data_list,C,doses):
            mice, tc = d[0:2]
            data = np.nanmean(tc, axis=0)
            yerr = np.nanstd(tc, axis=0)
            if self.ci=='sem':
                yerr /= np.sqrt(tc.shape[0])
            ax2.plot(t, data, color=c, linewidth=3, label=f'dose = {dose}')
            ax2.fill_between(t, data-yerr, data+yerr, color=c, alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x)
        ax2.set_xlabel('Time bins (normalized)')
        ax2.set_ylabel('P-waves/s' if mode=='pwaves' else '\u0394F/F')
        if len(doses)>1:
            ax2.legend()
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_dff_pwaves(self):
        """
        Plot averaged DF/F signal surrounding P-waves
        """
        self.fig.clear()
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        # get P-waves during all states (0) or specified brain state (1-6)
        istate = 0 if self.plotAllStates_btn.isChecked() else self.istate[0]
        self.setWindowTitle('Calculating DF/F timecourse ...')
        
        # get average DFF signal 
        z_win = list(self.dff_zwin) if self.dff_z==3 else False
        ddict = pwaves.dff_timecourse(self.ppath, rec, istate, dff_win=self.win,
                                            pzscore=min([self.dff_z,2]), 
                                            z_win=z_win, p_iso=self.p_iso,
                                            pcluster=self.pcluster, clus_event=self.clus_event,
                                            vm=self.dff_vm, psmooth=self.dff_psm, 
                                            dn=self.dff_dn, sf=self.sf, mouse_avg=self.mouse_avg, 
                                            ma_thr=self.ma_thr, ma_state=self.ma_state, 
                                            flatten_is=self.flatten_is, tstart=self.tstart, 
                                            tend=self.tend, print_stats=False, pplot=False)
        self.setWindowTitle('Done!')
        self.fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1)
        grid = GridSpec(2, 1, height_ratios=[2,1], figure=self.fig)
        # plot single-trial heatmap
        ax1 = self.fig.add_subplot(grid[0])
        hmap = pwaves.mx2d(ddict, mouse_avg='trial')[0]
        if self.dff_psm:
            hmap = AS.convolve_data(hmap, self.dff_psm)
        if self.dff_dn:
            hmap = AS.downsample_mx(hmap, self.dff_dn, axis='x')
        t = np.linspace(-np.abs(self.win[0]), np.abs(self.win[1]), hmap.shape[1])
        ntrial = np.arange(1, hmap.shape[0]+1)
        im = ax1.pcolorfast(t, ntrial, hmap, cmap='bwr')
        if len(self.dff_vm) == 2:
            im.set_clim(self.dff_vm)
        self.fig.colorbar(im, ax=ax1, pad=0.0)
        ax1.set_ylabel('Trial no.')
        
        # plot averaged DFF timecourse
        ax2 = self.fig.add_subplot(grid[1])
        tc_mx = pwaves.mx2d(ddict, mouse_avg=self.mouse_avg)[0]
        if self.sf:
            tc_mx = AS.convolve_data(tc_mx, self.sf, axis='x')
        if self.dff_dn:
            tc_mx = AS.downsample_mx(tc_mx, self.dff_dn, axis='x')
        data = np.nanmean(tc_mx, axis=0)
        yerr = np.nanstd(tc_mx, axis=0)
        if self.ci=='sem':
            yerr /= np.sqrt(tc_mx.shape[0])
        t2 = np.linspace(-np.abs(self.win[0]), np.abs(self.win[1]), len(data))
        ax2.plot(t2, data, color='black')
        ax2.fill_between(t2, data-yerr, data+yerr, color='black', alpha=0.3)
        ylab = '$\Delta$ F/F (z-scored)' if self.dff_z > 0 else '$\Delta$ F/F (%)'
        ax2.set_ylabel(ylab)
        ax2.set_xlabel('Time (s)')
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_single_dff_pwave(self):
        """
        Plot DF/F signal surrounding the P-wave currently selected in the main 
        annotation window
        """
        self.fig.clear()
        # get index of current P-wave
        pi = int(self.mainWin.curIdx)
        iwin1, iwin2 = pwaves.get_iwins(self.win, self.mainWin.sr)
        
        # load DFF, collect data window surrounding P-wave
        dff = so.loadmat(os.path.join(self.ppath, self.name, 'DFF.mat'), squeeze_me=True)['dff']*100
        if self.dff_z == 1:  # z-score DFF by recording
            dff = (dff-dff.mean()) / dff.std()
        data = dff[pi-iwin1 : pi+iwin2]
        # z-score DFF by local time window
        if self.dff_z == 2:
            data = (data - data.mean()) / data.std()
        elif self.dff_z == 3:
            zwin1, zwin2 = pwaves.get_iwins(self.dff_zwin, self.mainWin.sr)
            zdata = dff[pi-zwin1 : pi+zwin2]
            data = (data - zdata.mean()) / zdata.std()
            
        # downsample/smooth data
        if self.sf:
            data = AS.convolve_data(data, self.sf, axis='x')
        if self.dff_dn:
            data = AS.downsample_vec(data, self.dff_dn)
        
        # plot DFF surrounding single P-wave
        self.fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.35)
        x = np.linspace(-np.abs(self.win[0]), np.abs(self.win[1]), len(data))
        ax = self.fig.add_subplot(111)
        ax.plot(x, data, color='black', linewidth=5)
        ylab = '$\Delta$ F/F (z-scored)' if self.dff_z > 0 else '$\Delta$ F/F (%)'
        ax.set_ylabel(ylab)
        ax.set_xlabel('Time (s)')
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_dff_activity(self):
        self.fig.clear()
        # get average DFF signal in each state
        self.setWindowTitle('Calculating DF/F signal ... ')
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        
        df = AS.dff_activity(self.ppath, rec, self.istate, tstart=self.tstart, 
                             tend=self.tend, pzscore=self.pzscore, ma_thr=self.ma_thr, 
                             ma_state=self.ma_state, flatten_is=self.flatten_is,
                             mouse_avg=self.mouse_avg, pplot=False, print_stats=False)
        self.setWindowTitle('Done!')
    
        # plot average DFF signal in each state
        self.fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
        ax = self.fig.add_subplot(111)
        pal = {'REM':'cyan', 'Wake':'darkviolet', 'NREM':'darkgray', 'IS':'darkblue', 
               'IS-R':'navy', 'IS-W':'red', 'MA':'magenta'}
        yerr = ('ci',int(self.ci)) if self.ci.isnumeric() else 'se' if self.ci=='sem' else self.ci
        sns.barplot(data=df, x='state', y='dff', errorbar=yerr, palette=pal, ax=ax)
        if self.mouse_avg in ['mouse','recording']:
            lines = sns.lineplot(data=df, x='state', y='dff', hue=self.mouse_avg, 
                                 errorbar=None, markersize=0, legend=False, ax=ax)
            _ = [l.set_color('black') for l in lines.get_lines()]
        sns.despine()
        ax.set_xlabel('')
        ylab = '$\Delta$ F/F (z-scored)' if self.pzscore else '$\Delta$ F/F (%)'
        ax.set_ylabel(ylab)
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_sleep_timecourse(self):
        """
        Plot time-binned sleep data in each brain state
        """
        self.fig.clear()
        # get list(s) of recordings
        doses = ['0']
        rec_list = []
        if len(self.recordings)==0:
            rec_list.append([self.name])
        else:
            rec_list.append(self.recordings[0])
            for dose in list(self.recordings[1].keys()):
                rec_list.append(self.recordings[1][dose])
                doses.append(dose)
                
        # set y axis label
        labeldict = {'perc'       : ['% time','Percent time spent'],
                     'freq'       : ['Freq (1/h)', 'Frequency'],
                     'dur'        : ['Dur (s)', 'Duration'],
                     'is prob'    : ['% IS-R transitions', 'Percent IS--REM transitions'],
                     'pwave freq' : ['P-waves/s', 'P-wave frequency']}
        ylabel = '' if self.stat not in labeldict.keys() else labeldict[self.stat][0]
        title = '' if self.stat not in labeldict.keys() else labeldict[self.stat][1]
        # set x tick labels for time bins
        start = (self.tstart + self.tbin) / 3600.
        binHrs = [start + (i*self.tbin)/3600. for i in range(self.num_bins)]
        if self.num_bins == 1:  # plot single bar for each state in same graph
            xticklabels = [self.stateMap[s] for s in self.istate]
        if self.num_bins > 1:   # plot a separate timecourse graph for each state
            xticklabels = list(binHrs)
        x = np.arange(0, len(xticklabels))
        self.setWindowTitle('Calculating sleep timecourse ...')

        df = pd.DataFrame(columns=['mouse','dose','state','tbin','data']) 
        mcs = {}
        for rec, dose in zip(rec_list, doses):
            # get sleep data, store in dataframe
            mice, tc = pwaves.sleep_timecourse(self.ppath, rec, self.istate, self.tbin, 
                                               self.num_bins, stats=self.stat, 
                                               ma_thr=self.ma_thr, ma_state=self.ma_state, 
                                               flatten_is=self.flatten_is, tstart=self.tstart, 
                                               tend=self.tend, p_iso=self.p_iso, pcluster=self.pcluster,
                                               clus_event=self.clus_event, pzscore=False, pplot=False)
            for s in self.istate:
                for i,m in enumerate(mice):
                    ddict = {'mouse':m, 'dose':dose, 'state':s, 'tbin':binHrs, 'data':tc[s][i,:]}
                    df = pd.concat([df, pd.DataFrame(ddict)], axis=0, ignore_index=True)
                    mcs.update(AS.colorcode_mice(m))
        self.setWindowTitle('Done!')
        
        # calculate bar width and position
        width = 0.75 / len(doses)
        xadjust = (width/2) * (len(doses)-1)
        dosemap = {'0':'gray', '0.25':'blue', '5':'red'}
        mice = list(mcs.keys())
        ms_leg = [[],[]]
        
        # plot sleep data for each state
        nplts = len(self.istate) if self.num_bins>1 else 1
        grid = GridSpec(nplts, 1, figure=self.fig)
        
        for iplt in range(nplts):
            ax = self.fig.add_subplot(grid[iplt])
            if self.num_bins > 1:
                s = self.istate[iplt]  # only plot data for state $s
                x_iter = list(binHrs)
            else:
                x_iter = list(self.istate)
            for xi,xl in enumerate(x_iter):
                # locations of each bar surrounding main x tick
                xgrp = [x[xi]+xadj for xadj in np.linspace(-xadjust,xadjust,len(doses))]
                if self.num_bins > 1:     # get time bin $xl (current x-tick) during state $s (current graph)
                    irows = np.intersect1d(np.where(df.state==s)[0], np.where(df.tbin==xl))
                elif self.num_bins == 1:  # get state $xl (current x-tick only)
                    s = self.istate[xi]
                    irows = np.where(df.state==s)[0]
                # plot mean/yerr for each dose surrounding main x tick
                idoses = [np.intersect1d(irows, np.where(df.dose==d)[0]) for d in doses]
                data = [np.nanmean(df.data[idose]) for idose in idoses]
                yerr = [np.nanstd(df.data[idose]) for idose in idoses]
                if self.ci=='sem':
                    yerr = [y / np.sqrt(len(idose)) for y,idose in zip(yerr,idoses)]
                c = [self.mainWin.lut_brainstate[s]/255] if len(doses)==1 else [dosemap[d] for d in doses]
                ax.bar(xgrp, data, yerr=yerr, width=width, color=c, edgecolor='black')
                # plot data points for each mouse
                for m in mice:
                    midx = [np.intersect1d(idose, np.where(df.mouse==m)[0]) for idose in idoses]
                    mdata = [float(df.data[mi]) for mi in midx]
                    mpt = ax.plot(xgrp, mdata, marker='o', linewidth=3, color=mcs[m], 
                                  markeredgecolor='black')[0]
                    if m not in ms_leg[1]:
                        ms_leg[0].append(mpt); ms_leg[1].append(m)
            if iplt==0:
                legend1 = ax.legend(ms_leg[0], ms_leg[1], loc='center right', 
                                    bbox_to_anchor=(1,0.5))
                _ = ax.add_artist(legend1)
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
            ax.set_ylabel(ylabel)
            if self.num_bins>1:
                ax.set_title(self.stateMap[s])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if iplt==nplts-1 and self.num_bins>1:
                ax.set_xlabel('Time (hr)')
        self.fig.suptitle(title)
        self.canvas.draw()
        self.cleanup()
        
        
    def plot_sleep_spectrum(self):
        """
        Plot averaged power spectrum during each brain state
        """
        self.fig.clear()
        
        # get list(s) of recordings
        doses = ['0']; C = ['gray']
        rec_list = []
        pmode=bool(self.pmode)
        if len(self.recordings)==0:
            rec_list.append([self.name])
        else:
            rec_list.append(self.recordings[0])
            # if not all recordings have laser file, set laser mode off
            rcs = self.recordings[0]
            lpaths = [os.path.join(self.ppath, r, f'laser_{r}.mat') for r in rcs]
            if not all([os.path.exists(lpath) for lpath in lpaths]):
                pmode=False
            for dose in list(self.recordings[1].keys()):
                rcs = self.recordings[1][dose]
                rec_list.append(rcs)
                lpaths = [os.path.join(self.ppath, r, f'laser_{r}.mat') for r in rcs]
                if not all([os.path.exists(lpath) for lpath in lpaths]):
                    pmode=False
                doses.append(dose)
                C.append('blue' if dose=='0.25' else 'red' if dose=='5' else 'green')
        
        self.setWindowTitle('Calculating power spectrums ...')
        dfs = []
        for rec, dose in zip(rec_list, doses):
            for i,s in enumerate(self.istate):
                # get average power spectrum across recordings for each state 
                df = sleepy.sleep_spectrum_simple(self.ppath, rec, istate=s, fmax=self.fmax,
                                                  tstart=self.tstart, tend=self.tend, 
                                                  ma_thr=self.ma_thr, ma_state=self.ma_state, 
                                                  flatten_is=self.flatten_is, pmode=pmode, 
                                                  pnorm=bool(self.pnorm), harmcs=self.harmcs, 
                                                  harmcs_mode=self.harmcs_mode, 
                                                  exclusive_mode=self.exclusive_mode,
                                                  iplt_level=self.iplt_level, ci=self.ci, 
                                                  round_freq=True, noise_state=0, pplot=False)[2]
                # collect freq powers in dataframe
                df['state'] = s
                df['dose'] = dose
                df['Freq'] = np.round(df['Freq'],1)    
                dfs.append(df)
        if self.plotAllStates_btn.isChecked():
            dfs = [pd.concat(dfs, axis=0)]
        self.setWindowTitle('Done!')
        
        grid = GridSpec(len(self.istate), 1, figure=self.fig)
        for i,df in enumerate(dfs):
            # plot power spectrum(s)
            ax = self.fig.add_subplot(grid[i])
            yerr = ('ci',int(self.ci)) if self.ci.isnumeric() else 'se' if self.ci=='sem' else self.ci
                
            sns.lineplot(data=df, x='Freq', y='Pow', hue='Lsr', errorbar=yerr, ax=ax, 
                         palette={'yes':'blue', 'no':'gray'}, legend='auto' if i==0 else False)
            if self.plotAllStates_btn.isChecked():
                ax.set_title(' + '.join([self.stateMap[s] for s in self.istate]))
            else:
                ax.set_title(self.stateMap[self.istate[i]])
            ylab = 'Norm. power' if self.pnorm else 'Spectral density ($\mathrm{\mu V^2/Hz}$)'
            ax.set_ylabel(ylab)
            ax.set_xlabel('Freq. (Hz)' if len(self.istate)-1 else '')
        self.canvas.draw()
        self.cleanup()
    
    
    def plot_pwave_spectrum(self):
        """
        Plot averaged power spectrums of brain states with and without P-waves
        """
        self.fig.clear()
        rec = [self.name] if len(self.recordings)==0 else self.recordings[0]
        
        self.setWindowTitle('Calculating power spectrums ...')
        df = sleepy.sleep_spectrum_pwaves(self.ppath, rec, win_inc=self.win[1], 
                                          win_exc=self.excl_win[1], istate=self.istate[0], 
                                          pnorm=self.pnorm, nsr_seg=self.nsr_seg,
                                          perc_overlap=self.perc_overlap, fmax=self.fmax,
                                          recalc_highres=self.recalc_highres, 
                                          tstart=self.tstart, tend=self.tend, ci=self.ci,
                                          ma_thr=self.ma_thr, ma_state=self.ma_state,
                                          flatten_is=self.flatten_is, p_iso=self.p_iso,
                                          pcluster=self.pcluster, clus_event=self.clus_event, 
                                          exclude_noise=self.exclude_noise, pplot=False)
        self.setWindowTitle('Done!')
        
        # plot P-wave and non-P-wave power spectral densities
        self.fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1)
        grid = GridSpec(1, 2, width_ratios=[3,1], figure=self.fig)
        ax1 = self.fig.add_subplot(grid[0])
        yerr = ('ci',int(self.ci)) if self.ci.isnumeric() else 'se' if self.ci=='sem' else self.ci
        # create color palette for type(s) of P-waves analyzed
        pal = {'no':'gray'}; order=['no', 'yes']
        if self.p_iso and self.pcluster:
            pal.update({'single':'dodgerblue', 'cluster':'darkblue'})
            order = ['no', 'single', 'cluster']
        elif not self.p_iso and not self.pcluster:
            pal.update({'yes':'blue'})
        else:
            pal.update({'yes' : 'dodgerblue' if self.p_iso else 'darkblue'})
        # plot PSDs
        _ = sns.lineplot(data=df, x='freq', y='pow', hue='pwave', errorbar=yerr, 
                         linewidth=3, palette=pal, legend='auto', ax=ax1)
        sns.despine()
        ax1.set_xlabel('Freq (Hz)')
        ylabel = 'Norm. power' if self.pnorm else 'Spectral density ($\mathrm{\mu V^2/Hz}$)'
        ax1.set_ylabel(ylabel)
        
        # plot bar graph of high theta power
        ax2 = self.fig.add_subplot(grid[1])
        high_theta = [8,15]
        itheta = np.where((df.Freq >= high_theta[0]) & (df.Freq <= high_theta[1]))[0]
        df_theta = df.loc[itheta, ['idf', 'pow', 'pwave']]
        df_theta = df_theta.groupby(['Idf','P-wave']).sum().reset_index()
        _ = sns.barplot(x='P-wave', y='Pow', data=df_theta, errorbar=yerr, 
                        order=order, palette=pal, ax=ax2)
        lines = sns.lineplot(x='P-wave', y='Pow', hue='Idf', data=df_theta, 
                             errorbar=None, markersize=0, legend=False, ax=ax2)
        _ = [l.set_color('black') for l in lines.get_lines()]
        sns.despine()
        ax2.set_xlabel('P-wave')
        ax2.set_ylabel('')
        ax2.set_title(f'{high_theta[0]}-{high_theta[1]} Hz')
        
        self.canvas.draw()
        self.cleanup()
        
    
    def plot_lsr_brainstate(self):
        print('totally plotted laser brainstate')
    
    
    def plot_emg_twitches(self):
        self.fig.clear()
        
        # get list(s) of recordings
        doses = ['0']
        rec_list = []
        if len(self.recordings)==0:
            rec_list.append([self.name])
        else:
            rec_list.append(self.recordings[0])
            for dose in list(self.recordings[1].keys()):
                rec_list.append(self.recordings[1][dose])
                doses.append(dose)
        dicts = []
        for rec, dose in zip(rec_list, doses):
            # get EMG twitch frequency for list(s) of recordings
            tw_dict = pwaves.emg_twitch_freq(self.ppath, rec, thres=self.twitch_thres,
                                             recalc_twitches=self.recalc_twitches, 
                                             thres_type=self.twitch_thres_type, 
                                             thres_mode=self.twitch_thres_mode, 
                                             thres_first=self.twitch_thres_first, 
                                             min_twitchdur=self.min_twitchdur, 
                                             min_twitchsep=self.min_twitchsep, 
                                             min_REMdur=self.min_twitchREM, 
                                             rem_cutoff=self.twitch_REMcutoff, 
                                             recalc_amp=self.recalc_amp, 
                                             emg_source=self.emg_source, 
                                             nsr_seg=self.nsr_seg_msp, 
                                             perc_overlap=self.perc_overlap_msp, 
                                             recalc_highres=self.recalc_amp, 
                                             w0=self.w0_emg, w1=self.w1_emg, 
                                             r_mu=self.r_mu, dn=self.emg_dn, 
                                             smooth=self.emg_sm, pemg2=False,
                                             exclude_noise=self.exclude_noise, 
                                             tstart=self.tstart, tend=self.tend, 
                                             avg_mode=self.twitch_avg)
            dicts.append(tw_dict)
        # get subject x twitch frequency matrix
        tw_freq, labels = pwaves.mx1d(dicts[0], mouse_avg=self.mouse_avg)
        data = [np.nanmean(tw_freq, axis=0), None]
        yerr = [np.nanstd(tw_freq, axis=0), None]
        if self.ci=='sem':
            yerr = [yerr[0] / np.sqrt(len(tw_freq)), None]
        x = [[1]]; width=0.4; colors=['cyan']
        avgOver = 'each' if self.mouse_avg=='trial' else str(self.twitch_avg)
        pltTitle = 'each REM bout' if avgOver=='each' else 'all REM sleep'
        if len(dicts) > 1:
            tw_freq2, labels2 = pwaves.mx1d(dicts[1], mouse_avg=self.mouse_avg)
            data[1] = np.nanmean(tw_freq2, axis=0)
            yerr[1] = np.nanstd(tw_freq2, axis=0)
            if self.ci=='sem':
                yerr[1] = yerr[1] / np.sqrt(len(tw_freq2))
            x = [[0.8],[1.2]]; width=0.2; colors=['gray','blue']
        
        # plot average EMG twitch frequency for dataset(s)
        self.fig.set_constrained_layout_pads(w_pad=0.45, h_pad=2)
        ax = self.fig.add_subplot(111)
        c = None
        ax.bar(x[0], data[0], yerr=yerr[0], width=width, color=colors[0], edgecolor='black')
        if dose:
            ax.bar(x[1], data[1], yerr=yerr[1], width=width, color=colors[1], edgecolor='black')
        leg1=[]; leg2=[]
        if len(rec)==1 and len(tw_dict[rec[0]])>1 and not dose:
            # plot twitch freq for each REM period in 1 recording
            for i,r in enumerate(tw_dict[rec[0]]):
                c = list(np.random.choice(range(255), size=3)/255)
                leg1.append([ax.plot(x[0], r, color=c, marker='o', ms=12, 
                                     markeredgecolor='black')[0], f'REM period {i+1}'])
        elif self.mouse_avg != 'trial':
            # plot avg twitch freq for each mouse/recording
            for i,l in enumerate(labels):
                c = list(np.random.choice(range(255), size=3)/255)
                leg1.append([ax.plot(x[0], tw_freq[i], color=c, marker='o', ms=12, 
                                     markeredgecolor='black')[0], l])
            if dose:
                for i,l in enumerate(labels2):
                    c = list(np.random.choice(range(255), size=3)/255)
                    leg2.append([ax.plot(x[1], tw_freq2[i], color=c, marker='o', ms=12, 
                                         markeredgecolor='black')[0], l])
        ax.axes.xaxis.set_visible(False)
        ax.set_ylabel('Twitches/min')
        ax.set_title(f'Twitch freq averaged over {pltTitle}\nmouse_avg={self.mouse_avg}')
        if len(leg1)>0:
            h,l = zip(*leg1)
            legend1 = ax.legend(h, l, loc='upper left')
            _ = ax.add_artist(legend1)
        if len(leg2)>0:
            h2,l2 = zip(*leg2)
            legend2 = ax.legend(h2, l2, loc='upper right')
            _ = ax.add_artist(legend2)
        self.canvas.draw()
        self.cleanup()


    def plot_recording(self):
        """
        Plot data from a user-selected recording
        """
        # user selects recording directory
        ddir = QtWidgets.QFileDialog().getExistingDirectory(self, 
                                                            "Choose recording folder", 
                                                            self.ppath)
        
        if ddir:
            # get directory location and recording name
            rec = ddir.split('/')[-1]
            path = ddir[0:len(ddir)-len(rec)]
            # plot figure
            self.ppath = path
            self.recordings = [[rec], {}]
            self.plotFig_btn.click()
    
    
    def plot_experiment(self):
        """
        Plot data from a user-selected experiment .txt file
        """
        sfile = QtWidgets.QFileDialog().getOpenFileName(self, "Choose .txt file",
                                                        self.ppath, 
                                                        "Text files (*.txt)")[0]
        if sfile:
            with open(sfile, newline=None) as f:
                lines = f.readlines()
                f.close()
            
            # get rid of lines starting with # or $
            lines = [l for l in lines if not re.search('^\s+$', l) and not re.search('^\s*#', l)]
            
            # get starting character for each non-comment line
            slines = [re.split('\s+',l)[0] for l in lines]
            if set(slines) == set(('C','E')):
                # file includes control AND experimental recordings
                dose = True
                ridx = 1
            elif set(slines) == {'C'} or set(slines) == {'E'}:
                # file includes control OR experimental recordings
                dose = False
                ridx = 1
            elif all([s.split('_')[1][0] == 'n' for s in slines]):
                # file includes unlabeled recordings
                dose = False
                ridx = 0
            else:
                # file format is unknown
                return
            
            # get ending character for each non-comment line
            elines = [re.split('\s+',l)[-2] for l in lines]
            # check if experiment file contains P-wave information
            if np.setdiff1d(elines, ['S','C','1','2','1-2','2-1','X']).size == 0:
                pchannel_info = True
            else:
                pchannel_info = False
            # check if plot type involves analyzing P-waves
            if pchannel_info and self.plotTypeWidgets[self.plotType]['req_pwaves']:
                requirePwaves = True
            else:
                requirePwaves = False
            
            ctr_recordings = []
            exp_recordings = {}
            for l in lines:
                a = re.split('\s+', l)
                # eliminate recordings with no P-waves, if required
                if requirePwaves and a[-2] == 'X':
                    continue
                else:
                    # collect all recordings in one list
                    if not dose:
                        ctr_recordings.append(a[ridx])
                    elif dose:
                        # collect all control recordings
                        if re.search('C', a[0]):
                            ctr_recordings.append(a[1])
                        # assign experimental recordings to dose key in dictionary
                        elif re.search('E', a[0]):
                            if a[2] in exp_recordings.keys():
                                exp_recordings[a[2]].append(a[1])
                            else:
                                exp_recordings[a[2]] = [a[1]]

            txt_path = '/'.join(sfile.split('/')[0:-1])
            data_path = None
            while data_path is None:
                # data and txt file may be in different locations
                res = pqi.warning_msgbox('Select different location for data folders?')
                if res == -1:
                    return
                elif res == 0:
                    # use folder containing txt file as the default data path
                    dpath = txt_path
                elif res == 1:
                    # select path for recording folders
                    ddir = QtWidgets.QFileDialog().getExistingDirectory(self, 
                                                                        "Choose data directory", 
                                                                        txt_path)
                    if ddir:
                        dpath = ddir
                    else:
                        continue
                # raise error message if recording(s) not found in selected data path
                recs = ctr_recordings
                if len(exp_recordings) > 0:
                    recs += list(np.concatenate(list(exp_recordings.values())))
                missing = np.setdiff1d(recs, os.listdir(dpath))
                if missing.size > 0:
                    msg = 'Data folder(s) missing from selected directory'
                    _ = QtWidgets.QMessageBox.critical(self, '', msg)
                else:
                    data_path = dpath
            # update recordings in window, plot figure
            self.ppath = data_path
            self.recordings = [ctr_recordings, exp_recordings]
            self.txt_file = sfile.split('/')[-1]
            self.plotFig_btn.click()
            
        
    def cleanup(self):
        """
        Update figure settings, reset recordings and window title after each plot
        """
        self.plotSettings = self.dict_from_vars()
        self.ppath = str(self.mainWin.ppath)
        # change window title to the name of the recording/text file on graph
        if len(self.recordings) == 0:
            self.setWindowTitle(self.name)
        else:
            if len(self.recordings[0])==1 and len(self.recordings[1])==0:
                self.setWindowTitle(self.recordings[0][0])
            else:
                self.setWindowTitle(self.txt_file)
        self.recordings = []
    
    
    def closeEvent(self, event):
        """
        Save most recent plot settings, delete figure window upon closing
        """
        print("Closing figure ...")
        plt.close()
        self.mainWin.plotFigure_settings = self.plotSettings
        self.mainWin.EMG_amp_data = [None,None,None]
        self.deleteLater()
    
    
    def debug(self):
        pdb.set_trace()
        