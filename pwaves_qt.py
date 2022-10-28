#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Application for viewing and annotating data from experimental recordings

@author: fearthekraken
"""
import sys
import os
import re
import h5py
import time
import pyautogui
import scipy.io as so
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pdb
# custom modules
import sleepy
import pwaves
import AS
from gui_items import *
from gui_windows import FigureWindow
from detect_emg_twitches import EMGTwitchWindow


def get_cycles(ppath, name):
    """
    Extract the time points where dark/light periods start and end
    """
    # load recording time and duration from info file
    a = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'time')[0]
    hour, mi, sec = [int(i) for i in re.split(':', a)]
    a = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')[0]
    a,b,c = [int(i[0:-1]) for i in re.split(':', a)]
    total_dur = a*3600 + b*60 + c
    
    # number of light/dark switches
    nswitch = int(np.floor(total_dur / (12*3600)))
    switch_points = [0]
    cycle = {'light': [], 'dark':[]}
    
    if hour >= 7 and hour < 19:
        # recording starts during light cycle
        a = 19*3600 - (hour*3600+mi*60+sec)
        for j in range(nswitch):
            switch_points.append(a+j*12*3600)
        for j in range(1, nswitch, 2):
            cycle['dark'].append(switch_points[j:j+2])
        for j in range(0, nswitch, 2):
            cycle['light'].append(switch_points[j:j+2])
    else:
        # recording starts during dark cycle
        a = 0
        if hour < 24:
            a = 24 - (hour*3600+mi*60+sec) + 7*3600
        else:
            a = 7*3600 - (hour*3600+mi*60+sec)
        for j in range(nswitch):
            switch_points.append(a+j*12*3600)
        for j in range(0, nswitch, 2):
            cycle['dark'].append(switch_points[j:j+2])
        for j in range(1, nswitch, 2):
            cycle['light'].append(switch_points[j:j+2])
        
    return cycle


def load_stateidx(ppath, name):
    """
    Load the sleep state file of recording $ppath/$name
    @Returns
    M - sequence of sleep states (1=REM, 2=Wake, 3=NREM, etc)
    K - annotation history (0=bin not annotated, 1=bin visited by experimenter)
    """   
    # load sleep state file from recording folder
    file = os.path.join(ppath, name, 'remidx_' + name + '.txt')
    f = open(file, newline=None)    
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines:
        if re.match('\d', l):
            n = n+1
    M = np.zeros(n)  # brain state annotation
    K = np.zeros(n)  # annotation history
    i = 0
    for l in lines:
        if re.search('^\s+$', l):
            continue
        if re.search('\s*#', l):
            continue
        if re.match('\d+\s+-?\d+', l):
            a = re.split('\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i += 1
    return M,K


def rewrite_remidx(M, K, ppath, name, mode=0):
    """
    Update the sleep state file of recording $ppath/$name
    @Params
    M - sequence of sleep states (1=REM, 2=Wake, 3=NREM, etc)
    K - annotation history (0=bin not annotated, 1=bin visited by experimenter)
    mode - file naming convention
    @Returns
    None
    """
    # load current sleep state file
    if mode == 0 :
        outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
    else:
        outfile = os.path.join(ppath, name, 'remidx_' + name + '_corr.txt')
    # update annotation vectors
    f = open(outfile, 'w')
    s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
    f.writelines(s)
    f.close()


def get_snr(ppath, name) :
    """
    Load sampling rate of recording $ppath/$name from info file
    @Returns
    SR - sampling rate (Hz)
    """
    # open info.txt file
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)    
    lines = fid.readlines()
    fid.close()
    values = []
    # find sampling rate
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))     
    SR = float(values[0])
    return SR


def load_laser(ppath, name):
    """
    Load laser pulse train of recording $ppath/$name
    @Returns
    laser - stimulation vector (0=lsr down, 1=lsr up)
    """ 
    file = os.path.join(ppath, name, 'laser_'+name+'.mat')
    try:
        laser = np.squeeze(np.array(h5py.File(file,'r').get('laser')))
    except:
        laser = np.squeeze(so.loadmat(file)['laser'])
    return laser


def laser_start_end(laser, SR=1525.88, intval=5) :
    """
    Get start and end index of each laser stimulation period
    @Params
    laser - raw laser pulse train (0=lsr down, 1=lsr up)
    SR - sampling rate (Hz)
    intval - min. time (s) between separate laser stimulation periods
    @Returns
    (istart, iend) - tuple of (laser start indices, laser end indices)
    """
    # return empty lists if no instances of laser train
    idx = np.nonzero(laser > 0.1)[0]
    if len(idx) == 0 :
        return ([], [])
    # get starting and ending indices for each laser stimulation period
    idx2 = np.nonzero(np.diff(idx)*(1./SR) > intval)[0]
    istart = np.hstack([idx[0], idx[idx2+1]])
    iend   = np.hstack([idx[idx2], idx[-1]])    
    
    return (istart, iend)


def load_trigger(ppath, name):
    """
    Load sleep-state detection signal for closed-loop recording $ppath/$name
    @Returns
    triggered - detection vector (0=state not detected, 1=state detected online)
    """
    file = os.path.join(ppath, name, 'rem_trig_'+name+'.mat')
    try:
        triggered = np.squeeze(np.array(h5py.File(file,'r').get('rem_trig')))
    except:
        triggered = np.squeeze(so.loadmat(file)['rem_trig'])
    return triggered


def downsample_sd(x, length, nbin, noise_idx=[], replace_nan=np.nan):
    """
    Calculate standard deviation for $length sets of $nbin consecutive data points
    @Params
    x - data signal
    length - number of elements in downsampled vector
    nbin - number of data points per downsampled bin
    noise_idx - list of "noisy" indices in $x; conserved in output vector
    replace_nan - value to use for noisy bins
    @Returns
    sd_dn - downsampled vector of standard deviations
    """
    # replace noisy data indices with NaNs
    if len(noise_idx) > 0:
        x[noise_idx] = np.nan
    # calculate st. dev of each sample bin
    i = np.array(range(0,length+1)*nbin, dtype='int')
    sd_dn = np.array([np.std(x[a:b]) for a,b in zip(i[0:-1],i[1:])])
    # replace NaNs with alternative value
    inan = np.nonzero(np.isnan(sd_dn))
    sd_dn[inan] = replace_nan
    return sd_dn


class MainWindow(QtGui.QMainWindow):
    def __init__(self, ppath, name):
        """
        Instantiate main window, set initial variable values, load recording data
        @Params
        ppath - base folder
        name - recording folder
        """
        QtGui.QMainWindow.__init__(self)
        self.WIDTH, self.HEIGHT = pyautogui.size()
        self.setGeometry(QtCore.QRect(100, 20, self.WIDTH, self.HEIGHT))
        self.ppath = ppath
        self.name  = name
        
        ### Initialize plotting params ###
        self.index = 10            # index (FFT) of central timepoint for current EEG plot
        self.curPlotIndex = -1     # index (FFT) of central timepoint for previous EEG plot
        self.pcollect_index = False     # ignore or collect visited brain state indices?
        self.index_list = [self.index]  # collected brain state indices for bulk annotation
        self.curIdx = None        # currently selected Intan index
        self.plotBtns_index = 0   # index of currently visible set of plot buttons
        self.noiseBtns_index = 0  # index of currently visible set of noise buttons
        self.noiseStartIdx = None  # starting index of user-selected Intan noise
        self.noiseEndIdx = None    # ending index of user-selected Intan noise
        self.show_arrowStart = False  # show indicator arrow for start of noise sequence?
        self.show_arrowEnd = False    # show indicator arrow for end of noise sequence?
        self.tscale = 1   # time scale of plot (1=seconds, 2=minutes, 3=hours) 
        self.tunit = 's'  # units of time scale ('s', 'min', 'h')
        self.peeg2 = 0  # show EEG1 or EEG2?
        self.pemg2 = 0  # show EMG1 or EMG2?
        self.pplot_laser = False   # show laser?
        self.pplot_pidx = False    # show dots for detected P-waves?
        self.pplot_pthres = False  # show threshold for P-wave detection?
        self.pplot_ithres = False  # show indices used in threshold calculation?
        self.pplot_noise = False   # show noise in LFP/EMG/EEG signal?
        self.pplot_artifacts = [False, False, False, False]  # show invalid waveforms?
        self.pplot_elim_user = False  # show user-deleted waveforms?
        self.pplot_emgampl = True     # show EMG amplitude?
        self.pplot_LFPstd = False     # show LFP standard deviation?
        self.pplot_pfreq = False      # show P-wave frequency?
        self.pplot_dff = False        # show DF/F signal?
        self.pplot_bandpwr = False    # show power of EEG frequency bands?
        self.reset_noise = False      # reset noise or add newly detected indices to current noise?
        self.hasEMG = False           # recording contains EMG signal?
        self.hasDFF = False           # recording contains photometry signal?
        self.recordPwaves = False     # recording contains detected P-waves?
        self.lsrTrigPwaves = False    # recording contains laser-triggered P-waves?
        self.optoMode = ''            # recording contains optogenetic stimulation? OL or CL?
        
        # save plot settings between graphs
        self.plotFigure_settings = {}
        self.EMG_amp_data = [None, None, None]
        
        ### Initialize default parameter values ###
        # brainstate params
        self.ma_thr = 20     # max duration (s) for microarousals
        self.ma_state = 3    # brain state to assign microarousals
        self.flatten_is = 4  # treat IS-R & IS-W states as distinct (False) or identical (4)
        
        # LFP processing params
        self.channel = 'S'  # LFP referencing method
        self.w0 = 0.002  # lowest freq component
        self.w1 = 0.1    # highest freq component
        
        # LFP threshold params
        self.thres = 4.5         # threshold value for P-wave detection
        self.thres_type = 'std'  # thresholding method (raw value, st. deviations, or %iles)
        self.thres_states = {1:[False,0.],  # bool: use brain state to calculate threshold?
                             2:[False,0.],  # float: min. duration (s) of state episodes
                             3:[True, 60.], 
                             4:[False,0.]}
        self.thres_first = 10  # if > 0, use first X s of state episodes for thresholding
        
        # P-wave validation params
        self.amp_thres = 500  # amplitude threshold (uV) for waveform elimination
        self.amp_thres_type = 'raw'  # amplitude thresholding method
        self.hw_thres = 80    # half-width threshold (ms) for waveform elimination
        self.hw_thres_type = 'raw'  # half-width thresholding method
        self.dup_win = 40     # check neighboring spikes (within X ms) for duplicates
        self.post_stim = 0.1  # max latency (s) of 'laser-triggered' P-wave from laser onset
        
        # noise detection params
        self.noise_thres_up = 200    # positive threshold value for noise detection
        self.noise_thres_dn = 500    # negative threshold value for noise detection
        self.noise_thres_type = 'raw'  # noise thresholding method
        self.noise_win = 2  # eliminate waveforms within X s of detected noise
        self.noise_sep = 5  # connect noise sequences within X s of each other
        
        self.defaults = self.dict_from_vars()
        
        # create GUI, load recording info
        self.gen_layout()
        self.connect_buttons()
        self.load_recording()
        self.plotSettings = self.dict_from_vars()
        
        # draw initial data plots
        self.plot_brainstate()
        self.plot_treck()
        self.plot_eeg(findPwaves=True, findArtifacts=True)
        self.plot_session()
        self.graph_eeg.setRange(yRange=(-500, 500),  padding=None)
    
    #####################          GUI LAYOUT          #####################
        
    def gen_layout(self):
        """
        Layout for main window
        """
        try:
            self.centralWidget = QtWidgets.QWidget()
            self.centralLayout = QtWidgets.QVBoxLayout(self.centralWidget)
            
            ### LIVE GRAPHS LAYOUT ###
            self.plotView = pg.GraphicsLayoutWidget(parent=self)
            self.lay_brainstate  = self.plotView.addLayout() 
            # laser / annotation history / current timepoint
            self.graph_treck = pg.PlotItem()
            self.lay_brainstate.addItem(self.graph_treck)
            self.lay_brainstate.nextRow()
            # color-coded brain state
            self.graph_brainstate = self.lay_brainstate.addPlot()
            self.image_brainstate = pg.ImageItem() 
            self.graph_brainstate.addItem(self.image_brainstate)
            self.lay_brainstate.nextRow()
            # EEG spectrogram
            self.graph_spectrum = self.lay_brainstate.addPlot()
            self.image_spectrum = pg.ImageItem()     
            self.graph_spectrum.addItem(self.image_spectrum)
            self.lay_brainstate.nextRow()
            # Recording session data (FFT time; 2.5 s/bin)
            self.graph_emgampl = self.lay_brainstate.addPlot()          
            self.plotView.nextRow()    
            # Live data visualization (Intan time; 1000+ Hz sampling rate)
            self.arrowStart_btn = ArrowButton(parent=self)
            self.arrowEnd_btn = ArrowButton(parent=self)
            self.graph_eeg = GraphEEG(parent=self)
            self.graph_eeg.vb.setMouseEnabled(x=True, y=True)
            self.plotView.addItem(self.graph_eeg) 
            self.plotView.nextRow()
            # updatable plot items for viewing signals
            self.curData = pg.PlotDataItem()    # current Intan signal (LFP/EMG/EEG)
            self.curData.setPen((255,255,255),width=1)
            self.curLaser = pg.PlotDataItem()   # current laser signal
            self.curLaser.setPen((0,0,255),width=2.5)
            self.curDFF = pg.PlotDataItem()     # current DF/F signal
            self.curDFF.setPen((255,255,0),width=2.5)
            self.curThres = pg.InfiniteLine()   # threshold value for P-wave detection
            self.curThres.setAngle(0)
            self.curThres.setPen((0,255,0),width=1)
            labelOpts = {'position': 0.08, 'color':(0,255,0), 'movable':True}
            self.curThres.label = pg.InfLineLabel(self.curThres, text='', **labelOpts)
            self.curIThres = pg.PlotDataItem()  # current timepoints used for thresholding
            self.curIThres.setPen((240,159,60),width=2.5)
            self.centralLayout.addWidget(self.plotView)
            
            
            ### SETTINGS LAYOUT ###
            self.settingsWidget = QtWidgets.QFrame()
            self.settingsWidget.setFrameShape(QtWidgets.QFrame.Box)
            self.settingsWidget.setFrameShadow(QtWidgets.QFrame.Raised)
            self.settingsWidget.setLineWidth(5)
            self.settingsWidget.setMidLineWidth(3)
            self.settingsLayout = QtWidgets.QHBoxLayout(self.settingsWidget)
            # set fonts
            headerFont = QtGui.QFont()
            headerFont.setPointSize(10)
            headerFont.setBold(True)
            headerFont.setUnderline(True)
            font = QtGui.QFont()
            font.setPointSize(9)
            # set contents margins
            cmargins = QtCore.QMargins(px_w(11, self.WIDTH),
                                       px_h( 5, self.HEIGHT),
                                       px_w(11, self.WIDTH),
                                       px_h(20, self.HEIGHT))
            # get set of pixel widths and heights, standardized by monitor dimensions
            titleHeight = px_h(30, self.HEIGHT)
            wspace1, wspace5, wspace10, wspace15, wspace20 = [px_w(w, self.WIDTH) for w in [1,5,10,15,20]]
            hspace1, hspace5, hspace10, hspace15, hspace20 = [px_h(h, self.HEIGHT) for h in [1,5,10,15,20]]
            
            ### LFP processing params ###
            self.lfpWidget = QtWidgets.QWidget()
            width1 = int(self.WIDTH * 0.0938)
            self.lfpWidget.setFixedWidth(width1)
            self.lfpLayout = QtWidgets.QVBoxLayout(self.lfpWidget)
            self.lfpLayout.setContentsMargins(cmargins)
            self.lfpLayout.setSpacing(hspace10)
            lay1 = QtWidgets.QVBoxLayout()
            lay1.setContentsMargins(0,0,0,0)
            lay1.setSpacing(hspace20)
            title1 = QtWidgets.QLabel('LFP Processing')
            title1.setAlignment(QtCore.Qt.AlignCenter)
            title1.setFixedHeight(titleHeight)
            title1.setFont(headerFont)
            r1 = QtWidgets.QVBoxLayout()  # referencing method label + dropdown
            r1.setSpacing(hspace1)
            lfpsig_label = QtWidgets.QLabel('Referencing Method')
            lfpsig_label.setAlignment(QtCore.Qt.AlignCenter)
            lfpsig_label.setFont(font)
            self.lfpSig_type = QtWidgets.QComboBox()
            self.lfpSig_type.setFont(font)
            self.lfpSig_type.addItems(['Auto subtraction', 'Auto choice', 'LFP1', 
                                       'LFP2', 'LFP1 - LFP2', 'LFP2 - LFP1'])
            self.lfpSig_type.setCurrentIndex(-1)
            r1.addWidget(lfpsig_label)
            r1.addWidget(self.lfpSig_type)
            r2 = QtWidgets.QVBoxLayout()  # filter label + value boxes
            r2.setSpacing(hspace1)
            lfpFilt_label = QtWidgets.QLabel('Raw LFP Filter (Hz)')
            lfpFilt_label.setAlignment(QtCore.Qt.AlignCenter)
            lfpFilt_label.setFont(font)
            r2bot = QtWidgets.QHBoxLayout()  # low freq - hi freq filter boxes
            r2bot.setSpacing(wspace5)
            r2bot.setContentsMargins(0,0,0,0)
            self.lfpFreqLo_val = QtWidgets.QDoubleSpinBox()
            self.lfpFreqLo_val.setFont(font)
            self.lfpFreqLo_val.setDecimals(0)
            self.lfpFreqLo_val.setSuffix(' Hz')
            lfpFreq_dash = QtWidgets.QLabel('-')
            lfpFreq_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.lfpFreqHi_val = QtWidgets.QDoubleSpinBox()
            self.lfpFreqHi_val.setFont(font)
            self.lfpFreqHi_val.setDecimals(0)
            self.lfpFreqHi_val.setSuffix(' Hz')
            r2bot.addWidget(self.lfpFreqLo_val, stretch=2)
            r2bot.addWidget(lfpFreq_dash, stretch=0)
            r2bot.addWidget(self.lfpFreqHi_val, stretch=2)
            r2.addWidget(lfpFilt_label)
            r2.addLayout(r2bot)
            lay1.addLayout(r1)
            lay1.addLayout(r2)
            lay1.addSpacing(hspace10)
            self.lfpLayout.addWidget(title1, stretch=0)
            self.lfpLayout.addLayout(lay1, stretch=2)
            line_1 = vline()
            self.settingsLayout.addWidget(self.lfpWidget)
            self.settingsLayout.addWidget(line_1)
            
            ### LFP thresholding params ###
            self.thresWidget = QtWidgets.QWidget()
            width2 = int(self.WIDTH * 0.1615)
            self.thresWidget.setFixedWidth(width2)
            self.thresLayout = QtWidgets.QVBoxLayout(self.thresWidget)
            self.thresLayout.setContentsMargins(cmargins)
            self.thresLayout.setSpacing(hspace10)
            lay2 = QtWidgets.QHBoxLayout()
            lay2.setContentsMargins(0,0,0,0)
            lay2.setSpacing(wspace20)
            title2 = QtWidgets.QLabel('Threshold Parameters')
            title2.setAlignment(QtCore.Qt.AlignCenter)
            title2.setFixedHeight(titleHeight)
            title2.setFont(headerFont)
            c1 = QtWidgets.QGridLayout()  # brainstates + min. episode duration
            c1.setHorizontalSpacing(wspace5)
            c1.setVerticalSpacing(hspace10)
            thresstate_label = QtWidgets.QLabel('State(s)')
            thresstate_label.setAlignment(QtCore.Qt.AlignCenter)
            thresstate_label.setFont(font)
            thresMinDur_label = QtWidgets.QLabel('Min. Dur.')
            thresMinDur_label.setAlignment(QtCore.Qt.AlignCenter)
            thresMinDur_label.setFont(font)
            c1.addWidget(thresstate_label, 0, 0, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            c1.addWidget(thresMinDur_label, 0, 2, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            self.thresStates = {}
            for i,state in enumerate(['REM','Wake','NREM','IS']):
                # create brain state widgets, store in dictionary
                chk = QtWidgets.QCheckBox(state)
                chk.setFont(font)
                geq = QtWidgets.QLabel('\u2265')
                geq.setFont(font)
                val = QtWidgets.QDoubleSpinBox()
                val.setFont(font)
                val.setSuffix(' s')
                val.setDecimals(1)
                val.setMaximumWidth(px_w(75,self.WIDTH))
                self.thresStates[i+1] = [chk, val]
                # add widgets to layout
                c1.addWidget(chk, i+1, 0)
                c1.addWidget(geq, i+1, 1)
                c1.addWidget(val, i+1, 2)
            c2 = QtWidgets.QVBoxLayout()
            c2.setSpacing(hspace20)
            c2r1_w = QtWidgets.QWidget()
            c2r1_w.setFixedHeight(px_h(70,self.HEIGHT))
            c2r1 = QtWidgets.QVBoxLayout(c2r1_w)  # P-wave threshold value + method
            c2r1.setContentsMargins(0,0,0,0)
            c2r1.setSpacing(hspace1)
            thres_label = QtWidgets.QLabel('Detection Threshold')
            thres_label.setAlignment(QtCore.Qt.AlignCenter)
            thres_label.setFont(font)
            self.thresType = QtWidgets.QComboBox()
            self.thresType.setFont(font)
            self.thresType.addItems(['Raw value', 'Std. deviations', 'Percentile'])
            self.thres_val = QtWidgets.QDoubleSpinBox()
            self.thres_val.setFont(font)
            self.thres_val.setMaximum(1000)
            self.thres_val.setDecimals(1)
            self.thres_val.setSingleStep(0.1)
            c2r1.addWidget(thres_label, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            c2r1.addWidget(self.thresType)
            c2r1.addWidget(self.thres_val)
            c2r2 = QtWidgets.QGridLayout()  # use 1st X seconds of state
            c2r2.setHorizontalSpacing(px_w(3, self.WIDTH))
            c2r2.setVerticalSpacing(px_h(3, self.HEIGHT))
            self.thresFirst_btn = QtWidgets.QCheckBox('Use first')
            self.thresFirst_btn.setFont(font)
            self.thresFirst_label = QtWidgets.QLabel('    second(s) of state')
            self.thresFirst_label.setFont(font)
            self.thresFirst_val = QtWidgets.QDoubleSpinBox()
            self.thresFirst_val.setFont(font)
            self.thresFirst_val.setMinimum(0)
            self.thresFirst_val.setMaximum(60)
            self.thresFirst_val.setDecimals(0)
            c2r2.addWidget(self.thresFirst_btn, 0, 0, 1, 1)
            c2r2.addWidget(self.thresFirst_val, 0, 1, 1, 1)
            c2r2.addWidget(self.thresFirst_label, 1, 0, 1, 2)
            c2.addWidget(c2r1_w, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            c2.addLayout(c2r2)
            c2.addSpacing(hspace5)
            lay2.addLayout(c1)
            lay2.addLayout(c2)
            self.thresLayout.addWidget(title2, stretch=0)
            self.thresLayout.addLayout(lay2, stretch=2)
            line_2 = vline()
            self.settingsLayout.addWidget(self.thresWidget)
            self.settingsLayout.addWidget(line_2)
            
            ### P-wave validation params ###
            self.pvalidWidget = QtWidgets.QWidget()
            width3 = int(self.WIDTH * 0.1172)
            self.pvalidWidget.setFixedWidth(width3)
            self.pvalidLayout = QtWidgets.QVBoxLayout(self.pvalidWidget)
            self.pvalidLayout.setContentsMargins(cmargins)
            self.pvalidLayout.setSpacing(hspace10)
            lay3 = QtWidgets.QVBoxLayout()
            lay3.setContentsMargins(0,0,0,0)
            lay3.setSpacing(hspace20)
            title3 = QtWidgets.QLabel('P-wave Validation')
            title3.setAlignment(QtCore.Qt.AlignCenter)
            title3.setFixedHeight(titleHeight)
            title3.setFont(headerFont)
            r1 = QtWidgets.QHBoxLayout()
            r1.setSpacing(wspace15)
            r1c1_w = QtWidgets.QWidget()
            r1c1_w.setFixedHeight(px_h(70,self.HEIGHT))
            r1c1 = QtWidgets.QVBoxLayout(r1c1_w)  # amplitude threshold value + method
            r1c1.setContentsMargins(0,0,0,0)
            r1c1.setSpacing(hspace1)
            maxAmp_label = QtWidgets.QLabel('Max. Amplitude')
            maxAmp_label.setAlignment(QtCore.Qt.AlignCenter)
            maxAmp_label.setFont(font)
            self.maxAmp_thresType = QtWidgets.QComboBox()
            self.maxAmp_thresType.setFont(font)
            self.maxAmp_thresType.addItems(['None', 'Raw Value', 'Percentile'])
            self.maxAmp_val = QtWidgets.QDoubleSpinBox()
            self.maxAmp_val.setFont(font)
            r1c1.addWidget(maxAmp_label, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            r1c1.addWidget(self.maxAmp_thresType)
            r1c1.addWidget(self.maxAmp_val)
            r1c2_w = QtWidgets.QWidget()
            r1c2_w.setFixedHeight(px_h(70,self.HEIGHT))
            r1c2 = QtWidgets.QVBoxLayout(r1c2_w)  # half-width threshold value + method
            r1c2.setContentsMargins(0,0,0,0)
            r1c2.setSpacing(hspace1)
            maxHW_label = QtWidgets.QLabel('Max. Half-Width')
            maxHW_label.setAlignment(QtCore.Qt.AlignCenter)
            maxHW_label.setFont(font)
            self.maxHW_thresType = QtWidgets.QComboBox()
            self.maxHW_thresType.setFont(font)
            self.maxHW_thresType.addItems(['None', 'Raw Value', 'Percentile'])
            self.maxHW_val = QtWidgets.QDoubleSpinBox()
            self.maxHW_val.setFont(font)
            r1c2.addWidget(maxHW_label, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            r1c2.addWidget(self.maxHW_thresType)
            r1c2.addWidget(self.maxHW_val)
            r1.addWidget(r1c1_w, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            r1.addWidget(r1c2_w, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            r2 = QtWidgets.QVBoxLayout()  # duplicate waveform threshold
            r2.setSpacing(hspace1)
            dupWin_label = QtWidgets.QLabel('Check for duplicates within')
            dupWin_label.setAlignment(QtCore.Qt.AlignCenter)
            dupWin_label.setFont(font)
            self.dupWin_val = QtWidgets.QDoubleSpinBox()
            self.dupWin_val.setFont(font)
            self.dupWin_val.setDecimals(0)
            self.dupWin_val.setSuffix (' ms')
            r2.addWidget(dupWin_label)
            r2.addWidget(self.dupWin_val, alignment=QtCore.Qt.AlignCenter)
            lay3.addLayout(r1)
            lay3.addLayout(r2)
            self.pvalidLayout.addWidget(title3, stretch=0)
            self.pvalidLayout.addLayout(lay3, stretch=2)
            line_3 = vline()
            self.settingsLayout.addWidget(self.pvalidWidget)
            self.settingsLayout.addWidget(line_3)
            
            ### Noise detection params ###
            self.noiseWidget = QtWidgets.QWidget()
            width4 = int(self.WIDTH * 0.1146)
            self.noiseWidget.setFixedWidth(width4)
            self.noiseLayout = QtWidgets.QVBoxLayout(self.noiseWidget)
            cmargins2 = QtCore.QMargins(cmargins.left(), cmargins.top(), 
                                        cmargins.right(), cmargins.top())
            self.noiseLayout.setContentsMargins(cmargins2)
            self.noiseLayout.setSpacing(hspace10)
            noiseTitle_row = QtWidgets.QHBoxLayout()
            noiseTitle_row.setContentsMargins(0,0,0,0)
            noiseTitle_row.setSpacing(wspace5)
            self.noiseTitle4 = QtWidgets.QLabel('LFP Noise Detection')
            self.noiseTitle4.setAlignment(QtCore.Qt.AlignCenter)
            self.noiseTitle4.setFixedHeight(titleHeight)
            self.noiseTitle4.setFont(headerFont)
            # buttons to view noise options for additional signals
            self.noiseOptions = ['LFP', 'EMG', 'EEG']
            self.noiseBtns_bck = back_next_btns(parent=self, name='noiseBtns back')
            self.noiseBtns_bck.setIconSize(QtCore.QSize(0,0))
            self.noiseBtns_nxt = back_next_btns(parent=self, name='noiseBtns next')
            noiseTitle_row.addWidget(self.noiseBtns_bck, stretch=0, 
                                     alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft)
            noiseTitle_row.addWidget(self.noiseTitle4, stretch=2, 
                                     alignment=QtCore.Qt.AlignCenter)
            noiseTitle_row.addWidget(self.noiseBtns_nxt, stretch=0, 
                                     alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
            lay4 = QtWidgets.QVBoxLayout()
            lay4.setContentsMargins(0,0,0,0)
            lay4.setSpacing(hspace10)
            r1_w = QtWidgets.QWidget()
            r1_w.setFixedHeight(px_h(70,self.HEIGHT))
            r1 = QtWidgets.QVBoxLayout(r1_w)  # noise threshold values + method
            r1.setContentsMargins(0,0,0,0)
            r1.setSpacing(hspace1)
            noiseThres_label = QtWidgets.QLabel('Noise Thresholds')
            noiseThres_label.setAlignment(QtCore.Qt.AlignCenter)
            noiseThres_label.setFont(font)
            self.noiseThres_type = QtWidgets.QComboBox()
            self.noiseThres_type.setFont(font)
            self.noiseThres_type.addItems(['None', 'Raw Value', 'Percentile'])
            r1bot = QtWidgets.QHBoxLayout()
            r1bot.setContentsMargins(0,0,0,0)
            r1bot.setSpacing(wspace10)
            r1bot_b1 = QtWidgets.QHBoxLayout()
            r1bot_b1.setSpacing(wspace1)
            noiseThresUp_label = QtWidgets.QLabel('(+)')
            noiseThresUp_label.setAlignment(QtCore.Qt.AlignCenter)
            noiseThresUp_label.setFont(font)
            self.noiseThresUp_val = QtWidgets.QDoubleSpinBox()
            self.noiseThresUp_val.setFont(font)
            r1bot_b1.addWidget(noiseThresUp_label)
            r1bot_b1.addWidget(self.noiseThresUp_val)
            r1bot_b2 = QtWidgets.QHBoxLayout()
            r1bot_b2.setSpacing(wspace1)
            noiseThresDn_label = QtWidgets.QLabel('( - )')
            noiseThresDn_label.setAlignment(QtCore.Qt.AlignCenter)
            noiseThresDn_label.setFont(font)
            self.noiseThresDn_val = QtWidgets.QDoubleSpinBox()
            self.noiseThresDn_val.setFont(font)
            r1bot_b2.addWidget(noiseThresDn_label)
            r1bot_b2.addWidget(self.noiseThresDn_val)
            r1bot.addLayout(r1bot_b1)
            r1bot.addLayout(r1bot_b2)
            r1.addWidget(noiseThres_label, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            r1.addWidget(self.noiseThres_type, alignment=QtCore.Qt.AlignTop)
            r1.addLayout(r1bot)
            r2 = QtWidgets.QHBoxLayout()  # noise window + separation values
            r2.setSpacing(wspace20)
            r2c1 = QtWidgets.QVBoxLayout()
            r2c1.setSpacing(hspace1)
            noiseWin_label = QtWidgets.QLabel('Noise window')
            noiseWin_label.setAlignment(QtCore.Qt.AlignCenter)
            noiseWin_label.setFont(font)
            self.noiseWin_val = QtWidgets.QDoubleSpinBox()
            self.noiseWin_val.setFont(font)
            self.noiseWin_val.setDecimals(2)
            self.noiseWin_val.setSuffix(' s')
            r2c1.addWidget(noiseWin_label)
            r2c1.addWidget(self.noiseWin_val)
            r2c2 = QtWidgets.QVBoxLayout()
            r2c2.setSpacing(hspace1)
            noiseSep_label = QtWidgets.QLabel('Min. separation')
            noiseSep_label.setAlignment(QtCore.Qt.AlignCenter)
            noiseSep_label.setFont(font)
            self.noiseSep_val = QtWidgets.QDoubleSpinBox()
            self.noiseSep_val.setFont(font)
            self.noiseSep_val.setDecimals(2)
            self.noiseSep_val.setSuffix(' s')
            r2c2.addWidget(noiseSep_label)
            r2c2.addWidget(self.noiseSep_val)
            r2.addLayout(r2c1)
            r2.addLayout(r2c2)
            r3 = QtWidgets.QHBoxLayout()
            r3.setContentsMargins(wspace5,0,wspace5,0)
            r3.setSpacing(wspace20)
            r3c1 = QtWidgets.QHBoxLayout()  # save vs. reset noise buttons
            r3c1.setSpacing(wspace5)
            btn_grp = QtWidgets.QButtonGroup(self.noiseWidget)
            self.noiseReset_btn = update_noise_btn(top_parent=self, icon='reset',
                                                   parent=self.noiseWidget)
            self.noiseSave_btn = update_noise_btn(top_parent=self, icon='save',
                                                  parent=self.noiseWidget)
            btn_grp.addButton(self.noiseReset_btn)
            btn_grp.addButton(self.noiseSave_btn)
            # view EEG spectrogram with noise ignored (standard) or excluded
            self.noiseCalcSP_btn = update_noise_btn(top_parent=self, icon='calc',
                                                    parent=self.noiseWidget)
            self.noiseCalcSP_btn.hide()
            menu = QtWidgets.QMenu(self.noiseCalcSP_btn)
            calcAction = QtGui.QAction('Calculate noise-excluded SP', self.noiseCalcSP_btn)
            calcAction.setObjectName('calculate noise')
            calcAction.triggered.connect(self.switchSP)
            menu.addAction(calcAction)
            loadAction = QtGui.QAction('Load noise-excluded SP', self.noiseCalcSP_btn)
            loadAction.setObjectName('load noise')
            loadAction.triggered.connect(self.switchSP)
            menu.addAction(loadAction)
            resetAction = QtGui.QAction('Load standard SP', self.noiseCalcSP_btn)
            resetAction.setObjectName('load standard')
            resetAction.triggered.connect(self.switchSP)
            menu.addAction(resetAction)
            self.noiseCalcSP_btn.setMenu(menu)
            r3c1.addWidget(self.noiseReset_btn)
            r3c1.addWidget(self.noiseSave_btn)
            r3c1.addWidget(self.noiseCalcSP_btn)
            r3c1.addSpacerItem(QtWidgets.QSpacerItem(px_w(28,self.WIDTH),px_h(28,self.HEIGHT),
                                                     QtWidgets.QSizePolicy.Maximum))
            r3c2 = QtWidgets.QHBoxLayout()    # noise viewing buttons
            r3c2.setContentsMargins(0,0,0,0)
            r3c2.setSpacing(wspace5)
            btn = show_hide_event(parent=self)
            bck = back_next_event(parent=self, name='noise back')
            nxt = back_next_event(parent=self, name='noise next')
            bck.setEnabled(False)
            nxt.setEnabled(False)
            self.noiseShow_btn = [btn,bck,nxt]
            r3c2.addWidget(bck, stretch=0, alignment=QtCore.Qt.AlignCenter)
            r3c2.addWidget(btn, stretch=2, alignment=QtCore.Qt.AlignCenter)
            r3c2.addWidget(nxt, stretch=0, alignment=QtCore.Qt.AlignCenter)
            r3.addLayout(r3c1)
            r3.addLayout(r3c2)
            lay4.addWidget(r1_w, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
            lay4.addSpacing(hspace10)
            lay4.addLayout(r2)
            lay4.addLayout(r3)
            self.noiseLayout.addLayout(noiseTitle_row, stretch=0)
            self.noiseLayout.addLayout(lay4, stretch=2)
            line_4 = vline()
            self.settingsLayout.addWidget(self.noiseWidget)
            self.settingsLayout.addWidget(line_4)
            
            ### Artifact viewing params ###
            self.elimWidget = QtWidgets.QWidget()
            width5 = int(self.WIDTH * 0.1250)
            self.elimWidget.setFixedWidth(width5)
            self.elimLayout = QtWidgets.QVBoxLayout(self.elimWidget)
            self.elimLayout.setContentsMargins(cmargins)
            self.elimLayout.setSpacing(hspace10)
            lay5 = QtWidgets.QVBoxLayout()
            lay5.setContentsMargins(0,0,0,0)
            lay5.setSpacing(hspace10)
            title5 = QtWidgets.QLabel('Show Exclusions')
            title5.setAlignment(QtCore.Qt.AlignCenter)
            title5.setFixedHeight(titleHeight)
            title5.setFont(headerFont)
            # create viewing buttons for each category of eliminated waveform
            keys = ['elim_amp', 'elim_width', 'elim_dup', 'elim_noise', 'elim_user']
            labels = ['Amplitude outliers', 'Half-width outliers', 'Duplicate waves', 
                      'Motion artifacts', 'User-eliminated']
            self.elimView_btns = {}
            for key,label in zip(keys,labels):
                r = QtWidgets.QHBoxLayout()
                r.setSpacing(px_w(8, self.WIDTH))
                l = QtWidgets.QLabel(label)
                l.setFont(font)
                l.setFixedHeight(px_h(25,self.HEIGHT))
                l.setContentsMargins(wspace5,0,wspace5,0)
                l.setStyleSheet('QLabel'
                                '{ background-color : rgba(255,255,255,120);'
                                'border-color : rgba(255,255,255,170);'
                                'border-width : 2px;'
                                'border-radius : 5px }')
                # show/hide button
                btn = show_hide_event(parent=self)
                # next/back buttons
                bck = back_next_event(parent=self, name = key + ' back')
                nxt = back_next_event(parent=self, name = key + ' next')
                bck.setEnabled(False)
                nxt.setEnabled(False)
                if key == 'elim_user':
                    self.elimUser_btn = [btn,bck,nxt]
                else:
                    self.elimView_btns[key] = [btn,bck,nxt]
                r.addWidget(bck, stretch=0, alignment=QtCore.Qt.AlignCenter)
                r.addWidget(l, stretch=2, alignment=QtCore.Qt.AlignCenter)
                r.addWidget(btn, stretch=0, alignment=QtCore.Qt.AlignCenter)
                r.addWidget(nxt, stretch=0, alignment=QtCore.Qt.AlignCenter)
                lay5.addLayout(r)
            lay5.addSpacing(hspace5)
            self.elimLayout.addWidget(title5, stretch=0)
            self.elimLayout.addLayout(lay5, stretch=2)
            line_5 = vline()
            self.settingsLayout.addWidget(self.elimWidget)
            self.settingsLayout.addWidget(line_5)
            
            ### Data figure plotting ###
            self.figWidget = QtWidgets.QWidget()
            width6 = int(self.WIDTH * 0.0905)
            self.figWidget.setFixedWidth(width6)
            self.figLayout = QtWidgets.QVBoxLayout(self.figWidget)
            cmargins3 = QtCore.QMargins(int(cmargins.top()/2), cmargins.top(), 
                                        int(cmargins.top()/2), int(cmargins.top()*3))
            self.figLayout.setContentsMargins(cmargins3)
            self.figLayout.setSpacing(hspace5)
            figTitle_row = QtWidgets.QHBoxLayout()
            figTitle_row.setContentsMargins(0,0,0,0)
            figTitle_row.setSpacing(hspace5)
            title6 = QtWidgets.QLabel('Figure Plots')
            title6.setAlignment(QtCore.Qt.AlignCenter)
            title6.setFixedHeight(titleHeight)
            title6.setFont(headerFont)
            # buttons to view additional plotting options
            self.plotBtns_bck = back_next_btns(parent=self, name='plotBtns back')
            self.plotBtns_bck.setIconSize(QtCore.QSize(0,0))
            self.plotBtns_nxt = back_next_btns(parent=self, name='plotBtns next')
            figTitle_row.addWidget(self.plotBtns_bck, stretch=0, 
                                   alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft)
            figTitle_row.addWidget(title6, stretch=2, alignment=QtCore.Qt.AlignCenter)
            figTitle_row.addWidget(self.plotBtns_nxt, stretch=0, 
                                   alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
            # define name/color/enabling requirements of plot buttons
            plotBtn_ids = [('P-wave frequency', 'rgba(24,116,205,100)', ['recordPwaves']), 
                           ('P-waveforms', 'rgba(245,214,78,150)', ['recordPwaves']), 
                           ('P-wave spectrogram', 'rgba(154,50,205,100)', ['recordPwaves']), 
                           ('P-wave EMG', 'rgba(0,205,102,100)', ['recordPwaves', 'hasEMG']),
                           ('Lsr P-wave stats', 'rgba(84,44,45,70)', ['recordPwaves', 'lsrTrigPwaves']), 
                           ('P-wave transitions\n(time-normalized)', 'rgba(0,0,0,255)', ['recordPwaves']), 
                           ('P-wave DF/F', 'rgba(172,237,138,100)', ['recordPwaves', 'hasDFF']),
                           ('Sleep timecourse',  'rgba(196,52,24,100)', []), 
                           ('Sleep spectrum', 'rgba(250,250,250,150)', []), 
                           ('Opto brain state', 'rgba(70,49,148,100)', ['ol']), 
                           ('EMG twitches', 'rgba(86,245,234,100)', ['hasEMG'])]
            self.plotBtns_dict = {name:{'btn':None, 'label':None, 'color':color} for name,color,_ in plotBtn_ids}
            self.singlePlotBtns = ['P-wave spectrogram', 'P-wave EMG', 'P-wave DF/F']
            # one container widget for every 5 plot buttons
            self.plotBtn_widgets = []
            for iWidget in np.arange(0, len(plotBtn_ids), 5):
                # set container layout
                widget = QtWidgets.QWidget()
                lay = QtWidgets.QVBoxLayout(widget)
                lay.setContentsMargins(wspace10,0,wspace10,0)
                lay.setSpacing(hspace5)
                for iBtn in range(iWidget, min(iWidget+5, len(plotBtn_ids))):
                    name, color, reqs = plotBtn_ids[iBtn]
                    # create button and label
                    label = QtWidgets.QLabel(name)
                    label.setFont(font)
                    self.plotBtns_dict[name]['label'] = label
                    btn = PlotButton(parent=self, name=name, color=color, reqs=reqs)
                    self.plotBtns_dict[name]['btn'] = btn
                    r = QtWidgets.QHBoxLayout()
                    r.setSpacing(wspace5)
                    r.addWidget(btn)
                    r.addWidget(label)
                    lay.addLayout(r)
                self.plotBtn_widgets.append(widget)
            # add button widgets to settings layout
            self.figLayout.addLayout(figTitle_row, stretch=0)
            self.figLayout.addWidget(self.plotBtn_widgets[0], stretch=2)
            for widget in self.plotBtn_widgets[1:]:
                widget.hide()
                self.figLayout.addWidget(widget, stretch=2)
            line_6 = vline()
            self.settingsLayout.addWidget(self.figWidget)
            self.settingsLayout.addWidget(line_6)
            
            ### Action buttons ###
            self.btnsWidget = QtWidgets.QWidget()
            self.btnsLayout = QtWidgets.QVBoxLayout(self.btnsWidget)
            self.btnsLayout.setSpacing(hspace5)
            self.updatePlot_btn = QtWidgets.QPushButton('Apply')  # run P-wave detection
            self.updatePlot_btn.setFont(font)
            self.updatePlot_btn.setDefault(True)
            self.saveData_btn = QtWidgets.QPushButton('Save')     # save detection settings
            self.saveData_btn.setFont(font)
            self.showHelp_btn = QtWidgets.QPushButton('Help')
            self.showHelp_btn.setFont(font)
            self.moreSettings_btn = QtWidgets.QPushButton('More ...')
            self.moreSettings_btn.setFont(font)
            db = QtWidgets.QPushButton('DEBUG')
            db.setFont(font)
            db.clicked.connect(self.debug)
            self.btnsLayout.addWidget(self.updatePlot_btn)
            self.btnsLayout.addWidget(self.saveData_btn)
            self.btnsLayout.addWidget(self.showHelp_btn)
            self.btnsLayout.addWidget(self.moreSettings_btn)
            self.btnsLayout.addWidget(db)
            self.settingsLayout.addWidget(self.btnsWidget)
            
            # settings layout spacing
            self.settingsLayout.setContentsMargins(cmargins.left(), 0, 
                                                   cmargins.right(), 0)
            self.settingsLayout.setSpacing(hspace10)
            self.settingsWidget.setFixedHeight(int(self.HEIGHT*0.22))
            
            self.centralLayout.addWidget(self.settingsWidget)
            self.setCentralWidget(self.centralWidget)
            
            # brain state colormap
            pos = np.linspace(0, 1, 8)
            color = np.array([[0, 0, 0, 200], [0, 255, 255, 200], 
                              [150, 0, 255, 200], [150, 150, 150, 200], 
                              [66,86,219,200], [255,20,20,200], 
                              [0,255,43,200], [255,255,0,200]], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, color)
            self.lut_brainstate = cmap.getLookupTable(0.0, 1.0, 8)
            pos = np.array([0., 0.05, .2, .4, .6, .9])
            color = np.array([[0, 0, 0, 255], [0,0,128,255], [0,255,0,255], [255,255,0, 255], 
                              (255,165,0,255), (255,0,0, 255)], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, color)
            self.lut_spectrum = cmap.getLookupTable(0.0, 1.0, 256)
        
        except Exception as e:
            print('whoopsie')
            print(e)
            sys.exit()


    def connect_buttons(self):
        """
        Connect widgets to data analysis functions
        """
        # connect LFP processing buttons
        self.lfpSig_type.currentTextChanged.connect(self.update_lfp_params)
        self.lfpFreqLo_val.valueChanged.connect(self.update_lfp_params)
        self.lfpFreqHi_val.valueChanged.connect(self.update_lfp_params)
        
        # connect LFP thresholding buttons
        self.thres_val.valueChanged.connect(self.update_thres_params)
        self.thresType.currentTextChanged.connect(self.update_thres_params)
        for i,state in enumerate(['REM','Wake','NREM','IS']):
            self.thresStates[i+1][0].toggled.connect(self.update_thres_params)
            self.thresStates[i+1][1].valueChanged.connect(self.update_thres_params)
        self.thresFirst_btn.toggled.connect(self.update_thres_params)
        self.thresFirst_val.valueChanged.connect(self.update_thres_params)
        
        # connect P-wave validation buttons
        self.maxAmp_thresType.currentTextChanged.connect(self.update_pvalid_params)
        self.maxAmp_val.valueChanged.connect(self.update_pvalid_params)
        self.maxHW_thresType.currentTextChanged.connect(self.update_pvalid_params)
        self.maxHW_val.valueChanged.connect(self.update_pvalid_params)
        self.dupWin_val.valueChanged.connect(self.update_pvalid_params)
        
        # connect noise detection buttons
        self.noiseBtns_bck.clicked.connect(self.more_noise_btns)
        self.noiseBtns_nxt.clicked.connect(self.more_noise_btns)
        self.noiseThres_type.currentTextChanged.connect(self.update_noise_params)
        self.noiseThresUp_val.valueChanged.connect(self.update_noise_params)
        self.noiseThresDn_val.valueChanged.connect(self.update_noise_params)
        self.noiseWin_val.valueChanged.connect(self.update_noise_params)
        self.noiseSep_val.valueChanged.connect(self.update_noise_params)
        self.noiseReset_btn.toggled.connect(self.update_noise_params)
        
        # connect artifact viewing buttons
        self.noiseShow_btn[0].toggled.connect(self.show_hide_artifacts)
        self.noiseShow_btn[1].clicked.connect(self.plot_next_artifact)
        self.noiseShow_btn[2].clicked.connect(self.plot_next_artifact)
        for val in self.elimView_btns.values():
            val[0].toggled.connect(self.show_hide_artifacts)
            val[1].clicked.connect(self.plot_next_artifact)
            val[2].clicked.connect(self.plot_next_artifact)
        self.elimUser_btn[0].toggled.connect(self.show_hide_artifacts)
        self.elimUser_btn[1].clicked.connect(self.plot_next_artifact)
        self.elimUser_btn[2].clicked.connect(self.plot_next_artifact)
        
        # connect figure plotting buttons
        self.plotBtns_bck.clicked.connect(self.more_plot_btns)
        self.plotBtns_nxt.clicked.connect(self.more_plot_btns)
        for k in self.plotBtns_dict.keys():
            btn = self.plotBtns_dict[k]['btn']
            btn.clicked.connect(self.plotFigure)
        
        # connect action buttons
        self.updatePlot_btn.clicked.connect(self.update_plot)
        self.saveData_btn.clicked.connect(self.save_pdata)
        self.showHelp_btn.clicked.connect(self.show_help_window)
    
    
    ##################          UPDATING PARAMETERS          ##################
        
    
    def update_lfp_params(self):
        """
        Update LFP processing parameters from user input
        """
        # update LFP referencing method
        self.channel = ['S','C','1','2','1-2','2-1',None][self.lfpSig_type.currentIndex()]
        # update raw LFP bandpass filter
        self.w0 = float(self.lfpFreqLo_val.value() / (self.sr/2))
        self.w1 = float(self.lfpFreqHi_val.value() / (self.sr/2))
        self.disable_plot_update()
    
    
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
        
        # update brain states/min durations to use for thresholding
        for i,state in enumerate(['REM','Wake','NREM','IS']):
            self.thres_states[i+1][0] = self.thresStates[i+1][0].isChecked()
            self.thres_states[i+1][1] = float(self.thresStates[i+1][1].value())
            self.thresStates[i+1][1].setEnabled(self.thresStates[i+1][0].isChecked())
            self.thresStates[i+1][1].lineEdit().setVisible(self.thresStates[i+1][0].isChecked())
            
        # threshold by first X seconds of brain state(s)
        self.thres_first = float(self.thresFirst_val.value()) if self.thresFirst_btn.isChecked() else 0.
        self.thresFirst_val.setEnabled(self.thresFirst_btn.isChecked())
        self.thresFirst_val.setValue(self.thres_first)
        self.disable_plot_update()
    
    
    def update_pvalid_params(self):
        """
        Update P-wave validation parameters from user input
        """
        # update amplitude thresholding method, adjust spinbox settings
        i = self.maxAmp_thresType.currentIndex()
        d = [['none',False,'',5000,0,1],['raw',True,' uV',5000,0,100], ['perc',True,' %',100,1,1]][i]
        self.amp_thres_type, enabled, suffix, maxval, decimals, singlestep = d
        self.maxAmp_val.setEnabled(enabled)
        self.maxAmp_val.lineEdit().setVisible(enabled)
        self.maxAmp_val.setSuffix(suffix)
        self.maxAmp_val.setMaximum(maxval)
        self.maxAmp_val.setDecimals(decimals)
        self.maxAmp_val.setSingleStep(singlestep)
              
        # update half-width thresholding method, adjust spinbox settings
        j = self.maxHW_thresType.currentIndex()
        e = [['none',False,'',5000,0,1],['raw',True,' ms',5000,0,100], ['perc',True,' %',100,1,1]][i]
        self.hw_thres_type, enabled2, suffix2, maxval2, decimals2, singlestep2 = e
        self.maxHW_val.setEnabled(enabled2)
        self.maxHW_val.lineEdit().setVisible(enabled2)
        self.maxHW_val.setSuffix(suffix2)
        self.maxHW_val.setMaximum(maxval2)
        self.maxHW_val.setDecimals(decimals2)
        self.maxHW_val.setSingleStep(singlestep2)
        
        # update amplitude/half-width threshold values
        self.amp_thres = float(self.maxAmp_val.value())
        self.hw_thres = float(self.maxHW_val.value())
        # update duplicate threshold value
        self.dup_win = float(self.dupWin_val.value())
        
    
    def update_noise_params(self):
        """
        Update noise detection parameters from user input
        """
        # update noise threshold value and noise window (s)
        self.noise_thres_up = float(self.noiseThresUp_val.value())
        self.noise_thres_dn = float(self.noiseThresDn_val.value())
        self.noise_win = float(self.noiseWin_val.value())
        self.noise_sep = float(self.noiseSep_val.value())
        # update param for resetting vs. saving noise indices
        self.reset_noise = bool(self.noiseReset_btn.isChecked())
        
        # update noise threshold type
        i = self.noiseThres_type.currentIndex()
        d = [['none',None,None,None],['raw',' uV',5000,0],['perc',' %',100,1]][i]
        self.noise_thres_type, suffix, maxval, decimals = d
        # enable/disable noise threshold value boxes in GUI
        for valbox in [self.noiseThresUp_val, self.noiseThresDn_val]:
            valbox.setDisabled(i == 0)
            valbox.lineEdit().setVisible(i != 0)
            if i != 0:
                valbox.setSuffix(suffix)
                valbox.setMaximum(maxval)
                valbox.setDecimals(decimals)
    
    
    def update_vars_from_gui(self):
        """
        Update all parameter variables from user input
        """
        self.update_lfp_params()
        self.update_thres_params()
        self.update_pvalid_params()
        self.update_noise_params()
    
    
    def disable_plot_update(self):
        """
        Enable P-wave detection to run ONLY IF:
            1) LFP referencing method is set,
            2) Bandpass filtering frequencies are valid, and
            3) Thresholding method for P-wave detection is set
        """
        x1 = self.channel == None     # must choose referencing method
        x2 = self.w1 <= self.w0       # high freq in bp filter must be greater than low freq
        x3 = self.thres_type == None  # must choose threshold type for P-wave detection
        self.updatePlot_btn.setDisabled(any([x1,x2,x3]))
    
    
    def dict_from_vars(self):
        """
        Collect all current parameter values and return as dictionary
        """
        ddict = {'channel':None if not self.channel else str(self.channel),
                 'w0':float(self.w0),
                 'w1':float(self.w1),
                 'thres':float(self.thres),
                 'thres_type':None if not self.thres_type else str(self.thres_type),
                 'thres_states':{k:list(self.thres_states[k]) for k in self.thres_states.keys()},
                 'thres_first':float(self.thres_first),
                 'amp_thres':float(self.amp_thres),
                 'amp_thres_type':str(self.amp_thres_type),
                 'hw_thres':float(self.hw_thres),
                 'hw_thres_type':str(self.hw_thres_type),
                 'dup_win':float(self.dup_win),
                 'noise_thres_up':float(self.noise_thres_up),
                 'noise_thres_dn':float(self.noise_thres_dn),
                 'noise_thres_type':str(self.noise_thres_type),
                 'noise_win':float(self.noise_win),
                 'noise_sep':float(self.noise_sep)}
        return ddict
        
    
    def update_gui_from_vars(self):
        """
        Update all GUI widgets from current parameter variable values
        """
        ddict = self.dict_from_vars()
        # set LFP processing params
        if ddict['channel']:
            i = ['S','C','1','2','1-2','2-1'].index(ddict['channel'])
        else:
            i = -1
        self.lfpSig_type.setCurrentIndex(i)
        self.lfpFreqLo_val.setValue(ddict['w0'] * (self.sr/2))
        self.lfpFreqHi_val.setValue(ddict['w1'] * (self.sr/2))
        
        # set LFP thresholding params
        if ddict['channel']:
            if ddict['thres_type'] == 'au':
                i=1
            else:
                i = ['raw','std','perc'].index(ddict['thres_type'])
        else:
            i=-1
        self.thresType.setCurrentIndex(i)
        self.thres_val.setValue(ddict['thres'])
        for i,state in enumerate(['REM','Wake','NREM','IS']):
            self.thresStates[i+1][0].setChecked(ddict['thres_states'][i+1][0])
            self.thresStates[i+1][1].setValue(ddict['thres_states'][i+1][1])
        self.thresFirst_btn.setChecked(ddict['thres_first']>0)
        self.thresFirst_val.setValue(ddict['thres_first'])

        # set P-wave validation params
        i = ['none', 'raw', 'perc'].index(ddict['amp_thres_type'])
        self.maxAmp_thresType.setCurrentIndex(i)
        self.maxAmp_val.setValue(ddict['amp_thres'])
        j = ['none', 'raw', 'perc'].index(ddict['hw_thres_type'])
        self.maxHW_thresType.setCurrentIndex(j)
        self.maxHW_val.setValue(ddict['hw_thres'])
        self.dupWin_val.setValue(ddict['dup_win'])
        
        # set noise detection params
        i = ['none', 'raw', 'perc'].index(ddict['noise_thres_type'])
        self.noiseThres_type.setCurrentIndex(i)
        self.noiseThresUp_val.setValue(ddict['noise_thres_up'])
        self.noiseThresDn_val.setValue(ddict['noise_thres_dn'])
        self.noiseWin_val.setValue(ddict['noise_win'])
        self.noiseSep_val.setValue(ddict['noise_sep'])
        self.noiseReset_btn.setChecked(self.reset_noise)
        self.noiseSave_btn.setChecked(not self.reset_noise)
    
    
    def update_vars_from_dict(self, ddict={}):
        """
        Update all parameter variables from the inputted dictionary $ddict
        """
        try:
            self.channel = ddict['channel']
            self.w0 = ddict['w0']
            self.w1 = ddict['w1']
            self.thres = ddict['thres']
            self.thres_type = ddict['thres_type']
            self.thres_states = ddict['thres_states']
            self.thres_first = ddict['thres_first']
            self.amp_thres = ddict['amp_thres']
            self.amp_thres_type = ddict['amp_thres_type']
            self.hw_thres = ddict['hw_thres']
            self.hw_thres_type = ddict['hw_thres_type']
            self.dup_win =  ddict['dup_win']
            self.noise_thres_up = ddict['noise_thres_up']
            self.noise_thres_dn = ddict['noise_thres_dn']
            self.noise_thres_type = ddict['noise_thres_type']
            self.noise_win = ddict['noise_win']
            self.noise_sep = ddict['noise_sep']
        except KeyError:
            print('### ERROR: One or more params missing from settings dictionary; unable to update variables ###')
    

    def update_plot(self):
        """
        Run P-wave detection using the current settings, update plots 
        """
        self.setWindowTitle('Updating P-wave detection settings ...')
        self.update_vars_from_gui()  # update param variables from GUI
        self.process_LFP()       # calculate processed LFP signal
        self.get_noise()         # get indices of LFP noise
        self.get_thres_idx()     # get threshold indices
        self.threshold_LFP()     # calculate detection threshold, find P-waves
        self.validate_pwaves()   # eliminate LFP artifacts
        self.save_pdata()        # save settings and detected P-waves
        self.plotSettings = self.dict_from_vars()
        self.recordPwaves = True
        
        # update plots
        self.plot_eeg(findPwaves=True, findArtifacts=True)
        self.plot_session(scale=self.tscale, scale_unit=self.tunit)
        self.setWindowTitle('Updating P-wave detection settings ... Done!')
        time.sleep(1)
        self.setWindowTitle(self.name)
        
    @QtCore.pyqtSlot()
    def save_pdata(self, pfile='p_idx.mat'):
        """
        Save P-wave data and detection settings to .mat file
        """
        # get dictionary of current settings
        ddict = self.dict_from_vars()
        if self.sender() == self.saveData_btn and ddict != self.plotSettings:
            res = warning_dlg('Settings changed from current plot - continue?')
            if res == 0:
                return
        tmp = np.array([list(ddict['thres_states'].keys()), 
                        list(ddict['thres_states'].values())], dtype='object')
        ddict['thres_states'] = tmp

        # collect current P-wave indices and detection threshold
        ddict['p_idx'] = np.array(self.pi)
        ddict['p_thr'] = float(self.p_thr)
        # collect indices of automatically and manually eliminated waveforms
        tmp2 = np.array([list(self.elim_idx.keys()), 
                         list(self.elim_idx.values())], dtype='object')
        ddict['elim_idx'] = tmp2
        ddict['elim_user'] = np.array(self.elim_user)
        # collect noise indices
        ddict['noise_idx'] = np.array(self.noise_idx)
        ddict['emg_noise_idx'] = np.array(self.emg_noise_idx)
        ddict['eeg_noise_idx'] = np.array(self.eeg_noise_idx)
        ddict['thres_idx'] = np.array(self.thres_idx)
        
        # save P-wave info file
        fpath = os.path.join(self.ppath, self.name, pfile)
        so.savemat(fpath, ddict)
        # save brain state annotation
        rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
        
        if self.sender() == self.saveData_btn:
            self.setWindowTitle('P-wave detection saved !')
            time.sleep(1)
            self.setWindowTitle(self.name)
    
    
    @QtCore.pyqtSlot()
    def load_pdata(self, pfile='p_idx.mat', mode='default'):
        """
        Load P-wave data and detection settings from .mat file
        """
        # path to P-wave info file
        fpath = os.path.join(self.ppath, self.name, pfile)
        if mode=='default':
            ddict = dict(self.defaults)  # initialize widgets with default param values
        elif mode=='clear':
            ddict = {'channel':None,     # initialize "blank" widgets
                     'w0':0.,
                     'w1':0.,
                     'thres':0.,
                     'thres_type':None,
                     'thres_states':{s:[False,0] for s in [1,2,3,4]},
                     'thres_first':0,
                     'amp_thres':0,
                     'amp_thres_type':'none',
                     'hw_thres':0,
                     'hw_thres_type':'none',
                     'dup_win':0,
                     'noise_thres_up':0,
                     'noise_thres_dn':0,
                     'noise_thres_type':'none',
                     'noise_win':0.,
                     'noise_sep':0.}
        
        # load P-wave detection file
        if os.path.isfile(fpath):
            mfile = so.loadmat(fpath, squeeze_me=True)
        else:
            mfile = {}
        for key in self.defaults.keys():
            # for each param in file, update default/blank dictionary with loaded value
            if key in mfile.keys():
                if type(ddict[key]) == dict:
                    ddict[key] = {k:val for k,val in zip(mfile[key][0], mfile[key][1])}
                    if all([len(v)==2 for v in ddict[key].values()]):
                        ddict[key] = {k:[bool(val[0]), float(val[1])] for k,val in zip(ddict[key].keys(), ddict[key].values())}
                else:
                    ddict[key] = mfile[key]
        # update GUI and parameter variables from dictionary
        self.update_vars_from_dict(ddict)
        self.update_gui_from_vars()
        
        # load/initialize detected P-wave indices
        self.pi = np.array((), dtype='int') if 'p_idx' not in mfile.keys() else np.array(mfile['p_idx'])
        # calculate P-wave frequency
        if self.pi.size > 0:
            self.pfreq = pwaves.downsample_pwaves(self.LFP_processed, self.pi, 
                                                  self.sr, int(self.fbin), 
                                                  self.M.shape[1])[3]
            self.recordPwaves = True
        else:
            self.pfreq = np.zeros(self.M.shape[1])
        if len(self.pfreq) == self.M.shape[1]-1:
            self.pfreq = np.append(self.pfreq, 0.0)
        
        # load/initialize P-wave detection threshold value
        self.p_thr = 0.0 if 'p_thr' not in mfile.keys() else mfile['p_thr']
        
        # find laser events (if recording includes laser-triggered P-waves)
        if self.lsrTrigPwaves:
            tmp = pwaves.get_lsr_pwaves(self.pi, self.laser_raw, 
                                        post_stim=self.post_stim, sr=self.sr)
            self.lsr_pi, self.spon_pi, self.success_lsr, self.fail_lsr = tmp
        else:
            self.lsr_pi = np.array(())
            self.spon_pi = np.array(())
            self.success_lsr = np.array(())
            self.fail_lsr =  np.array(())
        
        # load/initalize automatically eliminated waveforms
        if 'elim_idx' in mfile.keys():
            self.elim_idx = {k:np.array(val) for k,val in zip(mfile['elim_idx'][0], mfile['elim_idx'][1])}
        else:
            self.elim_idx = {'elim_amp'   : np.array((), dtype='int'),
                             'elim_width' : np.array((), dtype='int'),
                             'elim_dup'   : np.array((), dtype='int'),
                             'elim_noise' : np.array((), dtype='int')}
        # load/initialize manually deleted waveforms
        self.elim_user = np.array((), dtype='int') if 'elim_user' not in mfile.keys() else np.atleast_1d(mfile['elim_user'])
        
        # load/initialize noise indices for LFP/EMG/EEG signals
        self.noise_idx = np.array((), dtype='int') if 'noise_idx' not in mfile.keys() else np.atleast_1d(mfile['noise_idx'])
        self.emg_noise_idx = np.array((), dtype='int') if 'emg_noise_idx' not in mfile.keys() else np.atleast_1d(mfile['emg_noise_idx'])
        self.eeg_noise_idx = np.array((), dtype='int') if 'eeg_noise_idx' not in mfile.keys() else np.atleast_1d(mfile['eeg_noise_idx'])
        # load/initialize indices used for LFP thresholding, save in Intan and FFT time for plotting
        self.thres_idx = np.array((), dtype='int') if 'thres_idx' not in mfile.keys() else np.atleast_1d(mfile['thres_idx'])
        self.thidx_raw = np.zeros(len(self.EEG))
        self.thidx_dn = np.zeros(self.M.shape[1])
        self.thidx_raw[self.thres_idx] = 1
        thdn = np.round(self.thres_idx / self.fbin).astype('int')
        self.thidx_dn[thdn] = 1
        # calculate binned STD of processed LFP (excluding noise indices)
        self.LFP_std_dn = downsample_sd(self.LFP_processed, length=self.M.shape[1],
                                         nbin=self.fbin, noise_idx=self.noise_idx, replace_nan=0)
        # enable option to calculate noise-excluded SP if EEG noise indices are marked
        calcAction = self.noiseCalcSP_btn.findChild(QtGui.QAction, 'calculate noise')
        calcAction.setEnabled(self.eeg_noise_idx.size > 0)
        
        
    ##################          DETECTING P-WAVES          ##################
        
    def process_LFP(self):
        """
        Load raw LFP(s) and filter/subtract to calculate the processed LFP
        """
        raw1 = np.array(self.LFP_raw)
        raw2 = np.array(self.LFP_raw2)
        
        # handle recordings with single LFP channel
        if all(raw1 == 0):
            if all(raw2 == 0):
                raise ValueError("### ERROR: No LFP files found")
            else:
                self.channel = '2'
        elif all(raw2 == 0):
            self.channel = '1'
        
        # filter LFP channels
        raw1 = sleepy.my_bpfilter(raw1, self.w0, self.w1)
        raw2 = sleepy.my_bpfilter(raw2, self.w0, self.w1)
        
        # for automatic LFP selection, determine "signal" vs. "reference" channel
        if self.channel == 'S' or self.channel == 'C':
            # get approx. number of P-waves on each channel (mean + 4*std)
            th1 = np.nanmean(raw1) + 4.0*np.nanstd(raw1)
            idx1 = pwaves.spike_threshold(raw1, th1)
            th2 = np.nanmean(raw2) + 4.0*np.nanstd(raw2)
            idx2 = pwaves.spike_threshold(raw2, th2)
            sig,ref = [raw1, raw2] if len(idx1) > len(idx2) else [raw2, raw1]
            if self.channel == 'S':
                self.LFP_processed = sig - ref
            elif self.channel == 'C':
                self.LFP_processed = sig
        # otherwise, select LFP or subtract signals
        elif self.channel == '1':
            self.LFP_processed = np.array(raw1)
        elif self.channel == '2':
            self.LFP_processed = np.array(raw2)
        elif self.channel == '1-2':
            self.LFP_processed = raw1 - raw2
        elif self.channel == '2-1':
            self.LFP_processed = raw2 - raw1
            
        # save processed LFP, add to LFP list
        so.savemat(os.path.join(self.ppath, self.name, 'LFP_processed.mat'), 
                   {'LFP_processed': self.LFP_processed})
        if all(self.LFP_list[0] == self.LFP_raw) or all(self.LFP_list[0] == self.LFP_raw2):
            self.LFP_list.insert(0,self.LFP_processed)
        else:
            self.LFP_list[0] = self.LFP_processed
        print('Done with processing LFP')
        
        
    def get_noise(self):
        """
        Find indices of noise in LFP/EMG/EEG signal
        """
        # detect noise for currently displayed Intan signal
        sig = self.noiseOptions[self.noiseBtns_index]
        if sig == 'LFP':
            data = self.LFP_processed; idx = self.noise_idx
        elif sig == 'EMG':
            data = self.EMG_list[0]; idx = self.emg_noise_idx
        elif sig == 'EEG':
            data = self.EEG_list[0]; idx = self.eeg_noise_idx
        
        if self.noise_thres_type == 'none':
            # no noise detection
            calc_noise_idx = np.array(())
        else:
            # detect by raw value
            if self.noise_thres_type == 'raw':
                nthres_up = float(self.noise_thres_up)
                nthres_dn = float(self.noise_thres_dn)
            # detect by percentile
            elif self.noise_thres_type == 'perc':
                nthres_up = np.percentile(data, self.noise_thres_up)
                nthres_dn = np.percentile(data, self.noise_thres_dn)
            calc_noise_idx = pwaves.detect_noise(data, self.sr, thres=[nthres_up, nthres_dn], 
                                                 win=self.noise_win, sep=self.noise_sep)
        
        if self.reset_noise:
            # delete previously detected/annotated noise, replace with new indices
            idx = calc_noise_idx.astype('int')
        else:
            # combine previously detected/annotated noise with new indices
            idx = np.sort(np.unique(np.append(idx, calc_noise_idx)))
            idx = np.array(idx, dtype='int')
        # update noise indices for the current signal
        if sig == 'LFP':
            self.noise_idx = idx
            self.LFP_std_dn = downsample_sd(self.LFP_processed, length=self.M.shape[1],
                                             nbin=self.fbin, noise_idx=idx, replace_nan=0)
        elif sig == 'EMG':
            self.emg_noise_idx = idx
        elif sig == 'EEG':
            self.eeg_noise_idx = idx
        # enable option to calculate noise-excluded SP if EEG noise indices are marked
        calcAction = self.noiseCalcSP_btn.findChild(QtGui.QAction, 'calculate noise')
        calcAction.setEnabled(self.eeg_noise_idx.size > 0)
        print('Done with finding noise')
        
        
    def get_thres_idx(self):
        """
        Get LFP indices to use for calculating P-wave detection threshold
        """
        # handle microarousals/transition states for thresholding purposes
        self.M_thres = AS.adjust_brainstate(self.M.flatten(), self.fdt, ma_thr=20, 
                                       ma_state=3, flatten_is=4)
        self.thres_idx = np.array(())
        self.thidx_raw = np.zeros(len(self.EEG))
        self.thidx_dn = np.zeros(self.M.shape[1])
        # state(s) to use for thresholding
        istate = [k for k in self.thres_states.keys() if self.thres_states[k][0]]
        # get downsampled indices of LFP noise
        noise = np.zeros(len(self.EEG))
        noise[self.noise_idx] = 1
        noise_dn = AS.downsample_vec(noise, int(self.fbin))
        noise_dn[np.where(noise_dn>0)[0]] = 1
        
        for s in istate:
            sseq = sleepy.get_sequences(np.where(self.M_thres==s)[0])
            # eliminate short brain state episodes
            sseq = [seq for seq in sseq if len(seq) >= (self.thres_states[3][1] / self.fdt)]
            if self.thres_first > 0:
                # get Intan indices of first X seconds in sequence
                sidx = [np.arange(seq[0]*self.fbin, seq[int(self.thres_first/self.fdt)]*self.fbin+self.fbin) for seq in sseq]
            else:
                # get Intan indices of entire sequence
                sidx = [np.arange(seq[0]*self.fbin, seq[-1]*self.fbin+self.fbin) for seq in sseq]
            sidx = np.concatenate(sidx)
            if np.where(sidx >= len(self.EEG))[0].size > 0:
                sidx = sidx[np.where(sidx < len(self.EEG))[0]]
            # eliminate bins with LFP noise
            sidx = np.setdiff1d(sidx, np.where(noise_dn==1)[0])
            self.thres_idx = np.concatenate((self.thres_idx, sidx))
        # if there are no qualifying threshold indices, use entire recording
        if self.thres_idx.size==0:
            print('### WARNING - No qualifying threshold indices found, using entire recording ...')
            self.thres_idx = np.arange(0,len(self.EEG))
        self.thres_idx = self.thres_idx.astype('int')
        # save threshold indices in Intan and FFT time for plotting
        self.thidx_raw[self.thres_idx] = 1
        thdn = np.round(self.thres_idx / self.fbin).astype('int')
        self.thidx_dn[thdn] = 1
        print('Done with finding threshold indices')
            
        
    def threshold_LFP(self):
        """
        Detect P-waves by thresholding LFP signal
        """
        # calculate detection threshold
        if self.thres_type == 'raw':
            self.p_thr = float(self.thres)
        elif self.thres_type == 'std':
            mn = np.nanmean(self.LFP_processed[self.thres_idx])
            std = np.nanstd(self.LFP_processed[self.thres_idx])
            self.p_thr = mn + self.thres*std
        elif self.thres_type == 'perc':
            self.p_thr = np.percentile(self.LFP_processed[self.thres_idx], self.thres)
        # get indices of waveforms crossing the calculated threshold
        self.pi = pwaves.spike_threshold(self.LFP_processed, self.p_thr)
        print('Done with finding P-waves')
    
    
    def validate_pwaves(self):
        """
        Eliminate non-P-wave artifacts by shape and size
        """
        # get amplitudes and half-widths of P-waves
        self.p_amps = [pwaves.get_amp(self.LFP_processed, i, self.sr) for i in self.pi]
        self.p_widths = [pwaves.get_halfwidth(self.LFP_processed, i, self.sr) for i in self.pi]
        df = pd.DataFrame({'idx':np.array(self.pi), 
                           'amp':np.array(self.p_amps), 
                           'halfwidth':np.array(self.p_widths)})
        df.dropna(inplace=True)
        self.pi = np.array(df['idx'])
        self.elim_idx = {'elim_amp'   : np.array((), dtype='int'),
                         'elim_width' : np.array((), dtype='int'),
                         'elim_dup'   : np.array((), dtype='int'),
                         'elim_noise' : np.array((), dtype='int')}
        
        # get amplitude elimination threshold
        if self.amp_thres_type != 'none':
            if self.amp_thres_type == 'raw':
                athres = float(self.amp_thres)
            elif self.amp_thres_type == 'perc':
                athres = np.percentile(np.array(self.p_amps), self.amp_thres)
            ea = np.array(df['idx'].iloc[np.where(df['amp'] > athres)[0]])
            self.elim_idx['elim_amp'] = ea.astype('int')
        
        # get half-width elimination threshold
        if self.hw_thres_type != 'none':
            if self.hw_thres_type == 'raw':
                hwthres = float(self.hw_thres)/1000.*self.sr
            elif self.hw_thres_type == 'perc':
                hwthres = np.percentile(np.array(self.p_widths), self.amp_thres)
            ew = np.array(df['idx'].iloc[np.where(df['halfwidth'] > hwthres)[0]])
            self.elim_idx['elim_width'] = ew.astype('int')
        
        # check whether closely neighboring P-waves are separate events
        pdifs = self.pi[1:] - self.pi[0:-1]
        tst_dups = np.where(pdifs < self.dup_win/1000.*self.sr)[0]
        dup_waves = []
        for di in tst_dups:
            p1 = self.pi[di]
            p2 = self.pi[di+1]
            # if first P-wave has already been classified as a duplicate, continue
            if p1 in dup_waves:
                continue
            else:  
                p1_mid = np.abs(self.LFP_processed[p1] - max(self.LFP_processed[p1:p2]))
                p2_mid = np.abs(self.LFP_processed[p2] - max(self.LFP_processed[p1:p2]))
                # if LFP deflects significantly upward between adjacent P-waves:
                if p1_mid > 100 and p2_mid > 75:
                    # count as separate waves
                    continue
                # if single wave was "double-detected"
                else:
                    # if 2nd threshold crossing is larger, classify 1st as duplicate
                    if self.LFP_processed[p2] < self.LFP_processed[p1]:
                        dup_waves.append(p1)
                    # if 1st threshold crossing is larger, classify 2nd as duplicate
                    elif self.LFP_processed[p1] < self.LFP_processed[p2]:
                        dup_waves.append(p2)
        self.elim_idx['elim_dup'] = np.array(dup_waves, dtype='int')
        
        # ignore waves within "noisy" LFP regions
        en = np.intersect1d(self.pi, self.noise_idx)
        self.elim_idx['elim_noise'] = en.astype('int')
        
        # eliminate all disqualified waves
        p_elim = np.concatenate(list(self.elim_idx.values()))
        self.pi = np.setdiff1d(self.pi, p_elim).astype('int')
        # calculate P-wave frequency
        if self.pi.size > 0:
            self.pfreq = pwaves.downsample_pwaves(self.LFP_processed, self.pi, 
                                                  self.sr, int(self.fbin), 
                                                  self.M.shape[1])[3]
        else:
            self.pfreq = np.zeros(self.M.shape[1])
        if len(self.pfreq) == self.M.shape[1]-1:
            self.pfreq = np.append(self.pfreq, 0.0)
        # find laser events (if recording includes laser-triggered P-waves)
        if self.lsrTrigPwaves:
            tmp = pwaves.get_lsr_pwaves(self.pi, self.laser_raw, 
                                        post_stim=self.post_stim, sr=self.sr)
            self.lsr_pi, self.spon_pi, self.success_lsr, self.fail_lsr = tmp
        # keep track of manually eliminated P-waves
        self.elim_user = np.array((), dtype='int')
        print('Done with validating P-waves')
    
    
    ##################           LIVE PLOTTING           ##################
        
        
    def plot_treck(self, scale=1):     
        """
        Plot overview data, annotated states, and current position in recording
        """
        # clear plot, load annotation history and current index
        self.graph_treck.clear()
        self.graph_treck.plot(self.ftime*scale, self.K*0.5, pen=(150,150,150))
        self.graph_treck.plot(self.ftime, np.zeros((self.ftime.shape[0],)), 
                              pen=pg.mkPen(width=10, color='w'))
        # plot currently annotated point
        self.graph_treck.plot([self.ftime[self.index]*scale + 0.5*self.fdt*scale], 
                              [0.0], pen=(0,0,0), symbolPen=(255,0,0), 
                              symbolBrush=(255, 0, 0), symbolSize=5)
        # plot laser overview
        if self.pplot_laser:
            self.graph_treck.plot(self.ftime*scale, self.laser, pen=(0,0,255))
        # plot online state detection overview
        if self.psuppl:
            self.graph_treck.plot(self.ftime*scale, self.suppl_treck*0.3, pen=(255,150,150))
        # plot overview of timepoints used for LFP threshold calculation
        if self.pplot_ithres:
            self.graph_treck.plot(self.ftime*scale, self.thidx_dn, pen=(240,159,60))
        # plot dark cycles
        for d in self.dark_cycle:
            a = int(d[0]/self.fdt)
            b = int(d[1]/self.fdt)
            self.graph_treck.plot(self.ftime[a:b+1], np.zeros((b-a+1,)), 
                                  pen=pg.mkPen(width=10, color=(100,100,100)))
        
        # set axis params
        self.graph_treck.setXLink(self.graph_spectrum.vb)
        self.graph_treck.vb.setMouseEnabled(x=True, y=False)
        limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 
                  'yMin': -1.1, 'yMax': 1.1}
        self.graph_treck.vb.setLimits(**limits)
        yax = self.graph_treck.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        yax.setLabel('Laser', units='', **labelStyle)
        yax.setTicks([[(0, ''), (1, '')]])
        xax = self.graph_treck.getAxis(name='bottom')
        xax.setTicks([[]])
    
    
    def plot_brainstate(self, scale=1):
        """
        Plot color-coded brain states
        """
        # clear plot, load annotation vector
        self.graph_brainstate.clear()
        self.image_brainstate = pg.ImageItem() 
        self.graph_brainstate.addItem(self.image_brainstate)       
        self.image_brainstate.setImage(self.M.T)
        # set time scale and color map
        self.image_brainstate.scale(self.fdt*scale,1)
        self.image_brainstate.setLookupTable(self.lut_brainstate)
        self.image_brainstate.setLevels([0, 7])
        
        # set axis params
        self.graph_brainstate.setXLink(self.graph_spectrum.vb)
        self.graph_brainstate.vb.setMouseEnabled(x=True, y=False)
        limits = {'xMin': -1*self.fdt*scale, 
                  'xMax': self.ftime[-1]*scale, 
                  'yMin': 0, 
                  'yMax': 1}
        self.graph_brainstate.vb.setLimits(**limits)
        yax = self.graph_brainstate.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        yax.setLabel('Brainstate', units='', **labelStyle)
        yax.setTicks([[(0, ''), (1, '')]])
        xax = self.graph_brainstate.getAxis(name='bottom')
        xax.setTicks([[]])
    
    
    def plot_session(self, scale=1, scale_unit='s'):
        """
        Plot EEG spectrogram and other FFT data for the recording session
            e.g. EEG spectrogram, EMG amplitude, P-wave frequency
        """
        # clear plot, load EEG spectrogram
        self.graph_spectrum.clear()
        self.image_spectrum = pg.ImageItem() 
        self.graph_spectrum.addItem(self.image_spectrum)
        self.image_spectrum.setImage(self.eeg_spec[0:self.ifreq[-1],:].T)
        # set time scale and color map
        self.image_spectrum.scale(self.fdt*scale, 1.0*self.fdx)
        self.image_spectrum.setLookupTable(self.lut_spectrum)
        # set axis params
        self.graph_spectrum.setXLink(self.graph_brainstate.vb)
        self.graph_spectrum.vb.setMouseEnabled(x=True, y=False)
        limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 
                  'yMin': 0, 'yMax': self.freq[self.ifreq[-1]]}
        self.graph_spectrum.vb.setLimits(**limits)
        ax = self.graph_spectrum.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('Freq', units='Hz', **labelStyle)
        ax.setTicks([[(0, '0'), (10, '10'), (20, '20')]])
        
        # plot data vector(s) in FFT time
        self.graph_emgampl.clear()
        l = 'EMG Ampl.'; u = 'V'
        # plot EMG amplitude
        if self.pplot_emgampl:
            self.graph_emgampl.plot(self.ftime*scale+scale*self.fdt/2.0, 
                                    self.EMGAmpl, pen=(255,255,255))
        # plot P-wave frequency
        if self.pplot_pfreq:
            self.graph_emgampl.plot(self.ftime*scale+scale*self.fdt/2.0, 
                                    self.pfreq*40, pen=(255,0,0))
            l = 'P-wave Freq'; u = ''
        # plot DF/F calcium signal
        if self.pplot_dff:
            self.graph_emgampl.plot(self.ftime*scale+scale*self.fdt/2.0, 
                                    self.dffdn, pen=(255,255,0))
            l  = 'DF/F Ampl.'; u = ''
        # plot standard deviation of LFP signal
        if self.pplot_LFPstd:
            self.graph_emgampl.plot(self.ftime*scale+scale*self.fdt/2.0, 
                                    self.LFP_std_dn, pen=(0,255,0))
            l = 'LFP S.D.'; u = 'V'
        # toggle through EEG frequency band powers
        if self.pplot_bandpwr:
            pwr = self.band_pwrs[self.band_pointer]
            pen = [(0,0,255),(0,255,0),(255,255,0),(255,0,255)][self.band_pointer]
            self.graph_emgampl.plot(self.ftime*scale+scale*self.fdt/2.0, pwr, pen=pen)
            l  = f'{list(self.bands.keys())[self.band_pointer].capitalize()} power'; u = ''
        
        # set axis params
        self.graph_emgampl.setXLink(self.graph_spectrum.vb)
        self.graph_emgampl.vb.setMouseEnabled(x=False, y=True)
        limits = {'xMin' : 0, 'xMax' : self.ftime[-1]*scale}
        self.graph_emgampl.vb.setLimits(**limits)
        yax = self.graph_emgampl.getAxis(name='left')
        yax.setLabel(l, units=u, **labelStyle)    
        xax = self.graph_emgampl.getAxis(name='bottom')
        xax.setLabel('Time', units=scale_unit, **labelStyle)
    
    
    def plot_eeg(self, findPwaves, findArtifacts):
        """
        Plot raw signal(s) and other Intan data for the currently visited timepoint. 
        The data window spans 5 FFT time bins, or ~12.5 s, centered on the current index
        
            Signals - EEG, EEG2, EMG, processed LFP, raw LFPs, DF/F, etc.
            Data    - laser pulses, P-wave indices, P-wave detection threshold, etc.
            
        The live data plot is updated following a number of user actions, including
        changing the viewed data signal, visiting a different timepoint, and annotating 
        a brain state. Some actions (i.e. changing a state from REM to wake) have no 
        effect on the "current" P-waves, since the same recording indices are being 
        plotted and no P-waves disappeared or were newly detected. For efficiency, 
        the "current" P-wave indices from the previous plot can be re-used as the new 
        "current" P-waves in the updated plot. 
        
        
        Fig. 1A - Current P-wave indices are [1,2,4] before AND after the plot update
        
        Index              *                                          *
                           
        P-waves    ____|___|_______|_                        _____|___|_______|_
                                             Annotation
        Signal     R   R   R   R   R      --------------->    R   R   R   R   W
        
        Idx        0   1   2   3   4                          0   1   2   3   4
        
        
        
        Other user actions (i.e. moving to a different timepoint in the recording) 
        require that a new set of "current" P-wave indices be found. Below, the
        user has shifted the current index two bins to the right, and the updated 
        sequence no longer includes the P-wave at index 1.

        
        Fig. 1B - The updated recording sequence has one fewer P-wave than the original
        
        Index              *                                          *
                           
        P-waves    ____|___|_______|_                        _|_______|_________ 
                                             Time shift
        Signal     R   R   R   R   R      --------------->    R   R   R   R   R
        
        Idx        0   1   2   3   4                          2   3   4   6   7
        
        
        Conversely, a user may remain at the same timepoint while changing the 
        P-wave detection threshold, resulting in a gain or loss of P-waves in the
        same time window after updating the graph. 
            When $plot_eeg is called after these actions, the $findPwaves parameter
        will be set to True, and the "current" P-waves will be found by taking the 
        intersection of all P-wace indices and the indices of the currently plotted 
        sequence. For actions that don't change the particular set of P-wave on the 
        graph, the $findPwaves parameter will be False to avoid the unnecessary
        calculation. The same logic applies to the $findArtifacts parameter, which
        keeps track of "artifact" waveforms which cross-the P-wave detection threshold
        but are invalidated due to shape/size abnormalities.
        """
        # if user moved to new recording index, update current plot indices
        if self.index != self.curPlotIndex:
            timepoint = self.ftime[self.index]+self.fdt  # current timepoint (s)
            self.twin = 2*self.fdt  # number of surrounding Intan samples to plot
        
            n = int(np.round((self.twin+self.fdt/2)/self.dt))
            i = int(np.round(timepoint / self.dt))       # timepoint idx in Intan signal
            self.ii = np.arange(i-n, i+n+1)              # Intan idx sequence
            self.tseq = np.arange(timepoint-n*self.dt, timepoint+n*self.dt+self.dt/2, self.dt)
            self.tseq = np.round(self.tseq, np.abs(int(np.floor(np.log10(self.dt)))))
            findPwaves = True
            findArtifacts = True
        self.eeg_amp = np.median(np.abs(self.EEG))
        
        # clear graph
        self.graph_eeg.clear()
        self.graph_eeg.setRange(xRange=(self.tseq[0],self.tseq[-1]), padding=None)
        
        # plot currently selected EEG/EMG/LFP signal
        self.curData.setData(self.tseq, self.EEG[self.ii], padding=None)
        self.graph_eeg.addItem(self.curData)
        
        # plot laser
        if self.pplot_laser and any(self.laser_raw != 0):
            self.curLaser.setData(self.tseq, self.laser_raw[self.ii]*self.eeg_amp*5)
            self.graph_eeg.addItem(self.curLaser)
        
        # plot DF/F signal
        if self.pplot_dff and any(self.dff != 0):
            self.curDFF.setData(self.tseq, self.dff[self.ii]*self.eeg_amp)
            self.graph_eeg.addItem(self.curDFF)
        
        # plot P-wave detection threshold
        if self.pplot_pthres and self.p_thr != 0 and self.pointers['LFP'] > -1:
            if self.plotSettings['thres_type'] == 'raw':
                txt = f'Threshold: -{self.plotSettings["thres"]} uV'
            elif self.plotSettings['thres_type'] == 'std':
                txt = f'Threshold: mean + {self.plotSettings["thres"]}*STD'
            elif self.plotSettings['thres_type'] == 'perc':
                txt = f'Threshold: {self.plotSettings["thres"]} percentile'
            self.curThres.setValue(-self.p_thr)
            self.curThres.label.setFormat(txt)
            self.graph_eeg.addItem(self.curThres)
        
        # highlight LFP signal used to calculate P-wave detection threshold
        if self.pplot_ithres and self.thres_idx.size > 0 and self.pointers['LFP'] > -1:
            self.curIThres.setData(self.tseq, self.thidx_raw[self.ii]*self.eeg_amp*7)
            self.graph_eeg.addItem(self.curIThres)
            
        # find P-wave/laser indices for time window
        if findPwaves:
            self.pi_seq = np.intersect1d(self.ii, self.pi)            # actual P-wave indices
            self.ti_seq = np.where(np.in1d(self.ii, self.pi_seq))[0]  # P-wave indices in current sequence
            if self.pi_seq.size > 0:
                self.pi_amps = self.EEG[self.pi_seq] - 25
            lsrp = np.concatenate((self.success_lsr, self.fail_lsr))
            self.iLsr_seq = np.intersect1d(self.ii, lsrp)                  # actual laser indices
            self.tiLsr_seq = np.where(np.in1d(self.ii, self.iLsr_seq))[0]  # laser indices in current sequence
        
        # find automatically/manually eliminated waveforms for time window
        if findArtifacts:
            qidx_seq = np.zeros((4,len(self.ii))).astype('int')
            keys = list(self.elim_idx.keys())
            for i,k in enumerate(keys):
                eidx = np.intersect1d(self.ii, self.elim_idx[k])  # actual waveform indices
                tidx = np.where(np.in1d(self.ii, eidx))[0]        # waveform indices in current sequence
                if eidx.size > 0:
                    qidx_seq[i,tidx] = eidx
            # find indices with at least one type of eliminated waveform
            cols = np.array([i for i in range(qidx_seq.shape[1]) if any(qidx_seq[:,i] != 0)])
            self.epi_seq =  np.array([max(qidx_seq[:,col]) for col in cols])   # actual waveform indices
            self.eti_seq = np.where(np.in1d(self.ii, self.epi_seq))[0]         # waveform indices in current sequence
            if self.epi_seq.size > 0:
                self.elim_pts = np.array([qidx_seq[:,i] for i in self.eti_seq])
            # find user-eliminated waves
            self.uepi_seq = np.intersect1d(self.ii, self.elim_user)       # actual waveform indices
            self.ueti_seq = np.where(np.in1d(self.ii, self.uepi_seq))[0]  # waveform indices in current sequence
            if self.uepi_seq.size > 0:
                self.uepi_amps = self.EEG[self.uepi_seq] - 25
            # find noise indices
            self.noise_seqs = sleepy.get_sequences(np.intersect1d(self.ii, self.noise_idx))
            self.tnoise_seqs = [np.where(np.in1d(self.ii, ns))[0] for ns in self.noise_seqs]
            self.emg_noise_seqs = sleepy.get_sequences(np.intersect1d(self.ii, self.emg_noise_idx))
            self.emg_tnoise_seqs = [np.where(np.in1d(self.ii, ens))[0] for ens in self.emg_noise_seqs]
            self.eeg_noise_seqs = sleepy.get_sequences(np.intersect1d(self.ii, self.eeg_noise_idx))
            self.eeg_tnoise_seqs = [np.where(np.in1d(self.ii, ens))[0] for ens in self.eeg_noise_seqs]
        
        # plot noise sequences
        if self.pplot_noise:
            if self.pointers['LFP'] != -1 and self.noise_seqs[0].size > 0:
                # LFP noise (deeppink)
                for tns in self.tnoise_seqs:
                    a = pg.PlotDataItem(self.tseq[tns], self.EEG[self.ii][tns], 
                                        pen=pg.mkPen((255,20,147,200),width=2))
                    self.graph_eeg.addItem(a)
            elif self.pointers['EMG'] != -1 and self.emg_noise_seqs[0].size > 0:
                # EMG noise (lightblue)
                for etns in self.emg_tnoise_seqs:
                    b = pg.PlotDataItem(self.tseq[etns], self.EEG[self.ii][etns], 
                                        pen=pg.mkPen((7,247,247,200),width=2))
                    self.graph_eeg.addItem(b)
            elif self.pointers['EEG'] != -1 and self.eeg_noise_seqs[0].size > 0:
                # EEG noise (gold)
                for eetns in self.eeg_tnoise_seqs:
                    c = pg.PlotDataItem(self.tseq[eetns], self.EEG[self.ii][eetns], 
                                        pen=pg.mkPen((252,186,3,200),width=2))
                    self.graph_eeg.addItem(c)
        
        # plot P-wave indices as dots (color-coded by brain state)
        if self.pplot_pidx and self.pi_seq.size > 0:
            self.pi_states = [int(self.M.flatten()[int(pi/self.fbin)]) for pi in self.pi_seq]
            pwaves = pg.PlotDataItem(self.tseq[self.ti_seq], self.pi_amps, 
                                     pen=None, symbol='o', symbolSize=15)
            brushes = [pg.mkColor(self.lut_brainstate[s]) for s in self.pi_states]
            pwaves.scatter.setBrush(brushes)
            pwaves.sigPointsClicked.connect(self.select_point)
            self.graph_eeg.addItem(pwaves)
        
        # plot P-wave triggering laser pulses as triangles (color-coded by success)
        if self.pplot_laser and self.lsrTrigPwaves and self.iLsr_seq.size > 0 and self.pi.size > 0:
            lsrpulses = pg.PlotDataItem(self.tseq[self.tiLsr_seq], 
                                        [self.eeg_amp*6+100]*len(self.iLsr_seq), 
                                        pen=None, symbol='t', symbolSize=10)
            brushes = []
            for l in self.iLsr_seq:
                if l in self.noise_idx:
                    brushes.append(pg.mkColor(220,220,220))  # lsr during noise = gray
                elif l in self.success_lsr:
                    brushes.append(pg.mkColor(167,240,10))   # successful lsr = green
                else:
                    brushes.append(pg.mkColor(242,157,169))  # failed lsr = red
            lsrpulses.scatter.setBrush(brushes)
            lsrpulses.sigPointsClicked.connect(self.select_point)
            self.graph_eeg.addItem(lsrpulses)

        # plot user-eliminated waveforms as stars (color-coded by brain state)
        if self.pplot_elim_user and self.uepi_seq.size > 0:
            self.uepi_states = [int(self.M.flatten()[int(uepi/self.fbin)]) for uepi in self.uepi_seq]
            user_elims = pg.PlotDataItem(self.tseq[self.ueti_seq], self.uepi_amps, 
                                         pen=None, symbol='star', symbolSize=15)
            brushes = [pg.mkColor(self.lut_brainstate[s]) for s in self.uepi_states]
            user_elims.scatter.setBrush(brushes)
            user_elims.sigPointsClicked.connect(self.select_point)
            self.graph_eeg.addItem(user_elims)
        
        # individually plot different eliminated waveform types
        if any(self.pplot_artifacts) and self.epi_seq.size > 0:
            artifacts = []
            for eti,epi,ei in zip(self.eti_seq, self.elim_pts, self.epi_seq):
                idx = np.intersect1d(np.where(epi!=0)[0], np.where(self.pplot_artifacts)[0])
                if len(idx) > 0:
                    elim_amps = [self.EEG[epi[j]] - 50 - 75*i for i,j in enumerate(idx)]
                    # A = amplitude outlier; W = half-width outlier; D = duplicate; N = noise artifact
                    symbols = np.array([pg_symbol(char) for char in ['A','W','D','N']])[idx]
                    a = pg.PlotDataItem([self.tseq[eti]]*len(elim_amps), elim_amps, 
                                        pen=None, symbol=symbols, symbolSize=12)
                    a.setSymbolPen((255,255,0),width=2)
                    a.sigPointsClicked.connect(self.select_point)
                    self.graph_eeg.addItem(a)
                    artifacts.append(a)
                else:
                    artifacts.append([])
        
        # check if currently selected point is still in plot window
        if self.curIdx in self.ii:
            # current point is a P-wave and P-waves are currently visible on graph
            if self.curIdx in self.pi_seq and self.pplot_pidx:
                i = int(np.where(self.pi_seq==self.curIdx)[0])
                self.curPoint = pwaves.scatter.points()[i]
                self.curItem = None
            # current point is a laser pulse and laser is currently visible on graph
            elif self.curIdx in self.iLsr_seq and self.pplot_laser and self.lsrTrigPwaves:
                i = int(np.where(self.iLsr_seq==self.curIdx)[0])
                self.curPoint = lsrpulses.scatter.points()[i]
                self.curItem = None
            # current point was manually eliminated and user-deleted waveforms are currently visible on graph
            elif self.curIdx in self.uepi_seq and self.pplot_elim_user:
                i = int(np.where(self.uepi_seq==self.curIdx)[0])
                self.curPoint = user_elim.scatter.points()[i]
                self.curItem = None
            # current point is a different artifact, which is currently visible on graph
            elif self.curIdx in self.epi_seq and any(self.pplot_artifacts):
                i = int(np.where(self.epi_seq==self.curIdx)[0])
                if artifacts[i] != []:
                    self.curPoint = artifacts[i].scatter.points()[0]
                    self.curItem = artifacts[i]
                else:
                    self.curPoint = None
                    self.curIdx = None
                    self.curItem = None
            else:
                print("How did you select a point that's not recognized as a waveform?")
        else:
            self.curPoint = None
            self.curIdx = None
            self.curItem = None
        
        # set red symbol pen for current point(s)
        if self.curItem is not None: 
            self.curItem.setSymbolPen((255,0,0),width=2)
        if self.curPoint is not None:
            self.curPoint.setPen((255,0,0),width=2)
        else:
            # reset color of single event plot buttons when point is cleared
            for name in self.singlePlotBtns:
                self.plotBtns_dict[name]['btn'].avg_mode()
                
        # plot user-interactive noise arrows
        if self.noiseStartIdx is not None or self.noiseEndIdx is not None:
            # show noise arrow indicators in top left of graph
            if self.show_arrowStart == True:
                self.arrowStart_btn.setPos(self.graph_eeg.vb.viewRange()[0][0]+0.2, 
                                           self.graph_eeg.vb.viewRange()[1][1]-200)
                self.graph_eeg.addItem(self.arrowStart_btn)
            if self.show_arrowEnd == True:
                self.arrowEnd_btn.setPos(self.graph_eeg.vb.viewRange()[0][0]+0.35, 
                                         self.graph_eeg.vb.viewRange()[1][1]-200)
                self.graph_eeg.addItem(self.arrowEnd_btn)
                
            # plot arrow marking noise start
            if self.noiseStartIdx in self.ii:
                opts = {'angle': -90, 'headLen':25, 'tipAngle':45, 'tailLen':25, 
                        'tailWidth':9, 'pen':pg.mkPen((255,255,255),width=2)}
                i = int(np.where(self.ii == self.noiseStartIdx)[0])
                arrow1 = pg.ArrowItem(pos=[self.tseq[i], self.EEG[self.ii][i]+150], **opts)
                if self.arrowStart_btn.active == True:
                    arrow1.setBrush(pg.mkBrush(255,255,255))
                else:
                    arrow1.setBrush(pg.mkBrush(0,0,0))
                self.graph_eeg.addItem(arrow1)
            # plot arrow marking noise end
            if self.noiseEndIdx in self.ii:
                opts = {'angle': -90, 'headLen':25, 'tipAngle':45, 'tailLen':25, 
                        'tailWidth':9, 'pen':pg.mkPen((255,255,255),width=2)}
                j = int(np.where(self.ii == self.noiseEndIdx)[0])
                arrow2 = pg.ArrowItem(pos=[self.tseq[j], self.EEG[self.ii][j]+150], **opts)
                if self.arrowEnd_btn.active == True:
                    arrow2.setBrush(pg.mkBrush(255,255,255))
                else:
                    arrow2.setBrush(pg.mkBrush(0,0,0))
                self.graph_eeg.addItem(arrow2)
        
        # plot user-selected segment of signal (yellow)
        if self.noiseStartIdx is not None and self.noiseEndIdx is not None:
            noiseSelectIdx = np.arange(self.noiseStartIdx, self.noiseEndIdx+1)
            ns_cand = np.intersect1d(self.ii, noiseSelectIdx)
            tns_cand = np.where(np.in1d(self.ii, ns_cand))[0]
            if ns_cand.size > 0:
                a = pg.PlotDataItem(self.tseq[tns_cand], self.EEG[self.ii][tns_cand], 
                                    pen=pg.mkPen((255,255,0),width=2.5))
                self.graph_eeg.addItem(a)
                
        # set axis parameters
        ax = self.graph_eeg.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel(self.ylabel, units='V', **labelStyle)
        ax = self.graph_eeg.getAxis(name='bottom')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('Time', units='s', **labelStyle)
        
        # save central index of current plot
        self.curPlotIndex = int(self.index)
    
    
    def switch_noise_boundary(self, arrow):
        """
        Update active ArrowButton for manually selecting segments of a plotted signal
        @Params
        arrow - ArrowButton object clicked by user
        
        See documentation for ArrowButton object in $gui_items.py for details
        """
        # clicked noise start indicator
        if arrow == self.arrowStart_btn:
            # noise end indicator exists
            if self.noiseEndIdx is not None:
                # if current noise boundary is end, switch to start
                if self.arrowStart_btn.active == False:
                    self.arrowStart_btn.active = True
                    self.arrowEnd_btn.active = False
                # if current noise boundary is start, bring start idx into plot window
                elif self.arrowStart_btn.active == True:
                    self.arrowEnd_btn.active = False
                    if self.noiseStartIdx not in self.ii:
                        self.index = int(round(self.noiseStartIdx/self.sr/self.fdt))
            # noise end indicator does not exist
            else:
                self.arrowStart_btn.active = False
                self.arrowEnd_btn.active = False
        # clicked noise end indicator
        elif arrow == self.arrowEnd_btn:
            # if current noise boundary is start, switch to end
            if self.arrowEnd_btn.active == False:
                self.arrowEnd_btn.active = True
                self.arrowStart_btn.active = False
            # if current noise boundary is end, bring end idx into plot window
            elif self.arrowEnd_btn.active == True:
                self.arrowStart_btn.active = False
                if self.noiseEndIdx not in self.ii:
                    self.index = int(round(self.noiseEndIdx / self.sr / self.fdt))
        # update arrow indicator fills, update EEG plots
        self.arrowStart_btn.update_style()
        self.arrowEnd_btn.update_style()
        self.plot_eeg(findPwaves=False, findArtifacts=False)
        self.plot_treck(self.tscale)
        

    def select_point(self, item, points):
        """
        Allow user to manually select a P-wave or laser pulse event
        """
        if len(points) > 1:
            print('Uh oh, you clicked on multiple points at once!')
            if self.curPoint:
                self.curPoint.setPen(None)
                if self.curItem: self.curItem.setSymbolPen(None)
            self.curIdx = None
            self.curPoint = None
            self.curItem = None
        else:
            pt = points[0]
            # clicked on currently selected point
            if self.curPoint and self.curPoint.pos()[0] == pt.pos()[0]:
                # remove red outline, deselect clicked point
                self.curPoint.setPen(None)
                if self.curItem:
                    self.curItem.setSymbolPen((255,255,0),width=2)
                self.curIdx = None
                self.curPoint = None
                self.curItem = None
            # clicked on a point not currently selected
            else:
                # deselect current point
                if self.curPoint:
                    self.curPoint.setPen(None)
                    if self.curItem:
                        self.curItem.setSymbolPen((255,255,0),width=2)
                # add red outline, select clicked point
                self.curPoint = pt
                # get current point index in relevant event sequence
                tidx = [np.where(self.tseq[x] == self.curPoint.pos()[0])[0] for x in [self.ti_seq, self.tiLsr_seq, self.eti_seq, self.ueti_seq]]
                seq,i = [(y,t) for (y,t) in zip([self.pi_seq, self.iLsr_seq, self.epi_seq, self.uepi_seq], tidx) if t.size==1][0]
                self.curIdx = int(seq[int(i)])
                if self.curIdx in self.epi_seq:
                    self.curItem = item
                else:
                    self.curItem = None
                self.curPoint.setPen((255,0,0),width=2)
                if self.curItem:
                    self.curItem.setSymbolPen((255,0,0),width=2)
        if self.curIdx is not None:
            # identify type of event selected
            if self.curIdx in self.lsr_pi:
                e = 'lsr_pi'
            elif self.curIdx in self.spon_pi:
                e = 'spon_pi'
            elif self.curIdx in self.success_lsr or self.curIdx in self.fail_lsr:
                e = 'lsr'
            else:
                e = 'other'
            # update color of single plot buttons for a selected point
            for name in self.singlePlotBtns:
                self.plotBtns_dict[name]['btn'].single_mode(e)
        else:
            # reset color of single plot buttons for averaging across events
            for name in self.singlePlotBtns:
                self.plotBtns_dict[name]['btn'].avg_mode()
        
        
    def show_hide_artifacts(self):
        """
        Allow user to view or hide indicators for each type of LFP artifact
        """
        # set show vs hide for Intan noise indices
        self.pplot_noise = self.noiseShow_btn[0].isChecked()
        # set show vs hide param for each artifact type
        self.pplot_artifacts = [val[0].isChecked() for val in self.elimView_btns.values()]
        self.pplot_elim_user = self.elimUser_btn[0].isChecked()
        # disable back/next buttons if artifact type is not plotted
        self.noiseShow_btn[1].setEnabled(self.pplot_noise)
        self.noiseShow_btn[2].setEnabled(self.pplot_noise)
        _ = [(val[1].setEnabled(b), val[2].setEnabled(b)) for val,b in zip(self.elimView_btns.values(), self.pplot_artifacts)]
        self.elimUser_btn[1].setEnabled(self.pplot_elim_user)
        self.elimUser_btn[2].setEnabled(self.pplot_elim_user)
        # update EEG graph
        self.plot_eeg(findPwaves=False, findArtifacts=False)
    
    
    def plot_next_artifact(self):
        """
        Allow user to scroll through each instance of an artifact or noise sequence
        """
        u = self.sender()
        key,direction = u.objectName().split(' ')
        if key == 'noise':
            # scroll through identified noise sequences in LFP/EMG/EEG signal
            sig = self.noiseOptions[self.noiseBtns_index]
            if sig == 'LFP':
                data = np.sort(self.noise_idx)
            elif sig == 'EMG':
                data = np.sort(self.emg_noise_idx)
            elif sig == 'EEG':
                data = np.sort(self.eeg_noise_idx)
        elif key == 'elim_user':
            # scroll through user-eliminated waveforms
            data = np.sort(self.elim_user)
        else:
            # scroll through amplitude outliers/half-width outliers/duplicates
            data = np.sort(self.elim_idx[key])
        
        if direction == 'back':
            # view previous artifact/noise sequence in recording
            iseq = np.where(data < self.ii[0])[0]
            if iseq.size > 0:
                i = data[iseq[-1]]
                self.index = int(round(i/self.sr/self.fdt))
                self.plot_treck()
                self.plot_eeg(findPwaves=True, findArtifacts=True)
                self.plot_session()
        elif direction == 'next':
            # view next artifact/noise sequence in recording
            iseq = np.where(data > self.ii[-1])[0]
            if iseq.size > 0:
                i = data[iseq[0]]
                self.index = int(round(i/self.sr/self.fdt))
                self.plot_treck()
                self.plot_eeg(findPwaves=True, findArtifacts=True)
                self.plot_session()
                
                
    def more_plot_btns(self):
        """
        Allow user to access buttons for plotting different figures 
        """
        u = self.sender()
        # hide current plot buttons
        self.plotBtn_widgets[self.plotBtns_index].hide()
        # show previous set of buttons
        if u.objectName() == 'plotBtns back' and self.plotBtns_index > 0: 
            self.plotBtns_index -= 1
        # show next set of buttons
        elif u.objectName() == 'plotBtns next' and self.plotBtns_index < len(self.plotBtn_widgets)-1:
            self.plotBtns_index += 1
        self.plotBtn_widgets[self.plotBtns_index].show()
        
        # update back/forward arrow icons for plot buttons
        sz = max(self.plotBtns_bck.iconSize().width(), self.plotBtns_nxt.iconSize().width())
        if self.plotBtns_index == 0:
            self.plotBtns_bck.setIconSize(QtCore.QSize(0,0))
        else:
            self.plotBtns_bck.setIconSize(QtCore.QSize(sz,sz))
        if self.plotBtns_index == len(self.plotBtn_widgets)-1:
            self.plotBtns_nxt.setIconSize(QtCore.QSize(0,0))
        else:
            self.plotBtns_nxt.setIconSize(QtCore.QSize(sz,sz))
    
    
    def more_noise_btns(self):
        """
        Allow user to change noise detection parameters for different Intan signals
        """
        # shift index 'back' to previous signal type or to 'next' signal type
        if self.sender() in [self.noiseBtns_bck, self.noiseBtns_nxt]:
            bdir = self.sender().objectName().split(' ')[1]
            if bdir == 'back' and self.noiseBtns_index > 0:
                self.noiseBtns_index -= 1
            elif bdir == 'next' and self.noiseBtns_index < len(self.noiseOptions)-1:
                self.noiseBtns_index += 1
        
        # get string for signal type currently being modified by noise params
        signal = self.noiseOptions[self.noiseBtns_index]
        self.noiseTitle4.setText(signal + ' Noise Detection')
        sz = max(self.noiseBtns_bck.iconSize().width(), self.noiseBtns_nxt.iconSize().width())
        bsz = 0 if self.noiseBtns_index == 0 else sz
        nsz = 0 if self.noiseBtns_index == len(self.noiseOptions)-1 else sz
        self.noiseBtns_bck.setIconSize(QtCore.QSize(bsz, bsz))
        self.noiseBtns_bck.setEnabled(bool(bsz))
        self.noiseBtns_nxt.setIconSize(QtCore.QSize(nsz, nsz))
        self.noiseBtns_nxt.setEnabled(bool(nsz))
        self.noiseCalcSP_btn.setVisible(signal=='EEG')
        
        # set current data & pointers for all signal types
        data_lists = [self.LFP_list, self.EMG_list, self.EEG_list]
        for i,opt in enumerate(self.noiseOptions):
            if opt==signal: 
                # if switching from other signal, choose first signal in $opt list
                if self.pointers[opt] == -1:
                    self.pointers[opt] = 0
                # set chosen signal as $EEG data
                self.EEG = data_lists[i][self.pointers[opt]]
                self.ylabel = opt + ' ' + str(self.pointers[opt]+1)
                if opt=='EEG':
                    self.eeg_spec = self.eeg_spec_list[self.pointers[opt]]
                elif opt=='EMG':
                    self.EMGAmpl = self.EMGAmpl_list[self.pointers[opt]]
                elif opt=='LFP':
                    if self.pointers[opt] == 0:
                        self.ylabel = 'LFP' if len(self.LFP_list) > 2 else 'Raw LFP 1'
                    else:
                        tmp = 0 if len(self.LFP_list) > 2 else 1
                        self.ylabel = 'Raw LFP ' + str(self.pointers[opt]+tmp)
            else:
                # set pointers for all other signals to 0
                self.pointers[opt] = -1

        # update plot
        self.plot_eeg(findPwaves=False, findArtifacts=False)
        self.plot_session(scale=self.tscale, scale_unit=self.tunit)
    
    
    def keyPressEvent(self, event):
        """
        Allow user to show/hide/shift/annotate data with keyboard inputs
        """
        ###   VIEW   ###
        
        # E - show EEG/switch EEG channel
        if event.key() == QtCore.Qt.Key_E:
            num_eeg = len(self.EEG_list)
            if self.pointers['EEG'] < num_eeg-1:
                self.pointers['EEG'] += 1               
            else:
                self.pointers['EEG'] = 0
            self.noiseBtns_index = 2
            self.more_noise_btns()
        
        # M - show EMG/switch EMG channel
        elif event.key() == QtCore.Qt.Key_M:
            num_emg = len(self.EMG_list)
            if self.pointers['EMG'] < num_emg-1:
                self.pointers['EMG'] += 1               
            else:
                self.pointers['EMG'] = 0
            self.noiseBtns_index = 1
            self.more_noise_btns()
        
        # L - show LFP/switch LFP channel
        elif event.key() == QtCore.Qt.Key_L:
            if len(self.LFP_list) > 0:                                
                num_lfp = len(self.LFP_list)
                if self.pointers['LFP'] < num_lfp-1:
                    self.pointers['LFP'] += 1
                else:
                    self.pointers['LFP'] = 0           
                self.noiseBtns_index = 0
                self.more_noise_btns()
        
        # T - show/hide (t)hreshold for detecting P-waves
        elif event.key() == QtCore.Qt.Key_T:
            if self.pplot_pthres == True:
                self.pplot_pthres = False
            else:
                self.pplot_pthres = True
            self.plot_eeg(findPwaves=False, findArtifacts=False)
        
        # P - show/hide detected [P]-waves
        elif event.key() == QtCore.Qt.Key_P:
            if self.pplot_pidx == True:
                self.pplot_pidx = False
            else:
                self.pplot_pidx = True
            self.plot_eeg(findPwaves=False, findArtifacts=False)
        
        # O - show/hide [o]ptogenetic laser stimulation
        elif event.key() == QtCore.Qt.Key_O:
            if self.pplot_laser == True:
                self.pplot_laser = False
            else:
                self.pplot_laser = True
            self.plot_eeg(findPwaves=False, findArtifacts=False)
            self.plot_treck(self.tscale)
        
        # U - show/hide [u]nderlying points for P-wave threshold calculation
        elif event.key() == QtCore.Qt.Key_U:
            if self.pplot_ithres == True:
                self.pplot_ithres = False
            else:
                self.pplot_ithres = True
            self.plot_eeg(findPwaves=False, findArtifacts=False)
            self.plot_treck(self.tscale)
        
        # A - show/hide EMG [a]mplitude
        elif event.key() == QtCore.Qt.Key_A:
            self.band_pointer = -1
            self.pplot_bandpwr = False
            if self.pplot_emgampl == True:
                self.pplot_emgampl = False
            else:
                self.pplot_emgampl = True
            self.plot_session()
        
        # D - show/hide LFP standard [d]eviation
        elif event.key() == QtCore.Qt.Key_D:
            self.band_pointer = -1
            self.pplot_bandpwr = False
            if self.pplot_LFPstd == True:
                self.pplot_LFPstd = False
            else:
                self.pplot_LFPstd = True
            self.plot_session()
        
        # F - show/hide P-wave [f]requency
        elif event.key() == QtCore.Qt.Key_F:
            self.band_pointer = -1
            self.pplot_bandpwr = False
            if self.pplot_pfreq == True:
                self.pplot_pfreq = False
            else:
                self.pplot_pfreq = True
            self.plot_session()
        
        # G - show/hide [g]camp6 calcium signal
        elif event.key() == QtCore.Qt.Key_G:
            self.band_pointer = -1
            self.pplot_bandpwr = False
            if self.pplot_dff == True:
                self.pplot_dff = False
            else:
                self.pplot_dff = True
            self.plot_session()
            self.plot_eeg(findPwaves=False, findArtifacts=False)
        
        # B - show EEG frequency [b]and power/switch frequency bands
        elif event.key() == QtCore.Qt.Key_B:
            if len(self.band_pwrs) > 0:
                self.pplot_bandpwr = True
                self.pplot_emgampl = False
                self.pplot_pfreq = False
                self.pplot_dff = False
                if self.band_pointer < len(self.band_pwrs)-1:
                    self.band_pointer += 1
                else:
                    self.band_pointer = 0
                self.plot_session()
            
        ###   BRAIN STATE ANNOTATION   ###
        
        # R - REM
        elif event.key() == QtCore.Qt.Key_R:            
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 1
            self.index_list = [self.index]
            self.pcollect_index = False
            self.plot_brainstate(self.tscale)
            self.plot_eeg(findPwaves=False, findArtifacts=False)
        
        # W - Wake
        elif event.key() == QtCore.Qt.Key_W:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 2
            self.index_list = [self.index]
            self.pcollect_index = False
            self.plot_eeg(findPwaves=False, findArtifacts=False)
            self.plot_brainstate(self.tscale)
        
        # N - NREM
        elif event.key() == QtCore.Qt.Key_N:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 3
            self.index_list = [self.index]
            self.pcollect_index = False
            self.plot_eeg(findPwaves=False, findArtifacts=False)
            self.plot_brainstate(self.tscale)
        
        # I - intermediate/transition sleep
        elif event.key() == QtCore.Qt.Key_I:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()]=4
            self.index_list = [self.index]
            self.pcollect_index = False
            self.plot_eeg(findPwaves=False, findArtifacts=False)
            self.plot_brainstate(self.tscale)
            
        # J - failed transition sleep
        elif event.key() == QtCore.Qt.Key_J:
            self.M_old = self.M.copy()
            self.M[0,self.index_range()] = 5
            self.index_list = [self.index]
            self.pcollect_index = False
            self.plot_eeg(findPwaves=False, findArtifacts=False)
            self.plot_brainstate(self.tscale)
        
        ###   SIGNAL/EVENT ANNOTATION   ###
        
        # 0 - manually eliminate P-wave
        elif event.key() == QtCore.Qt.Key_0:
            if self.curPoint and self.curIdx:
                if self.curIdx in self.pi:
                    print('deleted point ' + str(self.curIdx))
                    # remove idx from $self.pi, add to $self.elim_user
                    self.pi = np.delete(self.pi, np.where(self.pi==self.curIdx)[0])
                    self.elim_user = np.sort(np.append(self.elim_user, int(self.curIdx)))
                    # calculate new P-wave frequency
                    if self.pi.size > 0:
                        self.pfreq = pwaves.downsample_pwaves(self.LFP_processed, self.pi, 
                                                              self.sr, int(self.fbin), 
                                                              self.M.shape[1])[3]
                    else:
                        self.pfreq = np.zeros(self.M.shape[1])
                    if len(self.pfreq) == self.M.shape[1]-1:
                        self.pfreq = np.append(self.pfreq, 0.0)
                    # update list of laser-triggered P-waves
                    if self.lsrTrigPwaves:
                        tmp = pwaves.get_lsr_pwaves(self.pi, self.laser_raw, 
                                                    post_stim=self.post_stim, 
                                                    sr=self.sr)
                        self.lsr_pi, self.spon_pi, self.success_lsr, self.fail_lsr = tmp
                    # clear current point
                    self.curPoint = None
                    self.curIdx = None
                    self.plot_eeg(findPwaves=True, findArtifacts=True)
                    self.plot_session()
        
        # 9 - manually convert artifact waveform to P-wave
        elif event.key() == QtCore.Qt.Key_9:
            if self.curPoint and self.curIdx:
                if any([self.curIdx in x for x in self.elim_idx.values()]):
                    # remove idx from elimination dictionary
                    for k in self.elim_idx.keys():
                        if self.curIdx in self.elim_idx[k]:
                            self.elim_idx[k] = np.delete(self.elim_idx[k], 
                                                         np.where(self.elim_idx[k]==self.curIdx)[0])
                elif self.curIdx in self.elim_user:
                    # remove idx from $self.elim_user
                    self.elim_user = np.delete(self.elim_user, 
                                               np.where(self.elim_user==self.curIdx)[0])
                else:
                    return
                # add idx to $self.pi
                self.pi = np.sort(np.append(self.pi, int(self.curIdx)))
                # calculate new P-wave frequency
                if self.pi.size > 0:
                    self.pfreq = pwaves.downsample_pwaves(self.LFP_processed, 
                                                          self.pi, self.sr, int(self.fbin), 
                                                          self.M.shape[1])[3]
                else:
                    self.pfreq = np.zeros(self.M.shape[1])
                if len(self.pfreq) == self.M.shape[1]-1:
                    self.pfreq = np.append(self.pfreq, 0.0)
                # update list of laser-triggered P-waves
                if self.lsrTrigPwaves:
                    tmp = pwaves.get_lsr_pwaves(self.pi, self.laser_raw, 
                                                post_stim=self.post_stim, sr=self.sr)
                    self.lsr_pi, self.spon_pi, self.success_lsr, self.fail_lsr = tmp
                # clear current point
                self.curPoint = None
                self.curIdx = None
                self.plot_eeg(findPwaves=True, findArtifacts=True)
                self.plot_session()
        
        # X/C - annotate selected signal as noise/clean
        elif event.key() == QtCore.Qt.Key_X or event.key() == QtCore.Qt.Key_C:
            if self.noiseStartIdx is not None and self.noiseEndIdx is not None:
                noiseSelectIdx = np.arange(self.noiseStartIdx, self.noiseEndIdx+1)
                # annotate selected sequence as "noise"
                if event.key() == QtCore.Qt.Key_X:
                    if self.pointers['LFP'] != -1:
                        # add sequence to noise indices
                        self.noise_idx = np.sort(np.unique(np.append(self.noise_idx, 
                                                                     noiseSelectIdx)))
                        # remove P-waves in selected sequence from $self.pi
                        pi = np.intersect1d(self.pi, noiseSelectIdx)
                        self.pi = np.sort(np.setdiff1d(self.pi, pi))
                        # annotate all waveforms in selected sequence as "noise"
                        ielims = list(self.elim_idx.values())
                        ei = np.concatenate([np.intersect1d(x, noiseSelectIdx) for x in ielims])
                        w = np.concatenate((pi,ei))
                        ni = np.unique(np.append(self.elim_idx['elim_noise'], w))
                        self.elim_idx['elim_noise'] = np.sort(ni).astype('int')
                        # calculate new P-wave frequency
                        if self.pi.size > 0:
                            self.pfreq = pwaves.downsample_pwaves(self.LFP_processed, self.pi, 
                                                                  self.sr, int(self.fbin), 
                                                                  self.M.shape[1])[3]
                        else:
                            self.pfreq = np.zeros(self.M.shape[1])
                        if len(self.pfreq) == self.M.shape[1]-1:
                            self.pfreq = np.append(self.pfreq, 0.0)
                        # update list of laser-triggered P-waves
                        if self.lsrTrigPwaves:
                            tmp = pwaves.get_lsr_pwaves(self.pi, self.laser_raw, 
                                                        post_stim=self.post_stim, sr=self.sr)
                            self.lsr_pi, self.spon_pi, self.success_lsr, self.fail_lsr = tmp
                    elif self.pointers['EMG'] != -1:
                        # add sequence to EMG noise, don't change P-wave annotation
                        self.emg_noise_idx = np.sort(np.unique(np.append(self.emg_noise_idx, noiseSelectIdx)))
                    elif self.pointers['EEG'] != -1:
                        # add sequence to EEG noise, don't change P-wave annotation
                        self.eeg_noise_idx = np.sort(np.unique(np.append(self.eeg_noise_idx, noiseSelectIdx)))
                        # enable option to calculate noise-excluded SP if EEG noise indices are marked
                        calcAction = self.noiseCalcSP_btn.findChild(QtGui.QAction, 'calculate noise')
                        calcAction.setEnabled(self.eeg_noise_idx.size > 0)
                # annotate selected sequence as "clean"
                elif event.key() == QtCore.Qt.Key_C:
                    if self.pointers['LFP'] != -1:
                        # remove sequence from noise indices
                        self.noise_idx = np.sort(np.setdiff1d(self.noise_idx, noiseSelectIdx))
                        # remove waveforms in selected sequence from "noise" list
                        nni = np.intersect1d(self.elim_idx['elim_noise'], noiseSelectIdx)
                        ni = np.setdiff1d(self.elim_idx['elim_noise'], nni)
                        self.elim_idx['elim_noise'] = np.sort(ni).astype('int')
                        # check if noise waveforms were eliminated by amp/width/dup/user
                        awd = np.concatenate([val for val in list(self.elim_idx.values())[0:3]])
                        ci = np.setdiff1d(ni, np.concatenate((awd,self.elim_user)))
                        # add clean waveforms to $self.pi
                        self.pi = np.sort(np.append(self.pi, ci))
                        # calculate new P-wave frequency
                        if self.pi.size > 0:
                            self.pfreq = pwaves.downsample_pwaves(self.LFP_processed, self.pi, 
                                                                  self.sr, int(self.fbin), 
                                                                  self.M.shape[1])[3]
                        else:
                            self.pfreq = np.zeros(self.M.shape[1])
                        if len(self.pfreq) == self.M.shape[1]-1:
                            self.pfreq = np.append(self.pfreq, 0.0)
                        # update list of laser-triggered P-waves
                        if self.lsrTrigPwaves:
                            tmp = pwaves.get_lsr_pwaves(self.pi, self.laser_raw, 
                                                        post_stim=self.post_stim, sr=self.sr)
                            self.lsr_pi, self.spon_pi, self.success_lsr, self.fail_lsr = tmp
                    elif self.pointers['EMG'] != -1:
                        # remove sequence from EMG noise, don't change P-wave annotation
                        self.emg_noise_idx = np.sort(np.setdiff1d(self.emg_noise_idx, noiseSelectIdx))
                    elif self.pointers['EEG'] != -1:
                        # remove sequence from EEG noise, don't change P-wave annotation
                        self.eeg_noise_idx = np.sort(np.setdiff1d(self.eeg_noise_idx, noiseSelectIdx))
                        # enable option to calculate noise-excluded SP if EEG noise indices are marked
                        calcAction = self.noiseCalcSP_btn.findChild(QtGui.QAction, 'calculate noise')
                        calcAction.setEnabled(self.eeg_noise_idx.size > 0)
                # reset all noise boundaries
                self.noiseStartIdx = None
                self.arrowStart_btn.active = False
                self.show_arrowStart = False
                self.noiseEndIdx = None
                self.arrowEnd_btn.active = False
                self.show_arrowEnd = False
                self.plot_eeg(findPwaves=True, findArtifacts=True)
                self.plot_session()
        
        ###   ACTIONS   ###
        
        # cursor right - shift current index one bin right
        elif event.key() == QtCore.Qt.Key_Right:
            if self.index < self.num_bins-5:
                self.index += 1
            self.K[self.index] = 1
            self.plot_eeg(findPwaves=True, findArtifacts=True)
            self.plot_treck(self.tscale)
            if self.pcollect_index == 1:
                self.index_list.append(self.index)
            else:
                self.index_list = [self.index]
        
        # cursor left - shift current index one bin left
        elif event.key() == QtCore.Qt.Key_Left:
            if self.index >= 3:
                self.index -= 1
            self.K[self.index] = 1
            self.plot_eeg(findPwaves=True, findArtifacts=True)
            self.plot_treck(self.tscale)
            if self.pcollect_index == True:
                self.index_list.append(self.index)
            else:
                self.index_list = [self.index]
                
        # space - starting from current bin, collect indices visited with cursor
        elif event.key() == QtCore.Qt.Key_Space:
            self.pcollect_index = True
            self.index_list = [self.index]
        
        # cursor down - brighten spectrogram
        elif event.key() == QtCore.Qt.Key_Down:
            self.color_max -= self.color_max/10
            self.image_spectrum.setLevels((0, self.color_max))
        
        # cursor up - darken spectrogram
        elif event.key() == QtCore.Qt.Key_Up:
            self.color_max += self.color_max/10
            self.image_spectrum.setLevels((0, self.color_max))
        
        # 1 - seconds scale    
        elif event.key() == QtCore.Qt.Key_1:
            self.tscale = 1.0 
            self.tunit = 's'
            self.plot_session(scale=1, scale_unit='s')
            self.plot_brainstate(scale=1)
            self.plot_treck(scale=1)
            
        # 2 - minutes scale
        elif event.key() == QtCore.Qt.Key_2:
            self.tscale = 1/60.0 
            self.tunit = 'min'
            self.plot_session(scale=1/60.0, scale_unit='min')
            self.plot_brainstate(scale=1/60.0)
            self.plot_treck(scale=1/60.0)

        # 3 - hours scale
        elif event.key() == QtCore.Qt.Key_3:
            self.tscale = 1/3600.0 
            self.tunit = 'h'
            self.plot_session(scale=1/3600.0, scale_unit='h')
            self.plot_brainstate(scale=1/3600.0)
            self.plot_treck(scale=1/3600.0)  
        
        # s - save file
        elif event.key() == QtCore.Qt.Key_S:    
            rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
            self.plot_brainstate(self.tscale)
            self.plot_eeg(findPwaves=False, findArtifacts=False)
        
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
            or self.graph_treck.sceneBoundingRect().contains(pos)      \
            or self.graph_emgampl.sceneBoundingRect().contains(pos):
                mousePoint = self.graph_spectrum.vb.mapSceneToView(pos)
                # convert clicked point to recording index
                self.index = int(mousePoint.x()/(self.fdt*self.tscale))
                if self.pcollect_index == True:
                    self.index_list.append(self.index)
                else:
                    self.index_list = [self.index]
                # update EEG plot/currently visited timepoint
                self.plot_eeg(findPwaves=True, findArtifacts=True)
                self.plot_treck(self.tscale)
    
    
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
    
    
    def openFileNameDialog(self):
        """
        Allow user to choose recording folder in computer
        """
        fileDialog = QtWidgets.QFileDialog(self)
        fileDialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        name = fileDialog.getExistingDirectory(self, "Choose Recording Directory")
        (self.ppath, self.name) = os.path.split(name)        
        print("Setting base folder %s and recording %s" % (self.ppath, self.name))


    @QtCore.pyqtSlot()
    def load_recording(self):
        """
        Load recording data, including raw EEG/EMG/LFP/DFF signals, EEG spectrogram, 
        EMG amplitude, brain state annotation, laser stimulation train, and other 
        experiment info (sampling rate, light/dark cycles, etc)
        """
        # user selects recording folder in computer
        if self.name == '':
            self.openFileNameDialog()
        # set title for window
        self.setWindowTitle(self.name)
        
        # create lists to hold LFP/EMG/EEG signals
        self.EEG_list = []  
        self.EMG_list = []
        self.LFP_list = []
        self.EMGAmpl_list = []
        self.eeg_spec_list = []
        # initalize pointers to keep track of currently viewed signal
        self.pointers = {'LFP':0, 'EMG':-1, 'EEG':-1}           
        
        # load EEG1 and EMG1
        EEG1 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG.mat'))['EEG'])
        self.EEG_list.append(EEG1)
        if os.path.isfile(os.path.join(self.ppath, self.name, 'EMG.mat')):
            EMG1 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EMG.mat'))['EMG'])
            self.hasEMG = True
        else:
            EMG1 = np.zeros(len(EEG1))
        self.EMG_list.append(EMG1)
        # if existing, also load EEG2 and EMG2
        if os.path.isfile(os.path.join(self.ppath, self.name, 'EEG2.mat')):
            EEG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG2.mat'))['EEG2'])
            self.EEG_list.append(EEG2)
        if os.path.isfile(os.path.join(self.ppath, self.name, 'EMG2.mat')):
            EMG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 
                                                      'EMG2.mat'))['EMG2'])
            self.EMG_list.append(EMG2)
        
        # load raw LFPs
        if os.path.exists(os.path.join(self.ppath, self.name, 'LFP_raw.mat')):
            self.LFP_raw = so.loadmat(os.path.join(self.ppath, self.name, 'LFP_raw'), 
                                      squeeze_me=True)['LFP_raw']
            self.LFP_list.append(self.LFP_raw)
        else:
            self.LFP_raw = np.zeros(len(EEG1))
        if os.path.exists(os.path.join(self.ppath, self.name, 'LFP_raw2.mat')):
            self.LFP_raw2 = so.loadmat(os.path.join(self.ppath, self.name, 'LFP_raw2'), 
                                       squeeze_me=True)['LFP_raw2']
            self.LFP_list.append(self.LFP_raw2)
        else:
            self.LFP_raw2 = np.zeros(len(EEG1))
        # load processed LFP if it exists
        if os.path.exists(os.path.join(self.ppath, self.name, 'LFP_processed.mat')):
            self.LFP_processed = so.loadmat(os.path.join(self.ppath, self.name, 
                                                         'LFP_processed'), squeeze_me=True)['LFP_processed']
            self.LFP_list.insert(0, self.LFP_processed)
        else:
            self.LFP_processed = np.zeros(len(EEG1))
        self.ylabel = 'LFP' if len(self.LFP_list)>2 else 'LFP 1'
        # load all other available LFP files
        lfp_files = [f for f in os.listdir(os.path.join(self.ppath, self.name)) if re.match('^LFP', f)]
        lfp_files.sort()
        lfp_files = [f for f in lfp_files if f not in ['LFP_raw.mat', 'LFP_raw2.mat', 'LFP_processed.mat']]
        if len(lfp_files) > 0:
            for f in lfp_files:
                key = re.split('\\.', f)[0]
                LFP = so.loadmat(os.path.join(self.ppath, self.name, f), squeeze_me=True)[key]
                self.LFP_list.append(LFP)
        if len(self.LFP_list) > 0:
            self.EEG = self.LFP_list[0]  # EEG/EMG/LFP signal currently plotted
        else:
            self.EEG = self.EEG_list[0]
            self.pointers['LFP'] = -1
            self.pointers['EEG'] = 0
            self.ylabel = 'EEG 1'
        # median of Intan signal to scale the laser signal
        self.eeg_amp = np.median(np.abs(self.EEG))
        
        # load EEG spectrogram, set max color
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))):
            # if spectrogram does not exist, generate it
            sleepy.calculate_spectrum(self.ppath, self.name, fres=0.5)
            print("Calculating spectrogram for recording %s\n" % self.name)
        spec = so.loadmat(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))
        self.eeg_spec_list.append(spec['SP'])
        if 'SP2' in spec:
            self.eeg_spec_list.append(spec['SP2'])
        self.eeg_spec = self.eeg_spec_list[0]
        self.color_max = np.max(self.eeg_spec)
        # enable option to load noise-excluded SP if file exists
        npath = os.path.join(self.ppath, self.name, f'sp_nannoise_{self.name}.mat')
        loadAction = self.noiseCalcSP_btn.findChild(QtGui.QAction, 'load noise')
        loadAction.setEnabled(os.path.exists(npath))
        
        self.ftime = np.squeeze(spec['t'])        # vector of SP timepoints
        self.fdt = float(np.squeeze(spec['dt']))  # SP time resolution (e.g. 2.5)
        self.freq = np.squeeze(spec['freq'])      # vector of SP frequencies
        self.fdx = self.freq[1] - self.freq[0]    # SP frequency resolution (e.g. 0.5)
        self.ifreq = np.where(self.freq<=25)[0]   # freqs to show in EEG spectrogram
        self.mfreq = np.where((self.freq>=10) &   # freqs to use in EMG amplitude calculation
                              (self.freq<=500))[0]
        
        # calculate power of frequency bands
        self.band_pointer = -1
        self.bands = {'delta':[0.5, 4.5], 'sigma':[10, 15], 
                      'theta':[6, 9.5], 'beta':[15.5, 20]}
        self.band_pwrs = []
        for b in self.bands.values():
            f1 = np.where(self.freq >= b[0])[0]
            f2 = np.where(self.freq <= b[1])[0]
            ifreq = np.intersect1d(f1,f2)
            fpwr = np.nansum(self.eeg_spec_list[0][ifreq,:], axis=0)
            self.band_pwrs.append(fpwr)
        
        # load EMG spectrogram, calculate amplitude
        self.emg_spec = so.loadmat(os.path.join(self.ppath, self.name, 'msp_' + self.name + '.mat'))
        EMGAmpl1 = np.sqrt(self.emg_spec['mSP'][self.mfreq,:].sum(axis=0))
        self.EMGAmpl_list.append(EMGAmpl1)
        if 'mSP2' in self.emg_spec:
            EMGAmpl2 = np.sqrt(self.emg_spec['mSP2'][self.mfreq,:].sum(axis=0))
            self.EMGAmpl_list.append(EMGAmpl2)
        self.EMGAmpl = self.EMGAmpl_list[0]
        
        # set time bins, sampling rates etc.
        self.num_bins = len(self.ftime)             # no. SP time bins
        self.sr = get_snr(self.ppath, self.name)    # Intan sampling rate (e.g. 1000 Hz)
        self.dt = 1/self.sr                         # s per Intan sample (e.g. 0.001)
        self.fbin = np.round((1/self.dt)*self.fdt)  # no. Intan points per SP bin (e.g. 2500)
        if self.fbin % 2 == 1:
            self.fbin += 1

        # load brain state annotation
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
            # predict brain state from EEG/EMG data
            M,S = sleepy.sleep_state(self.ppath, self.name, pwrite=1, pplot=0)
        (A,self.K) = load_stateidx(self.ppath, self.name)
        # create 1 x nbin matrix for display
        self.M = np.zeros((1,self.num_bins))
        self.M[0,:] = A
        self.M_old = self.M.copy()
        # adjust flattened brainstate for LFP thresholding purposes
        self.M_thres = AS.adjust_brainstate(self.M.flatten(), self.fdt, ma_thr=20, 
                                                  ma_state=3, flatten_is=4)
                
        # load/plot laser
        self.laser = np.zeros((self.num_bins,))  # laser signal in SP time
        self.pplot_laser = False
        # supplementary treck signal (e.g. trigger signal from REM-online detection)
        self.suppl_treck = []
        self.psuppl = False
        lfile = os.path.join(self.ppath, self.name, 'laser_' + self.name + '.mat')
        if os.path.isfile(lfile):
            lsr = load_laser(self.ppath, self.name)
            (start_idx, end_idx) = laser_start_end(lsr, self.sr)
            # laser signal in Intan time
            self.laser_raw = lsr
            self.pplot_laser = True
            if len(start_idx) > 0:
                for (i,j) in zip(start_idx, end_idx) :
                    i = int(np.round(i/self.fbin))
                    j = int(np.round(j/self.fbin))
                    self.laser[i:j+1] = 1
                stim_idx = np.arange(start_idx[0], end_idx[0])
                # if laser stims are short step pulses (<100 ms)
                if len(stim_idx) < 100 and all(self.laser_raw[stim_idx]==1):
                    # classify as laser-triggered P-wave experiment
                    self.lsrTrigPwaves = True
                # if laser stims are longer pulse trains
                else:
                    # classify as open or closed-loop behavioral experiment
                    ifile = os.path.join(self.ppath, self.name, 'info.txt')
                    self.optoMode = sleepy.get_infoparam(ifile, 'mode')[0]
                
            # load online REM-detection signal
            tfile = os.path.join(self.ppath, self.name, 'rem_trig_' + self.name + '.mat')
            if os.path.isfile(tfile):
                self.psuppl = True
                self.suppl_treck = np.zeros((self.num_bins,))
                trig = load_trigger(self.ppath, self.name)
                (start_idx, end_idx) = laser_start_end(trig, self.sr)
                if len(start_idx) > 0:
                    for (i, j) in zip(start_idx, end_idx):
                        i = int(np.round(i / self.fbin))
                        j = int(np.round(j / self.fbin))
                        self.suppl_treck[i:j + 1] = 1
        else:
            self.laser_raw = np.zeros((len(self.EEG),))
        
        # load DF/F signal
        self.dff = np.zeros((len(self.EEG),))
        self.dffdn = np.zeros((self.num_bins,))  # DF/F signal in SP time
        dfile = os.path.join(self.ppath, self.name, 'DFF.mat')
        if os.path.isfile(dfile):
            self.dff = so.loadmat(dfile, squeeze_me=True)['dff']*100
            self.dffdn = so.loadmat(dfile, squeeze_me=True)['dffd']*100
            self.hasDFF = True  # classify as fiber photometry experiment
            
        # load information of light/dark cycles
        self.dark_cycle = get_cycles(self.ppath, self.name)['dark']
        
        # load P-wave information and detection settings
        self.load_pdata(mode='default')
        
        # disable plot buttons for which recording does not meet data requirements
        for key in self.plotBtns_dict.keys():
            self.plotBtns_dict[key]['btn'].enable_btn()
            c = 'black' if self.plotBtns_dict[key]['btn'].isEnabled() else 'gray'
            self.plotBtns_dict[key]['label'].setStyleSheet(f'color : {c}')
        
    def switchSP(self):
        """
        Switch between plots of standard and noise-excluded EEG spectrogram 
        """
        # load/calculate requested spectrogram
        n = self.sender().objectName()
        if 'noise' in n:
            if 'load' in n:
                print('Loading noise-excluded EEG spectrogram ...')
                SP = AS.noise_EEGspectrogram(self.ppath, self.name, recalc_sp=False)[0]
            elif 'calculate' in n:
                self.save_pdata()
                print('Calculating noise-excluded EEG spectrogram ...')
                SP = AS.noise_EEGspectrogram(self.ppath, self.name, recalc_sp=True,
                                             noise_idx=self.eeg_noise_idx)[0]
        elif 'standard' in n:
            print('Loading standard EEG spectrogram ...')
            SP = so.loadmat(os.path.join(self.ppath, self.name, 
                                         'sp_' + self.name + '.mat'))['SP']
        # replace main EEG spectrogram with loaded/calculated SP, recalculate freq bands
        self.eeg_spec_list[0] = SP
        for i,b in enumerate(self.bands.values()):
            f1 = np.where(self.freq >= b[0])[0]
            f2 = np.where(self.freq <= b[1])[0]
            ifreq = np.intersect1d(f1,f2)
            fpwr = np.nansum(self.eeg_spec_list[0][ifreq,:], axis=0)
            self.band_pwrs[i] = fpwr
        print('Done!')
        # show raw EEG1 signal and updated EEG1 SP on plot
        self.pointers['EEG'] = 0
        self.pointers['EMG'] = -1
        self.pointers['LFP'] = -1
        self.EEG = self.EEG_list[0]
        self.eeg_spec = self.eeg_spec_list[0]
        self.ylabel = 'EEG1'
        self.plot_eeg(findPwaves=False, findArtifacts=False)
        self.plot_session(scale=self.tscale, scale_unit=self.tunit)
        # enable option to load noise-excluded SP if file exists
        npath = os.path.join(self.ppath, self.name, f'sp_nannoise_{self.name}.mat')
        loadAction = self.noiseCalcSP_btn.findChild(QtGui.QAction, 'load noise')
        loadAction.setEnabled(os.path.exists(npath))
        
    
    def plotFigure(self):
        """
        Instantiate figure window for selected plot type
        """
        plotType = self.sender().name.replace('\n','')
        self.pw = FigureWindow(plotType, parent=self)
        self.pw.show()
        
    
    def twitch_annot_window(self):
        """
        Instantiate EMG amplitude annotation window
        """
        self.pw.setWindowTitle('Loading EMG twitch annotation window ...')
        self.tw = EMGTwitchWindow(self.ppath, self.name, parent=self, 
                                  settingsfile=self.pw.twitchFile)
        self.tw.show()
        self.pw.setWindowTitle('')
        self.tw.exec_()
    
    
    def show_help_window(self):
        """
        Instantiate informational help window
        """
        self.hw = HelpWindow()
        self.hw.show()
        
    
    # def closeEvent(self, event):
    #         # save brain state annotation
    #         rewrite_remidx(self.M, self.K, self.ppath, self.name, mode=0)
    #         #QtGui.QApplication.quit()
    
    
    def debug(self):
        pdb.set_trace()

# some input parameter management
params = sys.argv[1:]
if (len(params) == 0) :
    ppath = ''
    name = ''
elif len(params) == 1:
    if re.match('.*\/$', params[0]):
        params[0] = params[0][:-1]
    (ppath, name) = os.path.split(params[0])      
else:
    ppath = params[0]
    name  = params[1]
    
#ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
#ppath ='/media/fearthekraken/Mandy_HardDrive1/revision_data/'
#ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
#ppath = '/home/fearthekraken/Documents/Data/photometry'
#ppath ='/media/fearthekraken/Mandy_HardDrive1/ChR2_Open/'
#ppath ='/media/fearthekraken/Mandy_HardDrive1/ChR2_YFP_Open/

import platform
if platform.system() == 'Darwin':
    ppath ='/Users/amandaschott/Dropbox'
elif platform.system() == 'Linux':
    ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
    
name = 'Salinger_073020n1'


app = QtGui.QApplication([])
app.setStyle('Fusion')
w = MainWindow(ppath, name)
w.show()
sys.exit(app.exec_())