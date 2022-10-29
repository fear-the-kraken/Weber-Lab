#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:02:59 2022

@author: fearthekraken
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
import scipy.io as so
import pickle
import time
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sleepy
import AS
import pwaves
import pdb
from pqt_items import warning_dlg

class EMGTwitchFigure(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(EMGTwitchFigure, self).__init__(parent)
        
        self.mainWin = parent
        self.fig = plt.figure(figsize=(10,10), constrained_layout=True)
        self.fig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.5, wspace=0, hspace=0)
        
        # create plot window
        self.centralLayout = QtWidgets.QVBoxLayout(self)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.centralLayout.addWidget(self.toolbar)
        self.centralLayout.addWidget(self.canvas)
    
    def plot_data(self):
        self.fig.clear()
        # calculate twitch frequency
        REMdurs = []
        num_twitches = []
        
        for rseq in self.mainWin.EMG_remseq:
            noise_idx = np.where(self.mainWin.EMGdata_nidx[rseq]==1)[0]
            # get no. minutes of non-noisy REM sleep
            durREM = ((len(rseq) - len(noise_idx))*self.mainWin.mdt)/60
            REMdurs.append(durREM)
            # get EMG twitch sequences
            uA = sleepy.get_sequences(np.where(self.mainWin.usrAnnot[rseq]==1)[0])
            if uA[0].size > 0:
                num_twitches.append(len(uA))
            else:
                num_twitches.append(0)
        # eliminate REM periods that were entirely noise, calculate twitch frequency in each REM period
        num_twitches, REMdurs = zip(*[[i,j] for i,j in zip(num_twitches, REMdurs) if j!=0])
        num_twitches, REMdurs = [np.array(num_twitches), np.array(REMdurs)]
        twitch_freqs = np.divide(num_twitches, REMdurs)
        
        self.fig.suptitle('EMG Twitch Frequency')
        grid = GridSpec(2, 2, figure=self.fig)
        # plot twitch frequency averaged over all REM sleep
        ax1 = self.fig.add_subplot(grid[0,0])
        ax1.bar(['REM'], np.sum(num_twitches)/np.sum(REMdurs), color='cyan', edgecolor='black')
        ax1.set_ylabel('EMG twitches/min')
        ax1.set_title('Averaged across all REM sleep)')
        
        # plot twitch frequency averaged within each REM period
        ax2 = self.fig.add_subplot(grid[0,1])
        ax2.bar(['REM'], np.nanmean(twitch_freqs), yerr=np.nanstd(twitch_freqs)/np.sqrt(len(twitch_freqs)), color='cyan', edgecolor='black')
        for tw in twitch_freqs:
            c = list(np.random.choice(range(255), size=3)/255)
            ax2.plot(['REM'], tw, color=c, linewidth=0, marker='o')
        ax2.set_title('Averaged within each REM bout)')
        # make y axes equivalent
        y = [min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])]
        ax1.set_ylim(y)
        ax2.set_ylim(y)
        
        # plot twitch frequency for each REM period
        ax3 = self.fig.add_subplot(grid[1,:])
        ax3.bar(np.arange(1,len(twitch_freqs)+1), twitch_freqs, color='cyan', edgecolor='black')
        ax3.set_xticks(np.arange(1,len(twitch_freqs)+1))
        ax3.set_xticklabels(np.arange(1,len(twitch_freqs)+1))
        txt = [f'{i} tw.\n-----\n{int(j*60)} s' for i,j in zip(num_twitches, REMdurs)]
        AS.label_bars(ax3, above=0.2, text=txt)
        ax3.set_ylabel('EMG twitches/min')
        ax3.set_title('Twitch frequency during each REM period')
        
        self.canvas.draw()
    
    def closeEvent(self, event):
        self.hide()


class EMGTwitchWindow(QtWidgets.QDialog):
    def __init__(self, ppath, name, parent=None, settingsfile=[]):
        QtWidgets.QDialog.__init__(self)
        self.setGeometry( QtCore.QRect(50, 200, 1800, 500))
        
        self.ppath = ppath
        self.name  = name
        self.parent = parent
        
        # brainstate params
        self.ma_thr = 20
        self.ma_state = 3
        self.flatten_is = 4
        
        # random other stuff
        self.offset = 0
        self.saveThres = {'raw':{}, 'msp':{}}
        self.curFile = None
        
        # twitch annotation
        self.usrAnnot = np.array(())
        self.annotMap = {1:{'color':(53,81,119), 'sym':'o', 'vis':True},
                         2:{'color':(255,255,0), 'sym':'t', 'vis':False},
                         3:{'color':(0,0,255), 'sym':'x', 'vis':False}}
        
        # LFP plot view
        self.plot_pwaves = True
        self.show_pthres = False
        self.show_pidx = False
        self.show_noise = False
        self.show_laser = False
        
        # current plot
        self.idx = 0
        
        # create GUI
        self.pw = EMGTwitchFigure(parent=self)
        self.pw.setModal(False)
        self.setup_gui()
        self.connect_buttons()
        
        # load recording
        self.load_recording(self.ppath, self.name)
        
        # load specified settings file
        if settingsfile:
            if os.path.isfile(settingsfile):
                self.load_settings(settingsfile)
            else:
                settingsfile = []
        # look for any saved settings files
        if not settingsfile:
            sfiles = [f for f in os.listdir(os.path.join(self.ppath, self.name)) if f.endswith('.pkl')]
            if len(sfiles) == 1:
                self.load_settings(os.path.join(self.ppath, self.name, sfiles[0]))
            elif len(sfiles) > 1:
                self.load_settings()
        
            # set default param values
            elif len(sfiles) == 0:
                # EMG amplitude source param
                self.ampsrc = 'raw'
                
                # REM indexing params
                self.min_dur = 10    # min duration (s) of REM sleep
                self.rem_cutoff = 5  # time (s) to cut off the end of each REM period
                
                # raw EMG filtering params
                self.w0_raw = 0.02  # lowest freq component of raw EMG (highpass filter)
                self.w1_raw = -1    # highest freq component of raw EMG (lowpass filter)
                self.dn_raw = 50    # downsampling factor for raw EMG
                self.sm_raw = 100   # smoothing factor for raw EMG
                
                # EMG spectrogram params
                self.nsr_seg_msp = 0.2       # FFT window size (s) for mSP calculation
                self.perc_overlap_msp = 0.5  # FFT window overlap (%) for mSP calculation
                self.r_mu = [10,500]         # [low, high] freq cutoffs in mSP
                
                # EMG threshold params
                self.thres = 99           # threshold value (X)
                self.thres_type = 'perc'  # 'raw' = X, 'std' = X*std + mean, 'perc' = Xth percentile
                self.thres_mode = 1       # 1 = one threshold for all REM, 2 = indiv threshold for each REM bout
                self.thres_first = 0      # if > 0, set threshold based on first -- seconds of REM bouts
                
                # twitch detection params
                self.min_twitchdur = 0.1  # min duration (s) of twitch
                self.min_twitchsep = 0.2  # min time (s) between distinct twitches
                
                ### ADD VARIABLE (initialize)
                
                self.save_settings(os.path.join(self.ppath, self.name, 'default_twitch_settings.pkl'))
                self.load_settings(os.path.join(self.ppath, self.name, 'default_twitch_settings.pkl'))
        
        # save threshold info for current EMG source
        self.saveThres[self.ampsrc] = {'thres':float(self.thres),
                                       'thres_type':str(self.thres_type),
                                       'thres_mode':int(self.thres_mode),
                                       'thres_first':float(self.thres_first)}
        

        
        self.get_mSP(recalc=False, match_params=True)
        self.calculate_EMGAmpl()
        self.get_EMGdn()
        self.find_REM()
        self.detect_emg_twitches(calcAnnot=False)
        
        self.plotSettings = self.dict_from_vars()
        self.plot_data()
        
    
    def load_recording(self, ppath, name):
        self.setWindowTitle(name)
        
        # load sampling rate
        self.sr = sleepy.get_snr(ppath, name)
        self.nbin = int(np.round(self.sr) * 2.5)
        self.dt = (1.0 / self.sr) * self.nbin
        
        # load EEG, EMG
        self.EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), 
                              squeeze_me=True)['EEG']
        self.EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), 
                              squeeze_me=True)['EMG']
        
        # load processed LFP
        if os.path.isfile(os.path.join(ppath, name, 'LFP_processed.mat')):
            self.LFP = so.loadmat(os.path.join(ppath, name, 'LFP_processed.mat'),
                                  squeeze_me=True)['LFP_processed']
        else:
            self.LFP = np.zeros((len(self.EMG),))
        self.lfp_amp = np.median(np.abs(self.LFP))
        
        # load P-wave indices and detection threshold
        self.pidx = np.zeros((len(self.EMG),))
        if os.path.isfile(os.path.join(ppath, name, 'p_idx.mat')):
            pwave_info = so.loadmat(os.path.join(ppath, name, 'p_idx.mat'), squeeze_me=True)
            self.pi = pwave_info['p_idx']
            self.pidx[self.pi] = 1
        else:
            pwave_info = {}
            self.pi = []
        
        # load P-wave threshold
        self.pthres = 0 if 'p_thr' not in pwave_info.keys() else -float(pwave_info['p_thr'])
        
        # load noise indices
        self.lnidx = np.zeros((len(self.EMG),))
        self.enidx = np.zeros((len(self.EMG),))
        self.lni = np.array(()).astype('int') if 'noise_idx' not in pwave_info.keys() else np.array(pwave_info['noise_idx'])
        self.eni = np.array(()).astype('int') if 'emg_noise_idx' not in pwave_info.keys() else np.array(pwave_info['emg_noise_idx'])
        self.lnidx[self.lni] = 1
        self.enidx[self.eni] = 1
        
        # load laser
        if os.path.isfile(os.path.join(self.ppath, self.name, 'laser_' + self.name + '.mat')):
            lsr = sleepy.load_laser(self.ppath, self.name)
            self.laser_raw = lsr
            self.show_laser = True
        else:
            self.laser_raw = np.zeros((len(self.EEG),))
        
        # load and adjust brain state annotation
        self.M = sleepy.load_stateidx(ppath, name)[0]
        self.M = AS.adjust_brainstate(self.M, dt=2.5, ma_thr=self.ma_thr, 
                                      ma_state=self.ma_state, 
                                      flatten_is=self.flatten_is)
        
        # update GUI max values
        self.emgfiltLo_val.setMaximum(self.sr/2-1)
        self.emgfiltHi_val.setMaximum(self.sr/2-1)
        self.mspFreqLo_val.setMaximum(self.sr/2)
        self.mspFreqHi_val.setMaximum(self.sr/2)
        self.emgfiltDn_val.setMaximum(self.nbin*10)

    def get_mSP(self, mode='EMG', recalc=True, peeg2=False, pemg2=False, match_params=True):
        """
        @Params
        recalc - True to recalculate SPs, False to load SPs from saved files
        peeg2, pemg2 - False to use parietal EEG/ EMG1, True to use prefrontal EEG/ EMG2
        match_params - warn user if loaded SP params do not match current params
        """
        
        data = AS.highres_spectrogram(self.ppath, self.name, nsr_seg=self.nsr_seg_msp, 
                                      perc_overlap=self.perc_overlap_msp, recalc_highres=recalc, 
                                      mode=mode, exclude_noise=True, peeg2=peeg2, pemg2=pemg2, 
                                      match_params=match_params)
        if mode=='EMG':
            self.mSP, self.mfreq, self.mt, self.mnbin_msp, self.mdt_msp = data
        elif mode == 'EEG':
            self.eSP, self.efreq, self.et, self.enbin, self.edt = data
        
        # calculate EMG amplitude
        # self.calculate_EMGAmpl()
        
    def calculate_EMGAmpl(self):
        # calculate EMG amplitude
        i_mu = np.where((self.mfreq >= self.r_mu[0]) & (self.mfreq <= self.r_mu[1]))[0]
        self.EMGAmpl = np.sqrt(self.mSP[i_mu, :].sum(axis=0) * (self.mfreq[1] - self.mfreq[0]))
        # get processed signal without noise, high-pass filter to remove baseline fluctuations
        inoise = np.nonzero(np.isnan(self.EMGAmpl))
        ireal = np.setdiff1d(range(len(self.EMGAmpl)), inoise)
        sig = np.abs(sleepy.my_hpfilter(self.EMGAmpl[ireal], 0.1))
        self.EMGAmpl[ireal] = sig
    
    def get_EMGdn(self):
        # filter EMG
        if self.w0_raw != -1 or self.w1_raw != -1:
            if self.w1_raw == -1:    # highpass
                self.EMGfilt = np.abs(sleepy.my_hpfilter(self.EMG, self.w0_raw))
            elif self.w0_raw == -1:  # lowpass
                self.EMGfilt = np.abs(sleepy.my_lpfilter(self.EMG, self.w1_raw))
            else:                    # bandpass
                self.EMGfilt = np.abs(sleepy.my_bpfilter(self.EMG, self.w0_raw, self.w1_raw))
        else:
            self.EMGfilt = np.abs(np.array(self.EMG))
        
        self.EMGfilt[self.eni] = np.nan
        
        # smooth EMG
        if self.sm_raw > 0:
            #self.EMGfilt = AS.smooth_data(self.EMGfilt, self.sm_raw)
            self.EMGfilt = AS.convolve_data(self.EMGfilt, self.sm_raw, axis='x')
        
        # downsample EMG
        if self.dn_raw > 1:
            self.EMGdn = AS.downsample_nanvec(self.EMGfilt, self.dn_raw)
        else:
            self.EMGdn = np.array(self.EMGfilt)
        # get processed signal without noise, high-pass filter to remove baseline fluctuations
        inoise = np.nonzero(np.isnan(self.EMGdn))
        ireal = np.setdiff1d(range(len(self.EMGdn)), inoise)
        sig = np.abs(sleepy.my_hpfilter(self.EMGdn[ireal], 0.1))
        self.EMGdn[ireal] = sig
        
        
    def find_REM(self):
        # get qualifying REM sequences
        self.remseq = sleepy.get_sequences(np.where(self.M==1)[0])
        self.remseq = [rs for rs in self.remseq if len(rs)*self.dt >= self.min_dur]
        # cut off last bin(s) to eliminate muscle twitch from waking up
        if self.rem_cutoff > 0:
            icut = int(round(self.rem_cutoff / self.dt))
            self.remseq = [rs[0:-icut] for rs in self.remseq if len(rs)>icut]
        self.numrem = len(self.remseq)
        
        if self.ampsrc == 'msp':
            self.EMGdata = np.array(self.EMGAmpl)
            self.mnbin = float(self.mnbin_msp)
            self.mdt = float(self.mdt_msp)
            
        elif self.ampsrc == 'raw':
            self.EMGdata = np.array(self.EMGdn)
            self.mnbin = float(self.dn_raw)
            self.mdt = float(self.mnbin/self.sr)
            
        # get indices of REM sequences in EMG data
        cf = self.nbin/self.mnbin  # conversion factor from $M to EMG amp vector
        self.EMG_remseq = [np.arange(int(round(rs[0]*cf)), int(round(rs[-1]*cf+cf))+1) for rs in self.remseq]
        if self.EMG_remseq[-1][-1] == len(self.EMGdata):
            self.EMG_remseq[-1] = self.EMG_remseq[-1][0:-1]
        # get indices of REM sequences in raw EMG signal
        self.EMGraw_remseq = [np.arange(int(round(es[0]*self.mnbin)), int(round(es[-1]*self.mnbin+self.mnbin))+1) for es in self.EMG_remseq]
        
        # get indices of first X seconds in REM bouts for thresholding
        if self.thres_first > 0:
            ibins = int(round(self.thres_first*(self.sr/self.mnbin)))
            self.istart = [emgseq[0:ibins] for emgseq in self.EMG_remseq]
    
    def detect_emg_twitches(self, calcAnnot=True):
        # collect actual threshold used for each REM period
        self.thres_vals = []
        
        # threshold all REM bins
        if self.thres_mode==1:
            ridx = np.concatenate(self.EMG_remseq)
            
            # threshold by first X seconds of each REM bout OR whole REM bouts
            if self.thres_first > 0:
                th_ridx = np.concatenate(self.istart)
            else:
                th_ridx = np.array(ridx)
                
            # threshold using arbitrary units OR Xth percentile of EMG signal
            if self.thres_type == 'raw':
                th = float(self.thres)
            elif self.thres_type == 'std':
                mn = np.nanmean(self.EMGdata[th_ridx])
                std = np.nanstd(self.EMGdata[th_ridx])
                th = mn + self.thres*std
                
            elif self.thres_type == 'perc':
                th = np.nanpercentile(self.EMGdata[th_ridx], self.thres)
                

            # get indices of "peaks" and phasic sequences in EMG signal
            spk_idx = pwaves.spike_threshold(self.EMGdata[ridx], th, sign=-1)
            idx = np.where(self.EMGdata[ridx] > th)[0]
            
            self.twitch_idx = ridx[spk_idx]
            self.phasic_idx = ridx[idx]
            self.thres_vals = [th]*self.numrem
            
        # separately threshold each REM period to look for EMG activation
        elif self.thres_mode==2:
            self.twitch_idx = np.array(())
            self.phasic_idx = np.array(())
            
            for i in range(self.numrem):
                # indices of REM sequence in EMG signal
                seqi = self.EMG_remseq[i]
                if self.thres_first > 0:
                    th_ridx = self.istart[i]
                else:
                    th_ridx = np.array(seqi)
                
                # threshold using arbitrary units OR Xth percentile of EMG signal
                if self.thres_type == 'raw':
                    th = float(self.thres)
                elif self.thres_type == 'std':
                    mn = np.nanmean(self.EMGdata[th_ridx])
                    std = np.nanstd(self.EMGdata[th_ridx])
                    th = mn + self.thres*std
                elif self.thres_type == 'perc':
                    th = np.nanpercentile(self.EMGdata[th_ridx], self.thres)
                self.thres_vals.append(th)
                
                # get indices of "peaks" and phasic sequences in REM sequence
                spk_idx = pwaves.spike_threshold(self.EMGdata[seqi], th, sign=-1)
                idx = np.where(self.EMGdata[seqi] > th)[0]
                
                self.twitch_idx = np.concatenate([self.twitch_idx, seqi[spk_idx]])
                self.phasic_idx = np.concatenate([self.phasic_idx, seqi[idx]])
                
            self.twitch_idx = self.twitch_idx.astype(int)
            self.phasic_idx = self.phasic_idx.astype(int)
            
        # get trains (1's and 0's) of single twitches
        self.twitch_train = np.zeros(len(self.EMGdata))
        self.twitch_train[self.twitch_idx] = 1
        
        # combine neighboring twitch indices into sequences using $min_twitchsep param
        self.ibreak = int(round(self.min_twitchsep / self.mdt))
        self.phasic_seqs = sleepy.get_sequences(self.phasic_idx, self.ibreak)
        
        # fill in gaps of $ibreak indices, eliminate short twitches under $min_twitchdur s
        try:
            self.phasic_seqs = [np.arange(phseq[0],phseq[-1]+1) for phseq in self.phasic_seqs]
        except:
            pdb.set_trace()
        #self.phasic_seqi = np.concatenate([np.arange(phseq[0],phseq[-1]+1) for phseq in self.phasic_seqs])
        self.phasic_seqs = [phseq for phseq in self.phasic_seqs if len(phseq)*self.mdt >= self.min_twitchdur]
        self.phasic_seqi = np.concatenate(self.phasic_seqs)
        self.phasic_train = np.zeros(len(self.EMGdata))
        self.phasic_train[self.phasic_seqi] = 1
        self.phasic_remseqs = [self.phasic_train[i] for i in self.EMG_remseq]
        
        # get noise indices, make all Nans a flat line
        self.EMGdata_nidx = np.zeros((len(self.EMGdata),))
        self.EMGdata_ni = np.nonzero(np.isnan(self.EMGdata))[0]
        self.EMGdata_nidx[self.EMGdata_ni] = 1
        nseqs = sleepy.get_sequences(self.EMGdata_ni)
        if nseqs[0].size > 0:
            for ns in nseqs:
                self.EMGdata[ns] = self.EMGdata[ns[0]-1]
        
        # save automatically detected twitches
        self.autoAnnot = np.array(self.phasic_train).astype('int')
        # if self.ampsrc == 'msp':
        #     self.autoAnnot = np.array(self.phasic_train_msp).astype('int')
        # elif self.ampsrc == 'raw':
        #     self.autoAnnot = np.array(self.phasic_train_raw).astype('int')
        # if available, load user annotation and fill in -1's for noise indices
        if not calcAnnot and self.usrAnnot.size == self.autoAnnot.size:
            self.usrAnnot = np.array(self.usrAnnot).astype('int')
        else:
            self.usrAnnot = np.array(self.autoAnnot).astype('int')
        self.usrAnnot[self.EMGdata_ni] = -1
        
        # track changes to user annotation
        self.initAnnot = np.array(self.usrAnnot).astype('int')
        
    def plot_data(self):
        if self.idx > self.numrem:
            self.idx = self.numrem-1
            
        # get indices of current REM sequence in EMG data vector
        self.seqi = self.EMG_remseq[self.idx]
        
        if self.ampsrc == 'msp':
            # x axis from mSP calculation, adjust for indexing in Intan signal
            self.t = self.mt[self.seqi]
            self.tstart = self.t[0]
            self.tend = self.t[-1]
            #self.msp_adjust = np.linspace(self.sr,-self.sr, len(self.EMG))
            self.msp_adjust = np.zeros(len(self.EMG))
        elif self.ampsrc == 'raw':
            self.tstart = self.seqi[0]*((1.0/self.sr)*self.mnbin)
            self.tend = self.seqi[-1]*((1.0/self.sr)*self.mnbin)
            self.t = np.linspace(self.tstart, self.tend, len(self.seqi))
            self.msp_adjust = np.zeros(len(self.EMG))
        
        # get start and end points of REM sequence in raw Intan signals
        ri = int(round(self.seqi[0]*self.mnbin))
        re = int(round(self.seqi[-1]*self.mnbin+self.mnbin))
        # adjust indices to align with mSP
        rispi = int(round(ri + self.msp_adjust[ri]))
        respi = int(round(re + self.msp_adjust[re]))
        # get raw EMG
        self.remg = self.EMG[rispi:respi+1]
        clean_remg = self.remg[np.where(self.enidx[rispi:respi+1]==0)[0]]
        enidx_seqs = sleepy.get_sequences(np.where(self.enidx[rispi:respi+1]==1)[0])
        
        # get LFP signal, LFP noise, P-waves, laser
        self.LFP_seq = self.LFP[rispi:respi+1]
        clean_lfp = self.LFP_seq[np.where(self.lnidx[rispi:respi+1]==0)[0]]
        lnidx_seqs = sleepy.get_sequences(np.where(self.lnidx[rispi:respi+1]==1)[0])
        self.pidx_seqi = np.where(self.pidx[rispi:respi+1]==1)[0]
        if self.pidx_seqi.size > 0:
            self.pidx_amps = [self.LFP_seq[pi]-25 for pi in self.pidx_seqi]
        else:
            self.pidx_amps = []
        self.laser_seq = self.laser_raw[rispi:respi+1]
        
        # get EMG data and threshold value
        self.EMGdata_seq = self.EMGdata[self.seqi]
        clean_EMGdata = self.EMGdata_seq[np.where(self.EMGdata_nidx[self.seqi]==0)[0]]
        EMGdata_nseq = self.EMGdata_nidx[self.seqi]
        EMGdata_nseqs = sleepy.get_sequences(np.where(EMGdata_nseq==1)[0])
        
        self.seq_th = self.thres_vals[self.idx]
        # annotated sequences of phasic EMG activity
        self.annot_seq = self.usrAnnot[self.seqi]
        self.annot_seqs = sleepy.get_sequences(np.where(self.annot_seq>0)[0])
        if self.annot_seqs[0].size > 0:
            self.annot_amps = [max(self.EMGdata_seq[ph]) for ph in self.annot_seqs]
        else:
            self.annot_amps = []
        
        # clear plots
        self.curRawSeq.clear()
        self.curRawIdx.clear()
        self.curRawSeq2.clear()
        self.curRawIdx2.clear()
        self.curItem = None
        self.curPoint = None
        
        limits = {'xMin': self.tstart, 'xMax': self.tend}
        
        # collect EMG/LFP noise plot items
        self.cur_noise = []
        
        # plot raw EMG
        self.graph_rawEMG.clear()
        self.t1 = np.linspace(self.tstart, self.tend, len(self.remg))
        self.curRawEMG.setData(self.t1, self.remg)
        self.graph_rawEMG.addItem(self.curRawEMG)   # raw EMG for REM sequence  # raw EMG noise for REM sequence
        # plot raw EMG noise
        if enidx_seqs[0].size > 0:
            for en in enidx_seqs:
                a = pg.PlotDataItem(self.t1[en], self.remg[en], pen=pg.mkPen((7,247,247,200),width=2))
                a.setVisible(self.show_noise)
                self.graph_rawEMG.addItem(a)  
                self.cur_noise.append(a)
        self.graph_rawEMG.addItem(self.curRawSeq)   # raw EMG for currently selected twitch seq
        self.graph_rawEMG.addItem(self.curRawIdx)   # raw EMG for currently selected twitch idx
        self.graph_rawEMG.vb.setLimits(**limits)
        #self.graph_rawEMG.setRange(xRange=(self.tstart,self.tend))
        self.graph_rawEMG.setXRange(self.tstart, self.tend)
        self.graph_rawEMG.setYRange(min(clean_remg), max(clean_remg), padding=0.2)
        #self.graph_rawEMG.enableAutoRange(y=True)
        
        # plot LFP and P-waves
        self.graph_LFP.clear()
        self.curLFP.setData(self.t1, self.LFP_seq)  # LFP for REM sequence
        self.graph_LFP.addItem(self.curLFP)
        # plot LFP noise
        if lnidx_seqs[0].size > 0:
            for ln in lnidx_seqs:
                a = pg.PlotDataItem(self.t1[ln], self.LFP_seq[ln], pen=pg.mkPen((255,20,147,200),width=2))
                a.setVisible(self.show_noise)
                self.graph_LFP.addItem(a)  
                self.cur_noise.append(a)
        self.graph_LFP.addItem(self.curRawSeq2)
        self.graph_LFP.addItem(self.curRawIdx2)
        if len(self.pidx_amps) > 0:
            self.cur_pidx.setData(self.t1[self.pidx_seqi], self.pidx_amps)
        else:
            self.cur_pidx.clear()
        self.cur_pidx.setVisible(self.show_pidx)
        self.graph_LFP.addItem(self.cur_pidx)
        if self.pthres != 0:
            self.cur_pthres.setValue(self.pthres)
        else:
            self.cur_pthres.setValue(0)
        if self.show_pthres:
            self.cur_pthres.setPen((102,205,170),width=1)
        else:
            self.cur_pthres.setPen(None)
        self.graph_LFP.addItem(self.cur_pthres)
        self.cur_laser.setData(self.t1, self.laser_seq*self.lfp_amp*6)
        self.cur_laser.setVisible(self.show_laser)
        self.graph_LFP.addItem(self.cur_laser)
        
        self.graph_LFP.vb.setLimits(**limits)
        #self.graph_LFP.setRange(xRange=(self.tstart,self.tend))
        self.graph_LFP.setXRange(self.tstart, self.tend)
        #self.graph_LFP.enableAutoRange(y=True)
        self.graph_LFP.setYRange(min(clean_lfp), max(clean_lfp), padding=0.25)
        
        # plot EMG amplitude + noise and threshold
        self.graph_EMGdata.clear()
        self.curEMGdata.setData(self.t, self.EMGdata_seq)
        self.graph_EMGdata.addItem(self.curEMGdata)
        if EMGdata_nseqs[0].size > 0:
            for edn in EMGdata_nseqs:
                a = pg.PlotDataItem(self.t[edn], self.EMGdata_seq[edn], pen=pg.mkPen((7,247,247,200),width=2))
                a.setVisible(self.show_noise)
                self.graph_EMGdata.addItem(a)  
                self.cur_noise.append(a)
            
        if self.plotSettings['thres_type'] == 'std':
            txt = f'Threshold: mean + {self.plotSettings["thres"]}*STD'
        elif self.plotSettings['thres_type'] == 'perc':
            txt = f'Threshold: {self.plotSettings["thres"]} percentile'
        self.cur_thres.setValue(self.seq_th)
        self.cur_thres.label.setFormat(txt)
        self.graph_EMGdata.addItem(self.cur_thres)
        self.graph_EMGdata.vb.setLimits(**limits)
        #self.graph_EMGdata.setRange(xRange=(self.tstart,self.tend))
        self.graph_EMGdata.setXRange(self.tstart, self.tend)
        #self.graph_EMGdata.enableAutoRange(y=True)
        self.graph_EMGdata.setYRange(min(clean_EMGdata), max(clean_EMGdata), padding=0.3)
        
        # plot individual phasic indices/sequences
        for ph,amp in zip(self.annot_seqs, self.annot_amps):
            a = pg.PlotDataItem(self.t[ph], [amp+0.1]*len(ph))
            self.graph_EMGdata.addItem(a)
            annots = self.annot_seq[ph]
            # set symbol appearance
            syms = [self.annotMap[an]['sym'] for an in annots]
            a.setSymbol(syms)
            brushes = [pg.mkColor(self.annotMap[an]['color']) for an in annots]
            if len(annots) == 1:
                a.setSymbolBrush(brushes[0])
                a.setSymbolPen(None)
                a.setSymbolSize(15)
            else:
                a.scatter.setBrush(brushes)
                a.scatter.setPen(None)
                a.scatter.setSize(15)
            
            # set line appearance
            if any(annots==1): pencolor=self.annotMap[1]['color']
            elif any(annots==2): pencolor=self.annotMap[2]['color']
            else: pencolor=self.annotMap[3]['color']
            a.curve.setPen(pencolor,width=8)
            # connect items to updating functions
            a.sigPointsClicked.connect(self.select_point)
            a.curve.setClickable(True)
            a.curve.sigClicked.connect(self.select_item)
            
            # 0 = nothing, 1 = phasic sequence, 2 = noise, 3 = other
            #cols = [(53,81,119) if an==1 else (255,255,0) if an==2 else (0,0,255) for an in self.annot_seq[ph]]
            #syms = ['o' if an==1 else 't' if an==2 else 'x' for an in self.annot_seq[ph]]
            #brushes = [pg.mkColor(c) for c in cols]
            #penc = (53,81,119) if any(self.annot_seq[ph]==1) else (255,255,0) if any(self.annot_seq[ph]==2) else (0,0,255)
            #a = self.graph_EMGdata.plot(self.t[ph], [amp+1]*len(ph), pen=pg.mkPen(color=penc,width=8), symbol=syms, 
            #                            symbolPen=None, symbolBrush=brushes, symbolSize=15)
            
            
        # set title
        self.plotTitle.setText(f'REM Sequence {self.idx+1} of {self.numrem}')
    
    def select_item(self, item):
        """
        Select phasic EMG sequence
        """
        # get parent DataPlotItem of the curve
        item = item.parentItem()
        
        # deselect previous item and previous point
        self.click_updateCur(itemAction='deselect', pointAction='deselect')
        self.curPoint = None
        # no item selection
        if self.curItem == item:
            self.curItem = None
            return
        # select clicked item
        else:
            self.curItem = item
            self.click_updateCur(itemAction='select', pointAction='none')
            
    def select_point(self, item, points):
        """
        Select particular index in phasic EMG sequence
        """
        # number of points/indices in the clicked item
        ls = len(item.scatter.points())
        
        # clicked point on an item not currently selected
        if self.curItem != item:
            # number of points in the item to be deselected
            lc = len(self.curItem.scatter.points()) if self.curItem else 0
            
            # deselect previous item and previous point
            self.click_updateCur(itemAction='deselect', pointAction='deselect', lenSeq=lc)
            # select clicked item
            self.curItem = item
            
            if ls == 1:
                # select item and clicked point
                self.curPoint = points[0]
                self.click_updateCur(itemAction='select', pointAction='select', lenSeq=ls)
            else:
                # select item only, deselect current point
                self.click_updateCur(itemAction='select', pointAction='deselect', lenSeq=ls)
                self.curPoint = None
            return
        
        # 1+ clicked points on a currently selected item, do not select a point
        elif len(points) != 1:
            self.click_updateCur(itemAction='deselect', pointAction='deselect', lenSeq=ls)
            self.curPoint = None
            return
        else:
            pt = points[0]
            # clicked on currently selected point
            if self.curPoint and self.curPoint.pos() == pt.pos():
                # deselect clicked point and item
                if ls == 1:
                    self.click_updateCur(itemAction='deselect', pointAction='deselect', lenSeq=ls)
                    self.curItem = None
                # deselect point only
                else:
                    self.click_updateCur(itemAction='none', pointAction='deselect', lenSeq=ls)
                self.curPoint = None
            # clicked on new point in currently selected item
            else:
                # deselect previous point
                self.click_updateCur(itemAction='none', pointAction='deselect', lenSeq=ls)
                # select clicked point
                self.curPoint = pt
                self.click_updateCur(itemAction='none', pointAction='select', lenSeq=ls)
                
            
    def click_updateCur(self, itemAction, pointAction, lenSeq='-'):
        if self.curItem != None:
            if itemAction == 'deselect':
                self.curItem.curve.setShadowPen(None)   # remove line highlight 
                if lenSeq == 1:
                    self.curItem.setSymbolPen(None)  # remove single point outline
                self.curSeqi = None               # clear current seq idx
                self.curRawSeq.clear()            # clear raw EMG color
                self.curRawSeq.setVisible(False)
            elif itemAction == 'select':
                self.curItem.curve.setShadowPen(color='w', width=15)           # highlight line
                if lenSeq == 1:
                    self.curItem.setSymbolPen(color=(255,0,0), width=2)  # outline single point
                self.curSeqi = [i for i,x in enumerate(self.t) if x in self.curItem.xDisp]
                ei = np.arange(int(round(self.curSeqi[0]*self.mnbin)), int(round(self.curSeqi[-1]*self.mnbin+self.mnbin))+1)
                # plot current sequence in raw EMG
                self.curRawSeq.setData(self.t1[ei], self.remg[ei])
                self.curRawSeq.setVisible(True)
                if self.curPoint != None:
                    self.curRawIdx.setVisible(True)
            else:
                # these actions rely on currently selected items
                #ls = len(self.curItem.scatter.points())
                if itemAction == 'twitch':
                    # annotate item as clean
                    self.curItem.setSymbolBrush((53,81,119))
                    self.curItem.setSymbol('o')
                    self.curItem.curve.setPen(pg.mkPen(color=(53,81,119),width=8))
                    self.curItem.curve.setShadowPen(color='w', width=15)
                    self.annot_seq[self.curSeqi] = 1
                elif itemAction == 'noise':
                    # annotate item as noise
                    self.curItem.setSymbolBrush((255,255,0))
                    self.curItem.setSymbol('t')
                    self.curItem.curve.setPen(pg.mkPen(color=(255,255,0),width=5))
                    self.curItem.curve.setShadowPen(color='w', width=15)
                    self.annot_seq[self.curSeqi] = 2
        else:
            self.curSeqi = None
        
        if self.curPoint != None:
            if pointAction == 'deselect':
                self.curPoint.setPen(None)           # remove red outline
                self.curIdx = None                   # clear current idx
                self.curRawIdx.clear()               # clear raw EMG color
                self.curRawIdx.setVisible(False)
                if lenSeq == 1:
                    self.curRawSeq.clear()           # clear seq raw EMG
                    self.curRawSeq.setVisible(False)
            elif pointAction == 'select':
                self.curPoint.setPen((255,0,0),width=2)  # add red outline
                # get current point index
                self.curIdx = int(np.where(self.t == self.curPoint.pos()[0])[0])
                # plot current point in raw EMG
                ei = np.arange(int(round(self.curIdx*self.mnbin)), int(round(self.curIdx*self.mnbin+self.mnbin))+1)
                #self.curRawIdx.setData(self.t1[ei], self.remg[ei], pen=pg.mkPen(color=(255,0,0),width=5))
                self.curRawIdx.setData(self.t1[ei], self.remg[ei])
                self.curRawIdx.setVisible(True)     # plot seq raw EMG
                if lenSeq == 1:
                    self.curRawSeq.setData(self.t1[ei], self.remg[ei])
                    self.curRawSeq.setVisible(True)
            else:
                # these actions rely on currently selected items
                ls = len(self.curItem.scatter.points())
                if pointAction == 'twitch':
                    # annotate point as clean
                    self.curPoint.setBrush((53,81,119))
                    self.curPoint.setSymbol('o')
                    if ls == 1:
                        self.curItem.setSymbolBrush((53,81,119))
                        self.curItem.setSymbol('o')
                        self.curItem.scatter.setSize(15)
                    self.annot_seq[self.curIdx] = 1
                    
                elif pointAction == 'noise':
                    # annotate point as noise
                    self.curPoint.setBrush((255,255,0))
                    self.curPoint.setSymbol('t')
                    if ls == 1:
                        self.curItem.setSymbolBrush((255,255,0))
                        self.curItem.setSymbol('t')
                    self.annot_seq[self.curIdx] = 2
        else:
            self.curIdx = None
            
    def keyPressEvent(self, event):
        # 9 - annotate real EMG twitch
        if event.key() == QtCore.Qt.Key_9:
            if self.curPoint:
                print(f'{self.curIdx} is good!')
                self.click_updateCur(itemAction='none', pointAction='twitch')
            elif self.curItem:
                print(f'{self.curSeqi[0]} - {self.curSeqi[-1]} is a good sequence!')
                self.click_updateCur(itemAction='twitch', pointAction='none')
        # 0 - annotate noise
        elif event.key() == QtCore.Qt.Key_0:
            if self.curPoint:
                print(f'{self.curIdx} is a noise index')
                self.click_updateCur(itemAction='none', pointAction='noise')
            elif self.curItem:
                print(f'{self.curSeqi[0]} - {self.curSeqi[-1]} is a noise sequence')
                self.click_updateCur(itemAction='noise', pointAction='none')
        
        # N - show noise
        elif event.key() == QtCore.Qt.Key_N:
            if self.show_noise==True:
                self.show_noise = False
                #self.cur_pthres.setPen(None)
            elif self.show_noise==False:
                self.show_noise = True
                #self.cur_pthres.setPen((102,205,170),width=1) 
            for a in self.cur_noise:
                a.setVisible(self.show_noise)
        
        # L - show LFP
        elif event.key() == QtCore.Qt.Key_L:
            if self.plot_pwaves==True:
                self.plot_pwaves = False
                self.plotView.removeItem(self.graph_LFP)
            elif self.plot_pwaves==False:
                self.plot_pwaves = True
                self.plotView.addItem(self.graph_LFP, row=1, col=0)
        
        # K - show P-wave spi(k)e detection threshold
        elif event.key() == QtCore.Qt.Key_K:
            if self.show_pthres==True:
                self.show_pthres = False
                self.cur_pthres.setPen(None)
            elif self.show_pthres==False:
                self.show_pthres = True
                self.cur_pthres.setPen((102,205,170),width=1)
        # D - show (d)ots as P-wave indices
        elif event.key() == QtCore.Qt.Key_D:
            if self.show_pidx==True:
                self.show_pidx = False
            elif self.show_pidx==False:
                self.show_pidx = True
            self.cur_pidx.setVisible(self.show_pidx)
        # P - show laser (p)ulses
        elif event.key() == QtCore.Qt.Key_P:
            if self.show_laser==True:
                self.show_laser = False
            elif self.show_laser==False:
                self.show_laser = True
            self.cur_laser.setVisible(self.show_laser)
    
    def plot_figure(self):
        self.pw.plot_data()
        self.pw.show()
        #plotWindow = EMGTwitchFigure(parent=self)
        #plotWindow.exec_()
        
    def closeEvent(self, event):
        plt.close(self.pw.fig)
        # update params in parent window
        if self.parent is not None:
            ddict = self.dict_from_vars()
            self.parent.pw.update_emg_params(ddict=ddict)
            self.parent.pw.update_twitch_params(ddict=ddict)
            self.parent.pw.update_gui_from_vars()
            print('Twitch settings updated!')
    
    def setup_gui(self):
        try:
            #self.centralWidget = QtWidgets.QWidget()
            #self.centralLayout = QtWidgets.QHBoxLayout(self.centralWidget)
            self.centralLayout = QtWidgets.QHBoxLayout(self)
            
            # set up layout for main window
            self.plotWidget = QtWidgets.QWidget()
            self.plotLayout = QtWidgets.QVBoxLayout(self.plotWidget)
            # header
            self.plotHeadWidget = QtWidgets.QWidget()
            self.plotHeadLayout = QtWidgets.QHBoxLayout(self.plotHeadWidget)
            self.plotBack_btn = QtWidgets.QPushButton()
            self.plotBack_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
            self.plotBack_btn.setFixedSize(30,30)
            self.plotBack_btn.setIconSize(QtCore.QSize(25,25))
            self.plotBack_btn.setStyleSheet('background:darkgray')
            self.plotNext_btn = QtWidgets.QPushButton()
            self.plotNext_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
            self.plotNext_btn.setFixedSize(30,30)
            self.plotNext_btn.setIconSize(QtCore.QSize(25,25))
            self.plotNext_btn.setStyleSheet('background:darkgray')
            self.plotTitle = QtWidgets.QLabel()
            self.plotTitle.setAlignment(QtCore.Qt.AlignCenter)
            self.plotTitle.setStyleSheet('color:white')
            font = QtGui.QFont()
            font.setPointSize(15)
            self.plotTitle.setFont(font)
            self.plotHeadLayout.addWidget(self.plotBack_btn, stretch=0)
            self.plotHeadLayout.addWidget(self.plotTitle, stretch=2)
            self.plotHeadLayout.addWidget(self.plotNext_btn, stretch=0)
            self.plotHeadWidget.setStyleSheet('background-color:black')
            # main widget
            self.plotView = pg.GraphicsLayoutWidget()
            # plot raw EMG
            self.graph_rawEMG = pg.PlotItem()
            self.curRawEMG = pg.PlotDataItem()  # raw EMG for REM sequence
            self.curRawEMG.setPen((255,255,255),width=1)
            self.curRawSeq = pg.PlotDataItem()  # raw EMG for selected phasic seq
            self.curRawSeq.setPen((255,255,0,200),width=2)
            self.curRawIdx = pg.PlotDataItem()  # raw EMG for selected phasic idx
            self.curRawIdx.setPen((255,0,0),width=2)
            self.graph_rawEMG.vb.setMouseEnabled(x=True, y=True)
            ax = self.graph_rawEMG.getAxis(name='left')
            labelStyle = {'color': '#FFF', 'font-size': '12pt'}
            ax.setLabel('Raw EMG', units='uV', **labelStyle)
            # plot LFP
            self.graph_LFP = pg.PlotItem()
            self.curLFP = pg.PlotDataItem()  # LFP for REM sequence
            self.curLFP.setPen((255,255,255),width=1)
            self.curRawSeq2 = pg.PlotDataItem()  # LFP for selected phasic seq
            self.curRawSeq2.setPen((255,255,0,200),width=2)
            self.curRawIdx2 = pg.PlotDataItem()  # LFP for selected phasic idx
            self.curRawIdx2.setPen((255,0,0),width=2)
            self.cur_pidx = pg.PlotDataItem()    # P-wave indices in REM seq
            self.cur_pidx.setPen(None)
            self.cur_pidx.setSymbol('t1')
            self.cur_pidx.setSymbolPen((255,140,0),width=1)
            self.cur_pidx.setSymbolBrush((255,215,0))
            self.cur_pidx.setSymbolSize(7)
            self.cur_pthres = pg.InfiniteLine()  # P-wave detection threshold
            self.cur_pthres.setAngle(0)
            self.cur_pthres.setPen((102,205,170),width=1)
            self.cur_laser = pg.PlotDataItem()   # laser for REM sequence
            self.cur_laser.setPen((0,0,255),width=2)
            self.graph_LFP.vb.setMouseEnabled(x=True, y=True)
            self.graph_LFP.setXLink(self.graph_rawEMG.vb)
            ax = self.graph_LFP.getAxis(name='left')
            ax.setLabel('LFP', units='V', **labelStyle)
            # plot EMG amplitude
            self.graph_EMGdata = pg.PlotItem()
            self.curEMGdata = pg.PlotDataItem()  # EMG amplitude for REM sequence
            self.curEMGdata.setPen((255,255,255),width=1)
            #self.graph_EMGdata.addItem(self.curEMGdata)
            self.cur_thres = pg.InfiniteLine()  # twitch detection threshold
            self.cur_thres.setAngle(0)
            self.cur_thres.setPen((0,255,0),width=1)
            labelOpts = {'position': 0.08, 'color':(0,255,0), 'movable':True}
            self.cur_thres.label = pg.InfLineLabel(self.cur_thres, text='', **labelOpts)
            #self.graph_EMGdata.addItem(self.cur_thres)
            
            self.graph_EMGdata.vb.setMouseEnabled(x=True, y=True)
            self.graph_EMGdata.setXLink(self.graph_rawEMG.vb)
            ax = self.graph_EMGdata.getAxis(name='left')
            ax.setLabel('EMG Ampl.', units='V', **labelStyle)       
            ax = self.graph_EMGdata.getAxis(name='bottom')
            ax.setLabel('Time (s)', **labelStyle)
            self.plotView.addItem(self.graph_rawEMG, row=0, col=0)
            if self.plot_pwaves:
                self.plotView.addItem(self.graph_LFP, row=1, col=0)
            self.plotView.addItem(self.graph_EMGdata, row=2, col=0)
            self.plotView.ci.layout.setRowMaximumHeight(0,125)
            self.plotView.ci.layout.setRowMaximumHeight(1,125)
            self.plotLayout.addWidget(self.plotHeadWidget, stretch=0)
            self.plotLayout.addWidget(self.plotView, stretch=2)
            self.plotLayout.setSpacing(0)
            
            self.centralLayout.addWidget(self.plotWidget)
            
            ### LAYOUT FOR SETTINGS BUTTONS ###
            self.settingsWidget = QtWidgets.QWidget()
            self.settingsLayout = QtWidgets.QVBoxLayout(self.settingsWidget)
            
            # title
            self.settingsTitle = QtWidgets.QLabel('SETTINGS', self.settingsWidget)
            self.settingsTitle.setFixedHeight(40)
            self.settingsTitle.setAlignment(QtCore.Qt.AlignHCenter)
            font = QtGui.QFont()
            font.setPointSize(12)
            font.setBold(True)
            self.settingsTitle.setFont(font)
            self.settingsLayout.addWidget(self.settingsTitle)
            
            ### EMG amplitude source ###
            # header
            self.ampsrcHeadWidget = QtWidgets.QWidget()
            self.ampsrcHeadLayout = QtWidgets.QHBoxLayout(self.ampsrcHeadWidget)
            self.ampsrcTitle = QtWidgets.QLabel('EMG Source')
            self.ampsrcTitle.setAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(10)
            font.setBold(True)
            self.ampsrcTitle.setFont(font)
            self.ampsrcHeadLayout.addWidget(self.ampsrcTitle)
            # main widget
            self.ampsrcWidget = QtWidgets.QWidget()
            self.ampsrcLayout = QtWidgets.QVBoxLayout(self.ampsrcWidget)
            self.ampsrcLayout.setSpacing(0)
            self.ampsrcLayout.setContentsMargins(10,0,25,0)
            self.useRaw_btn = QtWidgets.QRadioButton()
            self.useRaw_btn.setText('  Raw EMG   ')
            self.useSP_btn = QtWidgets.QRadioButton()
            self.useSP_btn.setText('     EMG\nspectrogram')
            self.ampsrcLayout.addWidget(self.useRaw_btn, alignment=QtCore.Qt.AlignCenter)
            self.ampsrcLayout.addWidget(self.useSP_btn, alignment=QtCore.Qt.AlignCenter)
            
            ### REM sleep parameters ###
            # header
            self.remHeadWidget = QtWidgets.QWidget()
            self.remHeadLayout = QtWidgets.QHBoxLayout(self.remHeadWidget)
            self.remTitle = QtWidgets.QLabel('REM Sleep Parameters')
            self.remTitle.setAlignment(QtCore.Qt.AlignCenter)
            self.remTitle.setFont(font)
            self.remHead_update = QtWidgets.QPushButton()
            self.remHeadLayout.addWidget(self.remTitle)
            self.remHeadLayout.addWidget(self.remHead_update, alignment=QtCore.Qt.AlignRight)
            # main widget
            self.remWidget = QtWidgets.QWidget()
            self.remLayout = QtWidgets.QGridLayout(self.remWidget)
            self.remLayout.setHorizontalSpacing(20)
            self.mindur_label = QtWidgets.QLabel('Min. REM\nDuration (s)')
            self.mindur_label.setAlignment(QtCore.Qt.AlignHCenter)
            self.mindur_val = QtWidgets.QDoubleSpinBox()
            self.mindur_val.setMinimum(0)
            self.mindur_val.setMaximum(60)
            self.mindur_val.setDecimals(1)
            self.mindur_val.setSuffix(' s')
            self.cutoff_label = QtWidgets.QLabel('End REM\nCutoff (s)')
            self.cutoff_label.setAlignment(QtCore.Qt.AlignHCenter)
            self.cutoff_val = QtWidgets.QDoubleSpinBox()
            self.cutoff_val.setMinimum(0)
            self.cutoff_val.setMaximum(10)
            self.cutoff_val.setSingleStep(2.5)
            self.cutoff_val.setDecimals(1)
            self.cutoff_val.setSuffix(' s')
            self.remLayout.addWidget(self.mindur_label, 0, 0)
            self.remLayout.addWidget(self.mindur_val, 1, 0)
            self.remLayout.addWidget(self.cutoff_label, 0, 1)
            self.remLayout.addWidget(self.cutoff_val, 1, 1)
            
            self.ampsrcremHeadWidget = QtWidgets.QWidget()
            self.ampsrcremHeadWidget.setFixedHeight(42)
            self.ampsrcremHeadLayout = QtWidgets.QHBoxLayout(self.ampsrcremHeadWidget)
            self.ampsrcremHeadLayout.setContentsMargins(0,0,0,0)
            self.ampsrcremHeadLayout.addWidget(self.ampsrcHeadWidget, stretch=2)
            self.ampsrcremHeadLayout.addWidget(self.remHeadWidget, stretch=2)
            self.settingsLayout.addWidget(self.ampsrcremHeadWidget)
            self.ampsrcremWidget = QtWidgets.QWidget()
            self.ampsrcremWidget.setFixedHeight(95)
            self.ampsrcremLayout = QtWidgets.QHBoxLayout(self.ampsrcremWidget)
            line_1 = QtWidgets.QFrame(self.ampsrcremWidget)
            line_1.setFrameShape(QtWidgets.QFrame.VLine)
            line_1.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.ampsrcremLayout.addWidget(self.ampsrcWidget, stretch=1)
            self.ampsrcremLayout.addWidget(line_1, stretch=0)
            self.ampsrcremLayout.addWidget(self.remWidget, stretch=2)
            self.settingsLayout.addWidget(self.ampsrcremWidget)
            
            line_2 = QtWidgets.QFrame(self.settingsWidget)
            line_2.setFrameShape(QtWidgets.QFrame.HLine)
            line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.settingsLayout.addWidget(line_2, stretch=0)
            
            ### EMG filtering parameters ###
            # header
            self.emgfiltHeadWidget = QtWidgets.QWidget()
            self.emgfiltHeadWidget.setFixedHeight(42)
            self.emgfiltHeadLayout = QtWidgets.QHBoxLayout(self.emgfiltHeadWidget)
            self.emgfiltTitle = QtWidgets.QLabel('Raw EMG Parameters')
            self.emgfiltTitle.setAlignment(QtCore.Qt.AlignCenter)
            self.emgfiltTitle.setFont(font)
            self.emgfiltHead_update = QtWidgets.QPushButton()
            self.emgfiltHead_btn = QtWidgets.QPushButton()
            self.emgfiltHeadLayout.addWidget(self.emgfiltTitle, stretch=2)
            self.emgfiltHeadLayout.addWidget(self.emgfiltHead_update, stretch=0, alignment=QtCore.Qt.AlignRight)
            self.emgfiltHeadLayout.addWidget(self.emgfiltHead_btn, stretch=0, alignment=QtCore.Qt.AlignRight)
            # main widget
            self.emgfiltWidget = QtWidgets.QWidget()
            self.emgfiltWidget.setFixedHeight(90)
            self.emgfiltLayout = QtWidgets.QGridLayout(self.emgfiltWidget)
            self.emgfiltType_label = QtWidgets.QLabel('Filter Type')
            self.emgfiltType_label.setAlignment(QtCore.Qt.AlignCenter)
            self.emgfilt_type = QtWidgets.QComboBox()
            self.emgfilt_type.addItems(['None', 'Low-pass', 'High-pass', 'Band-pass'])
            self.emgfiltLo_val = QtWidgets.QDoubleSpinBox()
            self.emgfiltLo_val.setMinimum(1)
            self.emgfiltLo_val.setDecimals(0)
            self.emgfiltLo_val.setMaximumWidth(65)
            self.emgfiltLo_val.setSuffix('Hz')
            self.emgfiltHi_val = QtWidgets.QDoubleSpinBox()
            self.emgfiltHi_val.setMinimum(0)
            self.emgfiltHi_val.setDecimals(0)
            self.emgfiltHi_val.setMaximumWidth(65)
            self.emgfiltHi_val.setSuffix('Hz')
            self.emgfiltFreq_label1 = QtWidgets.QLabel('<')
            self.emgfiltFreq_label2 = QtWidgets.QLabel('freq')
            self.emgfiltFreq_label3 = QtWidgets.QLabel('<')
            font_math = QtGui.QFont('Cambria', 9, 1, True)
            self.emgfiltFreq_label1.setFont(font_math)
            self.emgfiltFreq_label2.setFont(font_math)
            self.emgfiltFreq_label3.setFont(font_math)
            self.emgfiltProc_label = QtWidgets.QLabel('Signal Processing')
            self.emgfiltProc_label.setAlignment(QtCore.Qt.AlignCenter)
            self.emgfiltDn_label = QtWidgets.QLabel('Downsample')
            self.emgfiltDn_val = QtWidgets.QDoubleSpinBox()
            self.emgfiltDn_val.setMinimum(1)
            self.emgfiltDn_val.setDecimals(0)
            self.emgfiltDn_val.setMaximumWidth(45)
            self.emgfiltSm_label = QtWidgets.QLabel('Smooth')
            self.emgfiltSm_val = QtWidgets.QDoubleSpinBox()
            self.emgfiltSm_val.setMinimum(0)
            self.emgfiltSm_val.setMaximum(1000)
            self.emgfiltSm_val.setDecimals(0)
            self.emgfiltSm_val.setMaximumWidth(45)
            self.emgfiltLayout.addWidget(self.emgfiltType_label, 0, 0, 1, 5)
            self.emgfiltLayout.addWidget(self.emgfilt_type, 1, 0, 1, 5)
            self.emgfiltLayout.addWidget(self.emgfiltLo_val, 2, 0)
            self.emgfiltLayout.addWidget(self.emgfiltFreq_label1, 2, 1)
            self.emgfiltLayout.addWidget(self.emgfiltFreq_label2, 2, 2)
            self.emgfiltLayout.addWidget(self.emgfiltFreq_label3, 2, 3)
            self.emgfiltLayout.addWidget(self.emgfiltHi_val, 2, 4)
            self.emgfiltLayout.addWidget(self.emgfiltProc_label, 0, 5, 1, 2)
            self.emgfiltLayout.addWidget(self.emgfiltDn_label, 1, 5, alignment=QtCore.Qt.AlignRight)
            self.emgfiltLayout.addWidget(self.emgfiltDn_val, 1, 6)
            self.emgfiltLayout.addWidget(self.emgfiltSm_label, 2, 5, alignment=QtCore.Qt.AlignRight)
            self.emgfiltLayout.addWidget(self.emgfiltSm_val, 2, 6)
            self.emgfiltLayout.setColumnStretch(5,2)
            self.settingsLayout.addWidget(self.emgfiltHeadWidget)
            self.settingsLayout.addWidget(self.emgfiltWidget)
            line_3 = QtWidgets.QFrame(self.settingsWidget)
            line_3.setFrameShape(QtWidgets.QFrame.HLine)
            line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.settingsLayout.addWidget(line_3, stretch=0)
            
            
            ### EMG spectrogram calculation settings ###
            # header
            self.mspHeadWidget = QtWidgets.QWidget()
            self.mspHeadWidget.setFixedHeight(42)
            self.mspHeadLayout = QtWidgets.QHBoxLayout(self.mspHeadWidget)
            self.mspTitle = QtWidgets.QLabel('EMG Spectrogram Parameters')
            self.mspTitle.setAlignment(QtCore.Qt.AlignCenter)
            self.mspTitle.setFont(font)
            self.mspHead_update = QtWidgets.QPushButton()
            self.mspHead_btn = QtWidgets.QPushButton()
            self.mspHeadLayout.addWidget(self.mspTitle, stretch=2)
            self.mspHeadLayout.addWidget(self.mspHead_update, stretch=0, alignment=QtCore.Qt.AlignRight)
            self.mspHeadLayout.addWidget(self.mspHead_btn, stretch=0, alignment=QtCore.Qt.AlignRight)
            # main widget
            self.mspWidget = QtWidgets.QWidget()
            self.mspWidget.setFixedHeight(80)
            self.mspLayout = QtWidgets.QGridLayout(self.mspWidget)
            self.mspLayout.setHorizontalSpacing(15)
            self.mspLayout.setVerticalSpacing(10)
            self.mspWindow_label = QtWidgets.QLabel('Window\nsize (s)')
            self.mspWindow_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mspWindow_val = QtWidgets.QDoubleSpinBox()
            self.mspWindow_val.setMinimum(0.1)
            self.mspWindow_val.setMaximum(10)
            self.mspWindow_val.setSingleStep(0.1)
            self.mspWindow_val.setSuffix(' s')
            self.mspWindow_val.setMaximumWidth(60)
            self.mspOverlap_label = QtWidgets.QLabel('%\noverlap')
            self.mspOverlap_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mspOverlap_val = QtWidgets.QDoubleSpinBox()
            self.mspOverlap_val.setMinimum(0)
            self.mspOverlap_val.setMaximum(100)
            self.mspOverlap_val.setDecimals(0)
            self.mspOverlap_val.setSuffix(' %')
            self.mspOverlap_val.setMaximumWidth(55)
            self.mspRes_label = QtWidgets.QLabel('mSP Res\n(s / bin)')
            self.mspRes_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mspRes_view = QtWidgets.QLabel()
            self.mspRes_view.setAlignment(QtCore.Qt.AlignCenter)
            self.mspRes_view.setFont(font)
            self.mspRes_label.setMaximumWidth(50)
            self.mspFreq_label = QtWidgets.QLabel('Freq. Band (Hz)')
            self.mspFreq_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mspFreqWidget = QtWidgets.QWidget()
            self.mspFreqLayout = QtWidgets.QHBoxLayout(self.mspFreqWidget)
            self.mspFreqLo_val = QtWidgets.QDoubleSpinBox()
            self.mspFreqLo_val.setMinimum(1)
            self.mspFreqLo_val.setDecimals(0)
            self.mspFreqLo_val.setSuffix(' Hz')
            self.mspFreq_dash = QtWidgets.QLabel('-')
            self.mspFreq_dash.setAlignment(QtCore.Qt.AlignCenter)
            self.mspFreqHi_val = QtWidgets.QDoubleSpinBox()
            self.mspFreqHi_val.setMinimum(0)
            self.mspFreqHi_val.setDecimals(0)
            self.mspFreqHi_val.setSuffix(' Hz')
            self.mspFreqLayout.addWidget(self.mspFreqLo_val, stretch=2)
            self.mspFreqLayout.addWidget(self.mspFreq_dash, stretch=0)
            self.mspFreqLayout.addWidget(self.mspFreqHi_val, stretch=2)
            self.mspFreqLayout.setContentsMargins(0,0,1,0)
            self.mspFreqLayout.setSpacing(5)
            self.mspLayout.addWidget(self.mspWindow_label, 0, 0)
            self.mspLayout.addWidget(self.mspOverlap_label, 0, 1)
            self.mspLayout.addWidget(self.mspRes_label, 0, 2)
            self.mspLayout.addWidget(self.mspFreq_label, 0, 3)
            self.mspLayout.addWidget(self.mspWindow_val, 1, 0)
            self.mspLayout.addWidget(self.mspOverlap_val, 1, 1)
            self.mspLayout.addWidget(self.mspRes_view, 1, 2)
            self.mspLayout.addWidget(self.mspFreqWidget, 1, 3)
            self.settingsLayout.addWidget(self.mspHeadWidget)
            self.settingsLayout.addWidget(self.mspWidget)
            
            line_4 = QtWidgets.QFrame(self.settingsWidget)
            line_4.setFrameShape(QtWidgets.QFrame.HLine)
            line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.settingsLayout.addWidget(line_4, stretch=0)
            
            ### EMG threshold settings ###
            # header
            self.thresHeadWidget = QtWidgets.QWidget()
            self.thresHeadWidget.setFixedHeight(42)
            self.thresHeadLayout = QtWidgets.QHBoxLayout(self.thresHeadWidget)
            self.thresTitle = QtWidgets.QLabel('Threshold Parameters')
            self.thresTitle.setAlignment(QtCore.Qt.AlignCenter)
            self.thresTitle.setFont(font)
            self.thresHead_update = QtWidgets.QPushButton()
            self.thresHead_btn = QtWidgets.QPushButton()
            self.thresHeadLayout.addWidget(self.thresTitle, stretch=2)
            self.thresHeadLayout.addWidget(self.thresHead_update, stretch=0, alignment=QtCore.Qt.AlignRight)
            self.thresHeadLayout.addWidget(self.thresHead_btn, stretch=0, alignment=QtCore.Qt.AlignRight)
            # main widget
            self.thresWidget = QtWidgets.QWidget()
            self.thresWidget.setFixedHeight(80)
            self.thresLayout = QtWidgets.QGridLayout(self.thresWidget)
            self.thresType = QtWidgets.QComboBox()
            self.thresType.addItems(['Raw value', 'Std. deviations', 'Percentile'])
            self.thres_val = QtWidgets.QDoubleSpinBox()
            self.thres_val.setMaximumWidth(75)
            self.thres_val.setMinimum(1)
            self.thres_val.setMaximum(100)
            self.thres_val.setSingleStep(0.1)
            self.thres_val.setDecimals(1)
            self.thmode1_btn = QtWidgets.QRadioButton()
            self.thmode1_btn.setText('All REM sleep')
            self.thmode2_btn = QtWidgets.QRadioButton()
            self.thmode2_btn.setText('Each REM period')
            self.thmode3_btn = QtWidgets.QCheckBox()
            self.thmode3_label1 = QtWidgets.QLabel('Use first')
            self.thmode3_label2 = QtWidgets.QLabel('second(s) of REM')
            self.thmode3_val = QtWidgets.QDoubleSpinBox()
            self.thmode3_val.setMinimum(0)
            self.thmode3_val.setMaximum(60)
            self.thmode3_val.setDecimals(0)
            self.thresLayout.addWidget(self.thresType, 0, 0, 1, 1)
            self.thresLayout.addWidget(self.thres_val, 1, 0, 1, 1, alignment=QtCore.Qt.AlignCenter)
            self.thresLayout.addWidget(self.thmode1_btn, 0, 1, 1, 1)
            self.thresLayout.addWidget(self.thmode2_btn, 1, 1, 1, 1)
            self.thresLayout.addWidget(self.thmode3_btn, 0, 2, 1, 1)
            self.thresLayout.addWidget(self.thmode3_label1, 0, 3, 1, 1)
            self.thresLayout.addWidget(self.thmode3_val, 0, 4, 1, 1)
            self.thresLayout.addWidget(self.thmode3_label2, 1, 3, 1, 2)
            self.settingsLayout.addWidget(self.thresHeadWidget)
            self.settingsLayout.addWidget(self.thresWidget)
            line_5 = QtWidgets.QFrame(self.settingsWidget)
            line_5.setFrameShape(QtWidgets.QFrame.HLine)
            line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.settingsLayout.addWidget(line_5, stretch=0)
            
            ### Twitch detection settings ###
            # header
            self.twitchHeadWidget = QtWidgets.QWidget()
            self.twitchHeadWidget.setFixedHeight(42)
            self.twitchHeadLayout = QtWidgets.QHBoxLayout(self.twitchHeadWidget)
            self.twitchTitle = QtWidgets.QLabel('Twitch Detection Parameters')
            self.twitchTitle.setAlignment(QtCore.Qt.AlignCenter)
            self.twitchTitle.setFont(font)
            self.twitchHead_update = QtWidgets.QPushButton()
            self.twitchHead_btn = QtWidgets.QPushButton()
            self.twitchHeadLayout.addWidget(self.twitchTitle, stretch=2)
            self.twitchHeadLayout.addWidget(self.twitchHead_update, stretch=0, alignment=QtCore.Qt.AlignRight)
            self.twitchHeadLayout.addWidget(self.twitchHead_btn, stretch=0, alignment=QtCore.Qt.AlignRight)
            # main widget
            self.twitchWidget = QtWidgets.QWidget()
            self.twitchWidget.setFixedHeight(60)
            self.twitchLayout = QtWidgets.QHBoxLayout(self.twitchWidget)
            self.twitchLayout.setSpacing(30)
            c1 = QtWidgets.QVBoxLayout()
            c1.setContentsMargins(0,0,0,0)
            c1.setSpacing(2)
            minTwitchDur_label = QtWidgets.QLabel('Min. Twitch Duration')
            minTwitchDur_label.setAlignment(QtCore.Qt.AlignHCenter)
            self.minTwitchDur_val = QtWidgets.QDoubleSpinBox()
            self.minTwitchDur_val.setMinimum(0)
            self.minTwitchDur_val.setMaximum(5000)
            self.minTwitchDur_val.setDecimals(2)
            self.minTwitchDur_val.setSingleStep(0.1)
            self.minTwitchDur_val.setSuffix(' s')
            c1.addWidget(minTwitchDur_label)
            c1.addWidget(self.minTwitchDur_val)
            c2 = QtWidgets.QVBoxLayout()
            c2.setContentsMargins(0,0,0,0)
            c2.setSpacing(2)
            minTwitchSep_label = QtWidgets.QLabel('Min. Twitch Separation')
            minTwitchSep_label.setAlignment(QtCore.Qt.AlignHCenter)
            self.minTwitchSep_val = QtWidgets.QDoubleSpinBox()
            self.minTwitchSep_val.setMinimum(0)
            self.minTwitchSep_val.setMaximum(5000)
            self.minTwitchSep_val.setDecimals(2)
            self.minTwitchSep_val.setSingleStep(0.1)
            self.minTwitchSep_val.setSuffix(' s')
            c2.addWidget(minTwitchSep_label)
            c2.addWidget(self.minTwitchSep_val)
            self.twitchLayout.addLayout(c1)
            self.twitchLayout.addLayout(c2)
            self.settingsLayout.addWidget(self.twitchHeadWidget)
            self.settingsLayout.addWidget(self.twitchWidget)
            line_6 = QtWidgets.QFrame(self.settingsWidget)
            line_6.setFrameShape(QtWidgets.QFrame.HLine)
            line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.settingsLayout.addWidget(line_6, stretch=0)
            
            ### ADD VARIABLE (add to gui)
            
            ### Action buttons ###
            self.btnsWidget = QtWidgets.QWidget()
            self.btnsWidget.setFixedHeight(40)
            self.btnsLayout = QtWidgets.QHBoxLayout(self.btnsWidget)
            self.updatePlot_btn = QtWidgets.QPushButton('Apply')
            self.updatePlot_btn.setDefault(True)
            self.plotFigure_btn = QtWidgets.QPushButton('Plot')
            self.saveSettings_btn = QtWidgets.QPushButton('Save')
            self.loadSettings_btn = QtWidgets.QPushButton('Load')
            self.moreSettings_btn = QtWidgets.QPushButton('More ...')
            self.btnsLayout.addWidget(self.updatePlot_btn)
            self.btnsLayout.addWidget(self.plotFigure_btn)
            self.btnsLayout.addWidget(self.saveSettings_btn)
            self.btnsLayout.addWidget(self.loadSettings_btn)
            #self.btnsLayout.addWidget(self.moreSettings_btn)
            self.btnsLayout.setSpacing(5)
            self.settingsLayout.addWidget(self.btnsWidget)
            
            self.debug_btn = QtWidgets.QPushButton('DEBUG')
            self.debug_btn.clicked.connect(self.debug)
            self.settingsLayout.addWidget(self.debug_btn)

            # settings layout spacing
            self.settingsLayout.setContentsMargins(0,0,0,0)
            self.settingsLayout.setSpacing(0)
            
            self.settingsLayout.setStretch(0,0)  # settings title
            self.settingsLayout.setStretch(1,0)  # EMG source/REM params # header
            self.settingsLayout.setStretch(2,2)                          # main widget
            self.settingsLayout.setStretch(3,0)                          # line
            self.settingsLayout.setStretch(4,0)  # raw EMG params
            self.settingsLayout.setStretch(5,2)
            self.settingsLayout.setStretch(6,0)
            self.settingsLayout.setStretch(7,0)  # mSP params 
            self.settingsLayout.setStretch(8,2)
            self.settingsLayout.setStretch(9,0)
            self.settingsLayout.setStretch(10,0) # threshold params
            self.settingsLayout.setStretch(11,2)
            self.settingsLayout.setStretch(12,0)
            self.settingsLayout.setStretch(13,0) # twitch params
            self.settingsLayout.setStretch(14,2)
            self.settingsLayout.setStretch(15,0)
            self.settingsLayout.setStretch(16,0) # action buttons
            
            self.settingsWidget.setFixedWidth(370)
            self.centralLayout.addWidget(self.settingsWidget)
            #self.setCentralWidget(self.centralWidget)
            
            # collect action buttons and widgets
            self.head_btns = {'ampsrc' : [False, False, self.ampsrcWidget],
                              'rem' : [self.remHead_update, False, self.remWidget],
                              'emgfilt' : [self.emgfiltHead_update, self.emgfiltHead_btn, self.emgfiltWidget],
                              'msp' : [self.mspHead_update, self.mspHead_btn, self.mspWidget],
                              'thres' : [self.thresHead_update, self.thresHead_btn, self.thresWidget],
                              'twitch' : [self.twitchHead_update, self.twitchHead_btn, self.twitchWidget]}
            for btn in self.head_btns.keys():
                u = self.head_btns[btn][0]
                if u:
                    u.setIcon(QtGui.QIcon('graph_icon.png'))
                    u.setFixedSize(25,25)
                    u.setIconSize(QtCore.QSize(20, 20))
                b = self.head_btns[btn][1]
                if b:
                    b.setCheckable(True)
                    b.setChecked(True)
                    b.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp))
                    b.setFixedSize(25,25)
                    b.setIconSize(QtCore.QSize(20,20))

        except Exception as e:
            print('whoopsie')
            print(e)
            sys.exit()
    
    def connect_buttons(self):
        # update EMG source variable
        self.useRaw_btn.toggled.connect(self.update_emg_source)
        self.useRaw_btn.toggled.connect(self.update_emg_source_gui)
        self.useRaw_btn.clicked.connect(self.save_emg_source_thres)
        self.useSP_btn.clicked.connect(self.save_emg_source_thres)
        
        # update REM sleep variables
        self.mindur_val.valueChanged.connect(self.update_rem_params)
        self.cutoff_val.valueChanged.connect(self.update_rem_params)
        
        # update raw EMG variables
        self.emgfilt_type.currentTextChanged.connect(self.update_raw_params)
        self.emgfilt_type.currentTextChanged.connect(self.update_emg_source_gui)
        self.emgfiltLo_val.valueChanged.connect(self.update_raw_params)
        self.emgfiltHi_val.valueChanged.connect(self.update_raw_params)
        self.emgfiltDn_val.valueChanged.connect(self.update_raw_params)
        self.emgfiltSm_val.valueChanged.connect(self.update_raw_params)
        
        # update EMG spectrogram variables
        self.mspWindow_val.valueChanged.connect(self.update_msp_params)
        self.mspOverlap_val.valueChanged.connect(self.update_msp_params)
        self.mspFreqLo_val.valueChanged.connect(self.update_msp_params)
        self.mspFreqHi_val.valueChanged.connect(self.update_msp_params)
        
        # update threshold variables
        self.thresType.currentTextChanged.connect(self.update_thres_params)
        self.thres_val.valueChanged.connect(self.update_thres_params)
        self.thmode1_btn.toggled.connect(self.update_thres_params)
        self.thmode3_btn.toggled.connect(self.update_thres_params)
        self.thmode3_val.valueChanged.connect(self.update_thres_params)
        
        ### ADDITIONAL SETTINGS 
        # update twitch detection variables
        self.minTwitchDur_val.valueChanged.connect(self.update_twitch_params)
        self.minTwitchSep_val.valueChanged.connect(self.update_twitch_params)
        
        ### ADD VARIABLE (connect to updating functions)
        
        # show/hide parameters, update plots
        for btn in self.head_btns.keys():
            u = self.head_btns[btn][0]
            if u:
                u.clicked.connect(self.update_plot)
            b = self.head_btns[btn][1]
            if b:
                b.toggled.connect(self.show_hide_settings)
        
        # connect action buttons
        self.updatePlot_btn.clicked.connect(self.update_plot)
        self.plotFigure_btn.clicked.connect(self.plot_figure)
        self.saveSettings_btn.clicked.connect(self.save_settings)
        self.loadSettings_btn.clicked.connect(self.load_settings)
        
        # connect plot back/forward buttons
        self.plotBack_btn.clicked.connect(self.plot_next_seq)
        self.plotNext_btn.clicked.connect(self.plot_next_seq)
        
                
    def show_hide_settings(self):
        b = self.sender()
        for btn in self.head_btns.keys():
            if b == self.head_btns[btn][1]:
                if b.isChecked():
                    b.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp))
                    hide = False
                else:
                    b.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowDown))
                    hide = True
                self.head_btns[btn][2].setHidden(hide)
                break
    
    @QtCore.pyqtSlot()
    def load_settings(self, sfile=[]):
        # load variable values from saved dictionary
        if not sfile:
            sfile = QtWidgets.QFileDialog().getOpenFileName(self, "Load your settings", 
                                                           os.path.join(self.ppath, self.name), 
                                                           "Dictionaries (*.pkl)")[0]
        if sfile:
            with open(sfile, mode='rb') as f:
                # load settings dict, set local variables, update GUI
                ddict = pickle.load(f)
                self.update_vars_from_dict(ddict)
                self.update_gui_from_vars()
                self.usrAnnot = np.array(ddict['annot'])
                
                if self.sender() == self.loadSettings_btn:
                    print('Settings loaded!')
                    #pdb.set_trace()
                    #self.update_plot()
                    self.updatePlot_btn.click()
            self.curFile = str(sfile)
    
    @QtCore.pyqtSlot()
    def save_settings(self, sfile=[]):
        if self.sender() == self.saveSettings_btn:
            # update variables from GUI values
            self.update_vars_from_gui()
            
        if not sfile:
            # choose directory & file name to save settings
            sfile = QtWidgets.QFileDialog().getSaveFileName(self, "Save your settings", 
                                                           os.path.join(self.ppath, self.name), 
                                                           "Dictionaries (*.pkl)")[0]
        if sfile:
            if sfile[-4:] != '.pkl':
                sfile = sfile + '.pkl' if '.' not in sfile else sfile[0:sfile.index('.')] + '.pkl'
            # get dictionary of current variable values, save as .pkl file
            ddict = self.dict_from_vars()
            if self.sender() == self.saveSettings_btn:
                self.usrAnnot[self.seqi] = self.annot_seq
                if ddict != self.plotSettings:
                    # warn user if GUI settings are different from plot settings
                    res = warning_dlg('Settings are different from current plot parameters - continue?')
                    if res == 0:
                        return
                    elif res == 1:
                        ddict['annot'] = []
                        msg = 'Settings saved!'
                else:
                    ddict['annot'] = np.array(self.usrAnnot)
                    msg = 'Settings and annotation saved!'
            else:
                ddict['annot'] = []
                msg = 'Default settings saved'
            # save settings file
            with open(sfile, mode='wb') as f:
                pickle.dump(ddict, f)
                print(msg)
            # save currently detected EMG twitches
            if self.sender() == self.saveSettings_btn:
                dpath = os.path.join(self.ppath, self.name, 'emg_twitches.mat')
                #twdata = np.concatenate([self.usrAnnot[rs] for rs in self.EMG_remseq])
                rdata = np.concatenate(self.EMG_remseq)
                edata = np.array(self.EMGdata)
                edata[self.EMGdata_ni] = np.nan
                so.savemat(dpath, {'twitches'     : self.usrAnnot,  # twitch annotation vector for EMG_amp
                                   'remidx'       : rdata,          # indices of REM sleep in twitch annot. vector
                                   'mnbin'        : self.mnbin,     # no. Intan samples per EMG amp bin
                                   'mdt'          : self.mdt,       # no. seconds per EMG amp bin
                                   'EMG_amp'      : edata,         # EMG amplitude vector (noise=NaNs)
                                   'settingsfile' : sfile})         # saved file with settings for twitches/EMG amp calculation
                # save EMG amplitude vector
                # epath = os.path.join(self.ppath, self.name, f'emg_amp_{self.name}.mat')
                # so.savemat(dpath, {'phasic_emg'   : np.array(self.phasic_remseqs, dtype='object'),
                #                    'mnbin'        : self.mnbin,
                #                    'mdt'          : self.mdt,
                #                    'settingsfile' : sfile})
            self.curFile = sfile
        
    def plot_next_seq(self):
        # previous REM sequence
        if self.sender() == self.plotBack_btn:
            if self.idx > 0:
                self.idx -= 1
        # next REM sequence
        elif self.sender() == self.plotNext_btn:
            if self.idx < self.numrem-1:
                self.idx += 1
        self.usrAnnot[self.seqi] = self.annot_seq  # update annotation
        self.ampsrc = self.plotSettings['ampsrc']  # keep plot EMG source
        self.plot_data()
            
    def update_emg_source_gui(self):
        if self.useRaw_btn.isChecked():
            # enable raw EMG settings
            self.emgfilt_type.setDisabled(False)
            # enable/disable freq inputs based on selected filter
            if self.emgfilt_type.currentText() == 'None':
                self.emgfiltLo_val.setDisabled(True)
                self.emgfiltHi_val.setDisabled(True)
                self.emgfiltFreq_label1.setStyleSheet('color:gray')
                self.emgfiltFreq_label2.setStyleSheet('color:gray')
                self.emgfiltFreq_label3.setStyleSheet('color:gray')
            else:
                self.emgfiltFreq_label2.setStyleSheet('color:black')
                if self.emgfilt_type.currentText() == 'Low-pass':
                    self.emgfiltLo_val.setDisabled(True)
                    self.emgfiltHi_val.setDisabled(False)
                    self.emgfiltFreq_label1.setStyleSheet('color:gray')
                    self.emgfiltFreq_label3.setStyleSheet('color:black')
                elif self.emgfilt_type.currentText() == 'High-pass':
                    self.emgfiltLo_val.setDisabled(False)
                    self.emgfiltHi_val.setDisabled(True)
                    self.emgfiltFreq_label1.setStyleSheet('color:black')
                    self.emgfiltFreq_label3.setStyleSheet('color:gray')
                elif self.emgfilt_type.currentText() == 'Band-pass':
                    self.emgfiltLo_val.setDisabled(False)
                    self.emgfiltHi_val.setDisabled(False)
                    self.emgfiltFreq_label1.setStyleSheet('color:black')
                    self.emgfiltFreq_label3.setStyleSheet('color:black')
            # disable mSP settings
            self.mspWindow_val.setDisabled(True)
            self.mspOverlap_val.setDisabled(True)
            self.mspRes_view.setStyleSheet('color:gray')
            self.mspFreqLo_val.setDisabled(True)
            self.mspFreq_dash.setStyleSheet('color:gray')
            self.mspFreqHi_val.setDisabled(True)
            # show raw EMG settings
            self.emgfiltHead_btn.setChecked(True)
            self.emgfiltWidget.setHidden(False)
            self.emgfiltHead_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp))
            # hide mSP settings
            self.mspHead_btn.setChecked(False)
            self.mspWidget.setHidden(True)
            self.mspHead_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowDown))
        elif self.useSP_btn.isChecked():
            # disable raw EMG settings
            self.emgfilt_type.setDisabled(True)
            self.emgfiltLo_val.setDisabled(True)
            self.emgfiltHi_val.setDisabled(True)
            self.emgfiltFreq_label1.setStyleSheet('color:gray')
            self.emgfiltFreq_label2.setStyleSheet('color:gray')
            self.emgfiltFreq_label3.setStyleSheet('color:gray')
            # enable mSP settings
            self.mspWindow_val.setDisabled(False)
            self.mspOverlap_val.setDisabled(False)
            self.mspRes_view.setStyleSheet('color:black')
            self.mspFreqLo_val.setDisabled(False)
            self.mspFreq_dash.setStyleSheet('color:black')
            self.mspFreqHi_val.setDisabled(False)
            # hide raw EMG settings
            self.emgfiltHead_btn.setChecked(False)
            self.emgfiltWidget.setHidden(True)
            self.emgfiltHead_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowDown))
            # show mSP settings
            self.mspHead_btn.setChecked(True)
            self.mspWidget.setHidden(False)
            self.mspHead_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp))
    
    def update_emg_source(self):
        if self.useRaw_btn.isChecked():
            self.ampsrc = 'raw'
        elif self.useSP_btn.isChecked():
            self.ampsrc = 'msp'
            
    def save_emg_source_thres(self):
        # current threshold params
        curThres = {'thres':float(self.thres),
                    'thres_type':str(self.thres_type),
                    'thres_mode':int(self.thres_mode),
                    'thres_first':float(self.thres_first)}
        # click on useRaw button --> save mSP thresholds, load raw thresholds
        if self.sender() == self.useRaw_btn:
            self.saveThres['msp'] = curThres
            tdict = self.saveThres['raw']

        # click on useSP button --> save raw thresholds, load mSP thresholds
        elif self.sender() == self.useSP_btn:
            self.saveThres['raw'] = curThres
            tdict = self.saveThres['msp']

        # update threshold params from saved dictionary
        if tdict:
            self.thres_val.setValue(tdict['thres'])
            if tdict['thres_type'] == 'raw':
                self.thresType.setCurrentIndex(0)
                self.thres_val.setSuffix(' uV')
            elif tdict['thres_type'] == 'std':
                self.thresType.setCurrentIndex(1)
                self.thres_val.setSuffix(' s.t.d')
            elif tdict['thres_type'] == 'perc':
                self.thresType.setCurrentIndex(2)
                self.thres_val.setSuffix(' %')
            if tdict['thres_mode'] == 1:
                self.thmode1_btn.setChecked(True)
                #self.thmode2_btn.setChecked(False)
            elif tdict['thres_mode'] == 2:
                #self.thmode1_btn.setChecked(False)
                self.thmode2_btn.setChecked(True)
            if tdict['thres_first'] > 0:
                self.thmode3_btn.setChecked(True)
                self.thmode3_val.setDisabled(False)
            else:
                self.thmode3_btn.setChecked(False)
                self.thmode3_val.setDisabled(True)
    
    def update_rem_params(self):
        self.min_dur = float(self.mindur_val.value())
        self.rem_cutoff = float(self.cutoff_val.value())
    
    def update_raw_params(self):
        if self.emgfilt_type.currentText() == 'None':
            self.w0_raw = -1
            self.w1_raw = -1
        elif self.emgfilt_type.currentText() == 'Low-pass':
            self.w0_raw = -1
            self.w1_raw = float(self.emgfiltHi_val.value() / (self.sr/2))
        elif self.emgfilt_type.currentText() == 'High-pass':
            self.w0_raw = float(self.emgfiltLo_val.value()  / (self.sr/2))
            self.w1_raw = -1
        elif self.emgfilt_type.currentText() == 'Band-pass':
            self.w0_raw = float(self.emgfiltLo_val.value()  / (self.sr/2))
            self.w1_raw = float(self.emgfiltHi_val.value()  / (self.sr/2))
        self.dn_raw = int(round(self.emgfiltDn_val.value()))
        self.sm_raw = float(self.emgfiltSm_val.value())
    
    def update_msp_params(self):
        self.nsr_seg_msp = float(self.mspWindow_val.value())
        self.perc_overlap_msp = float(self.mspOverlap_val.value()/100)
        self.mspRes_view.setText(str(round(self.nsr_seg_msp - self.nsr_seg_msp*self.perc_overlap_msp,2)))
        self.r_mu = [float(self.mspFreqLo_val.value()), float(self.mspFreqHi_val.value())]
    
    def update_thres_params(self):
        # threshold variable
        self.thres = float(self.thres_val.value())
        # threshold type
        if self.thresType.currentIndex() == 0:
            self.thres_type = 'raw'
            self.thres_val.setSuffix(' uV')
        elif self.thresType.currentIndex() == 1:
            self.thres_type = 'std'
            self.thres_val.setSuffix(' s.t.d')
        elif self.thresType.currentIndex() == 2:
            self.thres_type = 'perc'
            self.thres_val.setSuffix('  %')
        # threshold mode
        if self.thmode1_btn.isChecked():
            self.thres_mode = 1
        elif self.thmode2_btn.isChecked():
            self.thres_mode = 2
        # threshold by first X seconds
        if self.thmode3_btn.isChecked():
            self.thmode3_val.setDisabled(False)
            self.thres_first = float(self.thmode3_val.value())
        else:
            self.thmode3_val.setDisabled(True)
            self.thres_first = float(0)
        
    def update_twitch_params(self):
        self.min_twitchdur = float(self.minTwitchDur_val.value())
        self.min_twitchsep = float(self.minTwitchSep_val.value())
    
    ### ADD VARIABLE (update variable functions)
    
    def dict_from_vars(self):
        ### ADD VARIABLE (save in settings dictionary)
        ddict = {'ampsrc':str(self.ampsrc),
                 'min_dur':float(self.min_dur),
                 'rem_cutoff':float(self.rem_cutoff),
                 'w0_raw':float(self.w0_raw),
                 'w1_raw':float(self.w1_raw),
                 'dn_raw':int(self.dn_raw),
                 'sm_raw':float(self.sm_raw),
                 'nsr_seg_msp':float(self.nsr_seg_msp),
                 'perc_overlap_msp':float(self.perc_overlap_msp),
                 'r_mu':list(self.r_mu),
                 'thres':float(self.thres),
                 'thres_type':str(self.thres_type),
                 'thres_mode':int(self.thres_mode),
                 'thres_first':float(self.thres_first),
                 'min_twitchdur':float(self.min_twitchdur),
                 'min_twitchsep':float(self.min_twitchsep)}
        return ddict
    
    def update_gui_from_vars(self):
        ddict = self.dict_from_vars()
        # set EMG amplitude source
        if ddict['ampsrc'] == 'raw':
            self.useRaw_btn.setChecked(True)
            #self.useSP_btn.setChecked(False)
        elif ddict['ampsrc'] == 'msp':
            #self.useRaw_btn.setChecked(False)
            self.useSP_btn.setChecked(True)
        self.update_emg_source_gui()
        
        # set REM sleep params
        self.mindur_val.setValue(ddict['min_dur'])
        self.cutoff_val.setValue(ddict['rem_cutoff'])

        # set raw EMG params
        if ddict['w0_raw'] != -1 or ddict['w1_raw'] != -1:
            if ddict['w1_raw'] == -1:
                self.emgfilt_type.setCurrentIndex(2)
                self.emgfiltLo_val.setValue(ddict['w0_raw'] * (self.sr/2))
            elif ddict['w0_raw'] == -1:
                self.emgfilt_type.setCurrentIndex(1)
                self.emgfiltHi_val.setValue(ddict['w1_raw'] * (self.sr/2))
            else:
                self.emgfilt_type.setCurrentIndex(3)
                self.emgfiltLo_val.setValue(ddict['w0_raw'] * (self.sr/2))
                self.emgfiltHi_val.setValue(ddict['w1_raw'] * (self.sr/2))
        else:
             self.emgfilt_type.setCurrentIndex(0)
        self.emgfiltDn_val.setValue(ddict['dn_raw'])
        self.emgfiltSm_val.setValue(ddict['sm_raw'])
        
        # set EMG spectrogram params
        self.mspWindow_val.setValue(ddict['nsr_seg_msp'])
        self.mspOverlap_val.setValue(ddict['perc_overlap_msp']*100)
        self.mspRes_view.setText(str(round(ddict['nsr_seg_msp'] - ddict['nsr_seg_msp']*ddict['perc_overlap_msp'],2)))
        self.mspFreqLo_val.setValue(ddict['r_mu'][0])
        self.mspFreqHi_val.setValue(ddict['r_mu'][1])
        
        # set threshold params
        self.thres_val.setValue(ddict['thres'])
        if ddict['thres_type'] == 'raw':
            self.thresType.setCurrentIndex(0)
            self.thres_val.setSuffix(' uV')
        elif ddict['thres_type'] == 'std':
            self.thresType.setCurrentIndex(1)
            self.thres_val.setSuffix(' s.t.d')
        elif ddict['thres_type'] == 'perc':
            self.thresType.setCurrentIndex(2)
            self.thres_val.setSuffix(' %')
        if ddict['thres_mode'] == 1:
            self.thmode1_btn.setChecked(True)
            #self.thmode2_btn.setChecked(False)
        elif ddict['thres_mode'] == 2:
            #self.thmode1_btn.setChecked(False)
            self.thmode2_btn.setChecked(True)
        if ddict['thres_first'] > 0:
            self.thmode3_btn.setChecked(True)
            self.thmode3_val.setDisabled(False)
        else:
            self.thmode3_btn.setChecked(False)
            self.thmode3_val.setDisabled(True)
        
        # set twitch detection params
        self.minTwitchDur_val.setValue(ddict['min_twitchdur'])
        self.minTwitchSep_val.setValue(ddict['min_twitchsep'])
        
        ### ADD VARIABLE (update gui)
        
    def recalc_plot_data(self, calc_Raw=False, calc_mSP=False):
        if calc_Raw:
            # check if freq cutoffs or downsampling/smoothing factors changed from current plot
            x1 = self.w0_raw == self.plotSettings['w0_raw']
            x2 = self.w1_raw == self.plotSettings['w1_raw']
            x3 = self.dn_raw == self.plotSettings['dn_raw']
            x4 = self.sm_raw == self.plotSettings['sm_raw']
            if not all([x1,x2,x3,x4]):
                self.get_EMGdn()
        
        if calc_mSP:
            # recalculate mSP if FFT window size/overlap changed from current plot
            x1 = self.nsr_seg_msp == self.plotSettings['nsr_seg_msp']
            x2 = self.perc_overlap_msp == self.plotSettings['perc_overlap_msp']
            if not all([x1,x2]):
                self.get_mSP(recalc=True)
                self.calculate_EMGAmpl()
            else:
                # recalculate EMG amplitude if $r_mu changed from current plot
                if self.r_mu != self.plotSettings['r_mu']:
                    self.calculate_EMGAmpl()
            
    def update_vars_from_gui(self):
        ### ADD VARIABLE
        self.update_emg_source()
        self.update_rem_params()
        
        self.update_raw_params()
        self.update_msp_params()
        self.recalc_plot_data(calc_Raw=True, calc_mSP=True)
        
        self.update_thres_params()
        self.update_twitch_params()
        #self.update_offset()
    
    def update_vars_from_dict(self, ddict):
        try:
            ### ADD VARIABLE
            self.ampsrc = ddict['ampsrc']
            self.min_dur = ddict['min_dur']
            self.rem_cutoff = ddict['rem_cutoff']
            self.w0_raw = ddict['w0_raw']
            self.w1_raw = ddict['w1_raw']
            self.dn_raw = ddict['dn_raw']
            self.sm_raw = ddict['sm_raw']
            self.nsr_seg_msp = ddict['nsr_seg_msp']
            self.perc_overlap_msp = ddict['perc_overlap_msp']
            self.r_mu = ddict['r_mu']
            self.thres = ddict['thres']
            self.thres_type = ddict['thres_type']
            self.thres_mode = ddict['thres_mode']
            self.thres_first = ddict['thres_first']
            self.min_twitchdur = ddict['min_twitchdur']
            self.min_twitchsep = ddict['min_twitchsep']
        except KeyError:
            print('### ERROR: One or more params missing from settings dictionary; unable to update variables ###')
        
    def update_plot(self):
        self.setWindowTitle('Updating plot settings ...')
        # save annotation before exiting?
        #self.usrAnnot[self.seqi] = self.annot_seq
        
        # default - all param variables reset from saved plot settings
        self.update_vars_from_dict(self.plotSettings)
        
        # update subset of param variables based on input button
        u = self.sender()
        if u == self.remHead_update:
            self.update_rem_params()
        elif u == self.emgfiltHead_update:
            self.ampsrc = 'raw'
            self.update_raw_params()
            self.recalc_plot_data(self, calc_Raw=True, calc_mSP=False)
        elif u == self.mspHead_update:
            self.ampsrc = 'msp'
            self.update_msp_params()
            self.recalc_plot_data(self, calc_Raw=False, calc_mSP=True)
        elif u == self.thresHead_update:
            self.update_thres_params()
        elif u == self.twitchHead_update:
            self.update_twitch_params()
        elif u in [self.updatePlot_btn, self.loadSettings_btn]:
            self.update_vars_from_gui()
        else:
            print('idk man')
            
        # initialize from default values
        # update with GUI param values
        # load from settings file
        
        # get indices of REM sleep and EMG twitches 
        self.find_REM()
        self.detect_emg_twitches(calcAnnot=True)
        # plot data, save plot settings
        self.plotSettings = self.dict_from_vars()
        self.plot_data()
        self.update_gui_from_vars()
        self.setWindowTitle('Done!')
        time.sleep(1)
        self.setWindowTitle(self.name)


    
    def debug(self):
        pdb.Pdb(stdout=sys.__stdout__).set_trace()
        
# ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
# recordings = sleepy.load_recordings(ppath, 'pwaves_emg.txt')[1]
# name = recordings[0]

# ppath ='/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
# name = 'Dante_061722n1'

ppath ='/media/fearthekraken/Mandy_HardDrive1/revision_data/'
#name = 'Risk_072522n1'

#ppath = '/Users/amandaschott/Dropbox/'
name = 'Pumbaa_072522n1'

#ppath = '/media/fearthekraken/Mandy_HardDrive1/revision_data/'

#ppath = '/Users/amandaschott/Dropbox'
#name = 'King_071020n1'


# app = QtGui.QApplication([])
# w = EMGTwitchWindow(ppath, name, parent=None)
# w.show()
# sys.exit(app.exec_())
