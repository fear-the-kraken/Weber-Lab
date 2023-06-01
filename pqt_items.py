#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom widgets and functions used in main P-wave annotation window

@author: fearthekraken
"""
import sys
import os
import re
import h5py
import scipy.io as so
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import colorsys
from colour import Color
import pyautogui
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pdb
# custom modules
from sleepy import calculate_spectrum


#############      SUPPORT FUNCTIONS      #############

def str2num(x):
    """
    Convert numeric string to integer or float
    @Params
    x - string
    @Returns
    y - integer (e.g. '1.0' --> 1), float (e.g. '1.2' --> 1.2), or string (e.g. 'abc' --> 'abc')
    """
    try:
        y = int(x)
    except ValueError:
        try:
            y = float(x)
        except ValueError:
            y = x
    return y


def ordinal(n: int):
    """
    Convert integer to ordinal number (e.g. 1 --> 1st)
    @Params
    n - integer
    @Returns
    o - ordinal string
    """
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    o = str(n) + suffix
    return o


def hue(rgb, percent, mode=1):
    """
    rgb - 3 element tuple with values 0-255 representing (red, green, blue) components of color
    percent - % change in hue (percent=0 returns input color, percent=1 returns target color)
    mode - shift hue lighter (1), darker (0), or duller (0.5)
    """
    rgb = np.array(rgb)
    if mode == 1:
        target = np.array([255,255,255])  # lighten color
    elif mode == 0:
        target = np.array([0,0,0])        # darken color
    elif mode == 0.5:
        target = rgb.mean()               # de-intensify color
    distance = target - rgb
    step = distance * percent
    adjusted_rgb = tuple([int(x) for x in rgb + step])
    return adjusted_rgb


def ratio2rgb(ratio, color1, color2, scale=1):
    """
    @Params
    ratio - firing rate ratio between state1 and state2 (e.g. R:W)
    color1, color2 - rgb colors of state1 and state2
    scale - maximum value (1 or 255) in returned RGB tuple
    @Returns
    c - RGB color between white and color1 (if ratio > 1) or color2 (if ratio < 1)
    """
    # convert color name to RGB (e.g. purple --> (128,128,128))
    if isinstance(color1, str):
        color1 = np.multiply(Color(color1).get_rgb(), 255)
    if isinstance(color2, str):
        color2 = np.multiply(Color(color2).get_rgb(), 255)
    color1 = np.array(color1)
    color2 = np.array(color2)
    if ratio >= 1:
        perc = 1 - min([1, np.log(ratio)])
        c = hue(color1, perc, mode=1)
    else:
        perc  = 1 - (max([-1, np.log(ratio)]) * -1)
        c = hue(color2, perc, mode=1)
    # scale RGB values from 0 - 1
    if scale == 1:
        c = np.divide(c, 255.)
    
    return c


def pthres_txt(p_thr, thres_level, thres_type):
    """
    Create label for P-wave threshold item
    @Params
    p_thr - calculated threshold (uV) for P-wave detection
    thres_level - value (X) used to determine threshold
    thres_type - method of threshold calculation
                 * 'raw' = X uV, 'std' = mean * X st. deviations, 'perc' = Xth percentile
    @Returns
    txt - string representing threshold value and calculation mode
    """
    txt = f'-{round(np.abs(p_thr))} uV'
    if thres_type == 'std':
        txt += f' (\u03BC - \u03C3*{np.round(thres_level,1)})'
    elif thres_type == 'perc':
        txt += f' ({ordinal(int(np.round(thres_level,1)))} %ile)'
    return txt


def get_unit_icon(notes, widths):
    """
    Create QIcon for Neuropixel unit in graph context menu
    @Params
    notes - 3-element list with [state FR, P-wave FR, comment]
            e.g. ['R-max', 'P+', 'sharp FR peak at -0.6s, followed by sharp decrease']
    widths - 3-element list of icon widths (px) for each of the above notes
    @Returns
    icon - QIcon representing unit properties
    """
    assert type(notes) in [list, tuple]
    assert len(notes) == 3
    assert type(widths) in [list, tuple]
    assert len(widths) == 3
    
    # get text for each unit note
    state_note, p_note, comment = notes
    height = min(widths)
    w1, w2, w3 = widths
    
    comboPixmap = QtGui.QPixmap(np.sum(widths), height)
    comboPixmap.fill(QtGui.QColor(0,0,0,0))
    
    painter = QtGui.QPainter(comboPixmap)
    
    # if unit comment exists, draw pixmap of comment icon
    if comment and comment != '-':
        comment_icon = QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        comment_pix = comment_icon.pixmap(w1-2, height-2)
        painter.drawPixmap(1, 1, comment_pix)
    
    # if unit state note exists, draw pixmap of state dependence
    if state_note and state_note != '-':
        painter.drawText(w1, 0, w2, height, QtCore.Qt.AlignCenter, state_note)
        if p_note and p_note != '-':
            painter.drawLine(QtCore.QPointF(w1+w2, 0), QtCore.QPointF(w1+w2, height))
    
    # if unit P-wave note exists, draw pixmap of P-wave dependence
    if p_note and p_note != '-':
        painter.drawText(w1+w2, 0, w3, height, QtCore.Qt.AlignCenter, p_note)
    
    painter.end()
    
    icon = QtGui.QIcon(comboPixmap)
    return icon


#############     LOAD / PROCESS / SAVE DATA     #############

def get_snr_np(ppath, name):
    """
    Get Neuropixel sampling rate from info.txt file
    @Params
    ppath - base folder
    name - recording folder
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
        a = re.search("^" + 'SR_NP' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))     
    SR = float(values[0])
    return SR


def load_eeg_emg(ppath, name):
    """
    Load all EEG and EMG signals for a recording (.mat and/or h5py files)
    @Params
    ppath - base folder
    name - recording folder
    @Returns
    EEG_list - list of EEG signals as numpy arrays/h5py datasets
    EMG_list - list of EMG signals as numpy arrays/h5py datasets
    open_fid - list of open h5py files (if any EEG/EMG signals are in h5py format)
    """
    EEG_list = []
    EMG_list = []
    open_fid = []
    
    # load EEGs
    try:
        EEG1 = so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG']
    except:
        EEGfile = h5py.File(os.path.join(ppath, name, 'EEG.mat'), 'r+')
        open_fid.append(EEGfile)
        EEG1 = EEGfile['EEG']
    EEG_list.append(EEG1)
    if os.path.isfile(os.path.join(ppath, name, 'EEG2.mat')):
        try:
            EEG2 = so.loadmat(os.path.join(ppath, name, 'EEG2.mat'))['EEG2']
        except:
            EEG2file = h5py.File(os.path.join(ppath, name, 'EEG2.mat'), 'r+')
            open_fid.append(EEG2file)
            EEG2 = EEG2file['EEG2']
        EEG_list.append(EEG2)
        
    # load EMGs
    if os.path.isfile(os.path.join(ppath, name, 'EMG.mat')):
        try:
            EMG1 = so.loadmat(os.path.join(ppath, name, 'EMG.mat'))['EMG']
        except:
            EMGfile = h5py.File(os.path.join(ppath, name, 'EMG.mat'), 'r+')
            open_fid.append(EMGfile)
            EMG1 = EMGfile['EMG']
        EMG_list.append(EMG1)
    else:
        EMG1 = np.zeros(len(EEG1))
    EMG_list.append(EMG1)
    if os.path.isfile(os.path.join(ppath, name, 'EMG2.mat')):
        try:
            EMG2 = so.loadmat(os.path.join(ppath, name, 'EMG2.mat'))['EMG2']
        except:
            EMG2file = h5py.File(os.path.join(ppath, name, 'EMG2.mat'), 'r+')
            open_fid.append(EMG2file)
            EMG2 = EMG2file['EMG2']
        EMG_list.append(EMG2)
    
    return EEG_list, EMG_list, open_fid


def load_sp_msp(ppath, name):
    """
    Load/calculate EEG and EMG spectrograms for a recording (.mat files)
    @Params
    ppath - base folder
    name - recording folder
    @Returns
    SPEC - dictionary for EEG spectrogram (keys = SP, SP2, t, freq, dt)
    MSPEC - dictionary for EMG spectrogram
    """
    # if EEG/EMG spectrograms do not exist, generate them
    if not(os.path.isfile(os.path.join(ppath, name, 'sp_' + name + '.mat'))):
        calculate_spectrum(ppath, name, fres=0.5)
        print("Calculating spectrogram for recording %s\n" % name)
    # load EEG and EMG spectrograms
    SPEC = so.loadmat(os.path.join(ppath, name, 'sp_' + name + '.mat'))
    MSPEC = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))
    
    return SPEC, MSPEC


def load_lfp_files(ppath, name, keymatch=False, sort_grps=True):
    """
    Load all LFP signals for a recording (.mat and/or h5py files)
    @Params
    ppath - base folder
    name - recording folder
    keymatch - if True, the name of the LFP data array (e.g. .mat file key/h5py dataset name)
               must exactly match the file name
    sort_grps - if True, return LFP files sorted by group number in ascending order
    @Returns
    lfps - list of LFP signals as numpy arrays/h5py datasets
    lnames - list of LFP file names (e.g. 'LFP_PAG_320')
    lgrps - list of LFP groups (e.g. 320)
    open_fid - list of open h5py files (if any LFP signals are in h5py format)
    """
    LFP_list = []
    lfp_files = [f for f in os.listdir(os.path.join(ppath, name)) if re.match('^LFP', f)]
    lfp_files.sort()
    open_fid = []
    
    if len(lfp_files) > 0:
        for f in lfp_files:
            # get key for LFP data, load .mat file
            key = re.split('\\.', f)[0]
            grp = str(key)
            if sort_grps:
                grp_num = [x for x in key.split('_') if x.isnumeric()]
                if len(grp_num) > 0:
                    grp = grp_num[0]
            try:
                LFPdata = so.loadmat(os.path.join(ppath, name, f))
            except:
                LFPdata = h5py.File(os.path.join(ppath, name, f), 'r+')
            
            # if file contains a key matching the file name, load LFP data from that key
            if key in LFPdata.keys():
                LFP = LFPdata[key]
                LFP_list.append((key,LFP,grp))
            # otherwise, load data from another key IF param $keymatch is False
            else:
                if keymatch:
                    print('\nFile ' + f + ' has no keys that match the file name.')
                    print('To load data from other keys in this file, set the parameter $keymatch to False\n')
                    
                elif not keymatch:
                    if type(LFPdata) == dict:
                        # get rid of MATLAB params, try loading data from any remaining keys
                        _ = [LFPdata.pop(mat_key) for mat_key in ['__header__',
                                                                  '__version__',
                                                                  '__globals__']]
                    dkeys = list(LFPdata.keys())
                    LFP = []
                    if len(dkeys) > 0:
                        # sort to prioritize keys with "LFP" in the name
                        lsort = [0 if 'lfp' in dk.lower() else 1 for dk in dkeys]
                        dkeys = [dk for i,dk in sorted(zip(lsort, dkeys))]
                        for dk in dkeys:
                            try:
                                # if data is a 2-dimensional array, load as the LFP
                                data = LFPdata[dk]
                                if data.ndim == 2:
                                    LFP = data
                                    break
                            except:
                                continue
                        if len(LFP) == 0:
                            print('File ' + f + ' contains no data consistent with LFP signals')
                        else:
                            LFP_list.append((key,LFP,grp))
                    else:
                        print('\nFile ' + f + ' appears to be empty\n')
            if type(LFPdata) == h5py._hl.files.File:
                open_fid.append(LFPdata)
        if len(LFP_list) > 0:
            # sort by LFP name or group number
            grp_rks = [int(tup[2]) if tup[2].isnumeric() else np.inf for tup in LFP_list]
            LFPsort = [tup for i,tup in sorted(zip(grp_rks, LFP_list))]
            lnames, lfps, lgrps = zip(*LFPsort)
            return list(lfps), list(lnames), list(lgrps), open_fid
        else:
            print('No LFP data found in recording folder ' + name)
            return [],[],[],[]
    else:
        print('No LFP files found in recording folder ' + name)
        return [],[],[],[]


def load_unit_notes(ppath, name, auto=False):
    """
    Load notes for Neuropixel single units from .txt file
    @Params
    ppath - base folder
    name - recording folder
    auto - if True, load the file saved by the auto-classifier
           if False, load the file with the user's classifications and notes
    @Returns
    unotes - notes dictionary (keys=units, values=notes)
    """
    if auto:
        try:  # if auto file has not been created yet, return {}
            fid = open(os.path.join(ppath, name, 'auto_classifications.txt'), 'r')
        except:
            return {}
    else:
        fid = open(os.path.join(ppath, name, 'unit_notes.txt'), 'r')
    lines = fid.readlines()
    fid.close()
    
    unotes = {}
    for l in lines:
        a = re.search("^[0-9]+:", l)
        if a:
            params = l.strip().split('\t')
            u = int(params[0].replace(':',''))
            val = params[1:]
            unotes[u] = val
            
    return unotes


def save_unit_notes(ppath, name, unotes, auto=False):
    """
    Save notes for Neuropixel single units as .txt file
    @Params
    ppath - base folder
    name - recording folder
    unotes - notes dictionary (keys=units, values=notes)
    auto - if True, save the dictionary of auto-classifications
           if False, save the dictionary of user-edited classifications/notes
    @Returns
    None
    """
    if auto:  # save auto-generated values only
        fid = open(os.path.join(ppath, name, 'auto_classifications.txt'), 'w')
    else:  # include extra info in the heading for users
        fid = open(os.path.join(ppath, name, 'unit_notes.txt'), 'w')
        fid.write('#Unit' + '\t' + '#State' + '\t' + '#P-wave' + '\t' + '#Comments' + '\n')
        fid.write('#State-dependent activity: R-on, R-max, R-off, RW, W-max, W-min, N-max, X' + '\n')
        fid.write('#P-wave-dependent activity: P+, P-, P0' + '\n')
        fid.write('\n')
        if type(unotes) == list:
            print('Creating unit notes file ... ')
            unotes = {u : ['-','-','-'] for u in unotes}
    # save notes for each unit on a separate line
    for u,val in unotes.items():
        fid.write(str(u) + ':' + '\t' + '\t'.join(val) + '\t' + '\n')
    fid.close()
    print('Saved!')


def find_hidden_units(unit_list, notes_dicts, peak_fr, filt_dict):
    """
    Return list of units to exclude from group analysis
    @Params
    unit_list - single units to filter
    notes_dicts - list of 2 dictionaries with manual and automatic unit classification notes
                  * units will be filtered using classifications in the FIRST dictionary
    peak_fr - peak firing rate of each unit
    filt_dict - dictionary of unit classifications based on activity across brain
                states (e.g. 'R-max') or correlated with P-waves (e.g. P+). Values
                indicate whether units should be included (1) or hidden (0)
    """
    ddict1, ddict2 = notes_dicts
    hidden_unit_list = []
    # get peak firing rate bins
    pk_keys = [k for k in filt_dict.keys() if k[0].isnumeric()]
    pk_bins = np.array([float(i.split(' - ')[0].replace('+','')) for i in pk_keys])
    for unit, pk in zip(unit_list, peak_fr):
        # get unit notes: brainstate class, p-wave class, comments
        # if unit in ddict1.keys():
        #     notes = list(ddict1[unit])
        # else:
        #     notes = ['-','-','-']
        notes = ddict1.get(unit, ['-','-','-'])
        if notes[0] == '-':
            notes[0] = 'Unclassified state'
        if notes[1] == '-':
            notes[1] = 'Unclassified pwaves'
        # get unit peak frequency bin
        i = np.where(pk_bins <= pk)[0][-1]
        # get user/auto coherence
        other_notes = ddict2.get(unit, ['-','-','-'])
        if notes[0] == '-' or other_notes[0] == '-':
            cc_state = 'Missing brainstate classification(s)'
        else:
            cc_state = 'Same brainstate group' if notes[0] == other_notes[0] else 'Different brainstate groups'
        if notes[1] == '-' or other_notes[1] == '-':
            cc_pwave = 'Missing P-wave classification(s)'
        else:
            cc_pwave = 'Same P-wave group' if notes[1] == other_notes[1] else 'Different P-wave groups'
        
        a = filt_dict.get(notes[0], 1)  # show state classification group?
        b = filt_dict.get(notes[1], 1)  # show P-wave classification group?
        c = filt_dict.get(pk_keys[i], 1)  # show peak firing rate bin? 
        d = filt_dict.get(cc_state, 1)  # show user/auto results for state classification?
        e = filt_dict.get(cc_pwave, 1)  # show user/auto results for P-wave classification?
        if not (a and b and c and d and e):
            hidden_unit_list.append(unit)
    return hidden_unit_list


def downsample_signal(data, mode='accurate', nbin=None, sr=None, target_sr=None, 
                      nsamples=None, match_nsamples=False):
    """
    Downsample 1D signal vector
    @Params
    data - input vector
    mode - if 'accurate': downsample by calculating the mean of each consecutive 
                          group of $nbin samples
           if 'fast': downsample by taking every $nbin-th sample
    nbin - downsample $data to vector of bins representing $nbin samples each
    sr, target_sr - downsample $data from original sampling rate $sr to vector
                    with a target sampling rate $target_sr
    nsamples - downsample $data into vector of $nsamples total bins
    match_nsamples - if True, resize downsampled signal to exactly match the number
                     of bins given by $nsamples
    @Returns
    data_dn - downsampled vector
    """
    # clean data inputs
    if data.ndim > 2:
        print('Signal has too many dimensions')
        return
    if data.ndim == 2:
        if data.shape[0] == 1 or data.shape[1] == 1:
            data = np.squeeze(data)
        else:
            print('Signal must be a vector, not a matrix')
            return
            
    if nbin is None:
        if sr is not None and target_sr is not None:
            nbin = int(np.round(sr / target_sr))
        elif nsamples is not None:
            nbin = int(np.floor(len(data) / nsamples))
        else:
            print('Not enough parameters set')
            return
    
    # sample every Nth element of signal (super fast, accurately reflects low sampling rate)
    if mode == 'fast':
        data_dn = data[0::nbin]
    # average each consecutive group of N values (reasonable efficient, way cleaner result)
    else:
        # length of $data must be evenly divisibly by $nbin
        j = int(np.floor(len(data) / nbin) * nbin)
        x = data[0:j].reshape(-1,nbin).astype('float32')
        data_dn = np.mean(x, axis=1).astype('float32')
    
    if match_nsamples and (nsamples is not None) and (len(data_dn) != nsamples):
        if len(data_dn) > nsamples:
            data_dn = data_dn[0:nsamples]
            return data_dn
        if len(data_dn) < nsamples:
            data_dn = np.append(data_dn, np.zeros(nsamples-data_dn, dtype='float32'))
            return data_dn
    return data_dn


def downsample_signal2(data, nbin):
    """
    Downsample 1D signal vector
    @Params
    data - input vector
    nbin - calculate mean of each consecutive group of $nbin samples
    @Returns
    y - downsampled vector
    """
    n1 = int(np.floor(nbin))
    n2 = int(np.ceil(nbin))
    
    nrepeats = int(np.floor(len(data) / (n1+n2)))
    vals = np.tile(np.array([n1,n2]), nrepeats)
    x = np.split(data, np.cumsum(vals))
    y = np.array([np.mean(d) for d in x], dtype='float32')
    return y
    

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



#############     CREATE / ADJUST PYQT5 WIDGETS     #############

def px_w(width, screen_width, ref_width=1920):
    """
    Convert pixel width from reference computer monitor to current screen
    @Params
    width - integer pixel width to convert
    screen_width - width of current monitor in pixels
    ref_width - width of reference monitor; default value is from Weber lab computer
    @Returns
    new_width - proportional pixel width on current screen
    """
    new_width = int(round(screen_width * (width/ref_width)))
    new_width = min(max(1,new_width), screen_width)
    return new_width


def px_h(height, screen_height, ref_height=1080):
    """
    Scale pixel height from reference computer monitor to current screen
    """
    new_height = int(round(screen_height * (height/ref_height)))
    new_height = min(max(1,new_height), screen_height)
    return new_height


def vline(orientation='v'):
    """
    Create vertical or horizontal line to separate widget containers
    @Returns
    line - QFrame object
    """
    line = QtWidgets.QFrame()
    if orientation == 'v':
        line.setFrameShape(QtWidgets.QFrame.VLine)
    elif orientation == 'h':
        line.setFrameShape(QtWidgets.QFrame.HLine)
    else:
        print('###   INVALID LINE ORIENTATION   ###')
        return
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    line.setLineWidth(3)
    line.setMidLineWidth(3)
    line.setContentsMargins(0,15,0,15)
    return line


def pg_symbol(symbol, font=QtGui.QFont('San Serif')):
    """
    Create custom pyqtgraph marker from keyboard character
    @Params
    symbol - single character (e.g. @) to convert into plot marker
    font - character font
    @Returns
    mapped_symbol - customized marker for pyqtgraph plots
    """
    # max of one character for symbol
    assert len(symbol) == 1
    pg_symbol = QtGui.QPainterPath()
    pg_symbol.addText(0, 0, font, symbol)
    # scale symbol
    br = pg_symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    # center symbol in bounding box
    tr.translate(-br.x() - br.width() / 2., -br.y() - br.height() / 2.)
    mapped_symbol = tr.map(pg_symbol)
    return mapped_symbol


def set_xlimits(graph, tmin, tmax, padding=None):
    """
    Set x-limits for a pyqtgraph PlotItem
    @Params
    graph - pyqtgraph PlotItem
    tmin, tmax - minimum, maximum x-axis values
    padding - % of shown data to use as padding (None=default padding)
    @Returns
    None
    """
    graph.vb.setLimits(xMin=None, xMax=None)
    graph.vb.setXRange(tmin, tmax, padding=padding)
    xmin, xmax = graph.vb.viewRange()[0]
    graph.vb.setLimits(xMin=xmin, xMax=xmax)
    

def xlink_graphs(graph_list, target_graph=None):
    """
    Link the x axes of pyqtgraph PlotItems
    @Params
    graph_list - link the x axis of each PlotItem to the first graph in the list
    target_graph - optional PlotItem to link all graphs in list
    @Returns
    None
    """
    # unlink first graph from any other PlotItems
    graph_list[0].vb.linkView(graph_list[0].vb.XAxis, None)
    # link all subsequent items to first graph
    for graph in graph_list[1:]:
        graph.setXLink(graph_list[0].vb)
    # link first graph to target item, if given
    if target_graph is not None:
        graph_list[0].setXLink(target_graph.vb)


class UnitFiltWindow(QtWidgets.QDialog):
    """
    Pop-up window for filtering Neuropixel units by brain state and/or P-wave-triggered firing
    """
    def __init__(self, filt_dict):
        super(UnitFiltWindow, self).__init__()
        wpx, hpx = pyautogui.size()
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20 = [px_w(w, wpx) for w in [5,10,15,20]]
        hspace5, hspace10, hspace15, hspace20 = [px_h(h, hpx) for h in [5,10,15,20]]
        # set contents margins, central layout
        self.setContentsMargins(wspace10,hspace10,wspace10,hspace10)
        self.centralLayout = QtWidgets.QGridLayout(self)
        self.centralLayout.setHorizontalSpacing(wspace20)
        self.centralLayout.setVerticalSpacing(hspace20)
        #self.centralLayout = QtWidgets.QVBoxLayout(self)
        #self.centralLayout.setSpacing(hspace20)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)
        labelFont = QtGui.QFont()
        labelFont.setPointSize(12)
        labelFont.setUnderline(True)
        headerFont = QtGui.QFont()
        headerFont.setPointSize(12)
        headerFont.setWeight(70)
        #headerFont.setBold(True)
        
        
        self.filt_dict = dict(filt_dict)
        # filter by brain state activity
        brstate_box = QtWidgets.QGroupBox('Brainstate Firing Rate')
        brstate_box.setFont(headerFont)
        brstate_lay = QtWidgets.QVBoxLayout(brstate_box)
        # filter by P-wave-triggered activity
        pwave_box = QtWidgets.QGroupBox('P-wave Firing Rate')
        pwave_box.setFont(headerFont)
        pwave_lay = QtWidgets.QVBoxLayout(pwave_box)
        # filter by peak firing rate
        peak_fr_box = QtWidgets.QGroupBox('Peak Firing Rate')
        peak_fr_box.setFont(headerFont)
        peak_fr_lay = QtWidgets.QVBoxLayout(peak_fr_box)
        # filter by classification consistency (manual vs auto classifications)
        cc_box = QtWidgets.QGroupBox('User vs Auto-Classification')
        cc_box.setFont(headerFont)
        cc_lay = QtWidgets.QHBoxLayout(cc_box)
        cc_state_lay = QtWidgets.QVBoxLayout()
        cc_state_label = QtWidgets.QLabel('Brainstate')
        cc_state_label.setStyleSheet('QLabel {font : 12pt normal; text-decoration : underline}')
        cc_state_lay.addWidget(cc_state_label)
        cc_pwave_lay = QtWidgets.QVBoxLayout()
        cc_pwave_label = QtWidgets.QLabel('P-wave')
        cc_pwave_label.setStyleSheet('QLabel {font : 12pt normal; text-decoration : underline}')
        cc_pwave_lay.addWidget(cc_pwave_label)
        cc_lay.addSpacing(wspace20*2)
        cc_lay.addLayout(cc_state_lay)
        cc_lay.addLayout(cc_pwave_lay)
        
        
        for cat in list(self.filt_dict.keys()):
            # create checkbox, initialize state, connect to updating function
            if cat.startswith('Unclassified'):
                chk = QtWidgets.QCheckBox('Unclassified')
            elif cat[0].isnumeric():
                chk = QtWidgets.QCheckBox(cat + ' Hz')
            elif cat.split(' ')[0] in ['Same','Different','Missing']:
                chk = QtWidgets.QCheckBox(cat.split(' ')[0])
            else:
                chk = QtWidgets.QCheckBox(cat)
            chk.setObjectName(cat)
            chk.setStyleSheet('QCheckBox {font : 12pt normal}')
            #chk.setFont(font)
            chk.setChecked(bool(self.filt_dict[cat]))
            chk.stateChanged.connect(self.filt_update)
            # add to brainstate or P-wave groupbox
            if cat.startswith('Unclassified'):
                if cat.split(' ')[1] == 'state':
                    brstate_lay.addWidget(chk)
                elif cat.split(' ')[1] == 'pwaves':
                    pwave_lay.addWidget(chk)
            elif cat[0].isnumeric():
                peak_fr_lay.addWidget(chk)
            else:
                if cat.startswith('P'):
                    pwave_lay.addWidget(chk)
                elif cat.split(' ')[0] in ['Same','Different','Missing']:
                    if 'brainstate' in cat:
                        cc_state_lay.addWidget(chk)
                    elif 'P-wave' in cat:
                        cc_pwave_lay.addWidget(chk)
                else:
                    brstate_lay.addWidget(chk)
            
        # action buttons
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        bbox = QtWidgets.QDialogButtonBox(QBtn)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        bbox.setCenterButtons(True)
        
        self.centralLayout.addWidget(brstate_box, 0, 0, 3, 1)
        self.centralLayout.addWidget(pwave_box, 0, 1, 1, 1)
        self.centralLayout.addWidget(peak_fr_box, 1, 1, 2, 1)
        self.centralLayout.addWidget(cc_box, 3, 0, 1, 2)
        self.centralLayout.addWidget(bbox, 4, 0, 1, 2)
        self.setWindowTitle('Filter units by ...')
    
    def filt_update(self, x):
        k = self.sender().objectName()
        self.filt_dict[k] = int(x)


def warning_dlg(msg):
    """
    Execute dialog box with yes/no options
    @Returns
    1 ("yes") or 0 ("no")
    """
    # create dialog box and yes/no buttons
    dlg = QtWidgets.QDialog()
    font = QtGui.QFont()
    font.setPointSize(12)
    dlg.setFont(font)
    msg = QtWidgets.QLabel(msg)
    msg.setAlignment(QtCore.Qt.AlignCenter)
    QBtn = QtWidgets.QDialogButtonBox.Yes | QtWidgets.QDialogButtonBox.No
    bbox = QtWidgets.QDialogButtonBox(QBtn)
    bbox.accepted.connect(dlg.accept)
    bbox.rejected.connect(dlg.reject)
    # set layout, run window
    lay = QtWidgets.QVBoxLayout()
    lay.addWidget(msg)
    lay.addWidget(bbox)
    dlg.setLayout(lay)
    if dlg.exec():
        return 1
    else:
        return 0


def warning_msgbox(msg):
    """
    Execute message box with yes/no/cancel options
    @Returns
    1 ("yes"), 0 ("no"), or -1 ("cancel")
    """
    # create message box and yes/no/cancel options
    dlg = QtWidgets.QMessageBox()
    dlg.setText(msg)
    dlg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    dlg.setDefaultButton(QtWidgets.QMessageBox.No)
    res = dlg.exec()
    if res == QtWidgets.QMessageBox.Yes:
        return 1
    elif res == QtWidgets.QMessageBox.No:
        return 0
    else:
        return -1


def back_next_btns(parent, name):
    """
    Create "back" or "next" arrow buttons for viewing different sets of plotting options
    @Params
    parent - main annotation window
    name - string defining object name and symbol; includes either "back" or "next"
    @Returns
    btn - green QPushButton object with left or right-facing arrow
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bsize = px_w(20, wpx)
    isize = px_w(17, wpx)
    # set button properties
    btn = QtWidgets.QPushButton()
    btn.setObjectName(name)
    btn.setFixedSize(bsize,bsize)
    if 'back' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack))
    elif 'next' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
    btn.setIconSize(QtCore.QSize(isize,isize))
    btn.setStyleSheet('QPushButton {border : none; margin-top : 2px}')
    return btn


def back_next_event(parent, name):
    """
    Create "back" or "next" arrow buttons for viewing each artifact in live data plot
    @Params
    parent - main annotation window
    name - string defining object name and symbol; includes either "back" or "next"
    @Returns
    btn - black & white QPushButton object with left or right-facing arrow
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bw, bh = [px_w(25, wpx), px_h(15, hpx)]
    iw, ih = [px_w(22, wpx), px_h(13, hpx)]
    # set button properties
    btn = QtWidgets.QPushButton()
    btn.setObjectName(name)
    btn.setFixedSize(bw,bh)
    if 'back' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward))
    elif 'next' in name:
        btn.setIcon(parent.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekForward))
    btn.setIconSize(QtCore.QSize(iw,ih))
    return btn
    

def show_hide_event(parent):
    """
    Create toggle button for showing and hiding each artifact in live data plot
    @Params
    parent - main annotation window
    @Returns
    btn - QPushButton object with show/hide icons for checked/unchecked states
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bsize = px_w(25, wpx)
    radius = px_w(5, wpx)
    # set button properties
    btn = QtWidgets.QPushButton()
    btn.setCheckable(True)
    btn.setChecked(False)
    btn.setFixedSize(bsize,bsize)
    btn.setStyleSheet('QPushButton'
                      '{ border : 2px solid gray;'
                      'border-style : outset;'
                      f'border-radius : {radius}px;'
                      'padding : 1px;'
                      'background-color : lightgray;'
                      'image : url("icons/hide_icon.png") }'
                      'QPushButton:checked'
                      '{ background-color : white;'
                      'image : url("icons/show_icon.png") }')
    return btn
    

def update_noise_btn(top_parent, parent, icon):
    """
    Create checkable buttons for handling noise detection and visualization
    @Params
    top_parent - main annotation window
    parent - noise widget in main window
    icon - if "save": update current noise indices with newly detected noise
           if "reset": replace current noise indices with newly detected noise
           if "calc": calculate/show EEG spectrogram with EEG noise excluded
    @Returns
    btn - QPushButton object with specified icon
    """
    # determine button size
    wpx, hpx = pyautogui.size()
    bsize = px_w(28, wpx)
    isize = px_w(18, wpx)
    # set button properties
    btn = QtWidgets.QPushButton(parent)
    btn.setCheckable(True)
    btn.setFixedSize(bsize,bsize)
    if icon == 'save':
        btn.setIcon(top_parent.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
    elif icon == 'reset':
        btn.setIcon(top_parent.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
    elif icon == 'calc':
        btn.setIcon(top_parent.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
    btn.setIconSize(QtCore.QSize(isize,isize))  # 28,18
    btn.setStyleSheet('QPushButton:checked {background-color : rgb(200,200,200)}')
    return btn



#############          PYQT5 SUBCLASSES          #############

class FreqBandWindow(QtWidgets.QDialog):
    """
    Pop-up window for setting frequency band name, range, and plotting color
    """
    def __init__(self):
        super(FreqBandWindow, self).__init__()
        wpx, hpx = pyautogui.size()
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20 = [px_w(w, wpx) for w in [5,10,15,20]]
        hspace5, hspace10, hspace15, hspace20 = [px_h(h, hpx) for h in [5,10,15,20]]
        # set contents margins, central layout
        self.setContentsMargins(wspace10,hspace10,wspace10,hspace10)
        self.centralLayout = QtWidgets.QVBoxLayout(self)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)
        
        # band name
        row1 = QtWidgets.QVBoxLayout()
        row1.setSpacing(hspace10)
        name_label = QtWidgets.QLabel('Band name')
        name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_input = QtWidgets.QLineEdit()
        row1.addWidget(name_label)
        row1.addWidget(self.name_input)
        
        # min and max frequencies
        row2 = QtWidgets.QVBoxLayout()
        row2.setSpacing(hspace10)
        range_label = QtWidgets.QLabel('Frequency Range')
        range_label.setAlignment(QtCore.Qt.AlignCenter)
        freq_row = QtWidgets.QHBoxLayout()
        freq_row.setSpacing(wspace5)
        self.freq1_input = QtWidgets.QDoubleSpinBox()
        self.freq1_input.setMaximum(500)
        self.freq1_input.setDecimals(1)
        self.freq1_input.setSingleStep(0.5)
        self.freq1_input.setSuffix(' Hz')
        self.freq1_input.valueChanged.connect(self.enable_save)
        freq_dash = QtWidgets.QLabel('-')
        self.freq2_input = QtWidgets.QDoubleSpinBox()
        self.freq2_input.setMaximum(500)
        self.freq2_input.setDecimals(1)
        self.freq2_input.setSingleStep(0.5)
        self.freq2_input.setSuffix(' Hz')
        self.freq2_input.valueChanged.connect(self.enable_save)
        freq_row.addWidget(self.freq1_input)
        freq_row.addWidget(freq_dash)
        freq_row.addWidget(self.freq2_input)
        row2.addWidget(range_label)
        row2.addLayout(freq_row)
        
        # plotting color
        row3 = QtWidgets.QVBoxLayout()
        row3.setSpacing(hspace5)
        color_label = QtWidgets.QLabel('Color')
        color_label.setAlignment(QtCore.Qt.AlignCenter)
        self.color_input = QtWidgets.QComboBox()
        # get list of CSS colors sorted by name and hue
        colors = mcolors.CSS4_COLORS
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) \
                        for name,color in colors.items())
        colornames = [name for hsv, name in by_hsv]
        for cname in colornames:
            pixmap = QtGui.QPixmap(100,100)
            pixmap.fill(QtGui.QColor(cname))
            self.color_input.addItem(QtGui.QIcon(pixmap), cname)
        row3.addWidget(color_label)
        row3.addWidget(self.color_input)
        
        # save or cancel changes
        self.bbox = QtWidgets.QDialogButtonBox()
        self.save_btn = self.bbox.addButton(QtWidgets.QDialogButtonBox.Save)
        self.save_btn.setDisabled(True)
        self.cancel_btn = self.bbox.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.bbox.setCenterButtons(True)
        self.bbox.accepted.connect(self.accept)
        self.bbox.rejected.connect(self.reject)
        
        # add widgets
        self.centralLayout.addLayout(row1)
        self.centralLayout.addLayout(row2)
        self.centralLayout.addLayout(row3)
        self.centralLayout.addWidget(self.bbox)
        self.centralLayout.setSpacing(hspace20)
    
    def enable_save(self):
        """
        Allow parameters to be saved if frequency range is valid
        """
        # max band frequency must be greater than min frequency
        f1, f2 = self.freq1_input.value(), self.freq2_input.value()
        self.save_btn.setEnabled(f2 > f1)
        
    
class FreqBandLabel(pg.LabelItem):
    """
    Text label for currently plotted EEG frequency band power
    """
    def __init__(self, parent=None):
        super(FreqBandLabel, self).__init__()
        self.mainWin = parent
        
    def contextMenuEvent(self, event):
        """
        Set context menu to edit, delete, or add new frequency band
        """
        self.menu = QtWidgets.QMenu()
        # edit freq band properties
        editAction = QtWidgets.QAction('Edit band')
        editAction.setObjectName('edit')
        editAction.triggered.connect(self.mainWin.freq_band_window)
        self.menu.addAction(editAction)
        # delete freq band
        deleteAction = QtWidgets.QAction('Delete band')
        deleteAction.setObjectName('delete')
        deleteAction.triggered.connect(self.mainWin.freq_band_window)
        self.menu.addAction(deleteAction)
        # add new freq band
        addAction = QtWidgets.QAction('Add new band')
        addAction.setObjectName('add')
        addAction.triggered.connect(self.mainWin.freq_band_window)
        self.menu.addAction(addAction)
        self.menu.exec_(QtGui.QCursor.pos())
            
        
    def set_info(self, freq1, freq2, band_name='', color='white'):
        """
        Update label of the current frequency band plot
        @Params
        freq1, freq2 - min and max frequencies in band
        band_name - name of frequency band (e.g. delta, theta, sigma...)
        color - color of band power plot
        """
        txt = f'{freq1} - {freq2} Hz'
        if band_name:
            txt = band_name + ' (' + txt + ')'
        labelOpts = {'color':color, 'size':'12pt'}
        self.setText(txt, **labelOpts)

    
class PlotButton(QtWidgets.QPushButton):
    """
    Data plotting button; instantiates pop-up figure window for specific graph
    """
    def __init__(self, parent, name, color, reqs=[]):
        super(PlotButton, self).__init__()
        self.mainWin = parent
        self.base_name = name
        self.color = color
        self.reqs = reqs
        
        # determine button size
        wpx, hpx = pyautogui.size()
        bsize = px_w(20, wpx)
        self.radius = int(bsize/2)
        self.bwidth = int(bsize/5)
        self.setFixedSize(bsize,bsize)
        # set button style
        self.setStyleSheet('QPushButton'
                           '{ border-color : gray;'
                           'border-width : ' + str(self.bwidth) + 'px;'
                           'border-style : outset;'
                           'border-radius : ' + str(self.radius) + 'px;'
                           'padding : 1px;'
                           'background-color : ' + self.color + ' }'
                           'QPushButton:pressed'
                           '{ border-width : ' + str(self.bwidth) + 'px;'
                           'background-color : rgba(139,58,58,255) }'
                           'QPushButton:disabled'
                           '{ border-width : ' + str(self.bwidth*3) + 'px;'
                           'background-color : ' + self.color + ' }')
        self.name = str(self.base_name)
    
    
    def enable_btn(self):
        """
        Enable button if recording has all required plot elements
        """
        enable = True
        if 'hasEMG' in self.reqs and self.mainWin.hasEMG == False:
            enable = False
        if 'hasDFF' in self.reqs and self.mainWin.hasDFF == False:
            enable = False
        if 'recordPwaves' in self.reqs and self.mainWin.hasPwaves == False:
            enable = False
        if 'lsrTrigPwaves' in self.reqs and self.mainWin.lsrTrigPwaves == False:
            enable = False
        if 'opto' in self.reqs and self.mainWin.optoMode == '':
            enable = False
        if 'ol' in self.reqs and self.mainWin.optoMode != 'ol':
            enable = False
        if 'cl' in self.reqs and self.mainWin.optoMode != 'cl':
            enable = False
        self.setEnabled(enable)
    
    
    def single_mode(self, p_event):
        """
        Change appearance when single event is selected
        """
        if p_event == 'lsr_pi':
            c = 'rgba(0,0,255,200)'      # blue background
        elif p_event == 'spon_pi':
            c = 'rgba(255,255,255,200)'  # white background
        elif p_event == 'lsr':
            c = 'rgba(17,107,41,200)'    # green background
        elif p_event == 'other':
            c = 'rgba(104,15,122,200)'   # purple background
        
        if self.isEnabled():
            # update plot button style
            self.setStyleSheet('QPushButton'
                               '{ border-color : rgba(255,0,0,215);'
                               f'border-width : {self.bwidth}px;'
                               'border-style : outset;'
                               f'border-radius : {int(round(self.bwidth/2))}px;'
                               'padding : 1px;'
                               'background-color : ' + c + ' }'
                               'QPushButton:pressed'
                               '{ background-color : rgba(255,255,0,255) }')
            self.name = 'Single ' + str(self.base_name)
    
    
    def avg_mode(self):
        """
        Reset appearance when single event is deselected
        """
        if self.isEnabled():
            # reset plot button style when selected point is cleared
            self.setStyleSheet('QPushButton'
                               '{ border-color : gray;'
                               f'border-width : {self.bwidth}px;'
                               'border-style : outset;'
                               f'border-radius : {self.radius}px;'
                               'padding : 1px;'
                               'background-color : ' + self.color + ' }'
                               'QPushButton:pressed'
                               '{ background-color : rgba(139,58,58,255) }')
            self.name = str(self.base_name)


class HelpWindow(QtWidgets.QDialog):
    """
    Informational pop-up window listing possible options for user keyboard input
    """
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        wpx, hpx = pyautogui.size()
        
        # set contents margins, central layout
        cm = px_w(25, wpx)
        self.setContentsMargins(cm,cm,cm,cm)
        self.centralLayout = QtWidgets.QVBoxLayout(self)
        
        # set fonts
        subheaderFont = QtGui.QFont()
        subheaderFont.setPointSize(12)
        subheaderFont.setBold(True)
        subheaderFont.setUnderline(True)
        keyFont = QtGui.QFont()
        keyFont.setFamily('Courier')
        keyFont.setPointSize(18)
        keyFont.setBold(True)
        font = QtGui.QFont()
        font.setPointSize(12)
        # get set of widths and heights, standardized by monitor dimensions
        wspace5, wspace10, wspace15, wspace20 = [px_w(w, wpx) for w in [5,10,15,20]]
        hspace5, hspace10, hspace15, hspace20 = [px_h(h, hpx) for h in [5,10,15,20]]
        
        # create title widget
        self.title = QtWidgets.QPushButton()
        bh = px_h(40, hpx)
        self.title.setFixedHeight(bh)
        rad = int(bh/4)
        bw = px_h(3, hpx)
        # create gradient color for title button
        dkR, dkG, dkB = (85,233,255)
        mdR, mdG, mdB = (142,246,255)
        ltR, ltG, ltB = (184,254,255)
        self.title.setStyleSheet('QPushButton' 
                                 '{background-color : qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,'
                                                      f'stop:0    rgb({dkR},{dkG},{dkB}),'
                                                      f'stop:0.25 rgb({mdR},{mdG},{mdB}),'
                                                      f'stop:0.5  rgb({ltR},{ltG},{ltB}),'
                                                      f'stop:0.75 rgb({mdR},{mdG},{mdB}),'
                                                      f'stop:1    rgb({dkR},{dkG},{dkB}));'
                                 f'border : {bw}px outset gray;'
                                 f'border-radius : {rad};'
                                 '}')
        # create title text 
        txt_label = QtWidgets.QLabel('Keyboard Inputs')
        txt_label.setAlignment(QtCore.Qt.AlignCenter)
        txt_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        f = QtGui.QFont()
        f.setPointSize(14)
        f.setBold(True)
        txt_label.setFont(f)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(txt_label)
        self.title.setLayout(layout)
        
        # set up grid layout for displaying keyboard info
        self.keywidget = QtWidgets.QFrame()
        self.keywidget.setStyleSheet('QFrame {background-color : rgba(255,255,255,180)}')
        self.keygrid = QtWidgets.QGridLayout(self.keywidget)
        self.keygrid.setContentsMargins(wspace20,hspace20,wspace20,hspace20)
        
        # info for data viewing keys
        keyLayout1 = QtWidgets.QVBoxLayout()
        keyLayout1.setSpacing(hspace20)
        keylay1 = QtWidgets.QVBoxLayout()
        keylay1.setSpacing(hspace10)
        keytitle1 = QtWidgets.QLabel('Data Viewing')
        keytitle1.setAlignment(QtCore.Qt.AlignCenter)
        keytitle1.setFont(subheaderFont)
        keydict1 = {'e' : 'show EEG / switch EEG channel',
                    'm' : 'show EMG / switch EMG channel', 
                    'l' : 'show LFP / switch LFP channel',
                    't' : 'show / hide threshold for P-wave detection',
                    'p' : 'show / hide P-wave indices',
                    'o' : 'show / hide opotogenetic laser train',
                    'a' : 'show / hide EMG amplitude',
                    'f' : 'show / hide P-wave frequency',
                    'g' : 'show / hide GCaMP calcium signal',
                    'u' : 'show / hide signal underlying P-wave threshold',
                    'd' : 'show / hide standard deviation of LFP',
                    'b' : 'show / toggle through EEG freq. bands (<b>B</b> to hide)'}
        for key,txt in keydict1.items():
            row = self.key_def_row(key, txt, spacing=wspace20, uline=True, 
                                   keyFont=keyFont, txtFont=font)
            keylay1.addLayout(row)
        keyLayout1.addWidget(keytitle1, stretch=0)
        keyLayout1.addLayout(keylay1, stretch=2)
        
        # info for brain state keys
        keyLayout2 = QtWidgets.QVBoxLayout()
        keyLayout2.setSpacing(hspace20)
        keylay2 = QtWidgets.QVBoxLayout()
        keylay2.setSpacing(hspace10)
        keytitle2 = QtWidgets.QLabel('Brain State Annotation')
        keytitle2.setAlignment(QtCore.Qt.AlignCenter)
        keytitle2.setFont(subheaderFont)
        keydict2 = {'r' : 'REM sleep',
                    'w' : 'wake', 
                    'n' : 'non-REM sleep',
                    'i' : 'intermediate (transition) sleep',
                    'j' : 'failed transition sleep'}
        for key,txt in keydict2.items():
            row = self.key_def_row(key, txt, spacing=wspace20, uline=True, 
                                   keyFont=keyFont, txtFont=font)
            keylay2.addLayout(row)
        keyLayout2.addWidget(keytitle2, stretch=0)
        keyLayout2.addLayout(keylay2, stretch=2)
        
        # info for signals/event annotation keys
        keyLayout3 = QtWidgets.QVBoxLayout()
        keyLayout3.setSpacing(hspace20)
        keylay3 = QtWidgets.QVBoxLayout()
        keylay3.setSpacing(hspace10)
        keytitle3 = QtWidgets.QLabel('Signal / Event Annotation')
        keytitle3.setAlignment(QtCore.Qt.AlignCenter)
        keytitle3.setFont(subheaderFont)
        keydict3 = {'9' : 'annotate waveform as P-wave', 
                    '0' : 'eliminate waveform as P-wave',
                    'x' : 'annotate selected signal as noise',
                    'c' : 'annotate selected signal as clean'}
        for key,txt in keydict3.items():
            row = self.key_def_row(key, txt, spacing=wspace20, uline=False, 
                                   keyFont=keyFont, txtFont=font)
            keylay3.addLayout(row)
        keyLayout3.addWidget(keytitle3, stretch=0)
        keyLayout3.addLayout(keylay3, stretch=2)
        
        # info for plot adjustment keys
        keyLayout4 = QtWidgets.QVBoxLayout()
        keyLayout4.setSpacing(hspace20)
        keylay4 = QtWidgets.QGridLayout()
        keylay4.setVerticalSpacing(hspace10)
        keytitle4 = QtWidgets.QLabel('Plot Adjustment')
        keytitle4.setAlignment(QtCore.Qt.AlignCenter)
        keytitle4.setFont(subheaderFont)
        keydict4a = {'\u2190' : 'scroll to the left',
                     '\u2192' : 'scroll to the right',
                     '\u2193' : 'brighten EEG spectrogram',
                     '\u2191' : 'darken EEG spectrogram'}
        bsize, radius, bwidth = [px_w(w, wpx) for w in [30,8,3]]
        bfont = QtGui.QFont()
        bfont.setPointSize(15)
        bfont.setBold(True)
        bfont.setWeight(75)
        for i,(key,txt) in enumerate(keydict4a.items()):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(wspace20)
            # create icon for keyboard arrows
            btn = QtWidgets.QPushButton()
            btn.setFixedSize(bsize,bsize)
            btn.setStyleSheet('QPushButton'
                              '{ background-color : white;'
                              f'border : {bwidth}px solid black;'
                              f'border-radius : {radius};'
                              'padding : 1px }')
            btn.setText(key)
            btn.setFont(bfont)
            # create text label
            t = QtWidgets.QLabel(txt)
            t.setAlignment(QtCore.Qt.AlignLeft)
            t.setFont(font)
            keylay4.addWidget(btn, i, 0, alignment=QtCore.Qt.AlignCenter)
            keylay4.addWidget(t, i, 1, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft)
        keydict4b = {'space' : 'collect sequence for annotation',
                     '1' : 'seconds time scale',
                     '2' : 'minutes time scale',
                     '3' : 'hours time scale',
                     's' : 'save brain state annotation'}
        for j,(key,txt) in enumerate(keydict4b.items()):
            row = self.key_def_row(key, txt, uline=False, keyFont=keyFont, txtFont=font)
            # collect QLabel objects from row, manually insert into grid layout
            keylay4.addWidget(row.itemAt(0).widget(), j+i+1, 0, 
                              alignment=QtCore.Qt.AlignCenter)
            keylay4.addWidget(row.itemAt(1).widget(), j+i+1, 1, 
                              alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft)
        keyLayout4.addWidget(keytitle4, stretch=0)
        keyLayout4.addLayout(keylay4, stretch=2)
        # add layouts to main grid
        self.keygrid.addLayout(keyLayout1, 0, 0, 3, 1)
        self.keygrid.addWidget(vline(), 0, 1, 3, 1)
        self.keygrid.addLayout(keyLayout2, 0, 2, 1, 1)
        self.keygrid.addWidget(vline(orientation='h'), 1, 2, 1, 1)
        self.keygrid.addLayout(keyLayout3, 2, 2, 1, 1)
        self.keygrid.addWidget(vline(), 0, 3, 3, 1)
        self.keygrid.addLayout(keyLayout4, 0, 4, 3, 1)
        self.keygrid.setHorizontalSpacing(px_w(50, wpx))
        self.keygrid.setVerticalSpacing(px_h(30, hpx))
        self.centralLayout.addWidget(self.title)
        self.centralLayout.addWidget(self.keywidget)
        self.centralLayout.setSpacing(hspace20)
    
    
    def key_def_row(self, key, txt, spacing=10, uline=True, 
                    keyFont=QtGui.QFont(), txtFont=QtGui.QFont()):
        """
        Create informational text row
        @Params
        key - symbol on keyboard
        txt - text explanation of associated keypress action
        spacing - space between $key and $txt labels
        uline - if True, underline the first instance of $key in the $txt string
        keyFont, txtFont - fonts to use for key and text labels
        @Returns
        row - QHBoxLayout containing QLabel objects for key and text
        """
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(spacing)
        # create key label
        #k = QtWidgets.QLabel(key.upper())
        k = QtWidgets.QLabel(key)
        k.setAlignment(QtCore.Qt.AlignCenter)
        k.setFont(keyFont)
        # find and underline the first $key instance that begins a word in the $txt string
        if uline:
            ik = [i for i,l in enumerate(txt.lower()) if l==key.lower() and (i==0 or txt[i-1]==' ')]
            if len(ik) > 0:
                txt = txt[0:ik[0]] + '<u>' + txt[ik[0]] + '</u>' + txt[ik[0]+1 : ]
        # create text label
        t = QtWidgets.QLabel(txt)
        t.setAlignment(QtCore.Qt.AlignLeft)
        t.setFont(txtFont)
        row.addWidget(k, alignment=QtCore.Qt.AlignCenter, stretch=0)
        row.addWidget(t, alignment=QtCore.Qt.AlignCenter | QtCore.Qt.AlignLeft, stretch=2)
        return row


class GraphEEG(pg.PlotItem):
    """
    Custom graph allowing the user to select a segment of plotted data. 
    
    A selected sequence consists of the indices between two time points, which are 
    saved in the main window when the user double-clicks the corresponding point on 
    the plot. Once both the start and end points are established, the user can press
    the "X" key to annotate the selected sequence as "noise", or the "C" key to
    annotate the sequence as "clean". After the selected data is annotated, the
    bookend indices in the main window are cleared, and the next double-click
    within the plot boundaries is saved as the start index for a new sequence.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Instantiate plot object
        """
        super().__init__(*args, **kwargs)
        
        self.mainWin = parent
        self.window_mode = str(self.mainWin.objectName())  ### 'pwaves_qt' or 'pqt_neuropixel'
        
        # load ArrowButton objects from main window
        self.arrowStart_btn = self.mainWin.arrowStart_btn
        self.arrowEnd_btn = self.mainWin.arrowEnd_btn
        
        
    def mouseDoubleClickEvent(self, event):
        """
        Update start or end point of user-selected sequence
        """
        if event.button() == QtCore.Qt.LeftButton:
            if self.window_mode == 'pqt_neuropixel':
                SR = float(self.mainWin.sr_np)
                DT = float(self.mainWin.dt_np)
                TSEQ = np.array(self.mainWin.tseq_np)
            elif self.window_mode == 'pwaves_qt':
                SR = float(self.mainWin.sr)
                DT = float(self.mainWin.dt)
                TSEQ = np.array(self.mainWin.tseq)
            
            # convert mouse click position to Intan or Neuropixel index in recording
            point = self.vb.mapSceneToView(event.pos()).x()
            point = np.round(point, np.abs(int(np.floor(np.log10(DT)))))
            i = np.abs(TSEQ - point).argmin()
            idx = int(TSEQ[int(i)] * SR)
            
            self.mainWin.show_arrowStart = True
            self.mainWin.show_arrowEnd = True
            # one or zero noise boundaries set
            if self.mainWin.noiseEndIdx is None:
                # no noise boundaries set - clicked point is starting noise idx
                if self.mainWin.noiseStartIdx is None:
                    self.mainWin.show_arrowEnd = False
                    self.mainWin.noiseStartIdx = idx
                    self.arrowStart_btn.active = False
                    self.arrowStart_btn.update_style()
                # clicked point is later in recording than starting noise idx
                elif self.mainWin.noiseStartIdx <= idx:
                    self.mainWin.noiseEndIdx = idx
                    self.arrowStart_btn.active = False
                    self.arrowStart_btn.update_style()
                    self.arrowEnd_btn.active = True
                    self.arrowEnd_btn.update_style()
                # clicked point is earlier in recording than starting noise idx
                elif self.mainWin.noiseStartIdx > idx:
                    self.mainWin.noiseEndIdx = int(self.mainWin.noiseStartIdx)
                    self.arrowEnd_btn.active = False
                    self.arrowEnd_btn.update_style()
                    self.mainWin.noiseStartIdx = idx
                    self.arrowStart_btn.active = True
                    self.arrowStart_btn.update_style()
            # both noise boundaries set
            else:
                # adjust starting noise idx
                if self.arrowStart_btn.active == True:
                    # clicked point is earlier in recording than ending noise idx
                    if self.mainWin.noiseEndIdx >= idx:
                        self.mainWin.noiseStartIdx = idx
                    # clicked point is later in recording than ending noise idx
                    else:
                        self.mainWin.noiseStartIdx = int(self.mainWin.noiseEndIdx)
                        self.arrowStart_btn.active = False
                        self.arrowStart_btn.update_style()
                        self.mainWin.noiseEndIdx = idx
                        self.arrowEnd_btn.active = True
                        self.arrowEnd_btn.update_style()
                # adjust ending noise idx
                elif self.arrowEnd_btn.active == True:
                    # clicked point is later in recording than starting noise idx
                    if self.mainWin.noiseStartIdx <= idx:
                        self.mainWin.noiseEndIdx = idx
                    # clicked point is earlier in recording than ending noise idx
                    else:
                        self.mainWin.noiseEndIdx = int(self.mainWin.noiseStartIdx)
                        self.arrowEnd_btn.active = False
                        self.arrowEnd_btn.update_style()
                        self.mainWin.noiseStartIdx = idx
                        self.arrowStart_btn.active = True
                        self.arrowStart_btn.update_style()
        # if user double-clicks with right button, clear selected points
        else:
            self.mainWin.noiseStartIdx = None
            self.arrowStart_btn.active = False
            self.mainWin.show_arrowStart = False
            self.mainWin.noiseEndIdx = None
            self.arrowEnd_btn.active = False
            self.mainWin.show_arrowEnd = False
        # update live plot in main window
        self.mainWin.plot_eeg(findPwaves=False, findArtifacts=False)


class ArrowButton(pg.ArrowItem):
    """
    Arrow button representing the starting or ending point of a user-selected 
    data sequence in a GraphEEG object
    
    An ArrowButton is black when not selected; if its corresponding point is not 
    in the currently plotted time window, clicking the button will bring the point
    to the center of the graph. An additional click will "select" the arrow button,
    changing the color to white and determining whether the next double-clicked point 
    will update the starting or ending index of the selected sequence
    
    Example: Suppose the starting index of a currently selected sequence is 100, and
             the ending index is 200, so the sequence is [100, 101, 102 ... 199, 200].
             * If the user selects the start ArrowButton, the next double-clicked point
               (e.g. 150) will replace the current starting index, so the new selected
               sequence is [150, 151, 152 ... 199, 200]
             * If the user selects the end ArrowButton, the next double-clicked point
               (e.g. 150) will replace the current ending index, so the new selected
               sequence is [100, 101, 102 ... 149, 150]
    """
    def __init__(self, parent=None):
        super(ArrowButton, self).__init__()
        self.mainWin = parent
        # set arrow style
        wpx, hpx = pyautogui.size()
        hl, tl, tw = [px_w(n, wpx) for n in [10,10,4]]
        opts = {'angle': -90, 'headLen':hl, 'tipAngle':45, 'tailLen':tl, 'tailWidth':tw, 
                'pen':pg.mkPen((255,255,255),width=2), 'brush':pg.mkBrush(0,0,0)}
        self.setStyle(**opts)
        
        self.active = False   # initial button state is "unselected"
        self.pressPos = None  # clear mouse press position
        
    def mousePressEvent(self, event):
        """
        Save location of user mouse click
        """
        self.pressPos = event.pos()
        
    def mouseReleaseEvent(self, event):
        """
        Update ArrowButton selection in main window
        """
        if self.pressPos is not None and event.pos() in self.boundingRect():
            self.mainWin.switch_noise_boundary(self)
        self.pressPos = None
            
    def update_style(self):
        """
        Change color to white when selected, or black when unselected
        """
        if self.active == True:
            self.setBrush(pg.mkBrush(255,255,255))
        elif self.active == False:
            self.setBrush(pg.mkBrush(0,0,0))


class GraphEEG2(pg.PlotItem):
    
    def __init__(self, parent, *args, **kwargs):
        """
        Updated version of GraphEEG
        * The original GraphEEG relies on the main window to keep track of the
        user-selected signal, while GraphEEG2 contains all necessary functions for
        handling arrow displays and processing user input by itself. This allows
        GraphEEG2 to be easily "plugged in" to an application, without having to add
        any data handling functionality to the main window itself
        """
        super().__init__(*args, **kwargs)
        
        self.mainWin = parent
        self.sr_np = ''    # initialize no. of samples per second
        self.dt_np = ''    # initialize no. of seconds per sample
        self.fbin_np = ''  # initialize no. of samples per FFT bin
        
        self.LFP = []  # LFP signal with P-waves
        self.SD = []   # downsampled standard deviation of LFP signal
        self.pointer, self.i =  0,0
        #self.grp = ''    # electrode group number
        self.ch_name = '' # name of LFP channel
        self.f0, self.f1 = [-1,-1]  # bandpass filtering frequency band
        
        #self.color = ''  # signal plotting color
        
        # label for LFP signal ('LFP processed')
        self.label = LFPLabel(parent=self.mainWin.intanView)
        self.label.set_info(txt='LFP processed', color=(255,255,255))
        self.label.setParentItem(self)
        self.label.anchor(itemPos=(1,0), parentPos=(1,0), offset=(0,-12))
        # data item for LFP signal (white)
        self.data_item = pg.PlotDataItem()
        self.data_item.setPen((255,255,255),width=1)
        self.addItem(self.data_item)
        # data item for P-wave indices (lavender brush, blue outline)
        self.pidx_item = pg.PlotDataItem(pen=None, symbol='o', symbolSize=12, 
                                         symbolBrush=(175,175,250))
        self.pidx_item.setSymbolPen((0,0,255),width=1.5)
        self.addItem(self.pidx_item)
        self.pidx_item.setVisible(False)
        # data item for P-wave detection threshold (green)
        self.pthres_item = pg.InfiniteLine()
        self.pthres_item.setAngle(0)
        self.pthres_item.setPen((0,255,0),width=1)
        labelOpts = {'anchors':[(0,0),(0,0)], 
                     'position': 0, 'color':(0,255,0), 'movable':True}
        self.pthres_item.label = pg.InfLineLabel(self.pthres_item, text='', **labelOpts)
        self.addItem(self.pthres_item)
        self.pthres_item.setVisible(False)
        # data item for noisy signal (deeppink)
        self.noise_item = pg.PlotDataItem(connect='finite')
        self.noise_item.setPen((255,20,147),width=2)
        self.addItem(self.noise_item)
        self.noise_item.setVisible(False)
        # vector and data item for user-selected signal (yellow)
        self.iselect = []
        self.select_item = pg.PlotDataItem()
        self.select_item.setPen((255,255,0),width=2)
        self.addItem(self.select_item)
        
        # set axis params
        yax = self.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('LFP (uV)', units='', **labelStyle)
        yax_w = int(px_w(50, self.mainWin.WIDTH))
        yax.setWidth(yax_w)
        
        # arrow buttons in top left of graph
        self.arrowStart_btn = ArrowButton2(parent=self.mainWin.intanView, 
                                           parentGraph=self, boundary='start')
        self.arrowEnd_btn = ArrowButton2(parent=self.mainWin.intanView, 
                                         parentGraph=self, boundary='end')
        # (0/1, 0/1) = (left/right, top/bottom)
        self.arrowStart_btn.anchor(itemPos=(0,0), parentPos=(0,0), offset=(yax_w+10,-12))
        self.arrowEnd_btn.anchor(itemPos=(0,0), parentPos=(0,0), offset=(yax_w+60,-12))
        self.arrowStart_btn.hide()
        self.arrowEnd_btn.hide()
        
        # arrows indicating start/end points of user-selected noise in signal
        self.arrowStart = ArrowIndicator(parent=self)
        self.arrowEnd = ArrowIndicator(parent=self)
        self.addItem(self.arrowStart)
        self.addItem(self.arrowEnd)
        self.arrowStart.hide()
        self.arrowEnd.hide()
        
    
    def get_sampling_rate(self):
        # load Neuropixel sampling rates from main window
        self.sr_np = self.mainWin.sr_np
        self.dt_np = self.mainWin.dt_np
        self.fbin_np = self.mainWin.fbin_np
    
    
    def mouseDoubleClickEvent(self, event):
        """
        Update start or end point of user-selected sequence
        """
        if self.mainWin.lfp_plotmode == 0:
            TSEQ = np.array(self.mainWin.tseq_np)  # loaded vector of timepoints
            ISEQ = np.array(self.mainWin.iseq_np)  # loaded vector of indices
            
            # convert mouse click position to recording index
            point = self.vb.mapSceneToView(event.pos()).x()  # clicked timepoint
            i = np.abs(TSEQ - point).argmin()                # index in viewed signal
            idx = ISEQ[i]                                    # Neuropixel index
            
            # get x and y position of arrow label
            xpos, ypos = [d[i] for d in self.data_item.getData()]
            
            # one or zero noise boundaries set
            if self.arrowEnd.idx is None:
                # no noise boundaries set - clicked point is starting noise idx
                if self.arrowStart.idx is None:
                    # move arrowStart to clicked point
                    self.arrowStart.idx = idx
                    self.arrowStart.setPos(xpos, ypos+150)
                    self.arrowStart.show()
                    self.arrowStart_btn.show()
                    
                # clicked point is later in recording than starting noise idx
                elif self.arrowStart.idx <= idx:
                    # move arrowEnd to clicked point
                    self.arrowEnd.idx = idx
                    self.arrowEnd.setPos(xpos, ypos+150)
                    self.arrowEnd.show()
                    self.arrowEnd_btn.active = True
                    self.arrowEnd_btn.show()
                    
                # clicked point is earlier in recording than starting noise idx
                elif self.arrowStart.idx > idx:
                    # move arrowEnd to the position of arrowStart
                    self.arrowEnd.idx = int(self.arrowStart.idx)
                    self.arrowEnd.setPos(self.arrowStart.pos())
                    self.arrowEnd_btn.active = False
                    self.arrowEnd_btn.show()
                    
                    # move arrowStart to clicked point
                    self.arrowStart.idx = idx
                    self.arrowStart.setPos(xpos, ypos+150)
                    self.arrowStart_btn.active = True
                    
            # both noise boundaries set
            else:
                # adjust starting noise idx
                if self.arrowStart_btn.active == True:
                    # clicked point is earlier in recording than ending noise idx
                    if self.arrowEnd.idx >= idx:
                        # move arrowStart to clicked point
                        self.arrowStart.idx = idx
                        self.arrowStart.setPos(xpos, ypos+150)
                        
                    # clicked point is later in recording than ending noise idx
                    else:
                        # move arrowStart to the position of arrowEnd
                        self.arrowStart.idx = int(self.arrowEnd.idx)
                        self.arrowStart.setPos(self.arrowEnd.pos())
                        self.arrowStart_btn.active = False
                        
                        # move arrowEnd to clicked point
                        self.arrowEnd.idx = idx
                        self.arrowEnd.setPos(xpos, ypos+150)
                        self.arrowEnd_btn.active = True
                        
                # adjust ending noise idx
                elif self.arrowEnd_btn.active == True:
                    # clicked point is later in recording than starting noise idx
                    if self.arrowStart.idx <= idx:
                        # move arrowEnd to clicked point
                        self.arrowEnd.idx = idx
                        self.arrowEnd.setPos(xpos, ypos+150)
                        
                    # clicked point is earlier in recording than starting noise idx
                    else:
                        # move arrowEnd to the position of arrowStart
                        self.arrowEnd.idx = int(self.arrowStart.idx)
                        self.arrowEnd.setPos(self.arrowStart.pos())
                        self.arrowEnd_btn.active = False
                        
                        # move arrowStart to clicked point
                        self.arrowStart.idx = idx
                        self.arrowStart.setPos(xpos, ypos+150)
                        self.arrowStart_btn.active = True
            # update arrow button styles (active vs not active)
            self.arrowStart_btn.update_style()
            b1 = pg.mkBrush(255,255,0) if self.arrowStart_btn.active else pg.mkBrush(0,0,0)
            self.arrowStart.setBrush(b1)
            self.arrowEnd_btn.update_style()
            b2 = pg.mkBrush(255,255,0) if self.arrowEnd_btn.active else pg.mkBrush(0,0,0)
            self.arrowEnd.setBrush(b2)
            
            # update noise boundaries
            if (self.arrowStart.idx is not None) and (self.arrowEnd.idx is not None):
                # plot user-selected signal in yellow 
                self.iselect = np.arange(self.arrowStart.idx, self.arrowEnd.idx)
                tstart, tend = self.arrowStart.pos().x(), self.arrowEnd.pos().x()
                xdata = np.linspace(tstart, tend, len(self.iselect))
                ydata = self.LFP[0][0, self.iselect]
                self.select_item.setData(xdata, ydata, padding=None)
                
                
    def switch_noise_boundary(self, pressed):
        """
        Update active ArrowButton for manually selecting segments of a plotted signal
        @Params
        arrow - ArrowButton object clicked by user
        
        See ArrowButton documentation for details
        """
        if (self.arrowStart.idx is not None) and (self.arrowEnd.idx is not None):
            # starting indicator clicked
            if pressed == 'start':
                # starting indicator not active
                if self.arrowStart_btn.active == False:
                    self.arrowStart_btn.active = True  # activate start
                    self.arrowEnd_btn.active = False   # deactivate end
                # starting indicator already active
                elif self.arrowStart_btn.active == True:
                    point = self.arrowStart.idx / self.sr_np
                    tmin, tmax = self.vb.viewRange()[0]
                    # if starting timpoint is outside viewing window, update index
                    if point < tmin or point > tmax:
                        fft_idx = int(np.round(self.arrowStart.idx / self.fbin_np))
                        self.mainWin.index = fft_idx
                        self.mainWin.update_index()
                
            # ending indicator clicked
            elif pressed == 'end':
                # ending indicator not active
                if self.arrowEnd_btn.active == False:
                    self.arrowStart_btn.active = False  # deactivate start
                    self.arrowEnd_btn.active = True     # activate end
                # ending indicator already active
                elif self.arrowEnd_btn.active == True:
                    point = self.arrowEnd.idx / self.sr_np
                    tmin, tmax = self.vb.viewRange()[0]
                    # if ending timpoint is outside viewing window, update index
                    if point < tmin or point > tmax:
                        fft_idx = int(np.round(self.arrowEnd.idx / self.fbin_np))
                        self.mainWin.index = fft_idx
                        self.mainWin.update_index()
            # update arrow button styles (active vs not active)
            self.arrowStart_btn.update_style()
            b1 = pg.mkBrush(255,255,0) if self.arrowStart_btn.active else pg.mkBrush(0,0,0)
            self.arrowStart.setBrush(b1)
            self.arrowEnd_btn.update_style()
            b2 = pg.mkBrush(255,255,0) if self.arrowEnd_btn.active else pg.mkBrush(0,0,0)
            self.arrowEnd.setBrush(b2)
    
    
    def reset_selection(self):
        """
        Clear user selection, reset arrows
        """
        # clear "current index" for indicator arrows, reset brush to black, hide
        self.arrowStart.idx = None
        self.arrowEnd.idx = None
        self.arrowStart.setBrush(pg.mkBrush(0,0,0))
        self.arrowEnd.setBrush(pg.mkBrush(0,0,0))
        self.arrowStart.hide()
        self.arrowEnd.hide()
        # deactivate arrow buttons, reset styles, hide
        self.arrowStart_btn.active = False
        self.arrowEnd_btn.active = False
        self.arrowStart_btn.update_style()
        self.arrowEnd_btn.update_style()
        self.arrowStart_btn.hide()
        self.arrowEnd_btn.hide()
        # clear selected indices
        self.iselect = []
        self.select_item.clear()
        self.update()
    
    
    def updated_data(self):
        """
        Adjust y-range and graph label based on the type of signal plotted
        """
        if [self.f0, self.f1] != [-1,-1]:
            ftxt = f' ({self.f0} - {self.f1} Hz)' 
        else:
            fxts = ''
        if self.mainWin.lfp_plotmode == 0:  # plotting LFP signals
            self.setYRange(-1200, 800, padding=None)
            self.label.set_info(txt='LFP' + ftxt, color=(255,255,255))
        elif self.mainWin.lfp_plotmode == 1:  # plotting downsampled standard deviations
            self.setYRange(0, 300, padding=None)
            self.label.set_info(txt='S.D. LFP' + ftxt, color=(255,255,255))


class ArrowButton2(pg.LabelItem):
    """
    Updated version of ArrowButton to work with GraphEEG2
    """
    def __init__(self, parent=None, parentGraph=None, boundary='start'):
        super(ArrowButton2, self).__init__()
        self.parentView = parent
        self.setParentItem(parentGraph)
        self.parentGraph = parentGraph
        self.boundary = boundary
        if self.boundary == 'start':
            self.txt = '\u2190'  # left arrow
        elif self.boundary == 'end':
            self.txt = '\u2192'  # right arrow
        # initial "inactive" arrows are white, not bold
        labelOpts = {'color':(255,255,255),
                     'size':'25pt', 
                     'bold':False}
        self.setText(self.txt, **labelOpts)
        
        self.active = False   # initial button state is "inactive"
        self.pressPos = None  # clear mouse press position
        
        
    def mousePressEvent(self, event):
        """
        Save location of user mouse click
        """
        self.pressPos = event.pos()
        
    def mouseReleaseEvent(self, event):
        """
        Update ArrowButton selection in parent graph
        """
        if self.pressPos is not None and event.pos() in self.boundingRect():
            self.parentGraph.switch_noise_boundary(pressed=self.boundary)
        self.pressPos = None
            
    def update_style(self):
        """
        Change color to yellow when selected, or white when unselected
        """
        if self.active == True:
            labelOpts = {'color':(255,255,0),
                         'size':'25pt', 
                         'bold':True}
        elif self.active == False:
            labelOpts = {'color':(255,255,255),
                         'size':'25pt', 
                         'bold':False}
        self.setText(self.txt, **labelOpts)


class ArrowIndicator(pg.ArrowItem):
    """
    Vertical arrows denoting the boundaries of the user-selected signal in GraphEEG2
    """
    def __init__(self, parent=None):
        super(ArrowIndicator, self).__init__()
        self.parentGraph = parent  # GraphEEG2
        self.idx = None  # Neuropixel index corresponding to arrow position
        
        # set arrow style
        wpx, hpx = pyautogui.size()
        hl, tl, tw = [px_w(n, wpx) for n in [14,14,5]]
        opts = {'angle': -90, 'headLen':hl, 'tipAngle':45, 'tailLen':tl, 'tailWidth':tw, 
                'pen':pg.mkPen((255,255,255),width=2), 'brush':pg.mkBrush(0,0,0)}
        self.setStyle(**opts)
        

class LFPLabel(pg.LabelItem):
    """
    Text label for a plotted LFP signal
    """
    def __init__(self, parent=None):
        super(LFPLabel, self).__init__()
        self.mainWin = parent
    
    def set_info(self, txt='', color='white', size='12pt'):
        """
        Update label of the LFP signal
        @Params
        txt - name of the .mat file
        color - color of the plotted signal
        """
        labelOpts = {'color':color, 'size':size}
        self.setText(txt, **labelOpts)


class GraphLFP(pg.PlotItem):
    """
    Custom graph for managing and displaying Neuropixel LFP signals
    """
    def __init__(self, parent):
        """
        Instantiate plot object
        """
        super(GraphLFP, self).__init__()
        self.mainWin = parent
        self.LFP = []    # list of LFP arrays (rows=signals, columns=timepoints)
        self.SD = []     # downsampled standard deviations for LFP arrays
        self.grp = ''    # electrode group number
        self.name = ''   # name of LFP channel
        self.color = ''  # signal plotting color
        self.label = ''  # LFPLabel item for plot
        self.class_label = ''
        
        self.ptimes_item = pg.PlotDataItem(connect='pairs')  # data item for P-wave times
        self.ptimes_item.setPen((255,255,255),width=0.5)
        self.addItem(self.ptimes_item)
        self.ptimes_item.setVisible(False)
        
        self.data_item = pg.PlotDataItem()  # data item for plot
        self.addItem(self.data_item)
        
        self.f0, self.f1 = ['','']  # bandpass filtering frequency band
        
        self.pointer = 0  # index of currently plotted LFP array in list
        self.i = 0        # row of currently plotted signal in a given LFP array
        self.isSelected = False
        
        # set axis params
        yax = self.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('LFP (uV)', units='', **labelStyle)
        self.yax_w = int(px_w(50, self.mainWin.WIDTH))
        yax.setWidth(self.yax_w)
        
        
    def updated_data(self):
        """
        Adjust y-range and graph label based on the type of signal plotted
        """
        # add a star to LFPLabel if this LFP is being used as the P-wave channel
        star = '\u2605  ' if self.name == self.mainWin.graph_pwaves.ch_name else ''
        if self.mainWin.lfp_plotmode == 0:  # plotting LFP signals
            if self.i == 0:
                self.setYRange(-2000, 2000, padding=None)  # raw signal
                self.label.set_info(txt=star + self.name, color=self.color)
            elif self.i == 1:
                self.setYRange(-1000, 1000, padding=None)  # filtered signal
                self.label.set_info(txt=star + self.name + f' ({self.f0} - {self.f1} Hz)', color=self.color)
        elif self.mainWin.lfp_plotmode == 1:  # plotting downsampled standard deviations
            if self.i == 0:
                self.setYRange(-100, 1500, padding=None)   # S.D. of raw signal
                self.label.set_info(txt=star + 'S.D. ' + self.name, color=self.color)
            elif self.i ==  1:
                self.setYRange(0, 300, padding=None)       # S.D. of filtered signal
                self.label.set_info(txt=star + 'S.D. ' + self.name + f' ({self.f0} - {self.f1} Hz)', color=self.color)
            
        
    def paint(self, painter, *args):
        """
        Draw red outline around graph if selected by user
        """
        if self.isSelected:
            painter.setPen(pg.mkPen((255,0,0),width=1))
        else:
            painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(self.boundingRect())
        pg.PlotItem.paint(self, painter, *args)
        
        
    def mouseDoubleClickEvent(self, event):
        """
        Allow user to select/deselect graph by double clicking
        """
        #if event.button() == QtCore.Qt.LeftButton:
        if self.mainWin.annotation_mode == False:
            if self.isSelected:
                self.isSelected = False
            else:
                self.isSelected = True
            self.update()
            self.mainWin.enable_filtering()
                
    
    def contextMenuEvent(self, event):
        """
        Set context menu with the option to set this LFP as the P-wave channel
        """
        if self.mainWin.annotation_mode == False:
            if self.isSelected:
                self.menu = QtWidgets.QMenu()
                self.menu.setObjectName(self.name + '_menu')
                pwaveAction = QtWidgets.QAction('Set as P-wave channel')
                pwaveAction.setObjectName('pwave')
                pwaveAction.triggered.connect(self.mainWin.choose_pwave_channel)
                self.menu.addAction(pwaveAction)
                self.menu.exec_(QtGui.QCursor.pos())




class GraphUnitClassificationWindow(pg.PlotItem):
    
    showSignal = QtCore.pyqtSignal()
    hideSignal = QtCore.pyqtSignal()
    
    def __init__(self, unit, region, color):
        super(GraphUnitClassificationWindow, self).__init__()
        self.unit = unit
        self.region = region
        self.color = color
        
        self.label = ''
        self.class_label = ''
        self.isSelected = False
    
    
    def paint(self, painter, *args):
        """
        Draw outline around graph if selected by user
        """
        if self.isSelected:
            painter.setPen(pg.mkPen(self.color, width=3))
        else:
            painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(self.boundingRect())
        pg.PlotItem.paint(self, painter, *args)
        
        
    def mouseDoubleClickEvent(self, event):
        """
        Allow user to select/deselect graph by double clicking
        """
        if self.isSelected:
            self.isSelected = False
            self.hideSignal.emit()
        else:
            self.isSelected = True
            self.showSignal.emit()
        
        
        # add/remove selection outline
        self.update()
        
        # dlg = QtWidgets.QDialog()
        # lay = QtWidgets.QVBoxLayout(dlg)
        # label = 'Show data'
        
        # # action buttons
        # QBtn = QtWidgets.QDialogButtonBox.Yes | QtWidgets.QDialogButtonBox.No
        # bbox = QtWidgets.QDialogButtonBox(QBtn)
        # bbox.accepted.connect(self.accept)
        # bbox.rejected.connect(self.reject)
        # bbox.setCenterButtons(True)
        
        #if event.button() == QtCore.Qt.LeftButton:
        # if self.mainWin.annotation_mode == False:
        #     if self.isSelected:
        #         self.isSelected = False
        #     else:
        #         self.isSelected = True
        #     self.update()
        #     self.mainWin.enable_filtering()



class GraphUnit(pg.PlotItem):
    """
    Custom graph for managing and displaying Neuropixel single unit data from
    a given brain region
    """
    def __init__(self, parent):
        """
        Instantiate plot object
        """
        super(GraphUnit, self).__init__()
        self.mainWin = parent
        
        self.name = ''         # name of brain region (e.g. 'PAG')
        self.color = ''        # data plotting color
        self.label = ''        # LFPLabel item for plot (e.g. 52 PAG)
        self.class_label = ''  # LFPLabel item for unit classification (e.g. RW | P+)
        self.units = []        # list of units in brain region (e.g. [20,24,25,31,38])
        self.fr_mx = []        # array of unit firing rates (rows=units, columns=time bins)
        self.peak_fr = []      # peak firing rate for each unit
        self.curUnit = ''      # currently plotted unit
        self.i = ''            # row of current unit in firing rate data array
        self.hiddenUnits = []  # list of units to exclude from analysis (e.g. grouping by activity patterns)
        
        self.data_item = pg.PlotDataItem()  # data item for plot
        self.addItem(self.data_item)
        
        # set axis params
        yax = self.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        yax.setLabel('FR', units='', **labelStyle)
        self.yax_w = int(px_w(50, self.mainWin.WIDTH))
        yax.setWidth(self.yax_w)
    
    
    def contextMenuEvent(self, event):
        """
        Set context menu for plotting different units in a brain region graph
        """
        pos = event.pos()
        # show context menu only on click to LFPLabel
        if pos in self.vb.boundingRect() and pos not in self.label.boundingRect():
            return
        
        menu = QtWidgets.QMenu()
        menu.setObjectName(self.name + '_menu')
        
        # add sub-menu for switching the current unit for the graph
        subMenu_switch = menu.addMenu('Switch Unit')
        widths = [px_w(x, self.mainWin.WIDTH) for x in [20,50,30]]
        for unit in self.units:
            if unit == self.curUnit:
                continue
            if unit in self.hiddenUnits:
                continue
            unitAction = QtWidgets.QAction('     ' + str(unit), self)
            unitAction.triggered.connect(self.mainWin.switch_unit)
            if unit in self.mainWin.unotes.keys():
                notes = list(self.mainWin.unotes[unit])
            else:
                notes = ['-','-','-']
            if notes != ['-','-','-']:
                icon = get_unit_icon(notes, widths)
                unitAction.setIcon(icon)
                if notes[2] != '-':
                    unitAction.setIconText(notes[2])
            unitAction.setShortcutVisibleInContextMenu(False)
            subMenu_switch.addAction(unitAction)
            subMenu_switch.addSeparator()
        subMenu_switch.setStyleSheet('QMenu'
                                     '{'
                                     'background-color : rgba(240,240,240,255);'
                                     f'icon-size : {np.sum(widths)}px;'
                                     '}'
                                     'QMenu::item'
                                     '{'
                                     'background-color : rgba(0,0,0,0);'
                                     'color : rgba(0,0,0,255);'
                                     f'width : {px_w(200, self.mainWin.WIDTH)};'
                                     '}'
                                     'QMenu::item:selected'
                                     '{'
                                     'background-color : rgba(150,150,255,255);'
                                     '}')
        menu.addSeparator()
        
        # filter units by activity patterns/firing rate/etc
        #subMenu_filt = menu.addMenu('Filter Units')
        filtAction = QtWidgets.QAction('Filter Units', self)
        filtAction.triggered.connect(self.mainWin.filt_units)
        menu.addAction(filtAction)
        menu.addSeparator()
        
        # show unit raster plot/mean firing rate surrounding P-waves
        viewUnitRasterAction = QtWidgets.QAction('Show unit raster plot', self)
        viewUnitRasterAction.triggered.connect(self.mainWin.pplot_unit_raster)
        menu.addAction(viewUnitRasterAction)
        # show firing rates across recording for all units in the brain region
        viewRegionAction = QtWidgets.QAction('Show firing rates for all ' + self.name + ' units', self)
        viewRegionAction.triggered.connect(self.mainWin.pplot_region_units_FR)
        menu.addAction(viewRegionAction)
        # show firing rates surrounding P-waves for all units in the brain region
        viewRegionPAction = QtWidgets.QAction('Show P-wave-triggered firing for all ' + self.name + ' units', self)
        viewRegionPAction.triggered.connect(self.mainWin.pplot_region_units_PFR)
        menu.addAction(viewRegionPAction)
        # write new note for the current unit
        editUnitNoteAction = QtWidgets.QAction('Write unit note', self)
        editUnitNoteAction.triggered.connect(self.mainWin.edit_unit_note)
        menu.addAction(editUnitNoteAction)
        menu.exec_(QtGui.QCursor.pos())
    
    
class LFPView(pg.GraphicsLayoutWidget):
    """
    Custom pyqtgraph layout with vertically stacked GraphLFP or GraphUnit
    objects, contained within a scrollable window
    """
    def __init__(self, parent):
        """
        Instantiate GraphicsLayoutWidget to hold graphs
        """
        super(LFPView, self).__init__()
        self.mainWin = parent
        
        # create scrollable viewing window
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.qscroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.wheelEvent = lambda event: None  # allow scrolling with mouse wheel
        
        # set viewed widget to the pyqtgraph layout containing the LFP graphs
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self)
        
    
    def create_lfp_graphs(self, names, graph_type):
        """
        Create a pyqtgraph PlotItem for each LFP signal or brain region
        """
        wpx, hpx = pyautogui.size()
        self.graph_height = int(px_h(100, hpx))
        self.graph_width = self.mainWin.graph_width
        self.graph_spacing = int(px_h(10, hpx))
        self.xheight = int(px_h(50, hpx))
        
        # set size of the pyqtgraph layout widget (# of graphs * height of each graph + spacing)
        nplots = len(names)
        self.setFixedHeight(nplots*self.graph_height + self.graph_spacing*(nplots+1) + self.xheight)
        
        hsvs = [colorsys.hsv_to_rgb(hue,0.8,0.8) for hue in np.linspace(0,0.8,nplots)]
        colors = [(int(r*255), int(g*255), int(b*255)) for (r,g,b) in hsvs]
        
        graph_list = []
        for i,(name,color) in enumerate(zip(names,colors)):
            
            # create a graph for each LFP channel/distinct brain region
            if graph_type == 'unit':
                graph = GraphUnit(parent=self.mainWin)
            else:
                graph = GraphLFP(parent=self.mainWin)
            graph.name = name
            graph.color = color
            graph.setObjectName(name + '_pggraph')
            graph.setFixedWidth(self.graph_width)
            
            # set color and name of data item in plot
            graph.data_item.setObjectName(name + '_pgdata')
            graph.data_item.setPen(color, width=1)
            
            # set color and name of data label
            graph.label = LFPLabel(parent=self)
            graph.label.setObjectName(name + '_label')
            graph.label.set_info(txt=name, color=color)
            graph.label.setParentItem(graph)
            graph.label.anchor(itemPos=(1,0), parentPos=(1,0))
            
            # set color and size of the unit classification label
            graph.class_label = LFPLabel(parent=self)
            graph.class_label.setObjectName(name + '_classlabel')
            graph.class_label.setParentItem(graph)
            graph.class_label.anchor(itemPos=(0,0), parentPos=(0,0), offset=(graph.yax_w+5,0))
            graph.class_label.setVisible(False)
            
            # add graph to $graph_list container
            graph_list.append(graph)
            
            # set axis parameters
            if i < nplots-1:
                graph.setFixedHeight(self.graph_height)
                graph.hideAxis('bottom')
            else:
                graph.setFixedHeight(self.graph_height + self.xheight)
                graph.getAxis(name='bottom').setHeight(self.xheight)
                graph.getAxis(name='bottom').setLabel('Time (s)')
            
            # add graph to the layout container
            self.addItem(graph)
            self.nextRow()
        
        return graph_list


class UnitQListItem(QtWidgets.QListWidgetItem):
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except Exception:
            return QtWidgets.QListWidgetItem.__lt__(self, other)
    
    
class UnitQList(QtWidgets.QListWidget):
    
    #signal = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):
        super(UnitQList, self).__init__(parent)
        self.win = parent
        
        #self.model().rowsInserted.connect(self.emit_signal)
        #self.model().rowsRemoved.connect(self.emit_signal)
        self.itemSelectionChanged.connect(self.unit_selection_updated)
        
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        #self.setSortingEnabled(True)
        
        wpx, hpx = pyautogui.size()
        self.icon_widths = [px_w(x, wpx) for x in [20,50,30]]
        #self.icon_widths = [px_w(x, wpx) for x in [16,40,24]]
        self.setStyleSheet('QListWidget'
                           '{background-color : rgba(255,255,255,50);'
                           'border : 5px groove rgb(150,150,150);'
                           'border-radius : 5px;'
                           'font-size : 10pt;'
                           'font-weight : 700;'
                           f'icon-size : {np.sum(self.icon_widths)}px;'
                           'padding : 1px;'
                           '}'
                           'QListWidget::item'
                           '{background-color : white;'
                           'border : 2px solid rgb(200,200,200);'
                           'border-radius : 1px;'
                           'padding : 2px;'
                           '}'
                           'QListWidget::item:selected'
                           '{background-color : rgb(179,179,191);}'
                           )
        self.verticalScrollBar().setStyleSheet('QScrollBar:vertical'
                                               '{'
                                               'background : rgb(235,235,230);'
                                               'border : 1px solid rgb(150,150,150);'
                                               'margin : 15px 0 15px 0;'
                                               '}'
                                               'QScrollBar::handle:vertical'
                                               '{'
                                               'background : rgb(198,198,207);'
                                               'border : 1px solid rgb(96,96,105);'
                                               'min-height : 15px;'
                                               '}'
                                                'QScrollBar::add-line:vertical'
                                                '{'
                                                'height : 15px;'
                                                'subcontrol-position : bottom;'
                                                'subcontrol-origin : margin;'
                                                '}'
                                                'QScrollBar::sub-line:vertical'
                                                '{'
                                                'height : 15px;'
                                                'subcontrol-position : top;'
                                                'subcontrol-origin : margin;'
                                                '}'
                                                )
        self.setFixedWidth(px_w(200, wpx))
        
        
    def unit_selection_updated(self):
        self.selected_units = [int(item.text()) for item in self.selectedItems()]
        # enable "plot units" button if 1 or more units in list are selected
        self.win.plotUnits_btn.setEnabled(bool(len(self.selected_units) > 0))
        #self.signal.emit(selected_units)
        
    
    @QtCore.pyqtSlot(int, list, tuple)
    def update_qlist_from_btns(self, add, unit_list, color):
        """
        Add or remove region units from QListWidget when corresponding region button changes state
        """
        if add:
            # add region units to QListWidget
            listed_units = [int(self.item(i).text()) for i in range(self.count())]
            add_units = [u for u in unit_list if u not in listed_units and u not in self.win.hidden_units]
            #ddict = dict(self.win.unotes) if self.win.showManualClass_btn.isChecked() else dict(self.win.auto_unotes)
            for u in add_units:
                #item = QtWidgets.QListWidgetItem(str(u), self)
                item = UnitQListItem(str(u), self)
                item.setForeground(pg.mkBrush(color))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                notes = self.win.current_unotes.get(u, ['-','-','-'])
                notes[2] = '-'  # don't show notes icon
                icon = get_unit_icon(notes, self.icon_widths)
                item.setIcon(icon)
        else:
            # remove existing region units from QListWidget
            rm_items = [self.findItems(str(u), QtCore.Qt.MatchExactly) for u in unit_list]
            rm_items = [rmi for rmi in rm_items if rmi != []]
            _ = [self.takeItem(self.row(item[0])) for item in rm_items[::-1]]
            #_ = [self.takeItem(i) for i in range(self.count())[::-1]]
        
        
    def update_qlist_from_filt(self):
        #ddict = dict(self.win.unotes) if self.win.showManualClass_btn.isChecked() else dict(self.win.auto_unotes)
        
        # remove all listed units that appear in $self.win.hidden_units
        listed_units = [int(self.item(i).text()) for i in range(self.count())]
        rm_units = [u for u in listed_units if u in self.win.hidden_units]
        rm_items = [self.findItems(str(u), QtCore.Qt.MatchExactly) for u in rm_units]
        _ = [self.takeItem(self.row(item[0])) for item in rm_items[::-1]]
        
        # for checked regions, add any newly qualifying units
        for btn in self.win.region_btns:
            if btn.isChecked():
                add_units = [u for u in btn.units if u not in listed_units and u not in self.win.hidden_units]
                for u in add_units:
                    #item = QtWidgets.QListWidgetItem(str(u), self)
                    item = UnitQListItem(str(u), self)
                    item.setForeground(pg.mkBrush(btn.color))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    notes = self.win.current_unotes.get(u, ['-','-','-'])
                    notes[2] = '-'  # don't show notes icon
                    icon = get_unit_icon(notes, self.icon_widths)
                    item.setIcon(icon)
                
        

class FancyButton(QtWidgets.QPushButton):
    def __init__(self, text, color=None, opacity=255):
        super(FancyButton, self).__init__()
        
        if isinstance(color, tuple) and len(color) == 3:
            self.color = (color[0], color[1], color[2], opacity)
            self.color_pressed = (color[0], color[1], color[2], 255)
            self.color_disabled = (color[0], color[1], color[2], 20)
        else:
            self.color = (240,240,240,255)
            self.color_pressed = (160,160,160,255)
            self.color_disabled = (240,240,240,10)
        
        self.setText(text)
        self.setStyleSheet('QPushButton'
                           '{'
                           f'background-color : rgba{self.color};'
                           'border : 2px outset rgba(160,160,160,255);'
                           'border-radius : 4px;'
                           'color : rgba(0,0,0,255);'
                           'padding : 1px;'
                           '}'
                           'QPushButton:pressed'
                           '{'
                           f'background-color : rgba{self.color_pressed};'
                           'border : 2px outset rgba(100,100,100,255);'
                           'color : rgba(0,0,0,255);'
                           '}'
                           'QPushButton:disabled'
                           '{'
                           f'background-color : rgba{self.color_disabled};'
                           'border : 2px outset rgba(160,160,160,100);'
                           'color : rgba(0,0,0,50);'
                           '}'
                           )
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setWeight(99)
        self.setFont(font)
        
        
class FancyButton2(QtWidgets.QPushButton):
    def __init__(self, text, color=None, color_pressed=None, width=None, height=None, 
                 border_width=2, border_style='outset', border_radius=4, border_cparams=[0, 1], 
                 border_cparams_pressed=[0.7,0.5], text_color=(0,0,0), text_color_pressed=(85,85,85),
                 padding=2, font_size=10, font_weight=500):
        super(FancyButton2, self).__init__()
        
        # set button appearance
        self.setText(text)
        if isinstance(width, int):
            self.setFixedWidth(width)
        if isinstance(height, int):
            self.setFixedHeight(height)
        
        if color is None:
            self.color = (200,200,200)
        else:
            self.color = color

        # base button color
        dark = hue(self.color, 0.5, 1)
        medium = hue(self.color, 0.7, 1)
        light = hue(self.color, 0.8, 1)
        border = hue(self.color, border_cparams[0], border_cparams[1])
        
        # checked button color
        dark2 = hue(self.color, 0.2, 1)
        medium2 = hue(self.color, 0.35, 1)
        light2 = hue(self.color, 0.5, 1)
        border2 = hue(self.color, 0.6, 0)
        
        # pressed button color
        if color_pressed is None:
            self.color_pressed = (175,175,175)
        else:
            self.color_pressed = color_pressed
        border3 = hue(self.color, border_cparams_pressed[0], border_cparams_pressed[1])
        
        self.setStyleSheet('QPushButton' 
                            '{background-color : qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,'
                                                  f'stop:0    rgb{dark},'
                                                  f'stop:0.25 rgb{medium},'
                                                  f'stop:0.5  rgb{light},'
                                                  f'stop:0.75 rgb{medium},'
                                                  f'stop:1    rgb{dark});'
                            f'border : {border_width}px {border_style} rgb{border};'
                            f'border-radius : {border_radius}px;'
                            f'color : rgb{text_color};'
                            f'font-size : {font_size}pt;'
                            f'font-weight : {font_weight};'
                            f'padding : {padding}px'
                            '}'
                            'QPushButton:checked'
                            '{background-color : qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,'
                                                  f'stop:0    rgb{dark2},'
                                                  f'stop:0.25 rgb{medium2},'
                                                  f'stop:0.5  rgb{light2},'
                                                  f'stop:0.75 rgb{medium2},'
                                                  f'stop:1    rgb{dark2});'
                            f'border : 3px inset rgb{border2};'
                            f'color : rgb{text_color};'
                            '}'
                            'QPushButton:pressed'
                            '{'
                            f'background-color : rgb{self.color_pressed};'
                            f'color : rgb{text_color_pressed};'
                            f'border : {border_width}px solid rgb{border3};'
                            '}'
                            )
        


class RegionButton(QtWidgets.QPushButton):
    
    # emit 1/0 (add/remove units), list of units, color tuple
    signal = QtCore.pyqtSignal(int, list, tuple)
    
    def __init__(self, name, color, width=None, height=None):
        super(RegionButton, self).__init__()
        self.name = name        # name of brain region
        self.color = color
        self.units = []         # list of units in brain region
        
        # connect buttons
        self.setCheckable(True)
        self.toggled.connect(self.update_qlist)
        
        # set button appearance
        self.setText(name)
        if isinstance(width, int):
            self.setFixedWidth(width)
        if isinstance(height, int):
            self.setFixedHeight(height)

        # base button color
        dark = hue(self.color, 0.5, 1)
        medium = hue(self.color, 0.7, 1)
        light = hue(self.color, 0.8, 1)
        border = hue(self.color, 0, 1)
        
        # checked button color
        dark2 = hue(self.color, 0.2, 1)
        medium2 = hue(self.color, 0.35, 1)
        light2 = hue(self.color, 0.5, 1)
        border2 = hue(self.color, 0.6, 0)
        
        # pressed button color
        pressed_background = (175,175,175)
        border3 = hue(self.color, 0.7, 0.5)
        
        self.setStyleSheet('QPushButton' 
                            '{background-color : qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,'
                                                  f'stop:0    rgb{dark},'
                                                  f'stop:0.25 rgb{medium},'
                                                  f'stop:0.5  rgb{light},'
                                                  f'stop:0.75 rgb{medium},'
                                                  f'stop:1    rgb{dark});'
                            f'border : 2px outset rgb{border};'
                            'border-radius : 4px;'
                            'color : rgb(0,0,0);'
                            'font-size : 10pt;'
                            'font-weight : 500;'
                            'padding : 2px'
                            '}'
                            'QPushButton:checked'
                            '{background-color : qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,'
                                                  f'stop:0    rgb{dark2},'
                                                  f'stop:0.25 rgb{medium2},'
                                                  f'stop:0.5  rgb{light2},'
                                                  f'stop:0.75 rgb{medium2},'
                                                  f'stop:1    rgb{dark2});'
                            f'border : 3px inset rgb{border2};'
                            'color : rgb(0,0,0);'
                            '}'
                            'QPushButton:pressed'
                            '{'
                            f'background-color : rgb{pressed_background};'
                            'color : rgb(85,85,85);'
                            f'border : 3px solid rgb{border3};'
                            '}'
                            )
    
    def update_qlist(self):
        # add/remove units from QListWidget
        if self.isChecked():
            self.signal.emit(1, self.units, self.color)
        else:
            self.signal.emit(0, self.units, self.color)

        
        