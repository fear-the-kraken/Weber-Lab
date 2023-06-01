#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 05:16:22 2023

@author: fearthekraken
"""
import tkinter as Tk
import tkinter.filedialog as tkf
import sys
import re
import os.path
import numpy as np
import scipy.io as so
from shutil import copy2, move
from functools import reduce
import sleepy
import os
from datetime import datetime
import json
import h5py
import time
import pdb
import pandas as pd

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


def get_lowest_filenum(path, fname_base):
    """
    I assume that path contains files/folders with the name fname_base\d+
    find the file/folder with the highest number i at the end and then 
    return the filename fname_base(i+1)
    """
    files = [f for f in os.listdir(path) if re.match(fname_base, f)]
    l = []
    for f in files :
        a = re.search('^' + fname_base + "(\d+)", f)
        if a :
            l.append(int(a.group(1)))           
    if l: 
        n = max(l) + 1
    else:
        n = 1

    return fname_base+str(n)    

def parse_challoc(ch_alloc):
    """
    the channel allocation string must have one capital E,M,L (if present).
    If there are only lower-case e's, m's, or l's or only capital E's, M's, or L's,
    set the first e,m,l to upper-case and the rest to lower-case
    """
    
    # search for e's
    neeg = len(re.findall('[eE]', ch_alloc))
    nemg = len(re.findall('[mM]', ch_alloc))
    nlfp = len(re.findall('[lL]', ch_alloc))
    
    # only small e's
    if neeg == len(re.findall('e', ch_alloc)):
        ch_alloc = re.sub('e', 'E', ch_alloc, count=1)
    # only large E
    if neeg == len(re.findall('E', ch_alloc)):
        ch_alloc = re.sub('E', 'e', ch_alloc)
        ch_alloc = re.sub('e', 'E', ch_alloc, count=1)
    
    # only small m's
    if nemg == len(re.findall('m', ch_alloc)):
        ch_alloc = re.sub('m', 'M', ch_alloc, count=1)
    # only large M
    if nemg == len(re.findall('M', ch_alloc)):
        ch_alloc = re.sub('M', 'm', ch_alloc)
        ch_alloc = re.sub('m', 'M', ch_alloc, count=1)

    # only small l's
    if nlfp == len(re.findall('l', ch_alloc)):
        ch_alloc = re.sub('l', 'L', ch_alloc, count=1)
    # only large L
    if nlfp == len(re.findall('L', ch_alloc)):
        ch_alloc = re.sub('L', 'l', ch_alloc)
        ch_alloc = re.sub('l', 'L', ch_alloc, count=1)

    return ch_alloc

def get_param_file(ppath):
    """
    get the parameter file, i.e. the only .txt file within the specified
    folder $ppath
    """
    
    files = [f for f in os.listdir(ppath) if re.search('\.txt$', f)]
    if len(files)>1:
        print("Error more than one .txt files in specified folder %s" % ppath)
        sys.exit(1)
    if len(files) == 0:
        print("Error no parameter file in specified folder %s" % ppath)
    else:
        return files[0]

def get_infoparam(ppath, name):
    """
    name is a parameter/info text file, saving parameter values using the following
    syntax:
    field:   value 
    
    in regular expression:
    [\D\d]+:\s+.+    
    
    The function return the value for the given string field
    """
    fid = open(os.path.join(ppath, name), newline=None)    
    lines = fid.readlines()
    params = {}
    in_note = False
    fid.close()
    for l in lines :
        if re.search("^#[nN]otes:(.*)", l):
            #a = re.search("^#\s*(.*)", l)
            #params['note'] = [a.group(1)]
            #continue
            in_note = True
            params['note'] = []
            continue
        if in_note == True:
            if re.match("^[A-z_]+:", l):
                in_note=False
             
            if in_note and not(re.search("^\s+$", l)):
                params['note'].append(l)
        if re.search("^\s+$", l):
            continue
        if re.search("^[A-z_]+:" ,l):
            a = re.search("^(.+):" + "\s+(.*$)", l)
            if a :
                v = a.group(2).rstrip()
                v = re.split('\s+', v)
                params[a.group(1)] = v    
      
    # further process 'note' entry
    tmp = [i.strip() for i in params['note']]
    tmp = [i + ' ' for i in tmp]    
    if len(tmp)>0:
        f = lambda x,y: x+y
        tmp = reduce(f, tmp)
        tmp = re.split('//', tmp)
        tmp = ['#'+i for i in tmp if len(i)>0]

    #tmp = os.linesep.join(tmp)    
    params['note'] = assign_notes(params, tmp)
            
    return params

def assign_notes(params, notes):
    """
    check for each comment whether it was assigned to a specific mouse/mice using the 
    @ special sign; or (if not) assign it to all mice
    """
    comment = {} 
    
    mice = params['mouse_ID']
    for m in mice:
        comment[m] = []
    
    for l in notes:
        if re.match('@', l):
            for m in mice:
                if re.match('@' + m, l):
                    comment[m].append(l)
        else:
            comment[m].append(l)
                            
    return comment


#######################################################################################  
### START OF SCRIPT ###################################################################
#######################################################################################


raw_dir = '/media/fearthekraken/Mandy_HardDrive1/neuropixel/'  # directory containing raw data folders
PPATH = os.path.join(raw_dir, 'Processed_Sleep_Recordings')    # directory to save processed data folder
n_electrodes = 40   # number of neighboring Neuropixel electrodes to average for LFPs
load_win = 10       # chunks of time (min) to load at once when processing Neuropixel data
bp_filt = [5,30]    # [min,max] frequencies (Hz) used to bandpass filter LFPs
intan_dn_sr = 1000  # target sampling rate for Intan signals (False to avoid downsampling)
np_dn_sr = 2500     # target sampling rate for Neuropixel signals (False to avoid downsampling)


# choose directory containing raw data files/folders
root = Tk.Tk()
np_path = tkf.askdirectory(initialdir=raw_dir)
#root.update()
root.destroy()

# find "Neuropix-PXI-X.X" and "Rhythm_FPGA-X.X" folders
intanid = [f for f in os.listdir(np_path) if re.match('^Rhythm_FPGA', f)][0]
intan_dir = os.path.join(np_path, intanid)
npid = [f for f in os.listdir(np_path) if re.match('^Neuropix-PXI', f)][0]
np_dir = os.path.join(np_path, npid)

# load sync_messages.txt, get sampling rates
df = pd.read_csv(os.path.join(np_path, 'sync_messages.txt'),delimiter =":").reset_index()
# example dataframe:
# columns:  level_0        level_1           level_2       Software time  310292550@10000000Hz
#          Processor   Neuropix-PXI Id   100 subProcessor   0 start time  310292545@30000Hz
#          Processor   Neuropix-PXI Id   100 subProcessor   1 start time  25857713@2500Hz
#          Processor    Rhythm FPGA Id   101 subProcessor   0 start time  10343700@1000Hz
intan_row, np_row = 0,1
for i in range(len(df)):
    if 'Rhythm' in df['level_1'][i]:
        intan_row = i
    elif '1 start time' in df['Software time'][i]:
        np_row = i
        
intan_sr = int(df.iloc[intan_row,-1].split('@')[1].split('H')[0])
np_sr = int(df.iloc[np_row,-1].split('@')[1].split('H')[0])

# get timestamp and date for the end of the recording
ts = os.path.getmtime(os.path.join(np_path,'sync_messages.txt'))
cdate = datetime.utcfromtimestamp(ts).strftime('%m/%d/%y')

# load parameter .txt file, add sampling rates
param_file = get_param_file(intan_dir)
params = get_infoparam(intan_dir, param_file)
params['port'] = ['A', 'B', 'C', 'D']
# add date tag for recording
params['date'] = [cdate]
dtag = re.sub('/', '', cdate)
print(f'Using {dtag} as date tag')
# get conversion factor for scaling data to microvolts (uV)
FACTOR = 0.195
if 'conversion' in params:
    FACTOR = float(params['conversion'][0])
    print(f'Found conversion factor: {FACTOR}')
else:
    params['conversion'] = [str(FACTOR)]
    
# create name (mouse + dtag + n#) for processed recording folder
mouse = params['mouse_ID'][0]
fbase_name = mouse + '_' + dtag + 'n'
name = get_lowest_filenum(PPATH, fbase_name)

if not(os.path.isdir(os.path.join(PPATH,name))):
    print(f'Creating directory {name}\n')
    os.mkdir(os.path.join(PPATH,name))
    
print('\n#######################################')
print('####  PROCESSING EEG/EMG SIGNALS   ####')
print('#######################################\n')

# load amplifier file
data_amp = np.fromfile(os.path.join(intan_dir, 'continuous.dat'), 'int16')
data_amp = data_amp * FACTOR

# get channel allocation (e.g. EeMm) and number of channels (usually 4)
ch_alloc = parse_challoc(params['ch_alloc'][0])
nchannels = len(ch_alloc)

neeg, nemg, nlfp = 1,1,1
for ch_offset,c in enumerate(ch_alloc):
    dfile = ''
    # identify signal type (e.g. EEG, EEG2, EMG, EMG2)
    if re.match('E', c):
        dfile = 'EEG'
    if re.match('e', c):
        dfile = 'EEG' + str(neeg+1)
        neeg += 1
    if re.match('M', c):
        dfile = 'EMG'
    if re.match('m', c):
        dfile = 'EMG' + str(nemg+1)
        nemg += 1
    if re.match('L', c):
        dfile = 'LFP'
    if re.match('l', c):
        dfile = 'LFP' + (str(nlfp+1))
        nlfp += 1
    
    if len(dfile) > 0:
        # data_amp = [EEG EEG2 EMG EMG2 EEG EEG2 EMG EMG2 EEG ...]
        print(f'Saving {dfile} of mouse {mouse}')
        data = data_amp[ch_offset::nchannels]
        if intan_dn_sr != False and intan_dn_sr < intan_sr:
            data = downsample_signal(data, mode='accurate', sr=intan_sr, target_sr=intan_dn_sr)
            
        time.sleep(1)
        
        # save signal in h5py file
        with h5py.File(os.path.join(PPATH, name, dfile + '.mat'), 'w') as f:
            dset = f.create_dataset(dfile, shape=(1,len(data)), maxshape=(None,None), dtype='float32')
            dset[0,:] = data
        
if intan_dn_sr != False and intan_dn_sr < intan_sr:
    params['SR'] = [intan_dn_sr]
else:
    params['SR'] = [intan_sr]


print('\n#######################################')
print('###  PROCESSING NEUROPIXEL SIGNALS  ###')
print('#######################################\n')

# load position and estimated brain region of each electrode
regions = pd.read_json(os.path.join(np_path,'channel_locations.json')).T.iloc[0:-1]
regions['ch'] = regions.index.str.split('_').str[-1].astype('int64')

time.sleep(1)

ntotal_channels = 384  # total number of recording electrodes
vals = np.arange(0, ntotal_channels, n_electrodes)

# create key, get offsets, and initialize LFP file for each electrode group
grp_keys = []
grp_offsets = []
lfp_files = []
for x in range(0,len(vals)):
    # get offset of each signal from the start of $data_amp
    i = vals[x]
    j = ntotal_channels if i==vals[-1] else vals[x+1]
    grp_offsets.append(range(i,j))
    
    # find the brain region recorded by the most channels in the group
    xch = regions.iloc[i:j, :]
    regs = xch.brain_region.value_counts().reset_index()
    rname = '_' if i==vals[-1] else regs.iloc[0,0]
    key = 'LFP_' + rname + "_" + str(j)
    grp_keys.append(key)
    
    file = h5py.File(os.path.join(PPATH, name, key + '.mat'), 'a')
    lfp_files.append(file)

# for crazy high sampling rate, each data chunk must be shorter to avoid hanging computer
if np_dn_sr < np_sr:
    load_win = load_win / (np_sr / np_dn_sr)
nsamples = load_win*60*np_sr*ntotal_channels  # number of samples per chunk of loaded data


with open(os.path.join(np_dir, 'continuous.dat'), 'rb') as f:
    count = 0
    while True:
        # read in data chunk
        data_amp = np.fromfile(f, dtype='int16', count=nsamples)
        
        # print update
        w1 = load_win*count
        t1 = f'{w1}m' if w1<60 else f'{int(w1/60)}h{" "+str(w1%60)+"m" if w1%60 > 0 else ""}'
        w2 = w1 + int(len(data_amp)/ntotal_channels/np_sr/60)
        t2 = f'{w2}m' if w2<60 else f'{int(w2/60)}h{" "+str(w2%60)+"m" if w2%60 > 0 else ""}'
        print(f'Loading {t1} - {t2} ...')
        
        # calculate LFPs, update h5py files
        for key,offrange,lf in zip(grp_keys,grp_offsets,lfp_files):
            
            # create matrix of channels x timepoints, average across channels to get LFP
            lfp = np.array([data_amp[o :: ntotal_channels] for o in offrange]).mean(axis=0)
            
            if np_dn_sr != False and np_dn_sr < np_sr:
                lfp = downsample_signal(lfp, mode='accurate', sr=np_sr, target_sr=np_dn_sr)
            
            # band-pass filter LFP
            w0, w1 = [freq/(np_sr/2.0) for freq in bp_filt]
            lfp_filt = sleepy.my_bpfilter(lfp, w0=w0, w1=w1, N=4)
            
            # store LFPs in h5py file
            if count == 0:
                dset = lf.create_dataset(key, shape=(2,len(lfp)), maxshape=(None,None), dtype='float32')
                dset[0,:] = lfp
                dset[1,:] = lfp_filt
                dset.attrs['signals'] = ['raw', f'bp_{bp_filt[0]}_{bp_filt[1]}']
            else:
                lf[key].resize(lf[key].shape[1] + len(lfp), axis=1)
                lf[key][0,-len(lfp):] = lfp
                lf[key][1,-len(lfp):] = lfp_filt
        
        if len(data_amp) < nsamples:
            print(f'-- Calculated {len(vals)} LFPs from {n_electrodes} electrodes each\n')
            break
        count += 1

# close LFP files
for lf in lfp_files:
    lf.close()

if np_dn_sr != False and np_dn_sr < np_sr:
    params['SR_NP'] = [np_dn_sr]
else:
    params['SR_NP'] = [np_sr]

# save info file
fid = open(os.path.join(PPATH, name, 'info.txt'), 'w')    
comments = params['note'][mouse]  # write notes
for l in comments:
    fid.write(l + os.linesep)
for k in list(params.keys()):  # write other info tags
    v = params[k]
    if k == 'note':
        continue
    fid.write(k + ':' + '\t' + str(v[0]) + '\n')

# no colleagues
fid.write('colleagues:\t' + '')
fid.close()

print('Done!')
