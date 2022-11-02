#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 04:12:24 2022

@author: fearthekraken
"""
import os
import scipy
import scipy.io as so
import AS
import pwaves
import sleepy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison
import pingouin as ping
import seaborn as sns
import pickle
import re
import pdb

#%%
###   FIGURE 5D,E,F,G,H,I,J,K,N,P - DREADD vs mCherry statistics   ###
STAT = 'pwave freq'  # 'perc','freq','dur','is prob', 'pwave freq'
DREADD = 'hm3dq'  # 'hm3dq', 'hm4di'
istate=[1]

ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
PWAVES = True if 'pwave' in STAT else False
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=PWAVES); hm3dq_cno=hm3dq_cno['0.25']
(hm4di_sal, hm4di_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=PWAVES); hm4di_cno=hm4di_cno['5']
(mCherry_sal, mCherry_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=PWAVES)
mCherry_cno_025=mCherry_cno['0.25']; mCherry_cno_5=mCherry_cno['5']
#%%
if DREADD=='hm3dq':
    # hm3dq mice
    hm3dq_mice, hm3dq_cT = pwaves.sleep_timecourse(ppath, hm3dq_sal, istate=istate, tbin=18000, n=1, 
                                                   stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    hm3dq_mice, hm3dq_eT = pwaves.sleep_timecourse(ppath, hm3dq_cno, istate=istate, tbin=18000, n=1, 
                                                   stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    # mCherry mice
    mCherry_mice, mCherry_cT = pwaves.sleep_timecourse(ppath, mCherry_sal, istate=istate, tbin=18000, n=1, 
                                                       stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    mCherry_mice, mCherry_eT_025 = pwaves.sleep_timecourse(ppath, mCherry_cno_025, istate=istate, tbin=18000, n=1, 
                                                           stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    # hm3dq vs mCherry
    hm3dq_df = pwaves.df_from_timecourse_dict(tdict_list=[hm3dq_cT, hm3dq_eT, mCherry_cT, mCherry_eT_025],
                                               mice_list=[hm3dq_mice, hm3dq_mice, mCherry_mice, mCherry_mice],
                                               dose_list=['saline','cno','saline','cno'],
                                               virus_list=['hm3dq','hm3dq','mCherry','mCherry'])
    DF = hm3dq_df; C = 'blue'; DREADD = 'hm3dq'
elif DREADD=='hm4di':
    # hm4di mice
    hm4di_mice, hm4di_cT = pwaves.sleep_timecourse(ppath, hm4di_sal, istate=istate, tbin=18000, n=1, 
                                                   stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    hm4di_mice, hm4di_eT = pwaves.sleep_timecourse(ppath, hm4di_cno, istate=istate, tbin=18000, n=1, 
                                                   stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    # mCherry mice
    mCherry_mice, mCherry_cT = pwaves.sleep_timecourse(ppath, mCherry_sal, istate=istate, tbin=18000, n=1, 
                                                       stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    mCherry_mice, mCherry_eT_5 = pwaves.sleep_timecourse(ppath, mCherry_cno_5, istate=istate, tbin=18000, n=1, 
                                                         stats=STAT, flatten_is=4, exclude_noise=True, pplot=False)
    # hm4di vs mCherry
    hm4di_df = pwaves.df_from_timecourse_dict(tdict_list=[hm4di_cT, hm4di_eT, mCherry_cT, mCherry_eT_5],
                                                mice_list=[hm4di_mice, hm4di_mice, mCherry_mice, mCherry_mice],
                                                dose_list=['saline','cno','saline','cno'],
                                                virus_list=['hm4di','hm4di','mCherry','mCherry'])
    DF = hm4di_df; C = 'red'; DREADD = 'hm4di'
#%%
hm3dqsal_dur = AS.rem_duration(ppath, hm3dq_sal)
hm3dqsal_dur['dose'] = 'saline'
hm3dqcno_dur = AS.rem_duration(ppath, hm3dq_cno)
hm3dqcno_dur['dose'] = 'cno'
hm3dq_dur = pd.concat((hm3dqsal_dur, hm3dqcno_dur), axis=0, ignore_index=True)
[lodursal, hidursal] = [len(np.where(hm3dqsal_dur.dur < 60)[0]), len(np.where(hm3dqsal_dur.dur > 100)[0])]
[lodurcno, hidurcno] = [len(np.where(hm3dqcno_dur.dur < 60)[0]), len(np.where(hm3dqcno_dur.dur > 100)[0])]
plt.figure()
plt.bar(['saline', 'cno'], [lodursal, lodurcno], color=['gray','blue'])
plt.ylabel('# REM periods')
plt.title('# of short REM periods (< 60 s)')
plt.figure()
plt.bar(['saline', 'cno'], [hidursal, hidurcno], color=['gray','blue'])
plt.ylabel('# REM periods')
plt.title('# of long REM periods (> 100 s)')
#%%

sleepy.sleep_spectrum_simple(ppath, hm3dq_sal, istate=1, fmax=20, pmode=2, ci='sem', round_freq=True)
#%%
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True); hm3dq_cno=hm3dq_cno['0.25']
recordings = hm3dq_sal
#sleepy.sleep_spectrum_pwaves(ppath, hm3dq_sal, exclude_noise=True)


df = sleepy.sleep_spectrum_pwaves(ppath, recordings, win_inc=1, win_exc=1, istate=1, pnorm=False, 
                      nsr_seg=2, perc_overlap=0.75, recalc_highres=True, fmax=20, exclude_noise=True)
#%%
# plot figure
fig, axs = plt.subplots(nrows=1, ncols=2)
ydict = {1:{'perc':[0,15], 'freq':[0,10], 'dur':[0,100], 'is prob':[0,85], 'pwave freq':[0,1.1]},
         2:{'perc':[0,65], 'freq':[0,20], 'dur':[], 'pwave freq':[0,0.4]},
         3:{'perc':[0,75], 'freq':[], 'dur':[], 'pwave freq':[0,0.4]},
         4:{'perc':[0,8], 'freq':[], 'dur':[], 'pwave freq':[0,0.4]}}
for i,(ax,v) in enumerate(zip(axs,[DREADD, 'mCherry'])):
    df = DF.iloc[np.where(DF.virus==v)[0],:]
    bars = sns.barplot(x='dose', y='t0', ci=68, linewidth=1.5, edgecolor='black', capsize=0.15, errwidth=4, palette=['gray',C], data=df, ax=ax)
    lines = sns.lineplot(x='dose', y='t0', hue='Mouse', linewidth=2, legend=True, data=df, ax=ax)
    #[l.set_color('black') for l in lines.get_lines()]
    Y = [0,85] if STAT=='is prob' else ydict[istate[0]][STAT]
    #ax.set_ylim(Y)
    ax.set_xlabel('')
    ax.set_title(v)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i==0:
        ax.set_ylabel(STAT)
    else:
        ax.set_ylabel('')
        ax.get_yaxis().set_visible(False)
        ax.spines["left"].set_visible(False)
plt.show()
#%%
# plot figure for P-wave frequency differences
if STAT=='pwave freq':
    difdf = pd.DataFrame(columns=['mouse','virus','t0'])
    for v in [DREADD, 'mCherry']:
        vdf = DF.iloc[np.where(DF.virus==v)[0],:]
        isal, icno = [np.where(vdf.dose==d)[0]  for d in ['saline','cno']]
        if all(vdf.Mouse.iloc[isal] == vdf.Mouse.iloc[icno]):
            mice = list(vdf.Mouse.iloc[isal])
            difs = np.array(vdf.t0.iloc[icno]) - np.array(vdf.t0.iloc[isal])
            difdf = difdf.append(pd.DataFrame({'mouse':mice,
                                               'virus':[v]*len(mice),
                                               't0':difs}), ignore_index=True)
    plt.figure()
    sns.barplot(x='virus', y='t0', data=difdf, ci=68, palette={DREADD : C, 'mCherry':'gray'}, 
                edgecolor='black', linewidth=1.5, capsize=0.15, errwidth=4, order=[DREADD, 'mCherry'])
    sns.stripplot(x='virus', y='t0', data=difdf, color='black', size=8, order=[DREADD, 'mCherry'])
    plt.xlabel('')
    plt.ylabel(r'$\Delta$' + ' P-waves/s')
    plt.title('Dif. in P-wave frequency (CNO - saline)')
    plt.ylim([-0.6,0.4])
    plt.show()
    
    # stats
    data1, data2 = [np.array(difdf.t0.iloc[np.where(difdf.virus==v)[0]]) for v in set(difdf.virus)]
    p = stats.ttest_ind(data1, data2, nan_policy='omit')
    print(DREADD + ' vs mCherry (stat = ' + STAT + ')')
    print(f'T={round(p.statistic,3)}, p-value={round(p.pvalue,5)}, sig={"yes" if p.pvalue < 0.05 else "no"}')
#%%
# statistics: two-way mixed repeated measures ANOVA
res_anova = ping.mixed_anova(data=DF, dv='t0', within='dose', subject='Mouse', between='virus')
ping.print_table(res_anova)
if res_anova.loc[2,'p-unc'] < 0.05:
    # compare within virus group (saline vs. cno for DREADD mice and mCherry mice)
    mc = ping.pairwise_ttests(data=DF, dv='t0', within='dose', subject='Mouse', between='virus', 
                              padjust='holm', within_first=False)
    ping.print_table(mc)
    # compare between virus groups (DREADD vs. mCherry for saline and cno trials)
    mc2 = ping.pairwise_ttests(data=DF, dv='t0', within='dose', subject='Mouse', between='virus', 
                              padjust='holm', within_first=True)
    ping.print_table(mc2)
#%%
DREADD = 'hm4di'
mouse_avg='trial'; avg_mode='each'
### BOOTSTRAP DREADD P-WAVE FREQ DATA ###
if DREADD == 'hm3dq':
    df_hm3dq_sal = pwaves.state_freq(ppath, hm3dq_sal, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                     exclude_noise=True, pplot=False, return_mode='df')
    df_hm3dq_sal['virus'] = 'hm3dq'; df_hm3dq_sal['dose'] = 'saline'
    df_hm3dq_cno = pwaves.state_freq(ppath, hm3dq_cno, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                     exclude_noise=True, pplot=False, return_mode='df')
    df_hm3dq_cno['virus'] = 'hm3dq'; df_hm3dq_cno['dose'] = 'cno'
    df_hm3dq = pd.concat([df_hm3dq_sal, df_hm3dq_cno], axis=0, ignore_index=True)
    df_mCherry_sal = pwaves.state_freq(ppath, mCherry_sal, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                       exclude_noise=True, pplot=False, return_mode='df')
    df_mCherry_sal['virus'] = 'mCherry'; df_mCherry_sal['dose'] = 'saline'
    df_mCherry_cno_025 = pwaves.state_freq(ppath, mCherry_cno_025, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                           exclude_noise=True, pplot=False, return_mode='df')
    df_mCherry_cno_025['virus'] = 'mCherry'; df_mCherry_cno_025['dose'] = 'cno'
    df_mCherry_hm3dq = pd.concat([df_mCherry_sal, df_mCherry_cno_025], axis=0, ignore_index=True)
    DF1 = df_hm3dq; DF2 = df_mCherry_hm3dq; C='blue'; DREADD='hm3dq'

elif DREADD == 'hm4di':
    df_hm4di_sal = pwaves.state_freq(ppath, hm4di_sal, istate=[1,2,3,4], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                     exclude_noise=True, pplot=False, return_mode='df')
    df_hm4di_sal['virus'] = 'hm4di'; df_hm4di_sal['dose'] = 'saline'
    df_hm4di_cno = pwaves.state_freq(ppath, hm4di_cno, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                     exclude_noise=True, pplot=False, return_mode='df')
    df_hm4di_cno['virus'] = 'hm4di'; df_hm4di_cno['dose'] = 'cno'
    df_hm4di = pd.concat([df_hm4di_sal, df_hm4di_cno], axis=0, ignore_index=True)
    df_mCherry_sal = pwaves.state_freq(ppath, mCherry_sal, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                       exclude_noise=True, pplot=False, return_mode='df')
    df_mCherry_sal['virus'] = 'mCherry'; df_mCherry_sal['dose'] = 'saline'
    df_mCherry_cno_5 = pwaves.state_freq(ppath, mCherry_cno_5, istate=[1], flatten_is=4, mouse_avg=mouse_avg, avg_mode=avg_mode,
                                           exclude_noise=True, pplot=False, return_mode='df')
    df_mCherry_cno_5['virus'] = 'mCherry'; df_mCherry_cno_5['dose'] = 'cno'
    df_mCherry_hm4di = pd.concat([df_mCherry_sal, df_mCherry_cno_5], axis=0, ignore_index=True)
    DF1 = df_hm4di; DF2 = df_mCherry_hm4di; C='red'; DREADD='hm4di'
#%%
# bootstrap experimental and control mice
boot1 = AS.bootstrap_online_analysis(df=DF1, dv='freq', iv='dose', virus=DREADD, nboots=10000, 
                                     alpha=0.05, shuffle=False, seed=None, pplot=False)
boot2 = AS.bootstrap_online_analysis(df=DF2, dv='freq', iv='dose', virus='mCherry', nboots=10000, 
                                     alpha=0.05, shuffle=False, seed=None, pplot=False)
BOOT = pd.concat((boot1, boot2), axis=0, ignore_index=True)
#%%
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
file1 = os.path.join(ppath, 'hm3dq_boot.pkl')

AS.plot_bootstrap_online_analysis([], dv='freq', iv='dose', mode=1, plotType='violin', pload=file1, ylim=[-0.3, 0.3])
AS.compare_boot_stats([], mode=1, dv='freq', iv='dose', virus=['hm3dq','mCherry'], iv_val=[0,0], shuffled=[0,0], pload=file1)
#%%
file2 = os.path.join(ppath, 'hm4di_boot.pkl')

AS.plot_bootstrap_online_analysis([], dv='freq', iv='dose', mode=1, plotType='violin', pload=file2, ylim=[-0.3, 0.3])
AS.compare_boot_stats([], mode=1, dv='freq', iv='dose', virus=['hm4di','mCherry'], iv_val=[0,0], shuffled=[0,0], pload=file2)
#%%


#%%
###   FIGURE 5M,O - time-normalized P-wave frequency across brain state transitions   ###
DREADD = 'hm3dq'
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True); hm3dq_cno=hm3dq_cno['0.25']
(hm4di_sal, hm4di_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=True); hm4di_cno=hm4di_cno['5']
(mCherry_sal, mCherry_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=True)
mCherry_cno_025=mCherry_cno['0.25']; mCherry_cno_5=mCherry_cno['5']
sequence=[3,4,1,2]; state_thres=[(0,10000)]*len(sequence); nstates=[20,20,20,20]; cvm=[0.3,2.5]; evm= [0.28,2.2]  # NREM --> IS --> REM --> WAKE

if DREADD=='hm3dq':
    hm3dq_mice,hm3dq_cmx,hm3dq_cspe = pwaves.stateseq(ppath, hm3dq_sal, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
                                     vm=cvm, psmooth=[2,2], sf=4, mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    hm3dq_mice,hm3dq_emx,hm3dq_espe = pwaves.stateseq(ppath, hm3dq_cno, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # CNO
                                     vm=evm, psmooth=[2,2], sf=4, mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    # mCherry_mice,mCherry_cmx,mCherry_cspe = pwaves.stateseq(ppath, mCherry_sal, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
    #                                  vm=cvm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    mCherry_mice,mCherry_emx025,mCherry_cspe025 = pwaves.stateseq(ppath, mCherry_cno_025, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # CNO
                                     vm=evm, psmooth=[2,2], sf=4, mode='pwaves', mouse_avg='trial', pplot=False, print_stats=False)
    MICE1,CMX1,EMX1,MICE2,EMX2,C = [hm3dq_mice, hm3dq_cmx, hm3dq_emx, mCherry_mice, mCherry_emx025, 'blue']
    SP = np.concatenate([hm3dq_cspe, hm3dq_espe, mCherry_cspe025], axis=0).mean(axis=0)
elif DREADD=='hm4di':
    hm4di_mice,hm4di_cmx,hm4di_cspe = pwaves.stateseq(ppath, hm4di_sal, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
                                     vm=cvm, psmooth=[2,2], sf=4, mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    hm4di_mice,hm4di_emx,hm4di_espe = pwaves.stateseq(ppath, hm4di_cno, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # CNO
                                     vm=evm, psmooth=[2,2], sf=4, mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    # mCherry_mice,mCherry_cmx,mCherry_cspe = pwaves.stateseq(ppath, mCherry_sal, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # saline
    #                                  vm=cvm, psmooth=[2,2], mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    mCherry_mice,mCherry_emx5,mCherry_cspe5 = pwaves.stateseq(ppath, mCherry_cno_5, sequence=sequence, nstates=nstates, state_thres=state_thres, fmax=25, pnorm=1,  # CNO
                                     vm=evm, psmooth=[2,2], sf=4, mode='pwaves', mouse_avg='mouse', pplot=False, print_stats=False)
    MICE1,CMX1,EMX1,MICE2,EMX2,C = [hm4di_mice, hm4di_cmx, hm4di_emx, mCherry_mice, mCherry_emx5, 'red']
    SP = np.concatenate([hm4di_cspe, hm4di_espe, mCherry_cspe5], axis=0).mean(axis=0)

# plot average SP
plt.figure(); ax = plt.gca()
im = ax.pcolorfast(np.arange(0,SP.shape[1]), np.arange(0,SP.shape[0]/2,0.5), 
                   AS.adjust_spectrogram(SP, pnorm=0, psmooth=[2,2]), cmap='jet')
im.set_clim([0.2,2.1])
plt.colorbar(mappable=im, pad=0.0)
# plot average P-wave freq
pwaves.plot_activity_transitions([CMX1, EMX1, EMX2], [MICE1, MICE1, MICE2], plot_id=['gray', C, f'{"lightblue" if C=="blue" else "lightcoral"}'], 
                                 group_labels=[f'{DREADD} saline', f'{DREADD} cno', 'mCherry cno'], 
                                 xlim=nstates, xlabel='Time (normalized)', ylabel='P-waves/s', title='NREM-->IS-->REM-->Wake')
#%%
### EEG POWER SPECTRUM COMPARISONS ###
EXPERIMENT = 'opto'  # 'opto', 'dreadds'
DREADD = 'hm3dq'        # 'hm3dq', 'hm4di'

if EXPERIMENT == 'dreadds':
    ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
    (hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False); hm3dq_cno=hm3dq_cno['0.25']
    (hm4di_sal, hm4di_cno) = AS.load_recordings(ppath, 'crh_hm4di_tnrem.txt', dose=True, pwave_channel=False); hm4di_cno=hm4di_cno['5']
    (mCherry_sal, mCherry_cno) = AS.load_recordings(ppath, 'mCherry_all.txt', dose=True, pwave_channel=False)
    mCherry_cno_025=mCherry_cno['0.25']; mCherry_cno_5=mCherry_cno['5']
    if DREADD=='hm3dq':
        recording_lists = [hm3dq_sal, hm3dq_cno, mCherry_sal, mCherry_cno_025]
        viruses, conditions = list(zip(*[('hm3dq',0), ('hm3dq',0.25), ('mCherry',0), ('mCherry',0.25)])); C = 'blue'
        hm3dq_ps = pd.DataFrame()
    elif DREADD=='hm4di':
        recording_lists = [hm4di_sal, hm4di_cno, mCherry_sal, mCherry_cno_5]
        viruses, conditions = list(zip(*[('hm4di',0), ('hm4di',5), ('mCherry',0), ('mCherry',5)])); C = 'red'
        hm4di_ps = pd.DataFrame()
    ppaths = [ppath]*len(recording_lists)
    pmode = 0; harmcs=0; colname='dose'
elif EXPERIMENT == 'opto':
    path1 = '/media/fearthekraken/Mandy_HardDrive1/ChR2_Open'
    path2 = '/media/fearthekraken/Mandy_HardDrive1/ChR2_YFP_Open'
    ppaths = [path1,path2]
    recording_lists = [os.listdir(p) for p in ppaths]
    viruses, conditions = [['chr2','yfp'], ['no','yes']]
    opto_ps = pd.DataFrame(); C = 'blue'
    pmode = 1; harmcs=5; colname='Lsr'

istate = [1,2,3,4]; stateMap = {1:'REM', 2:'Wake', 3:'NREM', 4:'IS', 5:'IS-W', 6:'MA'}
#bands = {'delta':[0.5, 4.5],'sigma':[10, 15],'theta':[6, 9.5],'beta':[15.5, 20]}
bands = {'delta':[0.5, 4], 'theta':[6, 10], 'sigma':[11, 15], 'beta':[15.5, 20], 'gamma':[55, 99]}

for i,(path,recordings,virus,condition) in enumerate(zip(ppaths,recording_lists,viruses,conditions)):
    msg = f'## {virus} mice'
    if EXPERIMENT=='dreadds':
        msg += f', dose={condition} mg/kg CNO'
    print(msg)
    for s in istate:
        print(f'....... state = {s}')
        
        mx, freq, df, _ = sleepy.sleep_spectrum_simple(path, recordings, istate=s, pmode=pmode, harmcs=harmcs,
                                                       tstart=0, tend=-1, fmax=500, mu=[5,100], ci=95, pnorm=False, 
                                                       harmcs_mode='iplt', iplt_level=1, round_freq=True, peeg2=False, 
                                                       pemg2=False, exclusive_mode=0, ma_thr=20, ma_state=6, flatten_is=4, 
                                                       pplot=False)
        df['state'] = s
        df['virus'] = virus
        df['Freq'] = np.round(df['Freq'],1)
        if EXPERIMENT=='dreadds':
            df['dose'] = condition
            if DREADD=='hm3dq':
                hm3dq_ps = pd.concat((hm3dq_ps, df), ignore_index=True)
            elif DREADD=='hm4di':
                hm4di_ps = pd.concat((hm4di_ps, df), ignore_index=True)
        elif EXPERIMENT=='opto':
            opto_ps = pd.concat((opto_ps, df), ignore_index=True)
if EXPERIMENT=='dreadds':
    if DREADD=='hm3dq':
        df_ps = hm3dq_ps
    elif DREADD=='hm4di': 
        df_ps = hm4di_ps 
elif EXPERIMENT=='opto':
    df_ps = opto_ps
#%%
# get power of delta/theta/sigma/beta frequency bands for each mouse in each condition
df_bands = pd.DataFrame()
for b in bands:
    A = df_ps[(df_ps.Freq >= bands[b][0]) & (df_ps.Freq <= bands[b][1])].groupby(['virus', 'Idf', colname, 'state']).sum().reset_index()
    A['Pow'] = A['Pow'] * 0.5
    A['band'] = b
    A = A[['virus', 'Idf', colname, 'state', 'Pow', 'band']]
    # cno - saline
    if EXPERIMENT=='dreadds':
        D = np.array(A[A.dose>0]['Pow']) - np.array(A[A.dose==0]['Pow'])
        #B = pd.DataFrame(data={'virus':A.loc[:,'virus'], 'Idf':[], 'Lsr', 'state', 'Pow', 'band'}, columns=['virus', 'Idf', 'Lsr', 'state', 'Pow', 'band'])
        B = A[A.dose>0].copy()
    elif EXPERIMENT=='opto':
        D = np.array(A[A.Lsr=='yes']['Pow']) - np.array(A[A.Lsr=='no']['Pow'])
        #B = pd.DataFrame(data={'virus':A.loc[:,'virus'], 'Idf':[], 'Lsr', 'state', 'Pow', 'band'}, columns=['virus', 'Idf', 'Lsr', 'state', 'Pow', 'band'])
        B = A[A.Lsr=='yes'].copy()
    B[colname] = 'diff'
    B['Pow'] = D
    df_bands = pd.concat([df_bands, A, B], ignore_index=True)

# ANOVA to test for interaction between frequency & dose for experimental and control mice
df_cmp = pd.DataFrame([])
parallel_ttests = False

fmax = 20
for virus in df_ps.virus.unique():
    dfs = df_ps[(df_ps.virus==virus) & (df_ps.Freq <= fmax)]#.groupby(['Idf', 'Freq', 'Lsr', 'state']).mean().reset_index()
    res = ping.rm_anova(data=dfs, dv='Pow', within=['Freq',colname], subject='Idf')
    ping.print_table(res)
    freq = dfs.Freq.unique()
    if parallel_ttests:
        for f in freq:
            dfs_freq = dfs[dfs.Freq == f]
            if EXPERIMENT=='dreadds':
                r = scipy.stats.ttest_rel(dfs_freq.loc[dfs_freq.dose>0, 'Pow'], dfs_freq.loc[dfs_freq.dose==0, 'Pow'])
            elif EXPERIMENT=='opto':
                r = scipy.stats.ttest_rel(dfs_freq.loc[dfs_freq.Lsr=='yes', 'Pow'], dfs_freq.loc[dfs_freq.Lsr=='no', 'Pow'])
            pval = r.pvalue * len(freq)
            if pval > 1:
                pval = 1
            sig = 'no'
            if pval < 0.05:
                sig = 'yes'
            d = {'Freq':[f], 'p-corr':[pval], 'sig':[sig], 'virus':[virus]}
            df_cmp = pd.concat([df_cmp, pd.DataFrame(d)], ignore_index=True)
    else:
        res_tt = ping.pairwise_ttests(data=dfs, dv='Pow', within=['Freq',colname], subject='Idf', padjust='Holm')
        res_cmp = res_tt[res_tt.Contrast == f'Freq * {colname}'].copy()
        res_cmp['virus'] = virus
        res_cmp['sig'] = 'no'
        res_cmp.loc[res_cmp['p-corr']<0.05, 'sig'] = 'yes'
        df_cmp = pd.concat((df_cmp, res_cmp), ignore_index=True)
#%%
# plot saline and cno power spectrums for each brain state, for experimental and control mice
groups = df_ps.virus.unique()
ngroups = len(groups)
mmax = df_ps.loc[:,'Pow'].max()
Y = [0,2500]
for s in istate:
    fig, axes = plt.subplots(nrows=1, ncols=ngroups)
    for ax, virus in zip(axes, groups):
        dfs = df_ps[(df_ps.virus==virus) & (df_ps.Freq <= fmax) & (df_ps.state==s)]
        dfs = dfs.groupby(['Idf', 'Freq', colname, 'state']).mean().reset_index()
        sns.lineplot(data=dfs, x='Freq', y='Pow', hue=colname, ax=ax, palette={conditions[1]:C, conditions[0]:'grey'})
        y = df_cmp.loc[df_cmp.virus==virus, 'sig']
        idx = np.where(y=='yes')[0]
        tmp = np.zeros((len(y),))
        tmp[idx] = 1
        z = np.ones((len(y),)) * mmax    
        ax.plot(freq[idx], z[idx], 'r.')
        #ax.set_ylim([0, mmax+0.1*mmax])
        ax.set_ylim(Y)
        sns.despine()

# plot avg. power in freq bands during saline/laser-off vs cno/laser-on REM in experimental mice
viruses = df_bands.virus.unique()
plt.figure(); ax=plt.gca()
sns.barplot(data=df_bands[(df_bands.virus==viruses[0]) & (df_bands.state==1) & (df_bands[colname].isin([conditions[1], conditions[0]]))], 
            x='band', y='Pow', hue=colname, palette={conditions[1]:C, conditions[0]:'grey'}, ax=ax)
ax.set_title(f'{viruses[0]} mice: freq band power during REM')
ax.set_ylim([0,5000])
plt.show()

#%%
# ANOVA to test effect of laser/CNO on different power bands for experimental and control mice
df_bandstats = pd.DataFrame()
for s in istate:
    print('\n###   ' + stateMap[s] + ' ANOVA   ###')
    for virus in viruses:
        dfs = df_bands[(df_bands.virus==virus) & (df_bands.state==s) & (df_bands[colname].isin([conditions[1], conditions[0]]))]
        res = ping.rm_anova(data=dfs, subject='Idf', dv='Pow', within=[colname,'band'])
        res_tt = ping.pairwise_ttests(data=dfs, subject='Idf', dv='Pow', within=['band', colname], padjust='Holm')
        res_cmp = res_tt[res_tt.Contrast == f'band * {colname}'].copy()
        res_cmp['virus'] = virus
        res_cmp['sig'] = 'no'
        res_cmp['state'] = s
        res_cmp.loc[res_cmp['p-corr'] <0.05,'sig'] = 'yes'
        res_cmp['label'] = ''
        for i in range(res_cmp.shape[0]):
            p = res_cmp.loc[res_cmp.index[i], 'p-corr']
            if p < 0.001:
                res_cmp.loc[res_cmp.index[i], 'label'] = '***'
            elif p < 0.01:
                res_cmp.loc[res_cmp.index[i], 'label'] = '**'
            elif p < 0.05:
                res_cmp.loc[res_cmp.index[i], 'label'] = '*'
            else:
                pass
        df_bandstats = pd.concat([df_bandstats, res_cmp])
        ping.print_table(res_tt)

# plot saline and cno power spectrums, label freq. bands with significant differences
clrs = sns.color_palette("husl", len(bands))
for s in istate:
    j = 0
    fig, axes = plt.subplots(nrows=1, ncols=ngroups)
    for virus,ax in zip(viruses, axes):
        dfs = df_ps[(df_ps.virus==virus) & (df_ps.Freq <= fmax) & (df_ps.state == s)]
        #dfs = dfs.groupby(['Idf', 'Freq', 'Lsr', 'state']).mean().reset_index()
        g = sns.lineplot(data=dfs, x='Freq', y='Pow', hue=colname, palette={conditions[1]:C, conditions[0]:'grey'}, ax=ax, axes=ax)
        for i,b in enumerate(bands):
            ax.add_patch(matplotlib.patches.Rectangle((bands[b][0], bands[b][1]), bands[b][1]-bands[b][0], 
                                           mmax, facecolor=clrs[i], edgecolor=None, alpha=0.1))
            w = bands[b][1] - bands[b][0]
            label = df_bandstats[(df_bandstats.state==s) & (df_bandstats.band==b) & (df_bandstats.virus == virus)]['label'].iloc[0]
            ax.text(bands[b][0]+w/2, mmax, label, ha='center', va='center')
            ax.text(bands[b][0]+w/2, mmax-mmax*0.1, r'$\mathrm{\%s}$' % b, ha='center', va='center')
        sns.despine()        
        ax.set_xlim([0, fmax])
        ax.set_ylim([0, mmax])
        ax.set_ylabel('')
        ax.set_title(stateMap[s])
        if j == 0:
            ax.set_ylabel(r'PSD ($\mathrm{\mu V^2 / Hz}$)')
        if j < len(axes)-1:
            g.legend().remove()
        ax.set_xlabel('Freq. (Hz)')
        j += 1
    plt.subplots_adjust(wspace=0.4, left = 0.15, bottom=0.3)

#%%
# test whether dose-induced changes in freq. band power are different between experimental and control mice
df_bandbetween = pd.DataFrame()
for s in istate:
    print('\n###   ' + stateMap[s] + ' ANOVA   ###')
    dfs = df_bands[(df_bands.state==s) & (df_bands[colname].isin(['diff']))]
    res = ping.mixed_anova(data=dfs, subject='Idf', dv='Pow', within='band', between='virus')    
    ping.print_table(res)
    res_tt = ping.pairwise_ttests(data=dfs, subject='Idf', dv='Pow', within='band', between='virus', within_first=True, padjust='Holm')
    res_cmp = res_tt[res_tt.Contrast=='band * virus'].copy()
    res_cmp['state'] = s
    res_cmp['sig'] = 'no'
    res_cmp.loc[res_cmp['p-corr']<0.05, 'sig'] = 'yes'
    
    res_cmp['label'] = ''
    for i in range(res_cmp.shape[0]):
        p = res_cmp.loc[res_cmp.index[i], 'p-corr']
        if p < 0.001:
            res_cmp.loc[res_cmp.index[i], 'label'] = '***'
        elif p < 0.01:
            res_cmp.loc[res_cmp.index[i], 'label'] = '**'
        elif p < 0.05:
            res_cmp.loc[res_cmp.index[i], 'label'] = '*'
        else:
            pass
    res_cmp.drop('hedges', axis=1, inplace=True)
    df_bandbetween = pd.concat([df_bandbetween, res_cmp], ignore_index=True)

# plot dose-induced changes in different frequency bands for each state
cs = ['lightblue' if C=='blue' else 'lightcoral', 'lightgrey']
palette = {v:cs for v,cs in zip(viruses, cs)}
fig, axes = plt.subplots(figsize=(10,4.5), nrows=1, ncols=len(istate), constrained_layout=True)

j = 0
for s,ax in zip(istate, axes):
    mmax = np.max(np.abs(df_bands[df_bands[colname]=='diff']['Pow']))
    y = mmax+mmax*0.1
    sns.barplot(data=df_bands[(df_bands[colname]=='diff') & (df_bands.state==s)], x='band', y='Pow', hue='virus', 
                  palette=palette, errcolor='black', errwidth=2, order = list(bands.keys()), ax=ax)
    g = sns.stripplot(data=df_bands[(df_bands[colname]=='diff') & (df_bands.state==s)], x='band', y='Pow', hue='virus', 
                      dodge=True, palette={v:'black' for v in viruses}, linewidth=0, 
                      order=list(bands.keys()), size=3.5, jitter=1, ax=ax)
    ax.plot(ax.get_xticks(), [0]*len(ax.get_xticks()), color='lightgray', linewidth=2, zorder=0)
    #{v:'gray' for v in viruses}
    #if j == len(states):
    if j==0:
        ax.set_ylabel(r'Power ($\mathrm{\mu V^2}$)')
        #ax.set_xticklabels(['$\mathrm{\%s}$'%s for s in bands_list])
    else:
        ax.set_ylabel('')
        #ax.set_xticklabels(['' for s in bands_list])
    j+=1
    ax.set_xticklabels(['$\mathrm{\%s}$'%k for k in list(bands.keys())])
    ax.set_xlabel('')
    #ax.set_ylabel(r'Power ($\mathrm{\mu V^2}$)')
    ax.set_title(stateMap[s], fontdict={'fontsize':10})
    g.legend().remove()
    sns.despine()
    y = 1500
    ax.set_ylim([-y, y])
    for b,x in zip(list(bands.keys()), [0,1,2,3]):
        label = df_bandbetween[(df_bandbetween.band==b) & (df_bandbetween.state==s)]['label'].iloc[0]
        ax.text(x,mmax, label)        

#%%
###   FIGURE 2I,J - DF/F signal surrounding P-waves   ###
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
pzscore=[2,2,2]; p_iso=0; pcluster=0; ylim=[-0.3,0.8]; vm=[-1,1.3]
#%%
# original timecourse
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                          base_int=2.5, baseline_start=0, p_iso=0, pcluster=0, clus_event='waves', ylim=ylim, vm=vm, 
                          psmooth=(8,15), dn=1000, sf=1000, use405=False)[0]
#%%
# original timecourse, finer binning
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                          base_int=2.5, baseline_start=0, p_iso=0, pcluster=0, clus_event='waves', ylim=ylim, vm=vm, 
                          psmooth=(8,70), dn=250, sf=2000, use405=False)[0]

#%%
#recordings = ['McKinnon_021420n1']
# zoomed-in timecourse
_ = pwaves.dff_timecourse(ppath, recordings, istate=1, plotMode='t', dff_win=[1,1], pzscore=[2,2,2], mouse_avg='mouse',
                          base_int=0.25, baseline_start=0, p_iso=0, pcluster=0, clus_event='waves', ylim=[], vm=[], 
                          psmooth=[], dn=0, sf=150, use405=False)[0]
#%%
# single P-waves
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='t', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                          base_int=2.5, baseline_start=0, p_iso=0.8, pcluster=0, clus_event='waves', ylim=ylim, vm=vm, 
                          psmooth=(8,70), dn=250, sf=2000, use405=False)[0]
#%%
# cluster P-waves
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='t', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                          base_int=2.5, baseline_start=0, p_iso=0, pcluster=0.5, clus_event='waves', ylim=ylim, vm=vm, 
                          psmooth=(8,70), dn=250, sf=2000, use405=False)[0]
#%%
# jitter timecourse
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                          base_int=2.5, baseline_start=0, p_iso=0, pcluster=0, clus_event='waves', ylim=ylim, vm=vm, 
                          psmooth=(8,70), dn=250, sf=2000, use405=False, jitter=15)[0]
#%%
# 405 timecourse
_ = pwaves.dff_timecourse(ppath, recordings, istate=0, plotMode='ht', dff_win=[10,10], pzscore=pzscore, mouse_avg='mouse',
                          base_int=2.5, baseline_start=0, p_iso=0, pcluster=0, clus_event='waves', ylim=ylim, vm=vm, 
                          psmooth=(8,70), dn=250, sf=2000, use405=True)[0]
#%%
# DFF/LFP cross-correlation
ppath = '/home/fearthekraken/Documents/Data/photometry'
recordings = sleepy.load_recordings(ppath, 'pwaves_photometry.txt')[1]
# plot example
name = 'Fritz_032819n1'
AS.plot_example(ppath, 'Fritz_032819n1', tstart=4340.2, tend=4343.2, PLOT=['DFF','LFP_ANNOT'], dff_nbin=20, lfp_nbin=10)
#%%
#recordings = ['Fallon_072220n1']
# plot cross-correlation
_ = pwaves.dff_pwaves_corr(ppath, recordings, win=1, istate=1, dffnorm=True, ptrain=True, ptrial=False, 
                           dn=1, sf=400, ma_thr=20, ma_state=3, flatten_is=4, mouse_avg='trial', 
                           jitter=False, jtr_win=10, seed=0, base_int=0.2, baseline_start=0, baseline_end=-1, 
                           pplot=True, print_stats=True, p_iso=0)

#%%
###   LASER P-WAVES EMG ACTIVATION   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_emg.txt')[1]

data = pwaves.pwave_emg(ppath, recordings, win=[-1,1], istate=[1], recalc_amp=False, exclude_noise=True)[0]
lsr_pwaves, spon_pwaves, success_lsr, fail_lsr = data

#%%
###   DREADDS EMG TWITCHES   ###



#%%
###   eYFP EEG QUANTIFICATION   ###


#%%
###   CLOSED LOOP STATS   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
OPSIN = 'ic++'  # 'chr2', 'ic++'
if OPSIN == 'chr2':
    # ChR2
    chr2_rec = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]
    df_chr2 = AS.state_online_analysis(ppath, chr2_rec, istate=1, single_mode=False, plotMode=[], print_stats=False)
    df_chr2['virus'] = 'chr2'
    # eYFP (ChR2 protocol)
    yfp_chr2_rec = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]
    df_yfp_chr2 = AS.state_online_analysis(ppath, yfp_chr2_rec, istate=1, single_mode=False, plotMode=[], print_stats=False)
    df_yfp_chr2['virus'] = 'yfp'
    chr2_df = pd.concat([df_chr2, df_yfp_chr2], axis=0, ignore_index=True)
    DF = chr2_df; C='blue'; OPSIN='chr2'
elif OPSIN == 'ic++':
    # iC++
    ic_rec = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]
    df_ic = AS.state_online_analysis(ppath, ic_rec, istate=1, single_mode=False, plotMode=[], print_stats=False)
    df_ic['virus'] = 'ic++'
    # eYFP (iC++ protocol)
    yfp_ic_rec = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]
    df_yfp_ic = AS.state_online_analysis(ppath, yfp_ic_rec, istate=1, single_mode=False, plotMode=[], print_stats=False)
    df_yfp_ic['virus'] = 'yfp'
    ic_df = pd.concat([df_ic, df_yfp_ic], ignore_index=True)
    DF = ic_df; C='red'; OPSIN='ic++'
#%%
# plot bar graph
fig, axs = plt.subplots(nrows=1, ncols=2)
for i,(ax,v) in enumerate(zip(axs,[OPSIN, 'yfp'])):
    df = DF.iloc[np.where(DF.virus==v)[0],:]
    bars = sns.barplot(x='lsr', y='dur', ci=68, linewidth=1.5, edgecolor='black', capsize=0.15, errwidth=4, palette=['gray',C], data=df, ax=ax)
    lines = sns.lineplot(x='lsr', y='dur', hue='mouse', linewidth=2, legend=True, data=df, ax=ax)
    #[l.set_color('black') for l in lines.get_lines()]
    ax.set_title(v)
y = [min([ax.get_ylim()[0] for ax in axs]), max([ax.get_ylim()[1] for ax in axs])]
_ = [ax.set_ylim(y) for ax in axs]
plt.show()


# statistics: two-way mixed repeated measures ANOVAs
res_anova = ping.mixed_anova(data=DF, dv='dur', within='lsr', subject='mouse', between='virus')
ping.print_table(res_anova)

if res_anova.loc[2,'p-unc'] < 0.05:
    # compare within virus group (laser-off vs. laser-on for ChR2 mice and eYFP mice)
    mc = ping.pairwise_ttests(data=DF, dv='dur', within='lsr', subject='mouse', between='virus', 
                              padjust='holm', within_first=False)
    ping.print_table(mc)
    # compare between virus groups (ChR2 vs. eYFP for laser-off and laser-on trials)
    mc2 = ping.pairwise_ttests(data=DF, dv='dur', within='lsr', subject='mouse', between='virus', 
                              padjust='holm', within_first=True)
    ping.print_table(mc2)
#%%
### CLOSED LOOP BOOTSTRAPPING ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
OPSIN = 'ic++'  # 'chr2', 'ic++'
if OPSIN == 'chr2':
    # ChR2
    chr2_rec = sleepy.load_recordings(ppath, 'crh_chr2_cl.txt')[1]
    df_chr2_sm = AS.state_online_analysis(ppath, chr2_rec, istate=1, single_mode=True, plotMode=[], print_stats=False)
    # eYFP (ChR2 protocol)
    yfp_chr2_rec = sleepy.load_recordings(ppath, 'crh_yfp_chr2_cl.txt')[1]
    df_yfp_chr2_sm = AS.state_online_analysis(ppath, yfp_chr2_rec, istate=1, single_mode=True, plotMode=[], print_stats=False)
    DF1 = df_chr2_sm; DF2 = df_yfp_chr2_sm; OPSIN = 'chr2'

elif OPSIN == 'ic++':
    # iC++
    ic_rec = sleepy.load_recordings(ppath, 'crh_ic_cl.txt')[1]
    df_ic_sm = AS.state_online_analysis(ppath, ic_rec, istate=1, single_mode=True, plotMode=[], print_stats=False)
    # eYFP (iC++ protocol)
    yfp_ic_rec = sleepy.load_recordings(ppath, 'crh_yfp_ic_cl.txt')[1]
    df_yfp_ic_sm = AS.state_online_analysis(ppath, yfp_ic_rec, istate=1, single_mode=True, plotMode=[], print_stats=False)
    DF1 = df_ic_sm; DF2 = df_yfp_ic_sm; OPSIN = 'ic++'

# bootstrap experimental and control mice
boot1 = AS.bootstrap_online_analysis(df=DF1, dv='dur', iv='lsr', virus=OPSIN, nboots=10000, alpha=0.05, shuffle=True, seed=0, pplot=False)
boot2 = AS.bootstrap_online_analysis(df=DF2, dv='dur', iv='lsr', virus='yfp', nboots=10000, alpha=0.05, shuffle=True, seed=1, pplot=False)
BOOT = pd.concat((boot1, boot2), axis=0, ignore_index=True)

#%%
file2 = os.path.join(ppath, 'ic_cl_bootNEW4.pkl')
BOOT.to_pickle(file2)

#%%
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
file1 = os.path.join(ppath, 'chr2_cl_boot.pkl')

# show bootstrapped data
AS.plot_bootstrap_online_analysis([], dv='dur', iv='lsr', mode=1, plotType='violin', pload=file1, ylim=[-80,80])
AS.compare_boot_stats([], mode=1, dv='dur', iv='lsr', virus=['chr2','yfp'], iv_val=[0,0], shuffled=[0,0], pload=file1)

#%%
# iC++ plots
file2 = os.path.join(ppath, 'ic_cl_bootNEW.pkl')
AS.plot_bootstrap_online_analysis([], dv='dur', iv='lsr', mode=1, plotType='violin', pload=file2, ylim=[-80,80])
#%%
AS.compare_boot_stats([], mode=1, dv='dur', iv='lsr', virus=['ic++','yfp'], iv_val=[0,0], shuffled=[0,0], pload=file2)


#%%
###   LASER P-WAVES UPDATED GRAPHS 'N QUANTIFICATION   ###
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves2.txt')[1]
#%%
# top - averaged waveforms surrounding P-waves & laser
filename = 'wf_win025'; wform_win = [0.25,0.25]; istate=[1]
pwaves.avg_waveform(ppath, recordings, istate, mode='pwaves', win=wform_win, mouse_avg='trial',  # spontaneous & laser-triggered P-waves
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])
pwaves.avg_waveform(ppath, recordings, istate, mode='lsr', win=wform_win, mouse_avg='trial',     # successful & failed laser
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])
#%%
# waveforms
filename = 'wf_win025NEW'; wform_win = [0.25,0.25]; istate=[1]
pwaves.avg_waveform(ppath, recordings, istate, mode='pwaves', win=wform_win, mouse_avg='trial',  # spontaneous & laser-triggered P-waves
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])
pwaves.avg_waveform(ppath, recordings, istate, mode='lsr', win=wform_win, mouse_avg='trial',     # successful & failed laser
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, ylim=[-0.3,0.1])
#%%
# middle - averaged SPs surrounding P-waves & laser
filename = 'sp_win3NEW'; win=[-3,3]; pnorm=2
#null=True; null_win=0; null_match='lsr'
pwaves.avg_SP(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True, post_stim=0.1,  # spontaneous & laser-triggered P-waves
              mouse_avg='mouse', pnorm=pnorm, psmooth=[(8,8),(8,8)], vm=[(0.82,1.3),(0.8,1.4)], 
              nsr_seg=2, perc_overlap=0.95, fmax=25, recalc_highres=False, pload=filename, psave=filename)
#%%
pwaves.avg_SP(ppath, recordings, istate=[1], mode='lsr', win=win, plaser=True, post_stim=0.1,     # successful & failed laser
              mouse_avg='mouse', pnorm=pnorm, psmooth=[(8,8),(8,8)], vm=[(0.82,1.3),(0.7,2)], 
              fmax=25, recalc_highres=False, pload=filename, psave=filename)
#%%
# bottom - average high theta power surrounding P-waves & laser
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='pwaves', win=win, plaser=True,     # spontaneous & laser-triggered P-waves
                          post_stim=0.1, mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], 
                          pnorm=pnorm, psmooth=5, fmax=25, pload=filename, psave=filename, ylim=[0.5,1.5])  # 0.5, 1.5
_ = pwaves.avg_band_power(ppath, recordings, istate=[1], mode='lsr', win=win, plaser=True,        # successful & failed laser
                          post_stim=0.1, mouse_avg='mouse', bands=[(8,15)], band_colors=[('green')], 
                          pnorm=pnorm, psmooth=5, fmax=25, pload=filename, psave=filename, ylim=[0.5,1.5])
#%%
# spectral profiles: null vs spon vs success lsr vs fail lsr
filename = 'sp_win3NEW'
spon_win=[-0.5, 0.5]; lsr_win=[0,1]; collect_win=[-3,3]; frange=[0, 20]; pnorm=2; null=True; null_win=0; null_match='lsr'
df = pwaves.sp_profiles(ppath, recordings, spon_win=spon_win, lsr_win=lsr_win, collect_win=collect_win, frange=frange, 
                        null=null, null_win=null_win, null_match=null_match, plaser=True, post_stim=0.1, pnorm=pnorm, 
                        psmooth=12, mouse_avg='mouse', ci='sem', pload=filename, psave=filename)
#%%
# laser stats
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
#recordings = sleepy.load_recordings(ppath, 'lsr_pwaves.txt')[1]  # original recordings 
#recordings = sleepy.load_recordings(ppath, 'lsr_pwaves2.txt')[1] # all recordings
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_ol.txt')[1] # open loop recordings

df = pwaves.lsr_pwaves_sumstats(ppath, recordings, istate=[1,2,3,4], exclude_noise=True, lsr_iso=0)
#%%
df2 = df.groupby(['mouse','state'], as_index=False).agg({'lsr p-waves' : 'sum',
                                                         'spon p-waves' : 'sum',
                                                         'total p-waves' : 'sum',
                                                         '% lsr p-waves' : 'mean',
                                                         'success lsr' : 'sum',
                                                         'fail lsr' : 'sum',
                                                         'total lsr' : 'sum',
                                                         '% success lsr' : 'mean'})
#%%
# laser success rate in each brain state
sns.barplot(x='state', y='% success lsr', data=df2, ci=68, palette={1:'cyan',2:'purple',3:'gray',4:'darkblue'})
sns.pointplot(x='state', y='% success lsr', hue='mouse', data=df2, ci=None, markers='', color='black', legend=False)

#%%
# total successful and failed laser pulses in each state
state_sums = df2.groupby('state').sum()
lsr_sums = state_sums.loc[:, ['success lsr', 'fail lsr']]
lsr_sums.plot(kind='bar', stacked=True, color=['blue','red'])
lsr_sums_overall = pd.DataFrame(lsr_sums.sum(axis=1)).T
lsr_sums_overall.plot(kind='bar', stacked=True, color=['cyan','purple','gray','darkblue'])

#%%
# total laser-triggered and spontaneous P-waves in each state
pwave_sums = state_sums.loc[:, ['lsr p-waves', 'spon p-waves']]
pwave_sums.plot(kind='bar', stacked=True, color=['blue','green'])
pwave_sums_overall = pd.DataFrame(pwave_sums.sum(axis=1)).T
pwave_sums_overall.plot(kind='bar', stacked=True, color=['cyan','purple','gray','darkblue'])

#%%
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_emg.txt')[1]
filename = 'emg_win2'; wform_win = [3,3]; istate=[1]
pwaves.avg_waveform(ppath, recordings, istate, mode='pwaves', win=wform_win, mouse_avg='trial',  # spontaneous & laser-triggered P-waves
                    plaser=True, post_stim=0.1, pload=filename, psave=filename, exclude_noise=True, 
                    ylim=[], signal_type='EMG')
#%%
# raw EMG signal surrounding laser-triggered vs spontaneous P-waves
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_emg.txt')[1]
wform_win = [1,1]; istate=[1]
pwaves.avg_waveform(ppath, recordings, istate, mode='lsr', win=wform_win, mouse_avg='trial',  # spontaneous & laser-triggered P-waves
                    plaser=True, post_stim=0.1, pload=False, psave=False, exclude_noise=True, 
                    ylim=[-100,100], signal_type='EMG', dn=10)
#%%
# EMG amplitude surrounding laser events
ppath = '/home/fearthekraken/Documents/Data/sleepRec_processed/'
recordings = sleepy.load_recordings(ppath, 'lsr_pwaves_emg.txt')[1]
data = pwaves.pwave_emg(ppath, recordings, emg_source='raw', win=[-1, 1], istate=[1], rem_cutoff=True, 
                     recalc_amp=False, nsr_seg=0.2, perc_overlap=0.5, recalc_highres=False, 
                     r_mu=[5,500], w0=5/(1000./2), w1=-1, dn=50, smooth=100, pzscore=2, tstart=0, tend=-1, 
                     ma_thr=20, ma_state=3, flatten_is=4, exclude_noise=True, plaser=True, sf=0,
                     post_stim=0.1, lsr_iso=0.5, lsr_mode='lsr', mouse_avg='trial', pplot=True, ylim=[-0.5,1.5])
lsr_data, spon_data, success_data, fail_data = data[0]
#%%
df = pd.DataFrame(columns=['mouse', 'event', 'amp'])
ampbin = len(lsr_data[1][recordings[0]][0])
collect_win=1.0
ampdt = (collect_win*2)/ampbin
ibin = int(round(0.05/ampdt))
ictr = int(ampbin/2) if ampbin % 2 == 0 else int(round((ampbin-1)/2))

for rec in recordings:
    #win = 0.2; collect_win=1; ampbins=len()
    idf = rec.split('_')[0]
    for ev,ddict in zip(['lsr','spon','fail'], [lsr_data,spon_data,fail_data]):
        z = [scipy.stats.zscore(a) for a in ddict[1][rec]]
        #AMPDATA = [np.nanmean(x[ictr-ibin:ictr+ibin]) for x in z]
        #AMPDATA = [np.nanmax(x[ictr-ibin:ictr+ibin]) for x in z]
        AMPDATA = [np.nanmean(x[ictr-ibin : ictr+ibin+1]) for x in ddict[1][rec]]
        
        ddf = pd.DataFrame({'mouse':idf, 'event':ev, 'amp':AMPDATA})
        df = pd.concat([df, ddf], axis=0, ignore_index=True)
    # lsr = pd.DataFrame({'mouse':idf, 'event':'lsr', 'amp':np.nanmean(lsr_data[1][rec], axis=1)})
    # spon = pd.DataFrame({'mouse':idf, 'event':'spon', 'amp':np.nanmean(spon_data[1][rec], axis=1)})
    # success = pd.DataFrame({'mouse':idf, 'event':'success', 'amp':np.nanmean(success_data[1][rec], axis=1)})
    # fail = pd.DataFrame({'mouse':idf, 'event':'fail', 'amp':np.nanmean(fail_data[1][rec], axis=1)})
    # df = pd.concat([df, lsr, spon, success, fail], axis=0, ignore_index=True)
    
df = df.dropna(axis=0)
df2 = df.groupby(['mouse','event']).mean().reset_index()
#%%
df3 = pd.DataFrame(columns=['mouse','event','amp'])
np.random.seed(1)
for m in np.unique(df.mouse):
    nlsr = len(np.where((df.event=='lsr') & (df.mouse==m))[0])
    ispon = np.where((df.event=='spon') & (df.mouse==m))[0]
    irand = np.random.choice(ispon, size=nlsr, replace=False)
    ielse = np.where((df.event!='spon') & (df.mouse==m))[0]
    df3 = df3.append(df.iloc[np.concatenate((irand,ielse)),:], ignore_index=True)
#%%
sns.barplot(x='event', y='amp', data=df2, ci=68, order=['spon','lsr','fail'], 
            palette={'lsr':'blue','spon':'green','success':'pink','fail':'red'})
sns.pointplot(x='event', y='amp', hue='mouse', data=df2, order=['spon','lsr','fail'], ci=None, markers='', legend=False, color='black')
#%%
plt.figure()
sns.barplot(x='event', y='amp', data=df, ci=68, order=['spon','lsr','fail'])
sns.stripplot(x='event', y='amp', data=df3, order=['spon','lsr','fail'])
#%%
res = ping.rm_anova(data=df, dv='amp', within='event', subject='mouse')
ping.print_table(res)
res_tt = ping.pairwise_ttests(data=df, dv='amp', within='event', subject='mouse', padjust='holm')
ping.print_table(res_tt)
#%%
# example laser/P-wave/EMG change
#name = 'Olaf_111720n1'  # 3312 - 3314; 11650 - 11654; 11772 - 11774
#AS.plot_example(ppath, 'Olaf_111720n1', tstart=11771.5, tend=11774.5, PLOT=['LSR','LFP', 'EMG'], lfp_nbin=10, emg_nbin=1)
#AS.plot_example(ppath, 'Olaf_111720n1', tstart=11651.5, tend=11654.5, PLOT=['LSR','LFP', 'EMG'], lfp_nbin=10, emg_nbin=1)

#name = 'Olaf_111920n1'   # 6007 - 6010; 12977 - 12980; 13156 - 13159; 14116 - 14125
#AS.plot_example(ppath, 'Olaf_111920n1', tstart=6007, tend=6010, PLOT=['LSR','LFP', 'EMG'], lfp_nbin=10, emg_nbin=1)


AS.plot_example(ppath, 'Olaf_111920n1', tstart=12976.8, tend=12980.2, PLOT=['LSR','LFP', 'EMG'], 
                lfp_nbin=10, emg_nbin=1, emg_filt=[115,None])  # 180,None
# AS.plot_example(ppath, 'Olaf_111920n1', tstart=14115, tend=14127, PLOT=['LSR','LFP', 'EMG'], 
#                 lfp_nbin=10, emg_nbin=1, emg_filt=[])


#%%
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True); hm3dq_cno=hm3dq_cno['0.25']
recordings = hm3dq_sal

# middle - averaged SPs surrounding P-waves & laser
filename = 'DREADDsp_win3_wake_mas'; win=[-3,3]; pnorm=2
#null=True; null_win=0; null_match='lsr'
pwaves.avg_SP(ppath, recordings, istate=[2], win=win, plaser=False, mouse_avg='mouse',
              nsr_seg=2, perc_overlap=0.95, fmax=30, recalc_highres=False, 
              pnorm=pnorm, psmooth=[12,8], vm=[[0.8,1.5]], ma_state=3, ma_thr=20,
              pload=False, psave=filename)

#%%
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True); hm3dq_cno=hm3dq_cno['0.25']
recordings = hm3dq_sal
#rec = recordings[0]
ma_thr=20; ma_state=3; flatten_is=4; mouse_avg='mouse'
exclude_noise=True

istate = 2
r_theta = [6,10]
r_delta = [0.5,4]
thres = [40,60]

#%%
### histogram stuff in progress


ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/tmp'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
hm3dq_cno = hm3dq_cno['0.25']

hm3dqsal_dur = AS.rem_duration(ppath, hm3dq_sal)
hm3dqsal_dur['dose'] = 'saline'
hm3dqcno_dur = AS.rem_duration(ppath, hm3dq_cno)
hm3dqcno_dur['dose'] = 'cno'
pal = {'cno':'blue', 'saline':'gray'}
#%%
trialdf = pd.concat([hm3dqsal_dur, hm3dqcno_dur], axis=0, ignore_index=True)

fig = plt.figure(); ax = plt.gca()

# all durations (up to 300 s)
_ = sns.histplot(data=trialdf, bins=25, x='dur', hue='dose', stat='density', kde=True,
                 common_norm=False, log_scale=False, ax=ax, multiple='layer',
                 kde_kws=dict(bw_adjust=1.4), line_kws=dict(linewidth=2.5), hue_order=['saline','cno'])
dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/remdur_dreaddhist_commonnorm.svg'
#plt.savefig(dpath, format="svg")
#%%
fig = plt.figure(); ax = plt.gca()
_ = sns.histplot(data=trialdf, bins=13, x='dur', hue='dose', stat='density', kde=True,
                 common_norm=True, log_scale=False, ax=ax, multiple='layer',
                 kde_kws=dict(bw_adjust=1.4), line_kws=dict(linewidth=2.5), hue_order=['saline','cno'])
dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/remdur_dreaddhist_commonnormlargebins2.svg'
plt.savefig(dpath, format="svg")
#%%
fig = plt.figure(); ax = plt.gca()  # 10 bins from [130,275] is pretty okay
# Duration up to 100 s (long REM period)  # potential: 90 bins from [5,270]
# 105 bins from [4.2,270]; 100 bins from 4.6,275]; 
# 50 bins from [10,270]; 40 bins from [0,270]; 35 bins from [6,275]
# 14 bins from [10,270]; 12 bins from [6,270]; 10 bins from [1,270]; 8 bins from [10, 275]

nbins = 100; bin_range=[4.6,275]
#nbins = 35; bin_range=[6,275]
#nbins = 10; bin_range=[1,270]

_ = sns.histplot(data=trialdf, bins=nbins, binrange=bin_range, x='dur', hue='dose', stat='density',
                 common_norm=False, log_scale=False, ax=ax, multiple='layer', hue_order=['saline','cno'],
                 kde=True, kde_kws=dict(bw_adjust=1.4), line_kws=dict(linewidth=2.5))
plt.title(f'{nbins} bins, bin range = {bin_range}')

dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/remdur_dreaddhist_sepnormFINE.svg'
plt.savefig(dpath, format="svg")

#%%
salsort = hm3dqsal_dur.sort_values(by=['dur'], ascending=False)
cnosort = hm3dqcno_dur.sort_values(by=['dur'], ascending=False)

# top 25 longest and shortest REM periods in each group
n = 25
top_n = pd.concat([salsort.iloc[0:n, :], cnosort.iloc[0:n, :]], axis=0, ignore_index=True)
bot_n = pd.concat([salsort.iloc[-n:, :], cnosort.iloc[-n:, :]], axis=0, ignore_index=True)

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
_ = sns.boxplot(data=bot_n, x='dose', y='dur', linewidth=3, whis=np.inf, ax=ax1)
_ = sns.swarmplot(data=bot_n, x='dose', y='dur', hue='mouse', edgecolor='black',
                  linewidth=0.9, palette=sns.color_palette("deep"), size=7, ax=ax1)
ax1.get_legend().remove()

_ = sns.boxplot(data=top_n, x='dose', y='dur', linewidth=3, whis=np.inf, ax=ax2)
_ = sns.stripplot(data=top_n, x='dose', y='dur', hue='mouse', edgecolor='black',
                  linewidth=0.9, palette=sns.color_palette("deep"), size=7, jitter=0.1, ax=ax2)
ax2.get_legend().remove()
_ = fig.suptitle(f'The {n} longest and shortest REM periods in each group')
for ax in [ax1,ax2]:
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.5))

dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/25longshortdur.svg'
plt.savefig(dpath, format="svg")

# stats
s,c = [bot_n.dur[np.where(bot_n.dose=='saline')[0]], bot_n.dur[np.where(bot_n.dose=='cno')[0]]]
p = scipy.stats.ttest_ind(s, c, equal_var=False, nan_policy='omit')
print('')
print('Shortest REM periods - saline vs CNO')
print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')
print('')

s2,c2 = [top_n.dur[np.where(top_n.dose=='saline')[0]], top_n.dur[np.where(top_n.dose=='cno')[0]]]
p2 = scipy.stats.ttest_ind(s2, c2, equal_var=False, nan_policy='omit')
print('Longest REM periods - saline vs CNO')
print(f'T={round(p2.statistic,3)}, p-val={round(p2.pvalue,5)}')
#%%
# plot longest and shortest REM periods for each mouse in each condition
salsort = hm3dqsal_dur.sort_values(by=['dur','mouse'], ascending=False, ignore_index=True)
cnosort = hm3dqcno_dur.sort_values(by=['dur','mouse'], ascending=False, ignore_index=True)
n2 = 3

top_n_per = pd.DataFrame(columns=['mouse','dur','dose'])
bot_n_per = pd.DataFrame(columns=['mouse','dur','dose'])
mice = list(np.unique(salsort.mouse))
for m in mice:
    lsal, lcno = [dframe.dur[np.where(dframe.mouse==m)[0][0:n2]] for dframe in [salsort, cnosort]]
    ssal, scno = [dframe.dur[np.where(dframe.mouse==m)[0][-n2:]] for dframe in [salsort, cnosort]]
    top_n_per = pd.concat([top_n_per, pd.DataFrame({'mouse':m, 'dur':lsal, 'dose':'saline'}), 
                           pd.DataFrame({'mouse':m, 'dur':lcno, 'dose':'cno'})], axis=0, ignore_index=True)
    bot_n_per = pd.concat([bot_n_per, pd.DataFrame({'mouse':m, 'dur':ssal, 'dose':'saline'}), 
                           pd.DataFrame({'mouse':m, 'dur':scno, 'dose':'cno'})], axis=0, ignore_index=True)

# plot the n longest and shortest REM periods from each mouse
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
_ = sns.boxplot(data=bot_n_per, x='dose', y='dur', linewidth=3, whis=np.inf, ax=ax1)
_ = sns.stripplot(data=bot_n_per, x='dose', y='dur', hue='mouse', edgecolor='black',
                  linewidth=0.9, palette=sns.color_palette("deep"), size=7, jitter=0.1, ax=ax1)
ax1.get_legend().remove()

_ = sns.boxplot(data=top_n_per, x='dose', y='dur', linewidth=3, whis=np.inf, ax=ax2)
_ = sns.stripplot(data=top_n_per, x='dose', y='dur', hue='mouse', edgecolor='black',
                  linewidth=0.9, palette=sns.color_palette("deep"), size=7, jitter=0.1, ax=ax2)
ax2.get_legend().remove()
_ = fig.suptitle(f'The {n2} longest and shortest REM periods for each mouse')
for ax in [ax1,ax2]:
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.5))

dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/3longshortper_invdivid.svg'
plt.savefig(dpath, format="svg")

#%%
# plot mean of the n longest/shortest REM periods for each mouse
mean_bot = bot_n_per.groupby(['mouse','dose']).mean().reset_index()
mean_bot = mean_bot.sort_values(['mouse','dose'], ascending=False)
mean_top = top_n_per.groupby(['mouse','dose']).mean().reset_index()
mean_top = mean_top.sort_values(['mouse','dose'], ascending=False)

# for each mouse, plot the mean duration of a "long" and "short" REM period in each condition
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

_ = sns.boxplot(data=mean_bot, x='dose', y='dur', linewidth=3, whis=np.inf, order=['saline','cno'], ax=ax1)
lines = sns.lineplot(data=mean_bot, x='dose', y='dur', hue='mouse', sort=True,
                     linewidth=2.5, errorbar=None, markers=None, legend=False, ax=ax1)
_ = [l.set_color('black') for l in lines.get_lines()]

_ = sns.boxplot(data=mean_top, x='dose', y='dur', linewidth=3, whis=np.inf, order=['saline','cno'], ax=ax2)
lines = sns.lineplot(data=mean_top, x='dose', y='dur', hue='mouse', sort=True,
                     linewidth=2.5, errorbar=None, markers=None, legend=False, ax=ax2)
_ = [l.set_color('black') for l in lines.get_lines()]
for ax in [ax1,ax2]:
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.5))

dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/3longshortper_means.svg'
plt.savefig(dpath, format="svg")
#%%

s,c = [mean_bot.dur[1::2], mean_bot.dur[0:-1:2]]
p = scipy.stats.ttest_rel(s, c, nan_policy='omit')
print('')
print('Mean shortest REM period - saline vs CNO')
print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')
print('')

s2,c2 = [mean_top.dur[1::2], mean_top.dur[0:-1:2]]
p2 = scipy.stats.ttest_rel(s2, c2, nan_policy='omit')
print('')
print('Mean longest REM period - saline vs CNO')
print(f'T={round(p2.statistic,3)}, p-val={round(p2.pvalue,5)}')
print('')




    