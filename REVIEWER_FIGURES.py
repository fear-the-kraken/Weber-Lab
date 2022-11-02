#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:37:13 2022

@author: fearthekraken
"""
import scipy.io as so
import scipy
import AS
import pwaves
import sleepy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison
import pingouin as ping
import seaborn as sns
import pickle
import re
import pdb

#%%
###   REVIEWER FIGURE 1 - hM3dq frequency of short and long REM periods   ###
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=False)
hm3dq_cno = hm3dq_cno['0.25']
thres_short, thres_long = [45, 80]

hm3dqsal_dur = AS.rem_duration(ppath, hm3dq_sal)
hm3dqsal_dur['dose'] = 'saline'
hm3dqcno_dur = AS.rem_duration(ppath, hm3dq_cno)
hm3dqcno_dur['dose'] = 'cno'
#%%

mice = np.unique(hm3dqsal_dur.mouse)
DF = pd.DataFrame(columns=['mouse', 'dose', 'type', 'number'])
for m in mice:
    short_sal = len(np.where((hm3dqsal_dur.dur <= thres_short) & (hm3dqsal_dur.mouse == m))[0])
    short_cno = len(np.where((hm3dqcno_dur.dur <= thres_short) & (hm3dqcno_dur.mouse == m))[0])
    long_sal = len(np.where((hm3dqsal_dur.dur >= thres_long) & (hm3dqsal_dur.mouse == m))[0])
    long_cno = len(np.where((hm3dqcno_dur.dur >= thres_long) & (hm3dqcno_dur.mouse == m))[0])
    DF = pd.concat([DF, pd.DataFrame({'mouse':m, 
                                      'dose':['saline','cno','saline','cno'], 
                                      'type':['short','short','long','long'],
                                      'number':[short_sal, short_cno, long_sal, long_cno]})],
                    axis=0, ignore_index=True)
#%%
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
DF1 = DF.loc[np.where(DF.type=='short')[0], :]
_ = sns.barplot(x='dose', y='number', errorbar='se', data=DF1, ax=ax1)
lines = sns.lineplot(x='dose', y='number', hue='mouse', data=DF1, 
                     errorbar=None, markers=None, legend=False, ax=ax1)
_ = [l.set_color('black') for l in lines.get_lines()]
ax1.set_ylabel(f'# REM periods < {thres_short} s')
p1 = scipy.stats.ttest_rel(DF1.loc[DF1.dose=='saline', 'number'], DF1.loc[DF1.dose=='cno', 'number'])
print('')
print('Short REM periods - saline vs CNO')
print(f'T={round(p1.statistic,3)}, p-val={round(p1.pvalue,5)}')
print('')

DF2 = DF.loc[np.where(DF.type=='long')[0], :]
_ = sns.barplot(x='dose', y='number', errorbar='se', data=DF2, ax=ax2)
lines = sns.lineplot(x='dose', y='number', hue='mouse', data=DF2, 
                     errorbar=None, markers=None, ax=ax2)
_ = [l.set_color('black') for l in lines.get_lines()]
ax2.legend().remove()
ax2.set_ylabel(f'# REM periods > {thres_long} s')
p2 = scipy.stats.ttest_rel(DF2.loc[DF2.dose=='saline', 'number'], DF2.loc[DF2.dose=='cno', 'number'])
print('Long REM periods - saline vs CNO')
print(f'T={round(p2.statistic,3)}, p-val={round(p2.pvalue,5)}')


#%%
###   REVIEWER FIGURE 2 - non-normalized EEG features of P-waves   ###
### OPTION 1: Average raw EEG spectrogram surrounding P-waves
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True); hm3dq_cno=hm3dq_cno['0.25']
#%%
#filename = 'sp_win3_pnorm0'; win=[-3,3]; pnorm=0; psmooth=[]; vm=[[0,1600]]
filename = 'sp_win3_pnorm1'; win=[-3,3]; pnorm=1; psmooth=[]; vm=[[0.2,2.5]]

pwaves.avg_SP(ppath, hm3dq_cno, istate=[1], win=win, plaser=False, mouse_avg='mouse',
              nsr_seg=2, perc_overlap=0.95, fmax=25, recalc_highres=False, 
              pnorm=pnorm, psmooth=psmooth, vm=vm, pload=filename, psave=filename)

#%%
ISTATE = 2
### OPTION 2: PSD of REM sleep with and without P-waves
# single P-waves vs cluster P-waves vs no P-waves
df1 = sleepy.sleep_spectrum_pwaves(ppath, hm3dq_sal, win_inc=1, win_exc=1, istate=ISTATE, 
                                   pnorm=False, nsr_seg=2, perc_overlap=0.95,
                                   recalc_highres=False, fmax=15, exclude_noise=True,
                                   p_iso=0.8, pcluster=0.5, ma_state=3, ma_thr=10, pplot=False)
pal1 = {'single':'dodgerblue', 'cluster':'darkblue', 'no':'gray'}; order1=['no','single','cluster']
#%%
ISTATE = 2
# P-waves vs no P-waves
df2 = sleepy.sleep_spectrum_pwaves(ppath, hm3dq_sal, win_inc=1, win_exc=1, istate=ISTATE, 
                                   pnorm=False, nsr_seg=2, perc_overlap=0.95, 
                                   recalc_highres=False, fmax=15, exclude_noise=True,
                                   p_iso=0, pcluster=0, ma_state=3, ma_thr=10, pplot=False)
pal2 = {'yes':'blue', 'no':'gray'}; order2 = ['no','yes']

#%%
DF = df1; pal=pal1; order=order1
#DF = df2; pal=pal2; order=order2

df_theta = DF.loc[np.where((DF.Freq >= 8) & (DF.Freq <= 15))[0], ['Idf', 'Pow', 'P-wave']]
df_theta = df_theta.groupby(['Idf','P-wave']).sum().reset_index()
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, gridspec_kw={'width_ratios':[3,1]})
# plot PSDs
sns.lineplot(x='Freq', y='Pow', hue='P-wave', data=DF, errorbar=None, palette=pal, ax=ax1)
# plot bars
_ = sns.barplot(x='P-wave', y='Pow', data=df_theta, errorbar='se', 
                order=order, palette=pal, ax=ax2)
lines = sns.lineplot(x='P-wave', y='Pow', hue='Idf', data=df_theta,
                     errorbar=None, markersize=0, legend=False, ax=ax2)
_ = [l.set_color('black') for l in lines.get_lines()]
ax2.set_title('8-15 Hz')

dpath = '/home/fearthekraken/Dropbox/Weber/Data_Outputs/data/singleclusterpwaves_wakePSD.svg'
plt.savefig(dpath, format="svg")

#%%

# stats
if order == ['no','yes']:
    p = scipy.stats.ttest_rel(df_theta.loc[df_theta['P-wave']=='no', 'Pow'], 
                              df_theta.loc[df_theta['P-wave']=='yes', 'Pow'])
    print('')
    print('High theta (8-15 Hz) power - P-waves vs no P-waves')
    print(f'T={round(p.statistic,3)}, p-val={round(p.pvalue,5)}')
    print('')

if order == ['no','single','cluster']:
    res = ping.rm_anova(data=df_theta, dv='Pow', within='P-wave', subject='Idf')
    ping.print_table(res)
    res_tt = ping.pairwise_tests(data=df_theta, dv='Pow', within='P-wave', 
                                  subject='Idf', padjust='holm')
    ping.print_table(res_tt)
    

#%%
###   REVIEWER FIGURE 3 - P-wave association with theta during wake   ###
### 3A: Average EEG spectrogram surrounding P-waves during wake
ppath = '/media/fearthekraken/Mandy_HardDrive1/nrem_transitions/'
(hm3dq_sal, hm3dq_cno) = AS.load_recordings(ppath, 'all_hm3dq.txt', dose=True, pwave_channel=True); hm3dq_cno=hm3dq_cno['0.25']
filename = 'sp_win3_wake2'; win=[-3,3]; pnorm=2

pwaves.avg_SP(ppath, hm3dq_cno, istate=[2], win=win, plaser=False, mouse_avg='mouse',
              nsr_seg=2, perc_overlap=0.95, fmax=30, recalc_highres=False, 
              pnorm=pnorm, psmooth=[12,6], vm=[[0.72,1.5]], ma_state=3, pload=filename, psave=filename)

#%%
# non-normalized association with wake
filename = 'sp_win3_wake2'; win=[-3,3]; pnorm=0
pwaves.avg_SP(ppath, hm3dq_cno, istate=[2], win=win, plaser=False, mouse_avg='mouse',
              nsr_seg=2, perc_overlap=0.95, fmax=30, recalc_highres=False, 
              pnorm=pnorm, psmooth=[12,6], vm=[[0.72,1.5]], ma_state=3, pload=filename, psave=filename)

#%%
### 3B: P-wave frequency during high-theta vs low-theta wake
_ = pwaves.theta_pfreq(ppath, hm3dq_sal, istate=2, r_theta=[6,10], r_delta=[0.5,4], thres=[40,60])
#%%

[lodursal, hidursal] = [len(np.where(hm3dqsal_dur.dur < 60)[0]), 
                        len(np.where(hm3dqsal_dur.dur > 100)[0])]
[lodurcno, hidurcno] = [len(np.where(hm3dqcno_dur.dur < 60)[0]), 
                        len(np.where(hm3dqcno_dur.dur > 100)[0])]
# plot figure
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
plt.figure()
plt.bar(['saline', 'cno'], [lodursal, lodurcno], color=['gray','blue'])
plt.ylabel('# REM periods')
plt.title('# of short REM periods (< 60 s)')
plt.figure()
plt.bar(['saline', 'cno'], [hidursal, hidurcno], color=['gray','blue'])
plt.ylabel('# REM periods')
plt.title('# of long REM periods (> 100 s)')
