"""
##############################################################################
ShinGLMCC.Allneuron.py

  This module primarily contains functions
    for analyzing the behavior of the entire neuronal population.

  by Yasuhiro Tsubo
  modified ver. 2023.10.23

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import numpy as np
import pandas as pd

from scipy import signal


## ===================================================================
## Parameters
## ===================================================================

## REC_MS: The end time of the spike time sequences
##           to be used in this analysis [ms]
REC_MS = 3600*1000

## DELTA_MS: Symbolic parameters representing time resolution. [ms]
## WINHALF_MS: Symbolic parameters representing half the time width
##               of the correlation function. [ms]
DELTA_MS = {"J":0.1,"K":1.0,"N":10.0,"L":1/30.0}
WINHALF_MS = {"J":120.0,"K":500.0,"N":500.0,"C":50.0,"F":25.0}

## Parameters for Welch Method.
## NOVERLAP: The number of overlapping data points in consecutive segments.
## NPERSEG: The number of data points in each segment.
NOVERLAP = 500
NPERSEG = 1000


## ===================================================================
## Public Functions
## ===================================================================

## dic2all() -----------------------------------------------
## Merge and sort spike times of all neurons.
##  [input]
##   +"dicspk": (dict)
##    {
##     *key:   (int) Neuron ID.
##     *value: (list of float) Spike times of each neuron.
##    }
##  [output]
##   +"lsspk": (list of float) Merged and Sorted spike times of all neurons.

def dic2all(dicspk):

    lscls = list(dicspk.keys())
    lsspkt = []
    for c in lscls:
        lsspkt += dicspk[c]
    npspk = np.array(lsspkt,dtype=float)
    lsspk = np.sort(npspk).tolist()
    
    return lsspk


## cor2allcor() --------------------------------------------
## Merge crosscorrelation functions of all neuron pairs
##  [input]
##   +"dfac": (Pandas DataFrame) Auto-correlation histogram table
##                                             with time resolution "rsl".
##   +"dfcc": (Pandas DataFrame) Cross-correlation histogram table
##                                             with time resolution "rsl".
##     .........................................................
##   +"df*c": (Pandas DataFrame)
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##        Correlation histogram values at Time lags -WINHALF_MS to WINHALF_MS. 
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)
##  [output]
##   +"dfacc": (Pandas DataFrame)
##        Auto-correlation histogram of the all neurons' spikes
##         with time resolution "rsl"..
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##        Correlation histogram values at Time lags -WINHALF_MS to WINHALF_MS. 
##    }
##     Row: 1

def cor2allacr(dfac,dfcc):

    srac = dfac.sum()[:-2].T
    srcc = dfcc.sum()[:-2].T

    ## Autocorrelation histogram of the all neurons' spikes
    dfaac = pd.DataFrame(srac+srcc).T
    ## Summation of crosscorrelation histograms of all neuron pairs
    ##   This represents the value obtained by subtracting
    ##   the summation of autocorrelation functions of all neurons from dfaac.
    #dfaac = pd.DataFrame(srcc).T

    return dfaac


## dic2nrf() -----------------------------------------------
## Generate Coarse-grained neural response function with time bin "N"=10ms
##  [input]
##   +"dicspk": (dict)
##    {
##     *key:   (int) Neuron ID.
##     *value: (list of float) Spike times of each neuron.
##    }
##  [output]
##   +"srnrf": (Pandas Series) Coarse-grained neural response function.
##                              with time resolution "rsl" N=10ms K=1ms
##    {
##     * 0 to REC_MS/DELTA_MS: (float) with time resolution ("N"= 10ms)
##                  Neural response function (Coarse-grained histogram).
##    }

def dic2nrf(dicspk,rsl):
    
    binmax = int(REC_MS/DELTA_MS[rsl])
    hstlst = []
    for c in dicspk.keys():
        nptm = np.trunc(np.array(dicspk[c])/DELTA_MS[rsl]).astype(int)
        hst = [0 for j in range(binmax+1)]
        for k in range(len(nptm)):
            hst[nptm[k]] += 1
        hstlst += [hst[:-1]]
    dfnrf = pd.DataFrame(np.array(hstlst),index=dicspk.keys(),
                        columns=[int(t*DELTA_MS[rsl]) for t in range(binmax)])
    srnrf = dfnrf.sum()*(1000/DELTA_MS[rsl])

    return srnrf


## nrf2pow() -----------------------------------------------
## Compute the power spectrum of the neural response function
##  [input]
##   +"srnrf": (Pandas Series) Coarse-grained neural response function.
##                              with time resolution "rsl" N=10ms K=1ms
##    {
##     * 0 to REC_MS/DELTA_MS: (float) with time resolution ("N"= 10ms) [ms]
##                  Neural response function (Coarse-grained histogram).
##    }
##    +rsl:    (str) Symbolic parameter representing time resolution. [ms]
##                  see "DELTA_MS".
##  [output]
##   +"srpow": (Pandas Series)
##        The power spectrum of neural response function srnrf.
##    {
##     * 0 to 49.9: (float) with frequency step 0.1 [Hz] for "N"
##     * 0 to 499: (float) with frequency step 0.1 [Hz] for "K"
##                  Power spectrum (Coarse-grained histogram).
##    }
##

def nrf2pow(srnrf,rsl):

    npall=np.array(srnrf)
    sampr = int(1000/DELTA_MS[rsl])
    fq, powall = signal.welch(
        npall,fs=sampr,nperseg=NPERSEG,
        noverlap=NOVERLAP,window="hann")

    srpow = pd.Series(powall,index=fq).iloc[:-1]
    print(srpow)

    return srpow


## dithering_spikeseq() ------------------------------------
## Introduce temporal dithering to a spike time sequence
##              by applying Gaussian noise with SD "dit" [ms].
##  [input]
##   +"dicspk": (dict)
##    {
##     *key:   (int) Neuron ID.
##     *value: (list of float) Spike times of each neuron.
##    }
##   +"dit": (float) Gaussian temporal dithering standard deviation. [ms]
##  [output]
##   +"dicspk_d": (dict) The same format as dicspk, but dithered.
##    {
##     *key:   (int) Neuron ID.
##     *value: (list of float) Spike times of each neuron.
##    }

def dithering_spikeseq(dicspk, dit):

    lscls = list(dicspk.keys())
    dicspk_d = {}
    np.random.seed(dit)
    for c in lscls:
        npspk = np.array(dicspk[c],dtype=float)
        npd = npspk + np.random.normal(0,dit,len(npspk))
        boolmask = (npd>=0) & (npd<REC_MS)
        npdit = npd[boolmask]
        dicspk_d[c] = np.sort(npdit).tolist()

    return dicspk_d

