"""
##############################################################################
ShinGLMCC.Correlation.py

  This module primarily contains functions
    for computing correlation functions.

  by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
  modified ver 1.0. 2023.10.23
  modified ver 2.0. 2024.08.28
    * in dic2cor():
      Modified to output autocorrelation with ref sorted in ascending order.

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import os
import itertools
import joblib

import numpy as np
import pandas as pd


## ===================================================================
## Parameters
## ===================================================================

## PROCESS_NUM: The maximum number of concurrently running jobs in joblib.
##                -1: all CPUs are used.
PROCESS_NUM = -1

## REC_MS: The end time of the spike time sequences
##           to be used in this analysis [ms]
REC_MS = 3600*1000

## DELTA_MS: Symbolic parameters representing time resolution. [ms]
## WINHALF_MS: Symbolic parameters representing half the time width
##               of the correlation function. [ms]
DELTA_MS = {"J":0.1,"K":1.0,"N":10.0,"L":1/30.0}
WINHALF_MS = {"J":120.0,"K":500.0,"N":500.0,"C":50.0,"F":25.0}


## ===================================================================
## Private Functions
## ===================================================================

## _find_nearest_index() -----------------------------------
## Find the smallest index in "lst"
##  where the element is greater than "target"
##  starting the search from the provided "index".
##  [input]
##   +"lst":    (list of float) Spike times.
##   +"target": (float) Spike time to be compared against elements in "lst".
##   +"index":  (int) Starting index for the search.
##  [output]
##   +"result": (int) Index of the nearest value in "lst"
##                                         greater than "target".
##
##  <- _calc_correlogram()

def _find_nearest_index(lst, target, index):

    result = index if index>0 else 0
    while len(lst) > result and lst[result] <= target:
        result += 1

    return result


## _calc_correlogram() --------------------------------------
## Compute the correlation histogram
##  between "lsref"(reference,pre) and "lstar"(target,post).
##  [input]
##   +"lsref":  (list of float) Spike times a reference neuron.
##   +"lstar":  (list of float) Spike times a target neuron.
##             <lenght> N (N= Total number of neurons.)
##   +"ref":    (int) Reference neuron ID.
##   +"tar":    (int) Target neuron ID.
##   +"halfms": (float) Half the time width of the correlation histogram. [ms]
##   +"delta":  (float) Time resolution (time bin width) [ms]
##  [output]
##   +"lshst":  (list of int,(dim+2)) Correlation histogram,
##               followed by reference and target neuron IDs.
##            <lenght> dim+2 (dim= Bin number of correlation histogram.)
##
##  <- dic2cor()

def _calc_correlogram(lsref, lstar, ref, tar, halfms, delta):

    halfwin = int(halfms/delta)
    [jso,jeo] = [-1,-1]
    hst = [ 0 for j in range(2*halfwin+1)]
    
    for i in range(len(lsref)):
        js = _find_nearest_index(lstar,lsref[i]-halfms,jso)
        je = _find_nearest_index(lstar,lsref[i]+halfms,jeo)
        [jso,jeo] = [js,je]
        ## Quantization method to avoid bankers' rounding
        for t in [int((lstar[j]-lsref[i]+halfms)/delta+0.5)
                  for j in range(js,je)]:
            hst[t] += 1
        
    lshst = hst+[ref,tar]
    return lshst


## ===================================================================
## Public Functions
## ===================================================================

## dic2cor() -----------------------------------------------
## Calculate the correlation histogram from spike time series data
##  stored in a dict format.
##  [input]
##   +"dicspk": (dict)
##    {
##     *key:   (int) Neuron ID.
##     *value: (list of float) Spike times of each neuron.
##    }
##   +rsl:    (str) Symbolic parameter representing time resolution. [ms]
##                  see "DELTA_MS".
##  [output]
##   +"dfac": (Pandas DataFrame) Auto-correlation histogram table
##                                             with time resolution "rsl".
##   +"dfcc": (Pandas DataFrame) Cross-correlation histogram table
##                                             with time resolution "rsl".
##   .........................................................
##   +"df*c*": (Pandas DataFrame)
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##                   Correlation histogram values
##                   at Time lags -WINHALF_MS to WINHALF_MS. 
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)


def dic2cor(dicspk,rsl):

    lscls = dicspk.keys()
    itr = list(itertools.combinations(lscls,2))
    
    ## Compute the autocorrelation histogram
    joblibac = joblib.Parallel(n_jobs=PROCESS_NUM)(
        [joblib.delayed(_calc_correlogram)(
            dicspk[c],dicspk[c],c,c,WINHALF_MS[rsl],DELTA_MS[rsl])
         for c in lscls])
    dfac = pd.DataFrame(joblibac)
    lstime = [round(t*DELTA_MS[rsl]-WINHALF_MS[rsl],1)
              for t in range(len(dfac.T)-2)]
    dfac.columns = lstime+["ref","tar"]

    ## Compute the cross-correlation histogram (small idx to large idx)
    joblibcc = joblib.Parallel(n_jobs=PROCESS_NUM)(
        [joblib.delayed(_calc_correlogram)(
            dicspk[cr],dicspk[ct],cr,ct,WINHALF_MS[rsl],DELTA_MS[rsl])
         for cr,ct in itr])
    dfccp = pd.DataFrame(joblibcc)
    dfccp.columns = dfac.columns

    ## Get the rest cross-correlation histogram (large idx to small idx)
    dfiv = dfccp.copy()
    dfiv.columns = lstime[::-1]+["tar","ref"]


    ## Merge the cross-correlation histograms
    dfccr = pd.concat([dfccp,dfiv])
    dfcc = dfccr.sort_values(["ref","tar"]).reset_index(drop=True)
 
    return dfac, dfcc
