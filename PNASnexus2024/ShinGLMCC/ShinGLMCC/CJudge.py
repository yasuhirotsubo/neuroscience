"""
##############################################################################
ShinGLMCC.CJudge.py

  This module primarily contains functions
    related to determining the presence or absence of a combination.

  by Yasuhiro Tsubo
  modified ver. 2023.10.23

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm

import matplotlib.pyplot as plt


## ===================================================================
## Parameters
## ===================================================================

## DELTA_MS: Symbolic parameters representing time resolution. [ms]
## WINHALF_MS: Symbolic parameters representing half the time width
##               of the correlation function. [ms]
DELTA_MS = {"J":0.1,"K":1.0,"N":10.0,"L":1/30.0}
WINHALF_MS = {"J":120.0,"K":500.0,"N":500.0,"C":50.0,"F":25.0}



## ===================================================================
## Private Functions
## ===================================================================

## _eacheidominance() ------------------------------------------
## Compute the excitatory Inhibitory dominance for each models.
##  [input]
##   +"dfei": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"c": Number of putative connection.
##     *"e": Number of putative excitatory connection.
##     *"i": Number of putative inhibitory connection.
##    }
##     Rows: N: Total number of neurons.
##  [output]
##   +"aav": (float) Average absolute value
##                    of "Excitatory Inhibitory dominance".
##   +"pfc": (float) Perfect consistency
##                   Proportion of "Excitatory Inhibitory dominance"
##                    equaling exactly -1 or +1.
##   +"dfei": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"c": Number of putative connection.
##     *"e": Number of putative excitatory connection.
##     *"i": Number of putative inhibitory connection.
##     *"dom": Excitatory Inhibitory dominance.
##               (dom= (e+i)/(e+i) )
##    }
##     Rows: Number of neurons with non-zero putative connections.
##
##  <- EIdominance()

def _eacheidominance(dfei):

    dfei = dfei[dfei["c"]!=0]
    dfei = dfei.copy()
    dfei["dom"] = (dfei["e"]- dfei["i"])/ dfei["c"]

    aav = np.abs(dfei["dom"]).mean(numeric_only=True)
    pfc = len(dfei[np.abs(dfei["dom"])>0.999])/len(dfei)
    dfei = dfei.reset_index().rename(columns={"index":"ref"})

    return aav, pfc, dfei



## ===================================================================
## Public Functions
## ===================================================================

## classicalCC() ------------------------------------
## Check for significant differences in histogram bin values.
## Each histogram value is assumed to obey a normal distribution
##  with a standard deviation equal to the square root of the mean.
##
## Analytically, the mean of crosscorrelation histogram is calculated as: 
##   = (R_ref)x(R_tar)x(T)x(dt) 
##   R_ref= Firing rate of a reference neuron. [Hz]
##   R_tar= Firing rate of a target neuron. [Hz]
##   T= Measurement time. [s]
##   dt= Bin width of crosscorrelation histogram. [s]
##
## Note: In this function, we do not use the above formula directly.
## Instead, we estimate the mean directly from the crosscorrelation function.
##  [input]
##   +"dfccK": (Pandas DataFrame) Cross-correlation histogram table
##                                             with time resolution "K"=1ms.
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##                   Correlation histogram values
##                   at Time lags -WINHALF_MS to WINHALF_MS. 
##     * "ref": (int) Reference Neuron ID.
##     * "tar": (int) Target Neuron ID.
##                      (dim = The number of time bins.)
##    }
##     Rows: N(N-1), auto-indexed. (N: Total number of neurons.)
##   +threshold": (float) Significance threshold.
##  [output]
##   +"dfbsc": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"maxcc": (int) Maximum value of cross-correlation histogram.
##     *"mincc": (int) Minimum value of cross-correlation histogram.
##     *"avecc": (float) Average value of cross-correlation histogram.
##     *"Ze": (float) (maxcc-avecc)/sqrt(avecc)
##     *"Zi": (float) (mincc-avecc)/sqrt(avecc)
##     *"alphae": (float) Probability that
##                          a standard normal random variable exceeds Ze.
##     *"alphai": (float) Probability that
##                          a standard normal random variable is below Zi.
##     *"upcc": Upper cross-correlation value.
##                          when "alphae" equals the "threshold".
##     *"lowcc": Lower cross-correlation value.
##                          when "alphai" equals the "threshold".
##     *"upcc": Value of cross-correlation histogram.
##                   when the corresponding "alpha" equals the "threshold".
##     *"ext":   (int) Indicator of putative excitatory connection.
##     *"inh":   (int) Indicator of putative inhibitory connection.
##                                 (1 if significant, 0 otherwise)
##    }
##     Rows": N(N-1), auto-indexed. (N: Total number of neurons.)

def classicalCC(dfcck,threshold):

    ## Extract data for GLMCC estimation -------------------
    w = WINHALF_MS["C"]
    w0 = DELTA_MS["K"]
    dfk = dfcck.T[:-2].query("@w0<=index<=@w").T.reset_index(drop=True)
    
    dfcl = dfcck[["ref","tar"]]
    dfcl = dfcl.copy()   ##  avoid SettingWithCopyWarning
    dfcl["maxcc"] = dfk.max(axis=1)
    dfcl["mincc"] = dfk.min(axis=1)
    dfcl["avecc"] = dfk.mean(axis=1)
    srsqrt = np.sqrt(dfcl["avecc"])
    dfcl["Ze"] = (dfcl["maxcc"]-dfcl["avecc"])/srsqrt
    dfcl["Zi"] = (dfcl["mincc"]-dfcl["avecc"])/srsqrt
    dfcl["alphae"] = 2*norm.sf(dfcl["Ze"])
    dfcl["alphai"] = 2*norm.sf(-dfcl["Zi"])
    dfcl["upcc"] = dfcl["avecc"]+srsqrt*norm.ppf(1-threshold)
    dfcl["lowcc"] = dfcl["avecc"]-srsqrt*norm.ppf(1-threshold)
    dfcl.loc[dfcl["lowcc"]<0,"lowcc"] = 0
    dfcl["ext"] =(dfcl["alphae"]<=dfcl["alphai"])*(dfcl["alphae"]<threshold)*1
    dfcl["inh"] =(dfcl["alphai"]<dfcl["alphae"])*(dfcl["alphai"]<threshold)*1

    return dfcl



## Grouping() -----------------------------------------------
## Group by the presence or absence of connections estimated
##   by the three estimation methods.
##  [input]
##   +dfbsc": (Pandas DataFrame) Estimated by "Classical"
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"maxcc": (int) Maximum value of cross-correlation histogram.
##     *"mincc": (int) Minimum value of cross-correlation histogram.
##     *"avecc": (float) Average value of cross-correlation histogram.
##     *"Ze": (float) (maxcc-avecc)/sqrt(avecc)
##     *"Zi": (float) (mincc-avecc)/sqrt(avecc)
##     *"alphae": (float) P-value for the significance of
##                            maximum cross-correlation Z-score.
##     *"alphai": (float) P-value for the significance of
##                            minimum cross-correlation Z-score.
##     *"upcc": Upper cross-correlation value.
##                          when "alphae" equals the "threshold".
##     *"lowcc": Lower cross-correlation value.
##                          when "alphai" equals the "threshold".
##     *"upcc": Value of cross-correlation histogram.
##                   when the corresponding "alpha" equals the "threshold".
##     *"ext":   (int) Putative excitatory connection indicator.
##     *"inh":   (int) Putative inhibitory connection indicator.
##                                 (1 if significant, 0 otherwise)
##    }
##     Rows: N(N-1), auto-indexed. (N: Total number of neurons.)
##   +"dfbsg": (Pandas DataFrame) Estimated by "GLM"
##   +"dfbss": (Pandas DataFrame) Estimated by "ShinGLM"
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"J":     (float) Synaptic strength of right-half cross-correlation (Jp)
##     *"delay": (int) Delay parameter of synaptic function [ms]
##     *"tau":   (int) Decay time constant of synaptic function [ms]
##     *"alpha": (float) P-value for the significance of
##                                      cross-correlation Z-score.
##     *"ext":   (int) Indicator of putative excitatory connection.
##     *"inh":   (int) Indicator of putative inhibitory connection.
##    }
##     Rows: N(N-1), auto-indexed. (N: Total number of neurons.)
##  [output]
##   +dfgrp": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"group": (str) Connection codes
##         ("N" for none, "C" for Classical, "G" for GLMCC, "S" for ShinGLMCC,
##          e.g., "CG" indicates connections estimated by Clasical & GLM only.)
##     *"Classical, GLM, Shin: (int) Connectivity status by respective methods
##                                   (1 for detected, 0 for not detected)
##     *"ext-[c/g/s]", "inh-[c/g/s]":
##                (int) Putative excitatory ("ext") or
##                          inhibitory ("inh") connection indicator per method.
##                                   ([c] Classical, [c] GLMCC, [s] ShinGLMCC)
##                                   (1 if significant, 0 otherwise)
##     }
##      Rows: N(N-1), auto-indexed. (N: Total number of neurons.)

def Grouping(dfbsc,dfbsg,dfbss):

    dfgr = dfbsc[["ref","tar"]].copy()
    dfgr["Classical"] = dfbsc["ext"]+dfbsc["inh"]
    dfgr["GLM"] = dfbsg["ext"]+dfbsg["inh"]
    dfgr["Shin"] = dfbss["ext"]+dfbss["inh"]
    dfgr["groupN"] = dfgr["Classical"]+2*dfgr["GLM"]+4*dfgr["Shin"]
    dicgrp = {0:"N", 1:"C", 2:"G", 3:"CG", 4:"S", 5:"CS", 6:"GS", 7:"CGS"}
    dfgr["group"] = dfgr["groupN"].map(dicgrp)

    dfbsc = dfbsc[["ext","inh"]].rename(columns={"ext":"ext-c","inh":"inh-c"})
    dfbsg = dfbsg[["ext","inh"]].rename(columns={"ext":"ext-g","inh":"inh-g"})
    dfbss = dfbss[["ext","inh"]].rename(columns={"ext":"ext-s","inh":"inh-s"})

    dfgrp = pd.concat([dfgr,dfbsc,dfbsg,dfbss],axis=1)
    colg = dfgrp.pop("group")
    dfgrp.insert(2, colg.name, colg)
    del dfgrp["groupN"]

    return dfgrp


## EIdominance() --------------------------------------------
## Compute the excitatory inhibitory dominance.
##  [input]
##   +"dataid": (str) Data name where the original npy format files stored.
##   +"dfgrp": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"group": (str) Connection codes
##          ("N"= none, "C"= Classical, "G"= GLMCC, "S"= ShinGLMCC,
##            e.g., "CG" indicates connections estimated
##                                             by Clasical & GLM only.)
##     *"Classical", "GLM", "Shin": (int)
##          Connectivity status by respective methods
##                                 (1 for detected, 0 for not detected)
##     *"ext-[c/g/s]", "inh-[c/g/s]": (int)
##          Putative excitatory ("ext") or inhibitory ("inh")
##           connection indicator per method.
##            ([c] Classical, [c] GLMCC, [s] ShinGLMCC)
##            (1 if significant, 0 otherwise)
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##           (N= Total number of neurons.)
##  [output]
##   +"dfeid": (Pandas DataFrame)
##    {
##     *"average absolute values": (float)
##           Average absolute value of "Excitatory Inhibitory dominance".
##     *"expressing perfect consistency": (float)
##           Proportion of "Excitatory Inhibitory dominance"
##             equaling exactly -1 or +1.
##    }
##     Rows: "Classical","GLM","Shin"
##   +"dfgpn": (Pandas DataFrame)
##    {
##     *"num": (int)
##           Number of neurons assigned to each "Connection code".
##    }
##     Rows: Connection codes ("C","CG","CGS","CS","G","GS","N","S")
##   +"dfei[c/g/s]": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"c": Number of putative connection.
##     *"e": Number of putative excitatory connection.
##     *"i": Number of putative inhibitory connection.
##     *"dom": Excitatory Inhibitory dominance.
##               (dom= (e+i)/(e+i) )
##    }
##     Rows: Number of neurons with non-zero putative connections.

def EIdominance(dataid,dfgrp):

    grcls = dfgrp.groupby("ref")
    dfcls = grcls.sum(numeric_only=True).reset_index()

    dfeic = dfcls[["Classical","ext-c","inh-c"]]
    dfeic.columns = ["c","e","i"]
    dfeig = dfcls[["GLM","ext-g","inh-g"]]
    dfeig.columns = ["c","e","i"]
    dfeis = dfcls[["Shin","ext-s","inh-s"]]
    dfeis.columns = ["c","e","i"]

    aavc, pfcc, dfeic = _eacheidominance(dfeic)
    aavg, pfcg, dfeig = _eacheidominance(dfeig)
    aavs, pfcs, dfeis = _eacheidominance(dfeis)
    
    dfeid = pd.DataFrame([[aavc,pfcc],[aavg,pfcg],[aavs,pfcs]],
                         columns=["average absolute values",
                                  "expressing perfect consistency"],
                         index=["Classical","GLM","Shin"])
    dfgpn = pd.DataFrame(dfgrp.groupby("group").size(),columns=["num"])

    return dfeid,dfgpn,dfeic,dfeig,dfeis

