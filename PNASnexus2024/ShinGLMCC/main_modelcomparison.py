"""
##############################################################################
main_modelcomparison.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.28
##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import os
import sys
import pickle

import pandas as pd

import ShinGLMCC.CJudge


## ===================================================================
## Parameters
## ===================================================================


## DATAID: Name of target dataset
DATAID = sys.argv[1]

## CONNECTION_THRESHOLD: The significance level
##             for determining the presence or absence of a connection
CONNECTION_THRESHOLD = 0.0001

## ===================================================================
## Functions
## ===================================================================

if __name__ == "__main__":

    #"""
    ## ---------------------------------------------------------
    ## Call CJudge.ClassicalCC()   (3 sec)
    ## Check for significant differences in histogram bin values.
    ## Calculate the p-value for ClassicalCC method.
    ##  [input]          
    ##   (DATAID)_cor_K.pkl: dfccK  cross-correlation with resolution 1ms
    ##  [output]
    ##   (DATAID)_Classical_best.csv: dfbsc
    ## .........................................................
    ##   +dfccK: (Pandas DataFrame)
    ##    {
    ##     * -WINHALF_MS to WINHALF_MS: (float)
    ##                   Correlation histogram values
    ##                   at Time lags -WINHALF_MS to WINHALF_MS. 
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##          (N= Total number of neurons.)
    ## .........................................................
    ##   +dfbsc: (Pandas DataFrame)
    ##    {
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##     *"maxcc": (int) Maximum value of cross-correlation histogram.
    ##     *"mincc": (int) Minimum value of cross-correlation histogram.
    ##     *"avecc": (float) Average value of cross-correlation histogram.
    ##     *"Ze": (float) (maxcc-avecc)/sqrt(avecc) Z-score.
    ##     *"Zi": (float) (mincc-avecc)/sqrt(avecc) Z-score.
    ##     *"alphae": (float) P-value for the presence of putative connection
    ##                                                by Ze (Excitatory).
    ##     *"alphai": (float) P-value for the presence of putative connection
    ##                                                by Zi (Inhibitory).
    ##     *"upcc": Upper cross-correlation value
    ##                              when "alphae"="CONNECTION_THRESHOLD"
    ##     *"lowcc": Lower cross-correlation value
    ##                              when "alphai"="CONNECTION_THRESHOLD"
    ##     *"ext":   (int) Putative excitatory connection indicator.
    ##     *"inh":   (int) Putative inhibitory connection indicator.
    ##                                 (1 if significant, 0 otherwise)
    ##    }
    ##     Rows: N(N-1), auto-indexed. (N: Total number of neurons.)

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_cor_K.pkl","rb") as f:    
        dfccK = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call CJudge.ClassicalCC ...")
    dfbsc = ShinGLMCC.CJudge.classicalCC(dfccK,CONNECTION_THRESHOLD)
    print("done")

    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfbsc.to_csv(DATAID+"_Classical_best.csv",index=None)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""


    #"""
    ## ---------------------------------------------------------
    ## Call CJudge.Grouping()   (2 sec)
    ## Group by the presence or absence of connections
    ##  estimated by the three estimation methods.
    ##  [input]          
    ##   (DATAID)_Classical_best.csv:  dfbsc
    ##   (DATAID)_GLM_best.csv:        dfbsg
    ##   (DATAID)_Shin_best.csv:       dfbss
    ##  [output]
    ##   (DATAID)_group.csv:           dfgrp 
    ## .........................................................
    ##   +dfbsc: (Pandas DataFrame)
    ##    {
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##     *"maxcc": (int) Maximum value of cross-correlation histogram.
    ##     *"mincc": (int) Minimum value of cross-correlation histogram.
    ##     *"avecc": (float) Average value of cross-correlation histogram.
    ##     *"Ze": (float) (maxcc-avecc)/sqrt(avecc) Z-score.
    ##     *"Zi": (float) (mincc-avecc)/sqrt(avecc) Z-score.
    ##     *"alphae": (float) P-value for the presence of putative connection
    ##                                                by Ze (Excitatory).
    ##     *"alphai": (float) P-value for the presence of putative connection
    ##                                                by Zi (Inhibitory).
    ##     *"upcc": Upper cross-correlation value
    ##                              when "alphae"="CONNECTION_THRESHOLD"
    ##     *"lowcc": Lower cross-correlation value
    ##                              when "alphai"="CONNECTION_THRESHOLD"
    ##     *"ext":   (int) Putative excitatory connection indicator.
    ##     *"inh":   (int) Putative inhibitory connection indicator.
    ##                                 (1 if significant, 0 otherwise)
    ##    }
    ##     Rows: N(N-1), auto-indexed. (N: Total number of neurons.)
    ## .........................................................
    ##   +dfbs[g/s]: (Pandas DataFrame)
    ##    {
    ##     * 0 to WINHALF_MS: (float) Optimised values of parameter a(t)
    ##                                  at Time lags 0 to WINHALF_MS.
    ##         !! Note that only the right-half data was extracted.
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##     *"J":   (float) Optimised values of parameter JR
    ##                      (Synapric weight for the right side.)
    ##     *"logpost": (float) Log-posterior across all ranges
    ##                                 with the estimated parameters.
    ##     *"loglike": (float) Log-likelihood for right-half side data.
    ##     *"delay": (int) Delay parameter of synaptic function. [ms]
    ##     *"tau":   (int) Decay time constant of synaptic function. [ms]
    ##     *"alpha": (float) P-value for the presence of putative connection.
    ##     *"ext":   (int) Putative excitatory connection indicator.
    ##     *"inh":   (int) Putative inhibitory connection indicator.
    ##                                 (1 if significant, 0 otherwise)
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##           (N= Total number of neurons.)
    ## .........................................................
    ##   +dfgrp: (Pandas DataFrame)
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

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfbsc = pd.read_csv(DATAID+"_Classical_best.csv")
    dfbsg = pd.read_csv(DATAID+"_GLM_best.csv")
    dfbss = pd.read_csv(DATAID+"_Shin_best.csv")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call CJudge.Grouping ...")
    dfgrp = ShinGLMCC.CJudge.Grouping(dfbsc,dfbsg,dfbss)
    print("done")

    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfgrp.to_csv(DATAID+"_group.csv",index=None)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""


    #"""
    ## ---------------------------------------------------------
    ## Call CJudge.EIdominance()
    ## Compute the excitatory Inhibitory dominance.
    ##  [input]          
    ##   (DATAID)_group.csv:           dfgrp
    ##  [output]
    ##   (DATAID)_EId_Classical.csv:   dfeic
    ##   (DATAID)_EId_GLM.csv:         dfeig
    ##   (DATAID)_EId_Shin.csv:        dfeis
    ##   (DATAID)_EId.csv:             dfeid
    ##   (DATAID)_grpnum.csv:          dfgpn
    ## .........................................................
    ##   +dfgrp: (Pandas DataFrame)
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
    ## .........................................................
    ##   +dfei[c/g/s]: (Pandas DataFrame)
    ##    {
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"c": Number of putative connection.
    ##     *"e": Number of putative excitatory connection.
    ##     *"i": Number of putative inhibitory connection.
    ##     *"dom": Excitatory Inhibitory dominance.
    ##               (dom= (e+i)/(e+i) )
    ##    }
    ##     Rows: Number of neurons with non-zero putative connections.
    ## .........................................................
    ##   +dfeid: (Pandas DataFrame)
    ##    {
    ##     *"average absolute values": (float)
    ##           Average absolute value of "Excitatory Inhibitory dominance".
    ##     *"expressing perfect consistency": (float)
    ##           Proportion of "Excitatory Inhibitory dominance"
    ##             equaling exactly -1 or +1.
    ##    }
    ##     Rows: "Classical","GLM","Shin"
    ## .........................................................
    ##   +dfgpn: (Pandas DataFrame)
    ##    {
    ##     *"num": (int)
    ##           Number of neurons assigned to each "Connection code".
    ##    }
    ##     Rows: Connection codes ("C","CG","CGS","CS","G","GS","N","S")

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfgrp = pd.read_csv(DATAID+"_group.csv")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call CJudge.EIdominance ...")
    dfeid,dfgpn,dfeic,dfeig,dfeis = ShinGLMCC.CJudge.EIdominance(DATAID,dfgrp)
    print("done")

    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfeic.to_csv(DATAID+"_EId_Classical.csv",index=None)
    dfeig.to_csv(DATAID+"_EId_GLM.csv",index=None)
    dfeis.to_csv(DATAID+"_EId_Shin.csv",index=None)
    dfeid.to_csv(DATAID+"_EId.csv")
    dfgpn.to_csv(DATAID+"_grpnum.csv")
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""
