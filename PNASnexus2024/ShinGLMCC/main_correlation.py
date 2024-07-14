"""
##############################################################################
main_correlation.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.28

  Run this program sequentially to generate correlation histograms.

  The PICKLE module allows for intermediate variable data
    to be written out and saved as is.
  If loaded, computation can resume from that point.
  However, USE the PICKLE files COMPUTED on the SAME MACHINE,
    because there might be compatibility issues
    when loading the data in different versions or environments.

  This program requires a certain number of CPU cores and memory.
  Insufficient resources may lead to slowdowns or errors in the execution.

  [Program Environment (during development)]
  * iMacPro2017 2.3GHz 18core Intel Xeon W / 128GB 2666MHz DDR4
  *  MacPro2019 2.5GHz 28core Intel Xeon W / 640GB 2933MHz DDR4
  * Test data set: KH_phase3

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import sys
import pickle

import ShinGLMCC.Correlation


## ===================================================================
## Parameters
## ===================================================================

## DATAID: Name of target dataset
DATAID = sys.argv[1]

## ===================================================================
## Functions
## ===================================================================

if __name__ == "__main__":

    """
    ## ---------------------------------------------------------
    ## Call Correlation.npx2dic()    (4 sec)
    ## Convert neuropixels data from original npy format to dict format.
    ##  [input]
    ##   (DATAID)/spike_clusters.npy
    ##   (DATAID)/spike_times.npy
    ##   (DATAID)/cluster_groups.csv
    ##  [output]
    ##   (DATAID)_dic.pkl:  dicspk    (key: Original Neuron ID)
    ##   (DATAID)r_dic.pkl: dicspkr   (key: Renubmered Neuron ID)
    ## .........................................................
    ##   +dicspk: (dict)
    ##    {
    ##     *key:   (int) Neuron ID. (Renubmered ID for dicspk"r")
    ##     *value: (list of float) Spike times of each neuron.
    ##    }
    
    print("call ShinGLMCC.npx2dic ...")
    dicspk, dicspkr = ShinGLMCC.Correlation.npx2dic(DATAID)
    print("done")
 
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_dic.pkl","wb") as f:
        pickle.dump(dicspk,f)
    with open(DATAID+"r_dic.pkl","wb") as f:
        pickle.dump(dicspkr,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    """

    #"""
    ## ---------------------------------------------------------
    ## Call Correlation.dic2cor()    (8 min 23 sec)
    ## Compute the correlation histogram from spike time series data.
    ##  [input]
    ##   (DATAID)_dic.pkl: dicspk
    ##  [output]
    ##   (DATAID)_acr_J.pkl: dfacJ  auto-correlation with resolution 0.1ms.
    ##   (DATAID)_cor_J.pkl: dfccJ  cross-correlation with resolution 0.1ms.
    ##   (DATAID)_acr_K.pkl: dfacK  auto-correlation with resolution 1ms.
    ##   (DATAID)_cor_K.pkl: dfccK  cross-correlation with resolution 1ms.
    ##    Note: For ShinGLM/GLM analysis, the following correlation histogram
    ##           with a time resolution (bin width) of N=10ms is not required.
    ##   (DATAID)_acr_N.pkl: dfacN  auto-correlation with resolution 10ms
    ##   (DATAID)_cor_N.pkl: dfccN  cross-correlation with resolution 10ms
    ## .........................................................
    ##   +df*c*: (Pandas DataFrame)
    ##    {
    ##     * -WINHALF_MS to WINHALF_MS: (float)
    ##                   Correlation histogram values
    ##                   at Time lags -WINHALF_MS to WINHALF_MS. 
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##            (N= Total number of neurons.)
    
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_dic.pkl","rb") as f:
        dicspk = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call ShinGLMCC.dic2cor (0.1ms bin) ...")
    dfacJ, dfccJ = ShinGLMCC.Correlation.dic2cor(dicspk,"J")
    print("done")
 
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_acr_J.pkl","wb") as f:    
        pickle.dump(dfacJ,f)
    with open(DATAID+"_cor_J.pkl","wb") as f:    
        pickle.dump(dfccJ,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    print("call ShinGLMCC.dic2cor (1ms bin) ...")
    dfacK, dfccK = ShinGLMCC.Correlation.dic2cor(dicspk,"K")
    print("done")
   
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_acr_K.pkl","wb") as f:    
        pickle.dump(dfacK,f)
    with open(DATAID+"_cor_K.pkl","wb") as f:    
        pickle.dump(dfccK,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    print("call ShinGLMCC.dic2cor (10ms bin) ...")
    dfacN, dfccN = ShinGLMCC.Correlation.dic2cor(dicspk,"N")
    print("done")
   
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_acr_N.pkl","wb") as f:    
        pickle.dump(dfacN,f)
    with open(DATAID+"_cor_N.pkl","wb") as f:    
        pickle.dump(dfccN,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    
    ## ---------------------------------------------------------
    #"""
