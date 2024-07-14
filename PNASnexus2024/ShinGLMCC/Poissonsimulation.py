"""
##############################################################################
Poissonsimulation.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.17

  Generate spike time sequences from time-dependent Poisson processes
   in dict format, and compute the cross-correlation function for Figs 2 & S1.
  Original recording length: 100 hours
  for Fig 2: 10 hours
  for Fig S1: 1 hour, 10 hours, 100 hours

##############################################################################
"""

import pickle

import ShinGLMCC.Correlation


## ===================================================================
## Parameters
## ===================================================================

## DATAID: Name of simulation target dataset
DATAID = "PoissonSim"


## ===================================================================
## Functions
## ===================================================================


if __name__ == "__main__":


    #"""
    ## ---------------------------------------------------------
    ## Trim the original time sequence data in dict format
    ##  to the specified measurement duration of h [hour]
    ##   [input]
    ##     DATAID_dic.pkl     : dicspk
    ##   [input]
    ##     DATAID_dic_xh.pkl  : dicspkp

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_dic.pkl","rb") as f:
        dicspk = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    REC_H_LIST = [1,10,100]
    for h in REC_H_LIST:
        rec_ms = 3600*1000*h
        dicspkp = {cls:[spk for spk in spks if spk < rec_ms]
                   for cls, spks in dicspk.items()}
        
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        with open(DATAID+str(h)+"h_dic.pkl","wb") as f:
            pickle.dump(dicspkp,f)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------

    #"""

    #"""
    ## ---------------------------------------------------------
    ## Call Correlation.dic2cor()
    ## Calculate the correlation histogram from spike time series data
    ##   [input]
    ##     (DATAID)_dic_(rec_h)h.pkl: dicspkp
    ##   [output]
    ##     (DATAID)_acr_(rec_h)h_K.pkl: dfacK
    ##     (DATAID)_cor_(rec_h)h_K.pkl: dfccK

    REC_H_LIST = [1,10,100]
    for h in REC_H_LIST:
        
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        with open(DATAID+str(h)+"h_dic.pkl","rb") as f:
            dicspkp = pickle.load(f)
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        
        dfacK, dfccK = ShinGLMCC.Correlation.dic2cor(dicspkp,"K")
        dfacJ, dfccJ = ShinGLMCC.Correlation.dic2cor(dicspkp,"J")

        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        with open(DATAID+str(h)+"h_acr_K.pkl","wb") as f:
            pickle.dump(dfacK,f)
        with open(DATAID+str(h)+"h_cor_K.pkl","wb") as f:
            pickle.dump(dfccK,f)

        with open(DATAID+str(h)+"h_acr_J.pkl","wb") as f:
            pickle.dump(dfacJ,f)
        with open(DATAID+str(h)+"h_cor_J.pkl","wb") as f:
            pickle.dump(dfccJ,f)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""
