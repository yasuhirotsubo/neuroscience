"""
##############################################################################
neuralresponse.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.17

  Analyse data related to the neural response function,
   which is the time-direction histogram of spike time series.

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import sys
import pickle

import ShinGLMCC.Allneuron

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
    ## Call Allneuron.dic2all()    ( < 1 sec )
    ##  Consolidate spike time sequences for each neuron
    ##   described in dict format into a single list format.
    ##   [input]
    ##     DATAID_dic.pkl : dicspk
    ##   [input]
    ##     DATAID_all.pkl  : lsspk

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_dic.pkl","rb") as f:
        dicspk = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call Allneuron.dic2all ...")
    lsspk = ShinGLMCC.Allneuron.dic2all(dicspk)
    print("done")

    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_all.pkl","wb") as f:
        pickle.dump(lsspk,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    """

    """
    ## ---------------------------------------------------------
    ## Call Allneuron.cor2allacr()   ( < 1 sec )
    ##  Consolidate the correlation functions described for each neuron pair
    ##   by summing them all into a single correlation function.
    ##   described in dict format into a single list format.
    ##   [input]
    ##     DATAID_acr_J.pkl : dfacJ
    ##     DATAID_cor_J.pkl : dfccJ
    ##     DATAID_acr_K.pkl : dfacK
    ##     DATAID_cor_K.pkl : dfccK
    ##   [output]
    ##     DATAID_all.pkl  : lsspk
    
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_acr_J.pkl","rb") as f:    
        dfacJ = pickle.load(f)
    with open(DATAID+"_cor_J.pkl","rb") as f:    
        dfccJ = pickle.load(f)
    with open(DATAID+"_acr_K.pkl","rb") as f:    
        dfacK = pickle.load(f)
    with open(DATAID+"_cor_K.pkl","rb") as f:    
        dfccK = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call Allneuron.cor2allacr ...")
    dfaacJ = ShinGLMCC.Allneuron.cor2allacr(dfacJ,dfccJ)
    dfaacK = ShinGLMCC.Allneuron.cor2allacr(dfacK,dfccK)
    print("done")
   
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_allacr_J.pkl","wb") as f:
        pickle.dump(dfaacJ,f)
    with open(DATAID+"_allacr_K.pkl","wb") as f:
        pickle.dump(dfaacK,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    """

    #"""
    ## ---------------------------------------------------------
    ## Call Allneuron.dic2nrf()   ( 1 min 25 sec )
    ##  Compute the neural response function from the spike time sequences
    ##    described in dict format.
    ##   [input]
    ##     DATAID_dic.pkl : dicspk
    ##   [output]
    ##     DATAID_nrf_N.pkl : srnrfN   10ms bin
    ##     DATAID_nrf_K.pkl : srnrfK    1ms bin
    ##     Neural response function is represented
    ##      by the firing rate per unit time [Hz x (number of Neurons)]

    
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_dic.pkl","rb") as f:
        dicspk=pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call Allneuron.dic2nrf ...")
    srnrfN = ShinGLMCC.Allneuron.dic2nrf(dicspk,"N")
    srnrfK = ShinGLMCC.Allneuron.dic2nrf(dicspk,"K")
    print("done")
 
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_nrf_N.pkl","wb") as f:
        pickle.dump(srnrfN,f)
    with open(DATAID+"_nrf_K.pkl","wb") as f:
        pickle.dump(srnrfK,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""

    """
    ## ---------------------------------------------------------
    ## Call Allneuron.nrf2pow()   ( < 1 sec )
    ##  Compute the power spectrum of neural response function
    ##   [input]
    ##     DATAID_nrf_K.pkl : srnrf
    ##   [output]
    ##     DATAID_pow.pkl : srpow

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_nrf_K.pkl","rb") as f:
        srnrfk=pickle.load(f)
    with open(DATAID+"_nrf_N.pkl","rb") as f:
        srnrfn=pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call Allneuron.nrf2pow ...")
    srpowk = ShinGLMCC.Allneuron.nrf2pow(srnrfk,"K")
    srpown = ShinGLMCC.Allneuron.nrf2pow(srnrfn,"N")
    print("done")
   
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_pow_K.pkl","wb") as f:
        pickle.dump(srpowk,f)
    with open(DATAID+"_pow_N.pkl","wb") as f:
        pickle.dump(srpown,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    """
