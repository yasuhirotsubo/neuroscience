"""
##############################################################################
dithering.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.17

  Apply time-direction dithering to the spike time sequences
   represented in dict format.

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import sys
import pickle

import ShinGLMCC.Allneuron
import ShinGLMCC.Correlation

## DIT_MS_LIST: List of dit(float)s,
##                Gaussian temporal dithering standard deviations. [ms]
DIT_MS_LIST = [5, 10, 50, 100]

## ===================================================================
## Parameters
## ===================================================================

## DATAID: Name of target dataset
DATAID = sys.argv[1]

## ===================================================================
## Functions
## ===================================================================

if __name__ == "__main__":

    #"""
    ## ---------------------------------------------------------
    ## Call Allneuron.dithering_spikeseq()
    ## Introduce temporal dithering to a spike time sequence
    ##              by applying Gaussian noise with SD "dit" [ms].
    ##   [input]
    ##     DATAID_dic.pkl : dicspk
    ##   [input]
    ##     DATAIDd(dit)_dic.pkl  : dicspkd

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_dic.pkl","rb") as f:
        dicspk = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    for dit in DIT_MS_LIST:

        dicspkd = ShinGLMCC.Allneuron.dithering_spikeseq(dicspk, dit)

        dataidd = DATAID+"d"+str(dit)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        with open(dataidd+"_dic.pkl","wb") as f:
            pickle.dump(dicspkd,f)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

        ## ---------------------------------------------------------
        ## Call Correlation.dic2cor()    (8 min 23 sec)
        ## Calculate the correlation histogram from spike time series data
        ##   [input]
        ##     (DATAID_dic.pkl: dicspk
        ##   [output]
        ##     (DATAID)_acr_J.pkl: dfacJ
        ##     (DATAID)_cor_J.pkl: dfccJ
        ##     (DATAID)_acr_K.pkl: dfacK
        ##     (DATAID)_cor_K.pkl: dfccK
        
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        with open(dataidd+"_dic.pkl","rb") as f:
            dicspk = pickle.load(f)
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
            
        dfacJ, dfccJ = ShinGLMCC.Correlation.dic2cor(dicspk,"J")
        
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        with open(dataidd+"_acr_J.pkl","wb") as f:    
            pickle.dump(dfacJ,f)
        with open(dataidd+"_cor_J.pkl","wb") as f:    
            pickle.dump(dfccJ,f)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

        dfacK, dfccK = ShinGLMCC.Correlation.dic2cor(dicspk,"K")
    
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        with open(dataidd+"_acr_K.pkl","wb") as f:    
            pickle.dump(dfacK,f)
        with open(dataidd+"_cor_K.pkl","wb") as f:    
            pickle.dump(dfccK,f)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

        #"""
        ## ---------------------------------------------------------
        ## Call Allneuron.cor2allacr()   ( < 1 sec )
        ##  Consolidate the correlation functions described for each neuron pair
        ##   by summing them all into a single correlation function.
        ##   described in dict format into a single list format.
        ##   [input]
        ##     dataidd_acr_J.pkl : dfacJ
        ##     dataidd_cor_J.pkl : dfccJ
        ##     dataidd_acr_K.pkl : dfacK
        ##     dataidd_cor_K.pkl : dfccK
        ##   [output]
        ##     dataidd_all.pkl  : lsspk
    
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        with open(dataidd+"_acr_J.pkl","rb") as f:    
            dfacJ = pickle.load(f)
        with open(dataidd+"_cor_J.pkl","rb") as f:    
            dfccJ = pickle.load(f)
        with open(dataidd+"_acr_K.pkl","rb") as f:    
            dfacK = pickle.load(f)
        with open(dataidd+"_cor_K.pkl","rb") as f:    
            dfccK = pickle.load(f)
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

        dfaacJ = ShinGLMCC.Allneuron.cor2allacr(dfacJ,dfccJ)
        dfaacK = ShinGLMCC.Allneuron.cor2allacr(dfacK,dfccK)
    
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        with open(dataidd+"_allacr_J.pkl","wb") as f:
            pickle.dump(dfaacJ,f)
        with open(dataidd+"_allacr_K.pkl","wb") as f:
            pickle.dump(dfaacK,f)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    #"""
