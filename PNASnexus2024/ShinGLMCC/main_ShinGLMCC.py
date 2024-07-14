"""
##############################################################################
main_ShinGLMCC.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.28

  Run this program sequentially to perform the connection estimation.

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

import os
import sys
import pickle
import itertools

import pandas as pd

import ShinGLMCC.ShinGLMCC


## ===================================================================
## Parameters
## ===================================================================

## DATAID: Name of target dataset.
DATAID = sys.argv[1]

## MODE: The estimation method.
##       "Shin" for ShinGLMCC /"GLM" for GLMCC (NatCommn.2019)
MODE = sys.argv[2]

## CONNECTION_THRESHOLD: The significance level
##             for determining the presence or absence of a connection.
CONNECTION_THRESHOLD = 0.0001

## DELAYLIST_MS: The list of delay parameter of synaptic function. [ms]
## TAULIST_MS:   The list of decay time constant of synaptic function. [ms]
DELAYLIST_MS = [1,2,3]
TAULIST_MS = [2,3,4]


## ===================================================================
## Functions
## ===================================================================

if __name__ == "__main__":

    #"""
    ## ---------------------------------------------------------
    ## Call ShinGLMCC.cor2glm()
    ## Call a Levenberg-Marquard optimizer
    ##      to estimate synaptic connectivity between two neurons.
    ##                 (2 min 29 sec for "Shin" / 2 min 59 sec for "GLM")
    ##  [input]          
    ##   (DATAID)_cor_J.pkl: dfccJ
    ##   (DATAID)_cor_K.pkl: dfccK
    ##  [output]
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_par_0.pkl:    dfma0
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_lik_0.csv:    dfml0
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_par_DxTx.pkl: dfma
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_lik_DxTx.csv: dfml
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
    ## .........................................................
    ##   +dfma*: (Pandas DataFrame)
    ##    {
    ##     *-WINHALF_MS to WINHALF_MS: (float)
    ##                   Optimised values of parameter a(t)
    ##                   at Time lags -WINHALF_MS to WINHALF_MS. 
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##            (N= Total number of neurons.)
    ## .........................................................
    ##   +dfml*: (Pandas DataFrame)
    ##    {
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##     *"J":   (float) Optimised values of parameter JR
    ##                      (Synapric weight for the right side.)
    ##     *"logpost": (float) Log-posterior across all ranges
    ##                                 with the estimated parameters.
    ##     *"loglike": (float) Log-likelihood for right-half side data.
    ##     *"diff":  (float) Improvement in negative log-posterior
    ##                          at the last update step of the LM method.
    ##     *"s":     (int) number of iterations using the LM method
    ##                          at the last update step.
    ##     *"cLM":   (int) A parameter of LM method at the last update step.
    ##     -- The following two columns are not included in dfma0. --
    ##     *"delay": (int) Delay parameter of synaptic function. [ms]
    ##     *"tau":   (int) Decay time constant of synaptic function. [ms]
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##            (N= Total number of neurons.)

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_cor_J.pkl","rb") as f:    
        dfccJ = pickle.load(f)
    with open(DATAID+"_cor_K.pkl","rb") as f:    
        dfccK = pickle.load(f)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call ShinGLMCC.cor2glm synapseless ...")
    npa0, dfma0, dfml0 = ShinGLMCC.ShinGLMCC.cor2glm(MODE,dfccJ,dfccK,0)
    print("done")

    ## npa0: (2D numpy array)
    ##       Each row contains the parameters [a(t),J]
    ##        for only the neuron pairs "ref ID < tar ID"
    ##        estimated by the "synapseless" model
    ##       npa0 is used as the initial values
    ##        for the following models with synapses.
    
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dirname = DATAID+"_"+MODE+"_par"
    os.makedirs(dirname, exist_ok=True)
    parfilename = os.path.join(dirname,dirname)
    with open(parfilename+"_0.pkl","wb") as f:    
        pickle.dump(dfma0,f)
    dfml0.to_csv(parfilename.replace("_par_","_lik_")+"_0.csv",index=None)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    for d,t in itertools.product(DELAYLIST_MS,TAULIST_MS):

        print("call ShinGLMCC.cor2glm tau="+str(t)+" delay="+str(d)+" ...")
        npa, dfma, dfml = ShinGLMCC.ShinGLMCC.cor2glm(
            MODE,dfccJ,dfccK,2,d,t,npa0)
        print("done")

        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        dtfilename = parfilename+"_D"+str(d)+"T"+str(t)
        with open(dtfilename+".pkl","wb") as f:    
            pickle.dump(dfma,f)
        dfml.to_csv(dtfilename.replace("_par_","_lik_")+".csv",index=None)
        ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""

    #"""
    ## ---------------------------------------------------------
    ## Call ShinGLMCC.lik2best()
    ## Find the best parameter set and calculate the p-values for two methods.
    ##                 (11 sec for "Shin" / 10 sec for "GLM")
    ##  [input]          
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_par_0.pkl:    dfma0
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_lik_0.csv:    dfml0
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_par_DxTx.pkl: dfma
    ##   (DATAID)_(MODE)_par/(DATAID)_(MODE)_lik_DxTx.csv: dfml
    ##  [output]
    ##   (DATAID)_(MODE)_best.csv: dfbs
    ## .........................................................
    ##   +dfma*: (Pandas DataFrame)
    ##    {
    ##     * -WINHALF_MS to WINHALF_MS: (float)
    ##                   Optimised values of parameter a(t)
    ##                   at Time lags -WINHALF_MS to WINHALF_MS. 
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##            (N= Total number of neurons.)
    ## .........................................................
    ##   +dfml*: (Pandas DataFrame)
    ##    {
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##     * "J":   (float) Optimised values of parameter JR
    ##                      (Synapric weight for the right side.)
    ##     *"logpost": (float) Log-posterior across all ranges
    ##                                 with the estimated parameters.
    ##     *"loglike": (float) Log-likelihood for right-half side data.
    ##     *"diff":  (float) Improvement in negative log-posterior
    ##                          at the last update step of the LM method.
    ##     *"s":     (int) number of iterations using the LM method
    ##                          at the last update step.
    ##     *"cLM":   (int) A parameter of LM method at the last update step.
    ##     -- The following two columns are not included in dfma0. --
    ##     *"delay": (int) Delay parameter of synaptic function. [ms]
    ##     *"tau":   (int) Decay time constant of synaptic function. [ms]
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##            (N= Total number of neurons.)
    ## .........................................................
    ##   +dfbs*: (Pandas DataFrame)
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

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dirname = DATAID+"_"+MODE+"_par"
    parfilename = os.path.join(dirname,dirname)
    with open(parfilename+"_0.pkl","rb") as f:    
        dfa0 = pickle.load(f)
    dfl0 = pd.read_csv(parfilename.replace("_par_","_lik_")+"_0.csv")
    dfm0 = pd.merge(dfa0,dfl0,on=["ref","tar"])
    dfm0.drop(columns=["J","diff","s","cLM"],inplace=True)
 
    lst = []
    for d,t in itertools.product(DELAYLIST_MS,TAULIST_MS):
        dtfilename = parfilename+"_D"+str(d)+"T"+str(t)
        with open(dtfilename+".pkl","rb") as f:
            dfa = pickle.load(f)
        dfl = pd.read_csv(dtfilename.replace("_par_","_lik_")+".csv")
        dfla = pd.merge(dfa,dfl,on=["ref","tar"])
        lst.append(dfla)
    dfm = pd.concat(lst,axis=0,ignore_index=True)
    dfm.drop(columns=["diff","s","cLM"],inplace=True)
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call ShinGLMCC.ShinGLMCC.lik2best ...")
    dfbs = ShinGLMCC.ShinGLMCC.lik2best(dfm0,dfm,CONNECTION_THRESHOLD)
    print("done")
    
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfbs.to_csv(DATAID+"_"+MODE+"_best.csv",index=None)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## ---------------------------------------------------------
    #"""

    #"""
    ## ---------------------------------------------------------
    ## Call ShinGLMCC.figdata()
    ## Generate data for plotting from the estimated model parameters.
    ##                 (37 sec for "Shin" / 39 sec for "GLM")
    ##  [input]
    ##   (DATAID)_(MODE)_best.csv: dfbs
    ##  [output]
    ##   (DATAID)_(MODE)_fig.pkl:  dffig
    ## .........................................................
    ##   +dfbs*: (Pandas DataFrame)
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
    ##                                 (1 if significant, 0 otherwise.)
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##           (N= Total number of neurons.)
    ## .........................................................
    ##   +dffig: (Pandas DataFrame)
    ##    {
    ##     * -WINHALF_MS to WINHALF_MS: (float)
    ##                 Estimated cross-correlation function
    ##                  by using the optimized parameters
    ##                  at Time lags -WINHALF_MS to WINHALF_MS.
    ##     *"ref": (int) Reference Neuron ID.
    ##     *"tar": (int) Target Neuron ID.
    ##    }
    ##     Rows: N(N-1), all neuron pairs, auto-indexed.
    ##           (N= Total number of neurons.)

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    with open(DATAID+"_cor_K.pkl","rb") as f:    
        dfccK = pickle.load(f)
    with open(DATAID+"_cor_J.pkl","rb") as f:    
        dfccJ = pickle.load(f)
    dirname = DATAID+"_"+MODE+"_par"
    parfilename = os.path.join(dirname,dirname)
    with open(parfilename+"_0.pkl","rb") as f:    
        dfp0 = pickle.load(f)
    dfbs = pd.read_csv(DATAID+"_"+MODE+"_best.csv")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    print("call ShinGLMCC.figdata ...")
    dffig = ShinGLMCC.ShinGLMCC.figdata(dfbs,dfp0)
    print("done")

    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    with open(DATAID+"_"+MODE+"_fig.pkl","wb") as f:    
        pickle.dump(dffig,f)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    
    #"""

