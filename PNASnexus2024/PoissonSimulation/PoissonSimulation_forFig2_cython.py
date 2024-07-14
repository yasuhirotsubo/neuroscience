"""
#####################################################################
PoissonSimulation_forFig2_cython.py
by Yasuhiro Tsubo  modified ver. 2023.10.15
MacPro: 26 m 9 s
#####################################################################
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib

import matplotlib.pyplot as plt
import time

import cyPoisson_sim

## ===================================================================
## Parameters
## ===================================================================

## PROCESS_NUM: The maximum number of concurrently running jobs in joblib
##              -1: all CPUs are used
PROCESS_NUM = -1


## DATAID: Name of OUPsimulation dataset
DATAID = "PoissonSim"

## DELTA_S: symbolic parameters for time resolution [s]
DELTA_S = {"S":0.001,"P":0.001/30}
REC_S = 3600*100
HOUR_S = 3600

SEED = 20230307

## SIGMA : amplitude of (Gaussian noise/telegraphic noise/sinusoidal) modulation
## MU : mean of (Gaussian noise/telegraphic noise/sinusoidal) modulation
## TAU : time constant of OUP or updown shift [s]
SIGMA = 2
MU = 10
TAU = 1 /20

## NB : the number of waves in sinusoidal modulation
## OMEGA0 : the mean frequency of NB sinusoidal waves
## DOMEGA : the standard deviation of the frequencies of NB sinusoidal waves
NB = 10
OMEGA0 = 1 *20
DOMEGA = 0.1 *20

## SHIFT : the time difference in firing rates between Rate a and b [s]
## SHIFTN : the number of indices corresponding to SHIFT
SHIFT = 0.1
SHIFTN = int(SHIFT/DELTA_S["S"])

FIGID = ["a1","a2","b1","b2","c1","c2","d1","d2","e1","e2","f1","f2"]
RATEID = ["a","a","b","a","c","c","d","d","e","e","f","f"]
SEEDP = [1,2,1,3,1,2,1,2,1,2,1,2]

## ===========================================================================
## Functions
## ===========================================================================

## get_spiketime() --------------------------------------------
## Generate spike sequeinces by time-dependent rate functions

def get_spiketime(n,_rate,seed):
    T = int(HOUR_S/DELTA_S["P"])
    Rnd = np.random.rand(T).tolist()
    Stms = n*HOUR_S/DELTA_S["S"]
    npt = np.array(cyPoisson_sim.poisson(_rate,Rnd,DELTA_S["P"]))+Stms
    
    return [n, npt.tolist()]


def rate2spike(rate,seed):
    Rsp = np.array_split(rate,REC_S//HOUR_S)
    lsRsp = [ Rsp[n].tolist() for n in range(len(Rsp)) ]
    joblibR = joblib.Parallel(n_jobs=PROCESS_NUM)(
        [joblib.delayed(get_spiketime)(n,lsRsp[n],seed)
         for n in range(len(Rsp))])
    lstR = []
    for n in range(len(joblibR)):
        lstR += joblibR[n][1]

    return np.array(lstR)


if __name__ == '__main__':
    
    ## Generate Rate functions ================================
    ## a: smooth oscillation with Nb(=10) frequencies
    ## b: rate_a with a time lag
    ## c: Ornstein-Uhlenbeck process
    ## d: Markov switching process
    ## e: rate_a + rate_c
    ## f: rate_a + rate_d

    dtk = DELTA_S["S"]
    RT = int(REC_S/DELTA_S["S"])
    rt = np.linspace(0,REC_S,RT+1)[:-1]

    lsrate = []

    ## Generate Rate_a Rate_b ---------------------------------

    np.random.seed(SEED+1)
    Nb = NB
    omega = OMEGA0+DOMEGA*np.random.normal(0,1,Nb)
    Rb = SIGMA*np.sum(np.sin(np.tile(omega,
                (len(rt),1)).T*np.tile(rt,(Nb,1))),axis=0)/np.sqrt(Nb)+MU

    with open(os.path.join(DATAID+"_rate_b.pkl"),"wb") as f:    
        pickle.dump(Rb,f)

    Ra = np.append(Rb[SHIFTN:],Rb[:SHIFTN])

    with open(os.path.join(DATAID+"_rate_a.pkl"),"wb") as f:    
        pickle.dump(Ra,f)
    
    ## Generate Rate_d Rate_f ---------------------------------

    np.random.seed(SEED+2)
    Rnd = np.random.rand(RT).tolist()

    ## cyPoisson_sim.updowm()
    ##   [in] Rnd : numpy array of standard nornal random numbers
    ##                   (size: RT = REC_S/DELTA_S["S"]
    ##                               REC_S : recording time [s]
    ##                               DELTA_S : time bin width (0.001) [s])
    ##        TAU : time constant of updown shift
    ##        SIGMA : amplitude of telegraphic noise
    ##        MU : mean of telegraphic noise
    Rd = np.array(cyPoisson_sim.updown(Rnd,TAU,DELTA_S["S"],SIGMA,MU))

    with open(os.path.join(DATAID+"_rate_d.pkl"),"wb") as f:    
        pickle.dump(Rd,f)

    Rf = Rd+Ra-MU

    with open(os.path.join(DATAID+"_rate_f.pkl"),"wb") as f:    
        pickle.dump(Rf,f)

    ## Generate Rate_c Rate_e -----------------------------------------

    np.random.seed(SEED+3)
    Rnd = np.random.normal(0,1,RT).tolist()
    ## cyPoisson_sim.oup()
    ##   [in] Rnd : numpy array of standard nornal random numbers
    ##                   (size: RT = REC_S/DELTA_S["S"]
    ##                               REC_S : recording time [s]
    ##                               DELTA_S : time bin width (0.001) [s])
    ##        TAU : time constant of OUP
    ##        SIGMA : scale of Gaussian noise for OUP
    ##        MU : mean of Gaussian noise for OUP
    Rc = np.array(cyPoisson_sim.oup(Rnd,TAU,DELTA_S["S"],SIGMA,MU))
    
    with open(os.path.join(DATAID+"_rate_c.pkl"),"wb") as f:    
        pickle.dump(Rc,f)

    Re = Rc+Ra-MU

    with open(os.path.join(DATAID+"_rate_e.pkl"),"wb") as f:    
        pickle.dump(Re,f)


    ## Generate spike sequeinces by time-dependent rate functions
    ## a: smooth oscillation with Nb(10) frequencies
    ## b: rate_a with a time lag
    ## c: Ornstein-Uhlenbeck process
    ## d: Markov switching process
    ## e: rate_a + rate_c
    ## f: rate_a + rate_d

    dicspk = {}
    for fg in range(len(FIGID)):
        with open(os.path.join(DATAID+"_rate_"+RATEID[fg]+".pkl"),"rb") as f:    
            R = pickle.load(f)
        print(fg)
        npspk = rate2spike(R,SEED+SEEDP[fg])
        with open(os.path.join(DATAID+"_spike_"+FIGID[fg]+".pkl"),"wb") as f:
            pickle.dump(npspk,f)
        with open(os.path.join(DATAID+"_spike_"+FIGID[fg]+".pkl"),"rb") as f:               npspk = pickle.load(f)
        dicspk[FIGID[fg]] = npspk.tolist()
        
    with open(os.path.join(DATAID+"_dic.pkl"),"wb") as f:    
        pickle.dump(dicspk,f)
    
