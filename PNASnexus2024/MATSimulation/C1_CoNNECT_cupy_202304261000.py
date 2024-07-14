"""
#####################################################################
C1_CoNNECT.py
by Yasuhiro Tsubo ver.2023.04.10
#####################################################################
"""

import os
import sys
import numpy as np
import cupy as cp
import pickle
import joblib
import random

import pandas as pd

import matplotlib.pyplot as plt
import time

from scipy.sparse import csr_matrix


## for GPGPU CUPY ###############################################
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)
#################################################################

## ===================================================================
## Parameters
## ===================================================================

## Set Model Constants
## --------------------------------------------------------

SEED = 20230307

NE  = 800     ## Number of excitatory neurons
NI  = 200     ## Number of inhibitory neurons

CE  = 100     ## Number of synaptic inputs received from excitatory neurons
CI  = 50      ## Number of synaptic inputs received from inhibitory neurons

VL  = -70.0   ## Leak equilibrium potential  [mV]
VL0 =  10.0   ## (Initial range of leak potential  [mV]) [Not in Endo2021]
VE  =   0.0   ## EPSP equilibrium potential  [mV]
VI  = -80.0   ## IPSP equilibrium potential  [mV]

DT  =   0.1   ## Simulation time step  [ms]
TME =  20.0   ## Membrane time constant for excitatory neurons  [ms]
TMI =  10.0   ## Membrane time constant for inhibitory neurons  [ms]
TSE =   1.0   ## Synaptic time constant for excitatory inputs   [ms]
TSI =   2.0   ## Synaptic time constant for inhibitory inputs   [ms]
TBE =   2.7   ## Background input time constant for excitatory neurons  [ms]
TBI =   10.5  ## Background input time constant for inhibitory neurons  [ms]
TQ1 =   10.0  ## Threshold time constant 1  [ms]
TQ2 =  200.0  ## Threshold time constant 2  [ms]

GEMU = -5.543   ## mu for log-normal
GESG =  1.3     ## sigma for log-normal
GEMX = 10.0     ## Maximum synaptic strength [Not in Endo2021, but in Teramae2012]
GIMU =  0.0217  ## mu for normal distribution
GISG =  0.00171 ## sigma for normal distribution

SYED =  [3,5]
SYID =  [2,4]

MAT_EA1 = [10,3]  ## MAT alpha1 for excitatory cells [Gauss(1.5,0.25) in Endo2021]
MAT_EA2 =    1.0  ## MAT alpha2 for excitatory cells [0.5 in Endo2021]
MAT_EO  =  -56    ## MAT omega for excitatory cells [-55 in Endo2021]
MAT_IA1 =   10.0  ## MAT alpha1 for inhibitory cells [3 in Endo2021]
MAT_IA2 =    0.2  ## MAT alpha2 for inhibitory cells [0 in Endo2021]
MAT_IO  =  -57    ## MAT omega for inhibitory cells

NBG   = [320,480,80,120]  ## Neuron distribution for background input
BGFQA  = 0.006    ## Average frequency of oscillatory component in background input [kHz]
BGFQS  = 0.003    ## Standard deviation of frequency for oscillatory component in background input [kHz]
BGFQN  = 10       ## Number of frequency samples for oscillatory component in background input
BGB0   = 0.03     ## Mean of oscillatory component in background input (Updated on 2023.4.24 from 0.015 to 0.03)
BGEMU  = 0.123    ## Mean excitatory value for background input
BGESG  = 0.0163   ## Standard deviation for excitatory value in background input
BGIMU  = 0.322    ## Mean inhibitory value for background input
BGISG  = 0.0265   ## Standard deviation for inhibitory value in background input



## Set Simulation Constants
## --------------------------------------------------------

TMAXSY = 10               ## Synaptic input buffer length  [ms]

SIMSTEP = int(60000/DT)   ## Number of steps per loop, the number inside represents  [ms]

## Sub Constants
## --------------------------------------------------------

N = NE+NI     ## Total number of neurons

## Time constants for Euler-Maruyama Method (numpy_array dim N)

np.random.seed(seed=SEED+10)
ET_M   = DT/np.hstack([(np.random.rand(NE)+1)*0.5*TME,
                       (np.random.rand(NI)+1)*0.5*TMI])    ## Membrane time constants

ET_SE  = DT/np.full(N,TSE)        ## Excitatory synaptic time constant
ET_SI  = DT/np.full(N,TSI)        ## Inhibitory synaptic time constant
ET_Q1  = DT/np.full(N,TQ1)        ## Threshold time constant 1
ET_Q2  = DT/np.full(N,TQ2)        ## Threshold time constant 2
ET_BE  = DT/np.full(N,TBE)        ## Excitatory background input time constant
ET_BI  = DT/np.full(N,TBI)        ## Inhibitory background input time constant

SBMAX = int(round(TMAXSY/DT))     ## Integer length of synaptic input buffer

## ===================================================================
## Functions
## ===================================================================

def make_connection(fname):

    ## ------------------------------------------------------------
    ## Synaptic strength list from pre to post neuron.
    ## Generate random values for connections twice as needed,
    ##  then remove those that don't meet the condition
    ## RND_LN np.ndarray (CE*N,) np.float64
    ## RND_NM np.ndarray (CI*N,) np.float64

    np.random.seed(seed=SEED)
    RND_LN = np.random.lognormal(mean=GEMU, sigma=GESG, size=(2*CE*N))
    RND_NM = np.random.normal(loc=GIMU, scale=GISG, size=(2*CI*N))
    RND_LN=RND_LN[RND_LN<GEMX][:CE*N]
    RND_NM=RND_NM[RND_NM>0][:CI*N]
    ## ------------------------------------------------------------

    ## ------------------------------------------------------------
    ## List of neuron IDs for pre and post
    ## E_PRE  np.ndarray (CE*N,) np.int64
    ## E_POST np.ndarray (CI*N,) np.int64
    ## I_PRE  np.ndarray (CE*N,) np.int64
    ## I_POST np.ndarray (CI*N,) np.int64
    random.seed(SEED+8)
    E_PRE = np.reshape([random.sample([m for m in range(NE) if m!=n ],CE)
                          for n in range(N)],(-1))
    E_POST = np.reshape([[n for m in range(CE)] for n in range(N)],(-1))
    I_PRE = np.reshape([random.sample([m for m in range(NE,N) if m!=n ],CI)
                          for n in range(N)],(-1))
    I_POST = np.reshape([[n for m in range(CI)] for n in range(N)],(-1))
    ## ------------------------------------------------------------

    ## ------------------------------------------------------------
    ## List of delays from pre to post  [int]
    ## ISYE  np.ndarray (CE*N,) np.int64
    ## ISYI  np.ndarray (CI*N,) np.int64
    np.random.seed(seed=SEED+1)
    TSYE = ((SYED[1]-SYED[0])*np.random.rand(CE*N)+SYED[0])
    TSYI = ((SYID[1]-SYID[0])*np.random.rand(CI*N)+SYID[0])
    ISYE = ((TSYE/DT).round()).astype(int)
    ISYI = ((TSYI/DT).round()).astype(int)
    ## ------------------------------------------------------------

    ## ------------------------------------------------------------
    ## Convert to the format of the connection delay matrix
    ## npGE[pre][delay][post]

    dfE = pd.DataFrame(np.column_stack([E_PRE,ISYE,E_POST,RND_LN])
                       ,columns=["pre","D","post","GE"]).astype({"pre":int})
    dfI = pd.DataFrame(np.column_stack([I_PRE,ISYI,I_POST,RND_NM])
                       ,columns=["pre","D","post","GI"]).astype({"pre":int})
    grE = dfE.groupby("pre")
    grI = dfI.groupby("pre")
    npGE = np.zeros((N,SBMAX,N))
    npGI = np.zeros((N,SBMAX,N))

    for j in grE.groups.keys():
        GE_D = grE.get_group(j)["D"].values.astype(int)
        GE_POST = grE.get_group(j)["post"].values.astype(int)
        GE_GE = grE.get_group(j)["GE"].values
        npGE[j] = csr_matrix((GE_GE, (GE_D, GE_POST)),
                             shape=(SBMAX, N)).toarray()
    for j in grI.groups.keys():
        GI_D = grI.get_group(j)["D"].values.astype(int)
        GI_POST = grI.get_group(j)["post"].values.astype(int)
        GI_GI = grI.get_group(j)["GI"].values
        npGI[j] = csr_matrix((GI_GI, (GI_D, GI_POST)),
                             shape=(SBMAX, N)).toarray()
    ## ------------------------------------------------------------

    ## ------------------------------------------------------------
    ## Save experimental data
    ## In the so-called connection matrix format (to be saved)
    ## Whether with print or np.savetxt,
    ##  pre is rows (vertical) and post is columns (horizontal)
    ## Save dfE and dfI as they are.
    """
    GE = csr_matrix((RND_LN, (E_PRE, E_POST)), shape=(N, N)).toarray()
    GI = csr_matrix((RND_NM, (I_PRE, I_POST)), shape=(N, N)).toarray()
    np.savetxt("GE_"+fname+".csv",GE,delimiter=",")
    np.savetxt("GI_"+fname+".csv",GI,delimiter=",")

    with open("dfE_"+fname+".pkl","wb") as f:
        pickle.dump(dfE,f)
    with open("dfI_"+fname+".pkl","wb") as f:
        pickle.dump(dfI,f)
    """
    ##----------------------------------------------------------------
    
    return npGE, npGI

## Set the MAT parameters in the form MA*[N]
##--------------------------------------------------------
def set_MAT():

    np.random.seed(seed=SEED+2)
    
    MA1 = np.hstack([np.random.normal(MAT_EA1[0],MAT_EA1[1],NE),
                     np.full(NI,MAT_IA1)])
    MA2 = np.hstack([np.full(NE,MAT_EA2),np.full(NI,MAT_IA2)])
    MO = np.hstack([np.full(NE,MAT_EO),np.full(NI,MAT_IO)])
    
    return MA1, MA2, MO


if __name__ == '__main__':

    time_p=time.time()
    
    A = float(sys.argv[1])
    ## 0.6 0.8 1.0 1.2,1.4,1.6,1.8,2
    B = 15
    fname=sys.argv[1]+"_OFF"

    npGE, npGI = make_connection(fname)

    MA1,MA2,MO = set_MAT()

    ## ======================================================================
    
    np.random.seed(seed=SEED+6)
    v  = np.random.random_sample(N)*VL0+VL

    q1 = np.zeros(N)
    q2 = np.zeros(N)
    ae = np.zeros(N)
    ai = np.zeros(N)
    be = np.zeros(N)
    bi = np.zeros(N)
    get = np.zeros(N)
    git = np.zeros(N)

    ge = np.zeros([SBMAX,N])
    gi = np.zeros([SBMAX,N])

    ## for GPGPU CUPY ###############################################
    cp.random.seed(seed=SEED+7)
    cET_BE=cp.array(ET_BE)
    cET_BI=cp.array(ET_BI)
    #################################################################

    if(os.path.isfile("raster"+fname+".csv")):
        os.remove("raster"+fname+".csv")

    with open("raster"+fname+".csv","a") as f:
        for mn in range(-1,60):

            ## for GPGPU CUPY ###############################################
            cwne=cp.random.normal(0,BGESG*cp.sqrt(2*cET_BE),(SIMSTEP,N))
            cwni=cp.random.normal(0,BGISG*cp.sqrt(2*cET_BI),(SIMSTEP,N))
            cwna=cp.random.normal(0,BGB0*cp.sqrt(DT),(SIMSTEP,N))
            bgfq = cp.random.normal(2*np.pi*BGFQA,2*np.pi*BGFQS,BGFQN)
            tls = cp.arange(mn*SIMSTEP*DT,(mn+1)*SIMSTEP*DT,DT)
            cbgsin = cp.sum(cp.sin(cp.outer(tls,bgfq)),axis=1)/BGFQN
            nxt = cp.hstack([cp.repeat(cbgsin[:,None],NBG[0],axis=1),
                             cp.zeros((SIMSTEP,NBG[1])),
                             cp.repeat(cbgsin[:,None],NBG[2],axis=1),
                             cp.zeros((SIMSTEP,NBG[3]))])        
            wne = cp.asnumpy(cwne)
            wni = cp.asnumpy(cwni)
            wna = cp.asnumpy(cwna*nxt)
            #################################################################
            
            lst=[]
            for t in range(SIMSTEP):
                
                RIbg = be*(v-VE)+bi*(v-VI)
                SYin = ae*(v-VE)+ai*(v-VI)
            
                v += -(v-VL)*ET_M-SYin*DT*A-RIbg*ET_M*B
                
                ae += -ae*ET_SE+ge[0]
                ai += -ai*ET_SI+gi[0]
            
                be += -(be-BGEMU)*ET_BE+wne[t] #+ wna[t]
                bi += -(bi-BGIMU)*ET_BI+wni[t]
            
                q1 += -q1*ET_Q1
                q2 += -q2*ET_Q2
                q   = q1+q2+MO
                
                q1 += ((v>=q)*MA1)
                q2 += ((v>=q)*MA2)
                
                spkls = np.where(v>=q)[0].tolist()
            
                for j in spkls:
                    if j<NE:
                        ge += npGE[j]
                    else:
                        gi += npGI[j]
                    lst+=[[j,(t+mn*SIMSTEP)*DT]]

                ge = np.vstack([ge[1:],np.zeros(N)])
                gi = np.vstack([gi[1:],np.zeros(N)])

            print("file:"+fname+"   loop:",mn,"min")
            if mn>=0:
                np.savetxt(f,np.array(lst),delimiter=",")
