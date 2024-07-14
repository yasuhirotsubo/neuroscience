"""
##############################################################################
MATsimulationMCC.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.10.17

  Calculate the Matthews correlation coefficient (MCC)
    for a simulation data (with a known ground truth).

##############################################################################
"""


## ===================================================================
## External modules
## ===================================================================

import itertools

import numpy as np
import pandas as pd

## ===================================================================
## Parameters
## ===================================================================

## DATAID: Name of target dataset
## TRUTHID: Name of ground truth dataset
## A_LIST: List of parameter a (1.2 -> 12)
TRUTHID = "Data_SimMAT"
DATAID = "SM_MAT511_B15_A"
A_LIST = [6,8,10,12,14,16,18]

SYN_THRESHOLD = 0.01
NEURON = 1000
DOWN_SAMP = 10


## ===================================================================
## Functions
## ===================================================================

def TFPN(df,eori):
    TP = (df[(df["GT"+eori]==1) & (df["SY"+eori]==1)].shape[0])
    FN = (df[(df["GT"+eori]==1) & (df["SY"+eori]==0)].shape[0])
    FP = (df[(df["GT"+eori]==0) & (df["SY"+eori]==1)].shape[0])
    TN = (df[(df["GT"+eori]==0) & (df["SY"+eori]==0)].shape[0])
    P = TP + FN
    N = FP + TN
    FPR = FP/N
    FNR = FN/P
    MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TN+FN)*P*N)

    return [TP,FN,FP,TN,P,N,FPR,FNR,MCC]


def _GroundTruth():

    npe = np.loadtxt(TRUTHID+"_GE.csv", delimiter=",")
    npi = np.loadtxt(TRUTHID+"_GI.csv", delimiter=",")
    rowe,cole = np.where(npe[::DOWN_SAMP,::DOWN_SAMP]>SYN_THRESHOLD)
    rowi,coli = np.where(npi[::DOWN_SAMP,::DOWN_SAMP]>0)
    consete = set(zip(rowe, cole))
    conseti = set(zip(rowi, coli))
    itr = list(itertools.permutations(range(NEURON//DOWN_SAMP), 2))
    rese = [[i, j, 1 if (i, j) in consete else 0 ] for i, j in itr]
    resi = [[i, j, 1 if (i, j) in conseti else 0 ] for i, j in itr]
    dfe=pd.DataFrame(rese,columns=["ref","tar","GTe"])
    dfi=pd.DataFrame(resi,columns=["ref","tar","GTi"])
    df=pd.merge(dfe,dfi,on=["ref","tar"])
    df["ref"]=df["ref"]*DOWN_SAMP
    df["tar"]=df["tar"]*DOWN_SAMP
    
    return df
    
    
if __name__ == "__main__":

    dfgt = _GroundTruth()

    lsC = []
    lsG = []
    lsS = []
    
    for a in A_LIST:
        
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        dflik = pd.read_csv(DATAID+"%02d"%a+"sub_group.csv")
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        dfout=pd.merge(dflik,dfgt,on=["ref","tar"])

        dfC = (dfout[["GTe","GTi","ext-c","inh-c"]].rename(
            columns={"ext-c":"SYe","inh-c":"SYi"}))
        dfG = (dfout[["GTe","GTi","ext-g","inh-g"]].rename(
            columns={"ext-g":"SYe","inh-g":"SYi"}))
        dfS = (dfout[["GTe","GTi","ext-s","inh-s"]].rename(
            columns={"ext-s":"SYe","inh-s":"SYi"}))

        lsCe = TFPN(dfC,"e")
        lsCi = TFPN(dfC,"i")
        lsGe = TFPN(dfG,"e")
        lsGi = TFPN(dfG,"i")
        lsSe = TFPN(dfS,"e")
        lsSi = TFPN(dfS,"i")
    
        lsC += [[a]+lsCe+lsCi+[(lsCe[-1]+lsCi[-1])*0.5]]
        lsG += [[a]+lsGe+lsGi+[(lsGe[-1]+lsGi[-1])*0.5]]
        lsS += [[a]+lsSe+lsSi+[(lsSe[-1]+lsSi[-1])*0.5]]

    cole = ["TPe","FNe","FPe","TNe","Pe","Ne","FPRe","FNRe","MCCe"]
    coli = ["TPi","FNi","FPi","TNi","Pi","Ni","FPRi","FNRi","MCCi"]
    col = ["A"]+cole+coli+["MCC"]

    dfc = pd.DataFrame(lsC,columns=col)
    dfg = pd.DataFrame(lsG,columns=col)
    dfs = pd.DataFrame(lsS,columns=col)
    
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfc.to_csv(DATAID+"_Classical_MCC.csv",index=None)
    dfg.to_csv(DATAID+"_GLM_MCC.csv",index=None)
    dfs.to_csv(DATAID+"_Shin_MCC.csv",index=None)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

