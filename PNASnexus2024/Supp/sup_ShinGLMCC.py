"""
##############################################################################
sup_additional.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2024.5.1
##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import os
import sys
import pickle
import itertools


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import ShinGLMCC.ShinGLMCC2


## ===================================================================
## Parameters
## ===================================================================

## DATAID: Name of target dataset
DATAID = sys.argv[1]

## DELAYLIST_MS: The list of delay parameter of synaptic function. [ms]
## TAULIST_MS:   The list of decay time constant of synaptic function. [ms]
DELAYLIST_MS = [1,2,3]
TAULIST_MS = [2,3,4]

SF2CYT={"phase3r":([19000000,20000000,21000000],["1.9","2.0","2.1"]),
       "frontalr":([490000,510000,530000],["4.9","5.1","5.3"]),
       "posteriorr":([2900000,3000000],["2.9","3.0"])}
SF2CYL={"phase3r":(19000000,21500000),
       "frontalr":(4800000,5400000),
       "posteriorr":(29000000,30500000)}

SF2CL={"phase3r":["000000","#FF00FF","#FF44FF","#FF88FF","#FFCCFF"],
       "frontalr":["000000","#00CED1","#44DEDA","#88EEE1","#CCFEEA"],
       "posteriorr":["000000","#FF69B4","#FF99C4","#FFB9D4","#FFF9E4"]}


## ===================================================================
## Functions
## ===================================================================


def npx2idxtbl(dataid):

    
    dfcln = pd.DataFrame(np.load(os.path.join(dataid,"spike_clusters.npy")))
    dftpn = pd.DataFrame(np.load(os.path.join(dataid,"spike_templates.npy")))
    nptpf = np.load(os.path.join(dataid,"templates.npy"))
    dfdp = pd.DataFrame(np.load(os.path.join(dataid,"channel_positions.npy")))
    dfgrp = pd.read_csv(os.path.join(dataid,"cluster_groups.csv"),sep="\t",skiprows=[0],names=["cls","group"])

    ## find the template number corresponding to each cluster
    ## dfcltpn: cluster ID and template ID for each spike
    dfcltpn = pd.concat([dfcln,dftpn],axis=1)
    dfcltpn.columns=["cls","tmp"]

    ## dfcltpn: the number of each cluster-template pair
    dfcltpc = dfcltpn.groupby("cls")["tmp"].apply(lambda x: x.value_counts()).sort_index(level=0).reset_index()
    dfcltpc.columns = ["cls","tmp","num"]    

    ## get the template shapes and find the peak channel
    dfpst = pd.DataFrame(np.argmax(np.max(nptpf,axis=1)-np.min(nptpf,axis=1),axis=1)).reset_index()
    dfpst.columns=["tmp","pos"]
    
    ## get the position of that channel
    dfdpy = dfdp.reset_index()
    dfdpy.columns=["pos","x","depth"]
    del dfdpy["x"]

    dfdept = pd.merge(dfpst,dfdpy,on="pos",how="left")
    dfdepc = pd.merge(dfcltpc,dfdept,on="tmp",how="left")
    dfdep = dfdepc.groupby("cls")["depth"].median().reset_index()
    dfclu = pd.DataFrame(dfgrp.query("group=='good'")["cls"]).reset_index(drop=True)
    dfref = pd.merge(dfclu,dfdep,on="cls",how="left").reset_index()
    dfref.columns = ["ref","cls","depth"]
    dfdpi = dfref.sort_values("depth").reset_index(drop=True).reset_index()
    dfdpi.columns = ["dpi","ref","cls","depth"]

    return dfdpi


def _C_matrix_plot(dfl,dfdpi,outname):
    
    num = round((1+np.sqrt(1+4*len(dfl)))*0.5)


    dfx = dfl["ref"]
    dfy = dfl["tar"]

    dfx = dfl["ref"].map(dfdpi.set_index("ref")["dpi"])
    dfy = dfl["tar"].map(dfdpi.set_index("ref")["dpi"])

    #dfx = dfl["ref"].map(dfdpi.set_index("ref")["depth"])
    #dfy = dfl["tar"].map(dfdpi.set_index("ref")["depth"])
    
    plt.gca().set_aspect("equal",adjustable="box")
    plt.scatter(dfx,dfy,s=dfl["CDIF"],
                marker=",",color="green",edgecolors="green",alpha=0.4)

    plt.xlim(0,num)
    plt.ylim(0,num)
    #plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xticks()
    plt.yticks()
    plt.xlabel("Pre (Reference)")
    plt.ylabel("Post (Target)")
    
    plt.savefig(outname)
    plt.clf()


def _allacrfit_fig(dfo,dataid):
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    plt.plot(dfo["t"],dfo["data"],color="black",lw=6)
    plt.plot(dfo["t"],dfo["Shin"],color="blue",lw=6)
    plt.plot(dfo["t"],dfo["Cont"],color="brown",lw=6)

    #plt.yticks(SF2CYT[dataid][0],SF2CYT[dataid][1])
    plt.ylim(SF2CYL[dataid][0],SF2CYL[dataid][1])
    plt.xticks([-50,0,50])
    plt.xlim(-50,50)

    plt.savefig("KH_"+dataid+"_FigS3_allacr.pdf")
    
if __name__ == "__main__":


    print("call ShinGLMCC.npx2dxtbl ...")
    dfdpi = npx2idxtbl(DATAID)
    print("done")

    """
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfdpi.to_csv(DATAID+"_cluster_depth.csv",index=None)
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    dfdpi = pd.read_csv(DATAID+"_cluster_depth.csv")
    ## wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfbsg = pd.read_csv(DATAID+"r_GLM_best.csv",
                        usecols=["ref","tar","delay","tau"])
    dfbss = pd.read_csv(DATAID+"r_Shin_best.csv",
                        usecols=["ref","tar","delay","tau"])
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    lsg = []
    lss = []
    for d,t in itertools.product(DELAYLIST_MS,TAULIST_MS):
        sname = f"{DATAID}r_Shin_par/{DATAID}r_Shin_lik_D{d}T{t}.csv"
        gname = f"{DATAID}r_GLM_par/{DATAID}r_GLM_lik_D{d}T{t}.csv"
        #print(sname)
        dfglt = pd.read_csv(gname,usecols=["ref","tar","delay","tau","loglike"])
        dfslt = pd.read_csv(sname,usecols=["ref","tar","delay","tau","loglike"])
        lsg += [dfglt]
        lss += [dfslt]

    dfg = pd.concat(lsg,axis=0).reset_index(drop=True)
    dfs = pd.concat(lss,axis=0).reset_index(drop=True)

    dfgl = pd.merge(dfbsg,dfg,on=["ref","tar","delay","tau"],how="left")
    dfsl = pd.merge(dfbss,dfs,on=["ref","tar","delay","tau"],how="left")

    dfl = dfsl.copy()[["ref","tar"]]
    dfl["CDIF"] =(dfsl["loglike"]-dfgl["loglike"]).apply(lambda x:1 if x>0 else 0)

    _C_matrix_plot(dfl,dfdpi,f"{DATAID}r_FigXX_CDI.pdf")
    

    """

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfJ = pd.read_pickle(DATAID+"r_allacr_J.pkl")
    dfK = pd.read_pickle(DATAID+"r_allacr_K.pkl")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr


    MODE = "Shin"
    print(MODE)
    paSS,dfk = ShinGLMCC.ShinGLMCC2.all2glm(MODE,dfJ,dfK,0)
    MODE = "Cont"
    print(MODE)
    paSC,dfk = ShinGLMCC.ShinGLMCC2.all2glm(MODE,dfJ,dfK,0)
    
    dfp = dfk.T.reset_index()
    
    dfo = pd.concat([dfp,pd.DataFrame(paSS),pd.DataFrame(paSC)],axis=1)
    dfo.columns=["t","data","Shin","Cont"]

    _allacrfit_fig(dfo,DATAID[3:]+"r")
