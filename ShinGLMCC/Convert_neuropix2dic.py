"""
======================================================================
Convert_neuropix2dic.py
Author: Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
Version 1.0: 2024.07.13
Version 2.0: 2024.08.28
  * Changed to output only one neuronID's dictionary (pickle)
    based on the second argument.

Paper:
Yasuhiro Tsubo and Shigeru Shinomoto,
Nondifferentiable activity in the brain,
PNAS nexus, Volume 3, Issue 7, July 2024, pgae261,
https://doi.org/10.1093/pnasnexus/pgae261

Overview:
This script converts the neuropixel files (
spike_clusters.npy, spike_times.npy, and cluster_groups.csv)
stored in the DATAID folder into a dictionary and
save it as a pickle file named DATAID_dic.pkl

Development Environment:
  * iMacPro2017 - 2.3GHz 18-core Intel Xeon W / 128GB 2666MHz DDR4
  * MacPro2019 - 2.5GHz 28-core Intel Xeon W / 640GB 2933MHz DDR4

Usage: for {DATAID}_dic.pkl
$ python Convert_neuropix2dic.py DATAID [depth/original](optional)
     Only use clusters labeled as 'good' in cluster_groups.csv.
     neuronIDs (keys) are output as follows:
       "original": (cls) as listed in cluster_groups.csv.
       "depth": (dpi) sorted by cls in order of depth (deeper is smaller).
       default: (ref) reassigns cls to consecutive numbers starting from 0.

======================================================================
"""
import sys
import pickle
import os
import numpy as np
import pandas as pd

DELTA_MS = {"J":0.1,"K":1.0,"N":10.0,"L":1/30.0}
REC_MS = 3600*1000

if __name__ == "__main__":

    DATAID = sys.argv[1]
    MODE = sys.argv[2] if len(sys.argv) >2 else "consecutive"

    print(MODE)
    
    print("Converting neuropixels data...")

    ## Open data
    npcls = np.load(os.path.join(DATAID, "spike_clusters.npy")).reshape(-1, 1)
    nptime = np.load(os.path.join(DATAID, "spike_times.npy")) * DELTA_MS["L"]
    dfgrp = pd.read_csv(os.path.join(DATAID, "cluster_groups.csv"),
                        sep="\t", skiprows=[0], names=["cls", "group"])
    dfclu = dfgrp[dfgrp["group"]=="good"][["cls"]].reset_index(drop=True)
    
    dfspk = pd.DataFrame(np.concatenate([npcls, nptime], 1),
                         columns=["cls", "time"])  
    idxspnum = np.argmax(nptime >= REC_MS)
    dfspk = dfspk[:idxspnum]

    dfcls = pd.DataFrame(npcls,columns=["cls"])
    dftps = pd.DataFrame(np.load(os.path.join(DATAID,"spike_templates.npy")),columns=["tmp"])
    dfcts = pd.concat([dfcls,dftps],axis=1).groupby("cls")["tmp"].value_counts().rename("num").reset_index()

    nptpf = np.load(os.path.join(DATAID,"templates.npy"))
    dfpos = pd.DataFrame(np.argmax(np.max(nptpf,axis=1)-np.min(nptpf,axis=1),axis=1),
                         columns=["pos"]).reset_index().rename(columns={"index": "tmp"})
    dfdp = pd.DataFrame(np.load(os.path.join(DATAID,"channel_positions.npy")),
                        columns=["x", "depth"]).reset_index().astype(int).rename(columns={"index": "pos"})
    dftmd = pd.merge(dfpos,dfdp,on="pos",how="left")[["tmp","depth"]].sort_values("tmp")
    dfdep = pd.merge(dfcts,dftmd,on="tmp",how="left").groupby("cls")["depth"].median().reset_index()
    dfnum = dfcts.groupby("cls")["num"].sum().reset_index()
    dfdn = pd.merge(dfnum,dfdep,on="cls",how="left")
    
    dfref = pd.merge(dfclu,dfdn,on="cls",how="left").reset_index().rename(columns={"index": "ref"})
    dfdpi = dfref.sort_values("depth").reset_index(drop=True).reset_index().rename(
        columns={"index": "dpi", "cls": "cls", "depth": "depth"})
    
    dfdpi.to_csv(f"{DATAID}_indextable.csv",index=None)

    dicref = pd.Series(dfdpi["ref"].values, index=dfdpi["cls"]).to_dict()
    dicdpi = pd.Series(dfdpi["dpi"].values, index=dfdpi["cls"]).to_dict()
    

    grall = dfspk[dfspk["cls"].isin(dfdpi["cls"])].groupby("cls")

    dicspko = {int(cls):grp["time"].sort_values().tolist() 
               for cls, grp in grall}
 
    if MODE == "original":
        dicspk = dicspko
    elif MODE == "depth":
        dicspk = {dicdpi[cls]: times for cls, times in dicspko.items() if cls in dicref}
    else:
        dicspk = {dicref[cls]: times for cls, times in dicspko.items() if cls in dicref}

    ## Save the dictionary as a pickle file
    with open(f"{DATAID}_dic.pkl", "wb") as f:
        pickle.dump(dicspk, f)

    print("done")
