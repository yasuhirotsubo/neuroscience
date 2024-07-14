"""
======================================================================
ShinGLMCC_fig.py
Author: Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
Modified: 2024.07.13

Paper:
Yasuhiro Tsubo and Shigeru Shinomoto,
Nondifferentiable activity in the brain,
PNAS nexus, Volume 3, Issue 7, July 2024, pgae261,
https://doi.org/10.1093/pnasnexus/pgae261

$ python ShinGLMCC_fig.py DATAID_Shin/GLM_best.csv
======================================================================
"""

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    csvname = sys.argv[1]
    figname = csvname.replace("best.csv","J.pdf")
    s = 20

    dfbs = pd.read_csv(csvname)
    dfe = dfbs[["ref","tar","J","ext"]]
    dfi = dfbs[["ref","tar","J","inh"]]

    num = round((1+np.sqrt(1+4*len(dfe)))*0.5)
    lst = [dfe["ref"][0]]+dfe["tar"][:num-1].tolist()
    idxdic = {ref:idx for idx,ref in enumerate(lst)}
    dfe["ref"]=dfe["ref"].map(idxdic)
    dfe["tar"]=dfe["tar"].map(idxdic)
    dfi["ref"]=dfi["ref"].map(idxdic)
    dfi["tar"]=dfi["tar"].map(idxdic)

    dfce = dfe.query("ext==1")
    dfci = dfi.query("inh==1")
    maxJ = dfce["J"].max()
    dfce = dfce.copy()
    dfci = dfci.copy()
    dfce["Se"] = (dfce["J"]*dfce["ext"])*s/maxJ
    dfci["Si"] = (-dfci["J"]*dfci["inh"])*s/maxJ
    dfci.loc[dfci["Si"]>s,"Si"] = s

    plt.gca().set_aspect("equal",adjustable="box")
    plt.scatter(dfce["ref"],dfce["tar"],s=dfce["Se"],
                marker=",",color="magenta",edgecolors="magenta",alpha=0.4)
    plt.scatter(dfci["ref"],dfci["tar"],s=dfci["Si"],
                marker=",",color="aqua",edgecolors="aqua",alpha=0.4)

    plt.xlim(0,num)
    plt.ylim(0,num)
    plt.gca().invert_yaxis()
    plt.xticks()
    plt.yticks()
    plt.xlabel("Pre (Reference)")
    plt.ylabel("Post (Target)")
    
    plt.savefig(figname)
    plt.clf()
    
