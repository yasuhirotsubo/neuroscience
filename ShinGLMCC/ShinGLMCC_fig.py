"""
======================================================================
ShinGLMCC_fig.py
Author: Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
Version 1.0: 2024.07.13
Version 2.0: 2024.08.28
   * Adjusted the script's argument format to match that of ShinGLMCC_main.py.
   * Modified the program to avoid the "SettingWithCopyWarning".

Paper:
Yasuhiro Tsubo and Shigeru Shinomoto,
Nondifferentiable activity in the brain,
PNAS nexus, Volume 3, Issue 7, July 2024, pgae261,
https://doi.org/10.1093/pnasnexus/pgae261

$ python ShinGLMCC_fig.py DATAID Shin/GLM
======================================================================
"""

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":

    DATAID = sys.argv[1]
    MODE = sys.argv[2]
    csvname = f"{DATAID}_{MODE}_best.csv"
    figname = csvname.replace("best.csv","J.pdf")
    s = 20

    dfbs = pd.read_csv(csvname)
    dfe = dfbs[["ref","tar","J","ext"]]
    dfi = dfbs[["ref","tar","J","inh"]]

    num = round((1+np.sqrt(1+4*len(dfe)))*0.5)
    lst = [dfe["ref"][0]]+dfe["tar"][:num-1].tolist()
    idxdic = {ref:idx for idx,ref in enumerate(lst)}
    dfe.loc[:,"ref"]=dfe["ref"].map(idxdic)
    dfe.loc[:,"tar"]=dfe["tar"].map(idxdic)
    dfi.loc[:,"ref"]=dfi["ref"].map(idxdic)
    dfi.loc[:,"tar"]=dfi["tar"].map(idxdic)

    dfce = dfe.query("ext==1").copy()
    dfci = dfi.query("inh==1").copy()
    maxJ = dfce["J"].max()

    dfce.loc[:,"Se"] = (dfce["J"]*dfce["ext"])*s/maxJ
    dfci.loc[:,"Si"] = (-dfci["J"]*dfci["inh"])*s/maxJ
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
    
