"""
======================================================================
ShinGLMCC_main.py
Author: Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
Version 1.0: 2024.07.13
Version 2.0: 2024.08.28
    * Changed the default setting to exclude neuron clusters
       with fewer than [NUMSPK] spikes from the analysis.
       default: NUMSPK=10
    * Modified the program to set J=0 and alpha=1 when there are bins
       with a frequency of zero in the 1ms-bin cross-correlation histogram (K).
    * Modified the program to execute the corresponding process
       by passing CC/Shin/GLM/ as an argument.
       (Separated the creation of the cross-correlation histogram
              and the connectivity estimation.)
       CC: creation of the cross-correlation histogram
       Shin: connectivity estimation by ShinGLMCC
       GLM: connectivity by GLMCC
    * in ShinGLMCC.Correlation.dic2cor():
      Modified to output autocorrelation with ref sorted in ascending order.

Paper:
Yasuhiro Tsubo and Shigeru Shinomoto,
Nondifferentiable activity in the brain,
PNAS nexus, Volume 3, Issue 7, July 2024, pgae261,
https://doi.org/10.1093/pnasnexus/pgae261

Overview:
This script processes large-scale neural activity data,
formatted as spike time series for each neuron (.dic format),
and estimates their functional connectivity using the ShinGLMCC/GLMCC model.
Due to the intensive CPU and memory requirements for large datasets,
a high-performance computing setup is recommended.

Development Environment:
  * iMacPro2017 - 2.3GHz 18-core Intel Xeon W / 128GB 2666MHz DDR4
  * MacPro2019 - 2.5GHz 28-core Intel Xeon W / 640GB 2933MHz DDR4

Usage: for {DATAID}_dic.pkl
$ python ShinGLMCC_main.py DATAID Shin/GLM
======================================================================
"""

import os
import sys
import pickle
import pandas as pd
import itertools

import ShinGLMCC.ShinGLMCC
import ShinGLMCC.Correlation

# Constants
NUMSPK = 10        # The minimum spike count for analyzing a neuron cluster
CONNECTION_THRESHOLD = 0.0001  # Significance level for connection presence
DELAYLIST_MS = [1, 2, 3]  # Delay parameters in milliseconds
TAULIST_MS = [2, 3, 4]    # Decay time constants in milliseconds

def save_parameters_and_likelihood(dfma, dfml, dataid, mode,suffix):
    pardirname = f"{dataid}_{mode}_par"
    os.makedirs(pardirname, exist_ok=True)
    parfilename = os.path.join(pardirname, f"{pardirname}_{suffix}.pkl")
    dfma.to_pickle(parfilename)
    likfilename = os.path.join(pardirname, f"{dataid}_{mode}_lik_{suffix}.csv")
    dfml.to_csv(likfilename)
    print(f"Parameters and likelihood saved with suffix {suffix}")

if __name__ == "__main__":
    
    DATAID = sys.argv[1]
    MODE = sys.argv[2]    # Estimation method: "Shin" or "GLM" or "CC"


    if MODE=="CC":
        # Load data dictionary from a pickle file
        with open(f"{DATAID}_dic.pkl", "rb") as file:
            dicspk = pickle.load(file)
            print(f"Data loaded from {DATAID}_dic.pkl")
        # Calculate and save correlations at different resolutions
        # resolutions: J=0.1ms, K=1ms
        for rsl in ["J", "K"]:
            dfac, dfcc = ShinGLMCC.Correlation.dic2cor(dicspk, rsl)
            dfac.to_pickle(f"{DATAID}_acr_{rsl}.pkl")
            dfcc.to_pickle(f"{DATAID}_cor_{rsl}.pkl")
            print(f"Correlation calculation and saving done for resolution {rsl}")
    else:
        # Evaluate ShinGLMCC/GLMCC (MODE="Shin"/"GLM")
        # Load previously calculated correlation data
        _dfccK = pd.read_pickle(f"{DATAID}_cor_K.pkl")
        _dfccJ = pd.read_pickle(f"{DATAID}_cor_J.pkl")
        print(f"Correlation data loaded for resolutions 0.1ms and 1ms")

        # pickup rows where these bins have a frequency of zero
        dfccK, dfccJ = ShinGLMCC.ShinGLMCC.find0ccrow(_dfccK,_dfccJ)

        # Evaluate ShinGLMCC/GLMCC with no connection and
        # save parameters and likelihood
        print(f"ShinGLMCC.cor2glm (no connection) is being called... MODE:{MODE}")
        npa0, dfma0, dfml0 = ShinGLMCC.ShinGLMCC.cor2glm(MODE,dfccJ,dfccK,0)
        save_parameters_and_likelihood(dfma0, dfml0, DATAID, MODE, "0")
        dfm0 = pd.merge(dfma0,dfml0,on=["ref","tar"]).drop(
            columns=["J","diff","s","cLM"])
        print("Done")

        # Evaluate ShinGLMCC with connection and save parameters and likelihood
        lst = []
        for d,t in itertools.product(DELAYLIST_MS,TAULIST_MS):
            print(f"ShinGLMCC.cor2glm tau={t} delay={d} is being called... MODE:{MODE}")
            npa, dfma, dfml = ShinGLMCC.ShinGLMCC.cor2glm(
                MODE,dfccJ,dfccK,2,d,t,npa0)
            print("Done")
            save_parameters_and_likelihood(dfma, dfml, DATAID, MODE, f"D{d}T{t}")
            dfmt = pd.merge(dfma,dfml,on=["ref","tar"]).drop(
                columns=["diff","s","cLM"])
            lst.append(dfmt)
        print("All combinations of delay and decay evaluations done")

        dfm = pd.concat(lst,axis=0,ignore_index=True)

        print("ShinGLMCC.ShinGLMCC.lik2best is being called...")
        dfbs = ShinGLMCC.ShinGLMCC.lik2best(dfm0,dfm,CONNECTION_THRESHOLD)
        print("Done")

        dfbs.to_csv(f"{DATAID}_{MODE}_best.csv",index=None)
        print(f"Best likelihood parameters saved to {DATAID}_{MODE}_best.csv")

