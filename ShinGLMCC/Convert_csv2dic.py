"""
======================================================================
Convert_csv2dic.py
Author: Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
Modified: 2024.07.15

Paper:
Yasuhiro Tsubo and Shigeru Shinomoto,
Nondifferentiable activity in the brain,
PNAS nexus, Volume 3, Issue 7, July 2024, pgae261,
https://doi.org/10.1093/pnasnexus/pgae261

Overview:
This script converts a csv file (DATAID.csv) into a dictionary and
save it as a pickle file named DATAID_dic.pkl

Development Environment:
  * iMacPro2017 - 2.3GHz 18-core Intel Xeon W / 128GB 2666MHz DDR4
  * MacPro2019 - 2.5GHz 28-core Intel Xeon W / 640GB 2933MHz DDR4

Usage: for {DATAID}_dic.pkl
$ python Convert_csv2dic.py DATAID
======================================================================
"""
import sys
import pickle
import pandas as pd

if __name__ == "__main__":

    DATAID = sys.argv[1]
    
    print("Converting csv data...")

    ## Open data
    dfspk = pd.read_csv(f"{DATAID}.csv", names=["cls", "time"])

    grall = dfspk.groupby("cls")
    dicspk = {int(cls):grp["time"].sort_values().tolist() 
              for cls, grp in grall}

    print("done")
 
    ## Save the dictionary as a pickle file
    with open(f"{DATAID}_dic.pkl", "wb") as f:
        pickle.dump(dicspk, f)
