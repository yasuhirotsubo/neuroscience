##############################################################################
## PoissonSimulation_main.py
## by Yasuhiro Tsubo     modified ver. 2023.10.09
## iMacPro: 8 m 27 s
##  [in]
##  [out]
##    DATAID_PoissonSim_rate_*.pkl            6 m
##############################################################################

python setup.py build_ext --inplace
python PoissonSimulation_forFig2_cython.py
