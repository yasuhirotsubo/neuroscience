"""
##############################################################################
make_figures.py
by Yasuhiro Tsubo (tsubo@fc.ritsumei.ac.jp)
modified ver. 2023.11.6
##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import os
import sys
import pickle
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp




## ===================================================================
## Parameters
## ===================================================================

## PROCESS_NUM: The maximum number of concurrently running jobs in joblib
##              -1: all CPUs are used
PROCESS_NUM = -1

## DELTA_MS: Symbolic parameters representing time resolution. [ms]
## WINHALF_MS: Symbolic parameters representing half the time width
##               of the correlation function. [ms]
DELTA_MS = {"J":0.1,"K":1.0,"N":10.0,"L":1/30.0}
WINHALF_MS = {"J":120.0,"K":500.0,"N":500.0,"C":50.0,"F":25.0}




F3AXT={"phase3r":([190000,192000,194000,196000,198000,200000],
                  [190,192,194,196,198,200]),
       "frontalr":([420000,422000,424000,426000,428000,430000],
                  [420,422,424,426,428,430]),
       "posteriorr":([420000,422000,424000,426000,428000,430000],
                     [420,422,424,426,428,430]),
       "MAT511A":([190000,191000,192000],[190,191,192])}
F3AXL={"phase3r":(190000,200000),
       "frontalr":(420000,430000),
       "posteriorr":(420000,430000),
       "MAT511A":(190000,192000)}




F3BYT={"phase3r":([18000000,20000000,22000000],["1.8","2.0","2.2"]),
       "frontalr":([4800000,5200000,5600000],["4.8","5.2","5.6"]),
       "posteriorr":([29000000,30000000,31000000],["2.9","3.0","3.1"])}
F3BYL={"phase3r":(17000000,22500000),
       "frontalr":(4500000,5700000),
       "posteriorr":(28000000,31000000)}

F3CYT={"phase3r":([1000,10000,100000,1000000],["10^3","10^4","10^5","10^6"]),
       "frontalr":([1000,10000,100000,1000000],["10^3","10^4","10^5","10^6"]),
       "posteriorr":([1000,10000,100000,1000000],["10^3","10^4","10^5","10^6"])}
F3CYL={"phase3r":(1000,5000000),
       "frontalr":(100,500000),
       "posteriorr":(1000,5000000)}





#F4JFACT_EX = 0.39 # 1/mV
#F4JFACT_IN = 1.57 # 1/mV
F4JFACT_EX = 1 # 1/mV
F4JFACT_IN = 1 # 1/mV


SF1REFTAR={"OUP":["c1","c2"],"SIN":["a1","a2"]}
SF1MAG = {"1h":1,"10h":10,"100h":100}
SF1SIGMA = 2
SF1MU = 10
SF1OMEGA0 = 1 *20
SF1DOMEGA = 0.1 *20
SF1TAU = 1 /20



SF2CYT={"phase3r":([1900000,2000000,2100000],["1.9","2.0","2.1"]),
       "frontalr":([490000,510000,530000],["4.9","5.1","5.3"]),
       "posteriorr":([2900000,3000000],["2.9","3.0"])}
SF2CYL={"phase3r":(1900000,2150000),
       "frontalr":(480000,540000),
       "posteriorr":(2900000,3050000)}

SF2CL={"phase3r":["000000","#FF00FF","#FF44FF","#FF88FF","#FFCCFF"],
       "frontalr":["000000","#00CED1","#44DEDA","#88EEE1","#CCFEEA"],
       "posteriorr":["000000","#FF69B4","#FF99C4","#FFB9D4","#FFF9E4"]}


## ===================================================================
## Functions
## ===================================================================


## ---------------------------------------------------------
## for Figures 1,4,S4,S6
## ---------------------------------------------------------
def _figcc_plot(ccwidth,figdir,wF,cls,cKn,uplowcc,npgn,npsn):

    plt.rcParams["figure.subplot.bottom"] = 0.15
    plt.rcParams["figure.subplot.left"] = 0.15
    plt.rcParams["axes.linewidth"] = 2
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_linewidth(2)
    plt.gca().spines["left"].set_linewidth(2)
   
    if ccwidth == "F":
        ybox = [uplowcc[1],uplowcc[0],uplowcc[0],uplowcc[1]]
        plt.fill([-wF,-wF,wF,wF],ybox,color="lime",alpha=0.1)
        plt.hlines(uplowcc,-wF,wF,color="springgreen",lw=2)
        wK = int(wF/DELTA_MS["K"])
    else:
        wK = int(wF/DELTA_MS["N"])
        
    xK = np.linspace(-wF, wF, 2*wK+1)
    plt.step(xK[1:-1],cKn[1:-1],color="black",lw=2,where="mid")

    if ccwidth == "F":
        plt.step(xK[1:-1],cKn[1:-1],color="black",lw=2,where="mid")
        plt.title(str(cls[0])+" ==>> "+str(cls[1]))
        wJ = int(wF/DELTA_MS["J"])
        xJ = np.linspace(-wF, wF, 2*wJ+1)
        plt.plot(xJ,npgn,lw=4,color="orange")
        plt.plot(xJ,npsn,lw=4,color="blue")
        plt.xticks([-20,0,20],fontsize=24)
        plt.yticks([],fontsize=36)
    else:
        plt.step(xK[1:-1],cKn[1:-1],color="black",lw=5,where="mid")
        plt.xticks([-500,0,500],fontsize=48)
        plt.yticks([],fontsize=36)


    ymax = plt.ylim()[1]
    plt.vlines(0,0,ymax,color="gray",lw=2)
        
    plt.xlim(-wF,wF)
    plt.ylim(bottom=0)

    filename = os.path.join(figdir,figdir+"_"+
                    str(cls[0])+"_"+str(cls[1])+".pdf")
    plt.savefig(filename)
    plt.clf()



## ---------------------------------------------------------
## for Figures 3,S3,S5,5
## ---------------------------------------------------------

def _neuralresponse_plot(srnrf,dataid,outname):

    plt.figure(figsize=(15, 5))
    
    if dataid=="MAT511A":
        plt.plot(srnrf.index, srnrf, color="black",lw=3)
        plt.ylim(0,28000)
    else:
        plt.plot(srnrf.index, srnrf, color="black")
        plt.ylim(0,7000)
    plt.xticks(F3AXT[dataid][0],F3AXT[dataid][1])
    plt.xlim(F3AXL[dataid][0],F3AXL[dataid][1])
    plt.yticks([0,4000],[0,4000])

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(False)
    ax.spines['bottom'].set_linewidth(3)

    plt.savefig(outname)
    plt.close()

    
def _allautocorrelation_plot(dfacrK,dataid,outname):

    plt.plot(dfacrK.columns, dfacrK.iloc[0], color="black",lw=3)
    plt.yticks(F3BYT[dataid][0],F3BYT[dataid][1])
    plt.ylim(F3BYL[dataid][0],F3BYL[dataid][1])
    plt.xticks([-400,-200,0,200,400])
    plt.xlim(-400.0,400)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    plt.savefig(outname)
    plt.close()



def _pow_plot(srpow,dataid,outname):

    plt.loglog(srpow.index, srpow, color="black",lw=3)
    plt.yticks(F3CYT[dataid][0],F3CYT[dataid][1])
    plt.ylim(F3CYL[dataid][0],F3CYL[dataid][1])

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    plt.savefig(outname)
    plt.clf()


## ---------------------------------------------------------
## for Figures 4,S4,S6
## ---------------------------------------------------------

def _J_matrix_plot(dfe,dfi,s,outname):

    num = round((1+np.sqrt(1+4*len(dfe)))*0.5)

    dfce = dfe.query("ext==1")
    dfci = dfi.query("inh==1")
    maxJ = dfce["J"].max()
    dfce = dfce.copy()
    dfci = dfci.copy()
    dfce["Se"] = (dfce["J"]*dfce["ext"]/F4JFACT_EX)*s/maxJ
    dfci["Si"] = (-dfci["J"]*dfci["inh"]/F4JFACT_IN)*s/maxJ
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
    
    plt.savefig(outname)
    plt.clf()


## ---------------------------------------------------------
## for Figures 5
## ---------------------------------------------------------

def _ax_MCC_plot(figtype,dfc,dfg,dfs,s,n):
    
    if n==4:
        axt = plt.subplot(gs[n:,0])
    else:
        axt = plt.subplot(gs[n,0])
        
    axt.spines["right"].set_visible(False)
    axt.spines["top"].set_visible(False)
    
    axt.plot(dfc["A"],dfc[figtype],color="green")
    axt.plot(dfg["A"],dfg[figtype],color="orange")
    axt.plot(dfs["A"],dfs[figtype],color="blue")
    axt.scatter(dfc["A"],dfc[figtype],color="green",s=s)
    axt.scatter(dfg["A"],dfg[figtype],color="orange",s=s)
    axt.scatter(dfs["A"],dfs[figtype],color="blue",s=s)
    axt.set_ylabel(figtype)
    if n==4:
        axt.set_xlabel("Connection Strength")
        axt.set_xticks([6,8,10,12,14,16,18],[0.6,0.8,1.0,1.2,1.4,1.6,1.8])
    else:
        axt.set_xticks([])



## ---------------------------------------------------------
## for Figures S1
## ---------------------------------------------------------



def _sim_length_plot(func,hour):
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    with open("PoissonSim"+hour+"_cor_K.pkl","rb") as f:
        dfccK = pickle.load(f)

    r=SF1REFTAR[func][0]
    t=SF1REFTAR[func][1]
    dfccC =dfccK.query("(ref==@r)and(tar==@t)").iloc[0,:-2]

    t = np.linspace(-100,100,201)*0.001
    if func=="OUP":
        ccf = (np.exp(-2*np.fabs(t)/SF1TAU)*SF1SIGMA*SF1SIGMA+SF1MU*SF1MU)*3.6
    else:
        ccf = (np.exp(-t*t*SF1DOMEGA*SF1DOMEGA/2)*np.cos(SF1OMEGA0*t)
               *SF1SIGMA*SF1SIGMA/2+SF1MU*SF1MU)*3.6
 
        
    plt.plot(t*1000,ccf*SF1MAG[hour],"crimson",lw=4)
    plt.step(dfccC.index, dfccC, color="black",where="mid",lw=1)

    plt.ylim(330*SF1MAG[hour],390*SF1MAG[hour])
    plt.yticks([340*SF1MAG[hour],360*SF1MAG[hour],380*SF1MAG[hour]])
    plt.xticks([-100,0,100])
    plt.xlim(-100,100)
    plt.axvline(x=0, color="gray",lw=1)
        
    plt.savefig("FigureS1_"+func+"_"+hour+"_lengthdep.pdf")
    plt.clf()


## ---------------------------------------------------------
## for Figures S2
## ---------------------------------------------------------


def _dethering_fig(dataid):

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    n = 0
    for dth in ["","d5","d10","d50"]:
        with open("KH_"+dataid+dth+"_allacr_J.pkl","rb") as f:
            df =pickle.load(f)
            plt.plot(df.columns, df.iloc[0], color=SF2CL[dataid][n],lw=3)
            n += 1

    plt.yticks(SF2CYT[dataid][0],SF2CYT[dataid][1])
    plt.ylim(SF2CYL[dataid][0],SF2CYL[dataid][1])
    plt.xticks([-50,0,50])
    plt.xlim(-50,50)

    plt.savefig("FigureS2_"+dataid+"_dithering.pdf")
    plt.clf()


## ---------------------------------------------------------
## for Figures S7
## ---------------------------------------------------------
   
    
def _EIdominance_plot(dfei,col,outname):

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    plt.xticks([-1,0,1])
    plt.xlim(-1,1)
    plt.yticks([0,50,100])
    plt.ylim(0,100)

    plt.hist(dfei["dom"],range=(-1,1),bins=20,color=col)
    plt.savefig(outname)
    plt.cla()

        

if __name__ == "__main__":


    """
    ## ---------------------------------------------------------
    ## Cross-correlation graph
    ##   for Figure 1,4,S4,S6
    ##  [input]          
    ##   (DATAID)_Classical_best.csv:  dfbsc
    ##   (DATAID)_cor_K.pkl:           dfccK
    ##   (DATAID)_GLM_fig:             dffigg
    ##   (DATAID)_Shin_fig.pkl:        dffigs

    # Figure1 1st submittion
    #(137,153)  # O N->N     .
    #(1,152)    # O CGS->CGS .
    #(80,95)    # O CGS->CGS .
    #(62,156)   # O C->C     .
    #(3,228)    # X CGS->CG  .  Replace 
    #(79,133)   # O CGS->CGS .  Replace
    #(10,92)    # X CG->C    .  Replace
    #(110,116)  # X CS->C    .  Replace
    #(90,227)   # X CS->CG   .  Replace

    # Figure1 2st submittion
    #(137,153)  # O N->N     .
    #(1,152)    # O CGS->CGS .
    #(80,95)    # O CGS->CGS .
    #(62,156)   # O C->C     .
    #(209,174)  # X CGS->CG  .  Replace GS
    #(152,216)  # O CGS->CGS .  Replace GS
    #(92,136)   # X CG->C    .  Replace
    #(18,224)   # X CS->C    .  Replace
    #(147,128)  # X CS->CG   .  Replace

    # Figure4 1st submittion
    #(44,182)   # O G->G
    #(10,145)   # X CG->C
    #(129,136)  # O CG->CG
    #(20,152)   # O C->C
    #(24,7)     # O CGS->CGS
    #(25,54)    # O CGS->CGS
    #(241,222)  # O N->N
    #(112,31)   # X CS->C
    #(125,30)   # O S->S
    #(114,59)   # O GS->GS

    # Figure4 2nd submittion
    #(44,182)   # O G->G        .
    #(73,117)   # X CG->C       . Replace
    #(129,136)  # O CG->CG      .
    #(20,152)   # O C->C        .
    #(24,7)     # O CGS->CGS    .
    #(25,54)    # O CGS->CGS    .
    #(241,222)  # O N->N        .
    #(111,230)  # X CS->C       . Replace
    #(125,30)   # O S->S        .
    #(114,59)   # O GS->GS      .
    
    # Figure4 replace candidate for cLM 0.001 [NOT USED]================
    # cuspy CS
    #(46,227)   # replace (112,31)
    #(116,233)  # replace (112,31)
    #(125,231)  # replace (112,31)
    #(240,28)   # replace (112,31)
    #(240,78)   # replace (112,31)
    #(36,191)   # replace (90,227) inhibitory 
    # cuspy CG
    #(27,136)   # replace (10,145)
    #(92,136)   # replace (10,145)
    #(94,92)    # replace (10,92)
    #===================================================================
    
    # FigureS4 1st submittion
    #(281,342)   # O G->G      .
    #(195,389)   # O CG->CG    .
    #(275,383)   # O CG->CG    .
    #(190,418)   # X C->CG     . Replace
    #(281,274)   # O CGS->CGS  .
    #(372,286)   # O CGS->CGS  .
    #(313,287)   # O N->N      .
    #(306,235)   # X CS->CG    . Replace
    #(323,282)   # O S->S      .
    #(293,328)   # O GS->GS    .

    # FigureS4 2nd submittion
    #(281,342)   # O G->G      .
    #(195,389)   # O CG->CG    .
    #(275,383)   # O CG->CG    .
    #(391,152)   # X C->CG     . Replace
    #(281,274)   # O CGS->CGS  .
    #(372,286)   # O CGS->CGS  .
    #(313,287)   # O N->N      .
    #(307,366)   # X CS->CG    . Replace
    #(323,282)   # O S->S      .
    #(293,328)   # O GS->GS    .


    # FigureS4 replace candidate for cLM 0.001 [NOT USED]===============
    # cuspy CS
    #(79,289)    # replace (306,235)
    #(306,241)   # replace (306,235)
    #(307,366)   # replace (306,235)
    # G
    #(301,185)   # replace (281,342)
    #(413,62)    # replace (281,342)
    #===================================================================

    # FigureS6 1st submittion
    #(41,293)    # O G->G      .     
    #(4,263)     # O CG->CG    .  Replace
    #(309,342)   # O CG->CG    .
    #(4,207)     # O C->C      .
    #(142,289)   # O CGS->CGS  .
    #(7,61)      # X CGS->CG   .  Replace
    #(0,5)       # O N->N      .
    #(329,263)   # X CS->C     .  Replace
    #(270,83)    # X S->N      .  Replace
    #(240,237)   # X GS->G     .  Replace

    # FigureS6 2nd submittion
    #(41,293)    # O G->G      .     
    #(16,74)     # O CG->CG    .  Replace
    #(309,342)   # O CG->CG    .
    #(4,207)     # O C->C      .
    #(142,289)   # O CGS->CGS  .
    #(30,293)    # X CGS->CG   .  Replace
    #(0,5)       # O N->N      .
    #(4,254)     # X CS->C     .  Replace
    #(10,386)    # X S->N      .  Replace
    #(305,300)   # X GS->G     .  Replace


    # FigureS6 replace candidate for cLM 0.001 [NOT USED]===============
    # cuspy CS
    #(154,54)     # replace (329,263)
    #(252,346)    # replace (329,263)
    # G
    #(8,222)      # replace (41,293)
    # S
    #(1,273)      # replace (270,83) inhibitory
    #(168,269)    # replace (270,83) inhibitory
    #(199,114)    # replace (270,83) inhibitory
    #(264,255)    # replace (270,83) inhibitory
    #(353,225)    # replace (270,83) inhibitory
    #===================================================================

    partial=True

    ## for figure1
    #DATAID = "KH_phase3r"
    #idxlst = [(137,153),(1,152),(80,95),(62,156),(3,228),
    #           (79,133),(10,92),(110,116),(90,227)]
    #ccwidth = "F"
    #ccwidth = "N"
    #figdir = DATAID+"_Fig1_CC_"+ccwidth

    ## for figure1rev
    #DATAID = "KH_phase3r"
    #idxlst = [(137,153),(1,152),(80,95),(62,156),(209,174),
    #           (152,216),(92,136),(18,224),(147,128)]
    #ccwidth = "F"
    #ccwidth = "N"
    #figdir = DATAID+"_Fig1rev_CC_"+ccwidth
    
    ## for figure4
    #DATAID = "KH_phase3r"
    #idxlst = [(44,182),(10,145),(129,136),(20,152),(24,7),
    #           (25,54),(241,222),(112,31),(125,30),(114,59)]
    #ccwidth = "F"
    #figdir = DATAID+"_Fig4_CC_"+ccwidth
    
    ## for figure4rev
    #DATAID = "KH_phase3r"
    #idxlst = [(44,182),(73,117),(129,136),(20,152),(24,7),
    #          (25,54),(241,222),(111,230),(125,30),(114,59)]
    #ccwidth = "F"
    #figdir = DATAID+"_Fig4rev_CC_"+ccwidth
    
    ## for figureS4
    #DATAID = "KH_frontalr"
    #idxlst = [(281,342),(195,389),(275,383),(190,418),(281,274),
    #          (372,286),(313,287),(306,235),(323,282),(293,328)]
    #ccwidth = "F"
    #figdir = DATAID+"_FigS4_CC_"+ccwidth
    
    ## for figureS4rev
    #DATAID = "KH_frontalr"
    #idxlst = [(281,342),(195,389),(275,383),(391,152),(281,274),
    #          (372,286),(313,287),(307,366),(323,282),(293,328)]
    #ccwidth = "F"
    #figdir = DATAID+"_FigS4rev_CC_"+ccwidth

    ## for figureS6
    #DATAID = "KH_posteriorr"
    #idxlst = [(41,293),(4,263),(309,342),(4,207),(142,289),
    #          (7,61),(0,5),(329,263),(270,83),(240,237)]
    #ccwidth = "F"
    #figdir = DATAID+"_FigS6_CC_"+ccwidth
    
    ## for figureS6rev
    DATAID = "KH_posteriorr"
    idxlst = [(41,293),(16,74),(309,342),(4,207),(142,289),
              (30,293),(0,5),(4,254),(10,386),(305,300)]
    ccwidth = "F"
    figdir = DATAID+"_FigS6rev_CC_"+ccwidth

    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfbsc = pd.read_csv(DATAID+"_Classical_best.csv",
                        usecols=["ref","tar","upcc","lowcc"])
    with open(DATAID+"_cor_K.pkl","rb") as f:    
        dfccK = pickle.load(f)
    with open(DATAID+"_cor_N.pkl","rb") as f:    
        dfccN = pickle.load(f)
    with open(DATAID+"_GLM_fig.pkl","rb") as f:    
        dffigg = pickle.load(f)
    with open(DATAID+"_Shin_fig.pkl","rb") as f:    
        dffigs = pickle.load(f)
    dfgrp = pd.read_csv(DATAID+"_group.csv")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    wF = WINHALF_MS[ccwidth]
    os.makedirs(figdir, exist_ok=True)

    dfccK.set_index(["ref","tar"],inplace=True)
    dfccN.set_index(["ref","tar"],inplace=True)
    dfbsc.set_index(["ref","tar"],inplace=True)
    dffigg.set_index(["ref","tar"],inplace=True)
    dffigs.set_index(["ref","tar"],inplace=True)
    dfgrp.set_index(["ref","tar"],inplace=True)


    dfgrp =dfgrp.loc[dfgrp.index.isin(idxlst)]
    #dfgrp =dfgrp.reset_index().query("group=='GS'")[["ref","tar"]]
    #idxlst= dfgrp.values.tolist()
    print(dfgrp)

    
    dfccK = dfccK.T.query("-@wF<=index<=@wF").T.copy()
    dfccN = dfccN.T.query("-@wF<=index<=@wF").T.copy()
    dffigg = dffigg.T.query("-@wF<=index<=@wF").T.copy()
    dffigs = dffigs.T.query("-@wF<=index<=@wF").T.copy()
    
    if partial==True:
        dfccK =dfccK.loc[dfccK.index.isin(idxlst)]
        dfccN =dfccN.loc[dfccN.index.isin(idxlst)]
        dfbsc =dfbsc.loc[dfbsc.index.isin(idxlst)]
        dffigg =dffigg.loc[dffigg.index.isin(idxlst)]
        dffigs =dffigs.loc[dffigs.index.isin(idxlst)]

    dfcls = dfccK.reset_index()[["ref","tar"]].values.tolist()

    if ccwidth == "N":
        dfcc = dfccN
    else:
        dfcc = dfccK
    
    #for n in range(len(dfcls)):
    #    _figcc_plot(ccwidth,figdir,wF,dfcls[n],np.array(dfcc.iloc[n,:]),
    #              dfbsc.iloc[n,:].tolist(),np.array(dffigg.iloc[n,:]),
    #              np.array(dffigs.iloc[n,:]))

    
    
    joblib.Parallel(n_jobs=PROCESS_NUM)(
        [joblib.delayed(_figcc_plot)(ccwidth,figdir,wF,dfcls[n],
                                   np.array(dfcc.iloc[n,:]),
                                   dfbsc.iloc[n,:].tolist(),
                                   np.array(dffigg.iloc[n,:]),
                                   np.array(dffigs.iloc[n,:]))
         for n in range(len(dfcls))])
    
    ## ---------------------------------------------------------
    """


    """
    ## ---------------------------------------------------------
    ## Neural response function plot
    ## for Figure 3,S3,S5

    FIGN = {"phase3r":"3","frontalr":"S3","posteriorr":"S5"}
    for dataid in ["phase3r","frontalr","posteriorr"]:
        with open("KH_"+dataid+"_nrf_N.pkl","rb") as f:
            srnrf = pickle.load(f)

        outname = dataid+"_Fig"+FIGN[dataid]+"_NRF_N.pdf"
        _neuralresponse_plot(srnrf,dataid,outname)


    ## ---------------------------------------------------------

    """

    """
    ## ---------------------------------------------------------
    ## Auto-correlation plot
    ## for Figure 3,S3,S5

    FIGN = {"phase3r":"3","frontalr":"S3","posteriorr":"S5"}
    for dataid in ["phase3r","frontalr","posteriorr"]:
        with open("KH_"+dataid+"_allacr_K.pkl","rb") as f:
            dfnrf = pickle.load(f)

        outname = dataid+"_Fig"+FIGN[dataid]+"_ALLACR_K.pdf"
        _allautocorrelation_plot(dfnrf,dataid,outname)

    ## ---------------------------------------------------------
    """

    """
    ## ---------------------------------------------------------
    ## Powerspectrum plot
    ## for Figure 3,S3,S5

    FIGN = {"phase3r":"3","frontalr":"S3","posteriorr":"S5"}
    for dataid in ["phase3r","frontalr","posteriorr"]:
        with open("KH_"+dataid+"_pow_N.pkl","rb") as f:
            srpow = pickle.load(f)

        outname = dataid+"_Fig"+FIGN[dataid]+"_pow_N.pdf"
        _pow_plot(srpow,dataid,outname)

    ## ---------------------------------------------------------

    """


      

    
    """
    ## ---------------------------------------------------------
    ## matrix graph
    ##   for Figure 4,S4,S6

    
    FIGN = {"phase3r":"3","frontalr":"S4","posteriorr":"S6"}
    for dataid in ["phase3r","frontalr","posteriorr"]:
        
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        dfbsc = pd.read_csv("KH_"+dataid+"_Classical_best.csv")
        dfbsg = pd.read_csv("KH_"+dataid+"_GLM_best.csv")
        dfbss = pd.read_csv("KH_"+dataid+"_Shin_best.csv")
        ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

        dfce = dfbsc[["ref","tar","Ze","ext"]].rename(columns={"Ze":"J"})
        dfci = dfbsc[["ref","tar","Zi","inh"]].rename(columns={"Zi":"J"})
        dfge = dfbsg[["ref","tar","J","ext"]]
        dfgi = dfbsg[["ref","tar","J","inh"]]
        dfse = dfbss[["ref","tar","J","ext"]]
        dfsi = dfbss[["ref","tar","J","inh"]]

        _J_matrix_plot(dfce,dfci,10,
                      "KH_"+dataid+"_Fig"+FIGN[dataid]+"_Classical_matrix.pdf")
        _J_matrix_plot(dfge,dfgi,20,
                      "KH_"+dataid+"_Fig"+FIGN[dataid]+"_GLM_matrix.pdf")
        _J_matrix_plot(dfse,dfsi,20,
                      "KH_"+dataid+"_Fig"+FIGN[dataid]+"_Shin_matrix.pdf")

    ## ---------------------------------------------------------

    #"""

  
    ## ---------------------------------------------------------
    ## Neural response function plot
    ## for Figure 5

    for A in ["06","12","18"]:
        with open("SM_MAT511_B15_A"+A+"_nrf_N.pkl","rb") as f:
            srnrf = pickle.load(f)
        dataid= "MAT511A"
        outname = "Figure5_SM_MAT511_B15_A"+A+"_NRF_N.pdf"
        _neuralresponse_plot(srnrf,dataid,outname)


    ## ---------------------------------------------------------
    #"""

    """
    ## ---------------------------------------------------------
    ## MCC plot
    ## for Figure 5

    DATAID = "SM_MAT511_B15_A"
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfc = pd.read_csv(DATAID+"_Classical_MCC.csv")
    dfg = pd.read_csv(DATAID+"_GLM_MCC.csv")
    dfs = pd.read_csv(DATAID+"_Shin_MCC.csv")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    fig = plt.figure()    
    gs = gsp.GridSpec(6,1, wspace=0.1, hspace=0.3)

    ax1 = _ax_MCC_plot("FPRe",dfc,dfg,dfs,15,0)
    ax2 = _ax_MCC_plot("FNRe",dfc,dfg,dfs,15,1)
    ax3 = _ax_MCC_plot("FPRi",dfc,dfg,dfs,15,2)
    ax4 = _ax_MCC_plot("FNRi",dfc,dfg,dfs,15,3)
    ax5 = _ax_MCC_plot("MCC",dfc,dfg,dfs,15,4)
    
    plt.savefig("Figure5_MCC_20231105.pdf")
    plt.show()

    ## ---------------------------------------------------------

    """



    """
    ## ---------------------------------------------------------
    ## time dependence of CC
    ##   for Figure S1

    for hour in ["1h","10h","100h"]:
        _sim_length_plot("OUP",hour)
        _sim_length_plot("SIN",hour)
    ## ---------------------------------------------------------
    """

    """
    ## ---------------------------------------------------------
    ## dithering of CC
    ##   for Figure S2

    for dataid in ["phase3r","frontalr","posteriorr"]:
        _dethering_fig(dataid)

    ## ---------------------------------------------------------
    """


    
    
    """
    ## ---------------------------------------------------------
    ## EI dominance plot
    ## for Figure S7

    DATAID="KH_phase3r"
    
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    dfgrp = pd.read_csv(DATAID+"_group.csv")
    dfeic = pd.read_csv(DATAID+"_EId_Classical.csv")
    dfeig = pd.read_csv(DATAID+"_EId_GLM.csv")
    dfeis = pd.read_csv(DATAID+"_EId_Shin.csv")
    ## rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

    _EIdominance_plot(
        dfeic,"limegreen",DATAID+"_EId_Classical.pdf")
    _EIdominance_plot(
        dfeig,"orange",DATAID+"_EId_GLM.pdf")
    _EIdominance_plot(
        dfeis,"blue",DATAID+"_EId_Shin.pdf")

    """
    
