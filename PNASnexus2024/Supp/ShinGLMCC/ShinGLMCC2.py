"""
##############################################################################
ShinGLMCC.ShinGLMCC2.py

  This module primarily contains functions
    related to the connectivity estimation algorithms
    ShinGLMCC and GLMCC.

  by Yasuhiro Tsubo
  modified ver. 2024.5.14

##############################################################################
"""

## ===================================================================
## External modules
## ===================================================================

import joblib

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import chi2


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

## Parameters for setting the initial value of parameter a(t)
## EPSILON: A small value set to prevent logarithmic divergence
##            when a histogram has bins of zero value.
EPSILON = 0.001

## BETA: Hyperparameter for the log prior weight
## BETA = 2/GAMMA (in Articles)
##   "sGLM": for estimating correlation between populations using GLMCC.
##   "cGLM": for estimating cross-correlation between neurons using GLMCC.
##     GAMMA   SciRep2021:2*10^-4   NatComm2019:5*10^-4
##      BETA   SciRep2021:10000     NatComm2019:4000
##        SciRep2021:  Endo et al. Sci Rep 11, 12087 (2021)
##        NatComm2019: Kobayashi et al. Nat Commun 10, 4468 (2019)
##   "cShin": for estimating cross-correlation between neurons using ShinGLMCC.
BETA = {"sGLM":2410000,"cGLM":4000,"cShin":1000000,"aShin":10e8,"aCont":10e8}

## Parameters for Levenberg-Marquard optimization method
## The regularization parameter cLM balances
##   the Hessian and gradient descent in Levenberg-Marquardt optimization.
##   Larger cLM values favor gradient descent.
## ETA:      Reduction factor for cLM during optimization.
## CLM_INIT: Initial value for cLM.
## CLM_MAX:  Maximum value for cLM.
## LOOP_OUT: Convergence threshold for negative log-posterior changes.
## MAX_ITER: Maximum number of iterations for the optimization loop.

ETA = 0.1
CLM_INIT = 0.01
CLM_MAX = 1e8
LOOP_OUT = 0.0001
MAX_ITER = 2000


## ===================================================================
## Private Functions
## ===================================================================

## _get_string_cols() --------------------------------------
## Extract only the str type columns from a pandas dataframe into a list.
##  [input]
##   +"dfm": (Pandas DataFrame) Arbitrary.
##  [output]
##   +(list of strs) List of the str type columns.
##
##  <- like2best()

def _get_string_cols(dfm):
    return [col for col in dfm.columns if isinstance(col, str)]

## _get_rightcc_cols() -------------------------------------
## Extract only the columns of the right-side of cross-correlation.
##  from a pandas dataframe into a list.
##  [input]
##   +"dfm": (Pandas DataFrame) Arbitrary including cross-correlation columns.
##  [output]
##   +(list of strs) List of the columns of the right-side of cross-correlation.
##
##  <- like2best()

def _get_rightcc_cols(dfm):
    return [col for col in dfm.columns
            if isinstance(col, (int, float)) and col>=0 ]



## _eachfigdata() ------------------------------------------
## Generate data for plotting for each neuron pair.
##  [input]
##   +"x[J/K]": (1D numpy array of floats)
##                Numbers starting from 0 to WINHALF_MS,
##                 with increments ["J"=0.1ms/"K"=1ms],
##                 used for function interpolation.
##   +"wJ": (int) Number of time bins of the half of
##                  cross-correlation histogram with time resolution "J"=0.1ms.
##   +"npa":  (1D numpy array of floats)
##           Optimized values of parameters [a(t),JR,JL].
##   +"npa0":  (1D numpy array of floats)
##           Optimized values of parameters [a(t),JR,JL] for synapseless model.
##              @Exclude the last two elements: a(t)
##              @Last two elements: JR, JL
##            <length> dim+2 
##            (dim: Dimensionality of a(t), typically set to 101.)
##   +"syn":   (float) Optimised values of parameter JR.
##              Only if putative connection index =1, otherwise 0.
##                      (Synapric weight for the right side.)
##   +"delay": (int) Delay parameter of synaptic function. [ms]
##   +"tau":   (int) Decay time constant of synaptic function. [ms]
##  [output]
##   +"fj": (1D numpy array of floats)
##     Data for plotting for one cross-correlation histogram.
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##                 Estimated cross-correlation function
##                  by using the optimized parameters
##                  at Time lags -WINHALF_MS to WINHALF_MS.
##    }
##
##  <- figdata()

def _eachfigdata(xJ,xK,wJ,npa,npa0,syn,delay,tau):
    aJ = np.interp(xJ, xK, npa)
    a0J = np.interp(xJ, xK, npa0)
    yJ = syn*get_synapticfunc(DELTA_MS["J"],WINHALF_MS["C"],delay,tau)[0][wJ:]

    if syn!=0:
        fJ= np.exp(aJ+yJ)
    else:
        fJ= np.exp(a0J)

    return fJ


## get_synapticfunc() --------------------------------------
## Get synaptic_function data
##  [input]
##   +"dj":    (float) Time bin width (typically "J"=0.1ms)
##                        of the output correlation function. [ms]
##   +"win":   (float) Half of the time width
##                        of the output correlation function. [ms]
##   +"delay": (int) Delay parameter of synaptic function [ms]
##   +"tau":   (int) Decay time constant of synaptic function [ms]
##  [output]
##   +"fpj": (1D numpy array (dimJ)
##             fR(t) Values of synaptic function for time lag t > 0, else 0,
##                          with time resolution dj, typically "J"(=0.1ms).
##   +"fmj": (1D numpy array (dimJ))fL(t): (1D numpy array (dimJ))
##             fL(t) Values of synaptic function for time lag t < 0, else 0.
##                          with time resolution dj, typically "J"(=0.1ms).
##                  fL(t) = fR(-t)
##             (dimJ: Dimensionality of "hsJ", typically set to 1001.)
##   +"fpj+fmj": fR(t)+fL(t)
##
##  <- cor2glm()

def get_synapticfunc(dj,win,delay,tau):

    delayj = (delay+win)/dj
    tauj = tau/dj
    fpj = np.array([ np.exp(-(j-delayj)/tauj)
                     if (j>=delayj) else 0
                     for j in range(2*int(win/dj)+1)])
    fmj = fpj[::-1]
    return fpj, fmj, fpj+fmj


## set_initpar() --------------------------------------------
## Set initial values of parameter a(t) and JR, JL
##  [input]
##   +"hsK":  (1D numpy array (dimK))
##              Cross-correlation histogram with time resolution "K"(=1ms)
##              (dimK: Dimensionality of "hsK", typically set to 101.)
##  [output]
##   +"npp":  (2D numpy array,(N(N-1)/2,dim+2))
##              Initial values of parameters [a(t),JR,JL].
##              (N= Total number of neurons.)
##              (dim: Dimensionality of a(t), typically set to 101.)
##    {
##     *column: @Exclude the last two elements: a(t)
##              @Last two elements: JR, JL
##     *row: for each Neuron ID pair (reference ID, target ID) (ref<tar)
##    }

##  <- cor2glm()

def set_initpar(hsK):

    ## Case of 1D numpy array 

    if hsK.ndim == 1:
        hsKm = np.ones_like(hsK) * hsK.mean()
        npp = np.hstack((np.log(hsKm + EPSILON),(0, 0)))

    ## Case of 2D numpy array
    
    else:
        hsKm = np.ones_like(hsK) * np.array(hsK.mean(axis=1)).reshape(-1, 1)
        npp = np.hstack((np.log(hsKm + EPSILON), np.zeros((hsKm.shape[0], 2))))

    return npp


## _calc_surk() --------------------------------------------
## Calculate auxiliary values SK, UK, RK.
##  [input]
##   +"p": (1D numpy array (dim+dof))
##           List of updated parameters of ShinGLMCC/GLMCC.
##           @Exclude the last two elements: a(t)
##           @Last two elements: JR, JL (only for dof=2)
##             (Synapric weight for the right and left side, respectively.)
##           (dof: 0 for without synaptic function,
##                 2 for with synaptic function.)
##           (dim: Dimensionality of a(t), typically set to 101.)
##   +"JKrt": (int) Time resolution ratio, typically set to 10.
##   +"fn":   (tuple of 3 1D numpy arrays) The synaptic functions.
##              (fR(t), fL(t), fR(t)+fL(t))
##                +fR(t): (1D numpy array (dimJ))
##                   Values of synaptic function for time lag > 0, else 0.
##                +fL(t): (1D numpy array (dimJ))
##                   Values of synaptic function for time lag < 0, else 0.
##                   fL(t) = fR(-t)
##                +"fpj+fmj": fR(t)+fL(t)
##   +"f": 1 for SK, fn[2] for UK, fn*fn for RK.
##  [output]
##   +"sK": (tuple of 3 1D numpy arrays) 
##               Auxiliary functions with time resolution "K"= 1ms.
##                (sKp(t), sKm(t), fR(t)+fL(t))
##                *SK= Sum_j (1*exp(JR*fR(tj)+JL*fL(tj)))
##                *UK= Sum_j (f(tj)*exp(JR*fR(tj)+JL*fL(tj)))
##                *RK= Sum_j (f(tj)*f(tj)*exp(JR*fR(tj)+JL*fL(tj)))
##    { 
##     *"sKp": (1D numpy array ((dimJ+1)/2))
##              Values of auxiliary function for time lag > 0, else 0.
##     *"sKm": (1D numpy array ((dimJ+1)/2))
##              Values of auxiliary function for time lag < 0, else 0.
##     *"sK": (1D numpy array (dimJ)) sK(t)= sKp(t)+sKm(t)
##    }
##
## <- _calc_logpost_H()
## <- _calc_logpost_grad()
## <- _calc_logpost()

def _calc_surk(p,JKrt,f,fn):

    sJ = np.exp(p[-2]*fn[0]+p[-1]*fn[1]) * f

    ## JKrt =10: The Ratio between the duration of a time bin
    ##                    at resolution "K" and one at resolution "J".
    ## To get values for each time bin at resolution "K"
    ##   using data from resolution "J":
    ## 1. Group the resolution "J" data into sets of JKrt=10 points each.
    ## 2. Sum each set.
    ## 3. Multiply by dt.
    ## Note: The data with time resolution "J",
    ##   corresponding to the bins at both ends of the histogram
    ##   with time resolution "K", is lacking JKrt/2 data points.
    ##   To address this, we padded both ends of the data
    ##   with time resolution "J", adding JKrt/2 data points
    ##   at each end that duplicate the neighboring values.

    sJx = np.hstack((sJ[:JKrt//2], sJ, sJ[-JKrt//2:]))[:-1]
    sK = sJx.reshape((-1,JKrt)).sum(axis=1)/JKrt

    n = (len(sK)-1)//2
    
    sKp = np.hstack((np.zeros(n),sK[n:]))
    sKm = np.hstack((sK[:n+1],np.zeros(n)))

    return [sKp, sKm, sK]



## _get_prior_H() ------------------------------------------
## Calculate the part of the Hessian matrix
##  related to a(t) due to the log-prior.
##  [input]
##   +"mode": (str) Specify the estimation method.
##                ("Shin" for ShinGLMCC, "GLM" for GLMCC)
##   +"dim":  (int) Dimensionality of a(t), typically set to 101.
##  [output]
##   +"H":    (2D numpy array, shape (dim,dim) )
##                The constant matrix part of the Hessian matrix.
##
##  <- _calc_nlogpost_H()

def _get_prior_H(mode, dim):

    if mode=="Shin":
        H = np.diag(np.full(dim-2,1),k=2) \
            +np.diag(np.full(dim-1,-4),k=1) \
            +np.diag(np.full(dim,6),k=0) \
            +np.diag(np.full(dim-1,-4),k=-1)\
            +np.diag(np.full(dim-2,1),k=-2)

        Hhx = (dim-1)//2

        H[0:2,0:2]=[[1,-2],[-2,5]]
        H[-2:,-2:]=[[5,-2],[-2,1]]
        H[Hhx-1:Hhx+2,Hhx-1:Hhx+2]=[[5,-2,0],[-2,2,-2],[0,-2,5]]
    elif mode=="Cont":
        H = np.diag(np.full(dim-2,1),k=2) \
            +np.diag(np.full(dim-1,-4),k=1) \
            +np.diag(np.full(dim,6),k=0) \
            +np.diag(np.full(dim-1,-4),k=-1)\
            +np.diag(np.full(dim-2,1),k=-2)

        Hhx = (dim-1)//2

        H[0:2,0:2]=[[1,-2],[-2,5]]
        H[-2:,-2:]=[[5,-2],[-2,1]]
    else:
        H =np.diag(np.full(dim-1,-1),k=1) \
            +np.diag(np.full(dim,2),k=0) \
            +np.diag(np.full(dim-1,-1),k=-1)
        H[0,0]= 1
        H[-1,-1]= 1

    return H


## _calc_nlogpost_H() ---------------------------------------
## Calculate the Hessian matrix of the negative log-posterior.
##  [input]
##   +Parameters: The same as ShinGLMCC()
##  [output]
##   +"H": (2D numpy array (dim+dof,dim+dof))
##              Hessian matrix of negative log-posterior
##                with respect to the ShinGLMCC/GLMCC parameters.
##            (dof: 0 for without synaptic function,
##                  2 for with synaptic function.)
##            (dim: Dimensionality of a(t), typically set to 101.)
##
## <- _update_p()

def _calc_nlogpost_H(mode, p, hsK, hsJ, fn, beta, dof):

    ## exp_aK: (1D numpy array) exp(a(t)) with time resolution "K"=1ms
    
    exp_aK=np.exp(p[:-2])


    ## _calc_surk(): Calculate auxiliary values SK, UK, RK.
    ## JKrt: (int) Time resolution ratio, typically set to 10.

    JKrt = int((len(hsJ)-1)/(len(hsK)-1))
    sK = _calc_surk(p,JKrt,1,fn)
    uK = _calc_surk(p,JKrt,fn[2],fn)
    rK = _calc_surk(p,JKrt,fn[2]*fn[2],fn)
    

    ## Haa: Part of the Hessian matrix related to a(t).
    ##   The first term is due to the log-likelihood,
    ##   and the second term is due to the log-prior.
    
    Haa = np.diag(exp_aK*sK[2])+beta*_get_prior_H(mode,len(exp_aK))


    ## If considering the synaptic function,
    ##   add two dimensions related to the synaptic weight J.
    
    if dof == 2:
        haJ = np.array([exp_aK*uK[0],exp_aK*uK[1]])
        hJa = haJ.T
        hJ = np.array([[np.dot(exp_aK,rK[0]),0],[0,np.dot(exp_aK,rK[1])]])
        H = np.hstack([np.vstack([Haa,haJ]),np.vstack([hJa,hJ])])
    else:
        H = Haa

    return H


## _calc_nlogpost_grad --------------------------------------
## Calculate the gradient vector of the negative log-posterior.
##  [input]
##   +Parameters: The same as ShinGLMCC()
##  [output]
##   +"grad": (1D numpy array (dim+dof))
##              List of partial derivative of negative log-posterior
##                with respect to the ShinGLMCC/GLMCC parameters.
##              @Exclude the last two elements: a(t)
##              @Last two elements: JR, JL (only for dof=2)
##                (Synapric weight for the right and left side, respectively.)
##            (dof: 0 for without synaptic function,
##                  2 for with synaptic function.)
##            (dim: Dimensionality of a(t), typically set to 101.)
##
## <- _update_p()

def _calc_nlogpost_grad(mode, p, hsK, hsJ, fn, beta, dof):

    ## exp_aK: (1D numpy array) exp(a(t)) with time resolution "K"=1ms
    
    exp_aK=np.exp(p[:-2])

    JKrt = int((len(hsJ)-1)/(len(hsK)-1))
    sK = _calc_surk(p,JKrt,1,fn)
    uK = _calc_surk(p,JKrt,fn[2],fn)

    sf = [np.dot(hsJ,fn[0]), np.dot(hsJ,fn[1])]


    ## d1: the 1st-order difference of a(t)
    ## d2: the 2nd-order difference of a(t)
    ## d4: the 4th-order difference of a(t)

    d1 = np.diff(p[:-2]).tolist()
    d2 = np.diff(p[:-2],n=2).tolist()
    d4 = np.diff(p[:-2],n=4).tolist()
    nd1 = (-np.diff(p[:-2])).tolist()
    nd2 = (-np.diff(p[:-2],n=2)).tolist()

    if mode=="Shin":
        a_2dfK = np.array([d2[0],-2*d2[0]+d2[1]]+d4\
                          +[-2*d2[-1]+d2[-2],d2[-1]])
        a_mid = (len(a_2dfK)-1)//2
        d2_mid = (len(d2)-1)//2
        a_2dfK[a_mid-1:a_mid+2]-=np.array((1,-2,1))*d2[d2_mid]
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ## weaken the component of the log-prior by inverting the sign.
        #a_2dfK = -a_2dfK
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    elif mode=="Cont":
        a_2dfK = np.array([d2[0],-2*d2[0]+d2[1]]+d4\
                          +[-2*d2[-1]+d2[-2],d2[-1]])
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ## weaken the component of the log-prior by inverting the sign.
        #a_2dfK = -a_2dfK
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    else:
        a_2dfK = np.array([nd1[0]]+nd2+[d1[-1]])

    grad = -hsK+exp_aK*sK[2]+a_2dfK*beta
    
    if dof==2:
        grad = np.append(grad,[-sf[i]+np.dot(exp_aK,uK[i])
                               for i in range(dof)])
        
    return grad



## update_p() ----------------------------------------------
## Update the parameters of ShinGLMCC/GLMCC
##  [input]
##   +Parameters: The same as ShinGLMCC()
##  [output]
##   +"p_new": (1D numpy array (dim+dof))
##              List of updated parameters of ShinGLMCC/GLMCC.
##              @Exclude the last two elements: a(t)
##              @Last two elements: JR, JL (only for dof=2)
##                (Synapric weight for the right and left side, respectively.)
##            (dof: 0 for without synaptic function,
##                  2 for with synaptic function.)
##            (dim: Dimensionality of a(t), typically set to 101.)
##
## <- _my_LM_optimizer()

def _updata_p(mode, p_old, hsK, hsJ, cLM, fn, beta, dof):

    grad = _calc_nlogpost_grad(mode,p_old,hsK,hsJ,fn,beta,dof)
    H = _calc_nlogpost_H(mode,p_old,hsK,hsJ,fn,beta,dof)
    T = np.diag(np.diag(H),k=0)
    Hinv = np.linalg.solve(H+cLM*T, grad)
    p_new = p_old - np.append(Hinv,np.zeros(2-dof))

    return p_new



## _calc_logpost() ------------------------------------------
## Calculate the log-posterior
##  [input]
##   +Parameters: The same as ShinGLMCC()
##  [output]
##   +"logpost":  (float) Log-posterior with the estimated parameters.
##   +"loglikep": (float) Log-likelihood for right-half side data.
##   +"loglikem": (float) Log-likelihood for left-half side data.
##
## <- _my_LM_optimizer()
## <- _calc_nlogpost()

def _calc_logpost(mode, p, hsK, hsJ, fn, beta, dof):

    ## exp_aK: (1D numpy array) exp(a(t)) with time resolution "K"=1ms
    
    exp_aK=np.exp(p[:-2])

    ## _calc_surk(): Calculate auxiliary values SK, UK, RK.
    ## JKrt: (int) Time resolution ratio, typically set to 10.

    JKrt = int((len(hsJ)-1)/(len(hsK)-1))
    sK = _calc_surk(p,JKrt,1,fn)

    sf = [np.dot(hsJ,fn[0]), np.dot(hsJ,fn[1])]

    
    ## Calculate the log likelihood
    
    loglike = np.dot(hsK,p[:-2])+p[-2]*sf[0]+p[-1]*sf[1]-np.dot(exp_aK,sK[2])

    
    ## Calculate the log prior

    if mode=="Shin":
        pdft = np.diff(p[:-2],n=2)
        pdf = np.delete(pdft,[(len(pdft)-1)//2])
    else:
        pdf = np.diff(p[:-2])        

    logprior = -beta*np.dot(pdf,pdf)/2


    ## Calculate the log posterior
    
    logpost = loglike + logprior

    
    # Recalculate the log-likelihood for both the right and left halves.
    
    midid = (len(hsK)-1)//2
    loglikep = np.dot(hsK[midid:],p[midid:-2])+p[-2]*sf[0]-np.dot(
        exp_aK[midid:],sK[2][midid:])
    loglikem = np.dot(hsK[:midid+1],p[:midid+1])+p[-1]*sf[1]-np.dot(
        exp_aK[:midid+1],sK[2][:midid+1])

    loglike = np.dot(hsK[midid+1:],p[midid+1:-2])-np.dot(exp_aK[midid+1:],sK[2][midid+1:])+np.dot(hsK[:midid],p[:midid])-np.dot(exp_aK[:midid],sK[2][:midid])
    print(loglike)

    return logpost, loglikep, loglikem



## _calc_nlogpost() -----------------------------------------
## Calculate the negative value of log-posterior
##          for Levenberg-Marquard optimizer.
##  [input]
##   +Parameters: The same as ShinGLMCC()
##  [output]
##   +"-logpost": (float) Negative value of log-posterior
##
## <- _my_LM_optimizer()

def _calc_nlogpost(mode,p, hsK, hsJ, fn, beta, dof):

    logpost, loglikep, loglikem = _calc_logpost(mode,p,hsK,hsJ,fn,beta,dof)

    return -logpost



## _my_LM_optimizer() ---------------------------------------
## Estimate the parameters of GLMCC/ShinGLMCC
##  to maximize the posterior probability by Levenberg-Marquard method.
##  [input]
##   +Parameters: The same as ShinGLMCC()
##  [output]
##   +Parameters: The same as ShinGLMCC()
##
##  <- ShinGLMCC()

def _my_LM_optimizer(mode, p, hsK, hsJ, fn, beta, dof):

    ## Initialization of the Levenberg-Marquardt regularization parameter
    
    cLM = CLM_INIT

    
    ## Initial parameter values and the corresponding negative log-posterior
    
    p0 = p
    nlogpost0 = _calc_nlogpost(mode,p0,hsK,hsJ,fn,beta,dof)

    ## The difference between current and previous negative log-posterior
    
    diff = 0

    
    ## Initial output values
    
    p_out,nlogpost_out = p0,nlogpost0

    
    ## Loop to iteratively update parameters
    ##   and minimize negative log-posterior

    for s in range(MAX_ITER):

        ## Propose new parameter values using two different cLM values
        ## Calculate negative log-posterior for the proposed parameter values
        
        p1 = _updata_p(mode,p0,hsK,hsJ,cLM,fn,beta,dof)
        p2 = _updata_p(mode,p0,hsK,hsJ,cLM*ETA,fn,beta,dof)
        nlogpost1 = _calc_nlogpost(mode,p1,hsK,hsJ,fn,beta,dof)
        nlogpost2 = _calc_nlogpost(mode,p2,hsK,hsJ,fn,beta,dof)

        ## Decide on the best parameter values and update cLM accordingly
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if mode=="Shin":
            if nlogpost2 <= nlogpost0:
                cLM = cLM*ETA
                p_out,nlogpost_out = p2,nlogpost2
        else:
            if nlogpost2 <= nlogpost0:
                if nlogpost2 <= nlogpost1:
                    cLM = cLM*ETA
                    p_out,nlogpost_out = p2,nlogpost2
                else:
                    p_out,nlogpost_out = p1,nlogpost1
            elif nlogpost1 <= nlogpost0:
                p_out,nlogpost_out = p1,nlogpost1
            else:
                if cLM <= CLM_MAX:
                    cLM = cLM/ETA
                    continue
        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
        ## Calculate the difference in negative log-posterior
        ##   from the previous iteration

        diff = nlogpost_out-nlogpost0


        ## Break conditions based on
        ##    change in negative log-posterior or the cLM value
        
        if ((abs(diff) < LOOP_OUT) or (cLM > CLM_MAX)):
            break
        
        ## Update parameter values for the next iteration
        
        p0,nlogpost0 = p_out,nlogpost_out

    ## Calculate log-likelihood for the optimized parameter values
    logpost, loglikep, loglikem = _calc_logpost(mode,p_out,hsK,hsJ,fn,beta,dof)
       
    return p_out, logpost, loglikep, loglikem, diff, s, cLM



## ===================================================================
## Public Functions
## ===================================================================

## ShinGLMCC() ---------------------------------------------
## Call a Levenberg-Marquard optimizer
##  to estimate synaptic connectivity between two neurons.
##  [input]
##   +"mode": (str) Specify the estimation method.
##                ("Shin" for ShinGLMCC, "GLM" for GLMCC)
##   +"p_in": (1D numpy array (dim+dof))
##              List of (initial) parameters of ShinGLMCC/GLMCC.
##                @Exclude the last two elements: a(t)
##                @Last two elements: JR, JL (only for dof=2)
##                (Synapric weight for the right and left side, respectively.)
##            (dof: 0 for without synaptic function,
##                  2 for with synaptic function.)
##            (dim: Dimensionality of a(t), typically set to 101.)
##   +"hsK":  (1D numpy array (dimK))
##              Cross-correlation histogram with time resolution "K"(=1ms)
##             (dimK: Dimensionality of "hsK", typically set to 101.)
##   +"hsJ":  (1D numpy array (dimJ))
##              Cross-correlation histogram with time resolution "J"(=0.1ms)
##             (dimJ: Dimensionality of "hsJ", typically set to 1001.)
##   +"fn":   (tuple of 3 1D numpy arrays) The synaptic functions.
##              (fR(t), fL(t), fR(t)+fL(t))
##                +fR(t): (1D numpy array (dimJ))
##                   Values of synaptic function for time lag > 0, else 0.
##                +fL(t): (1D numpy array (dimJ))
##                   Values of synaptic function for time lag < 0, else 0.
##                   fL(t) = fR(-t)
##                +"fpj+fmj": fR(t)+fL(t)
##   +"beta": Hyperparameter that determines the weight of the log-prior.
##   +"dof": Additional degree of freedom of GLMCC/ShinGLMCC parameters
##            (0 for without synaptic function, 2 for with synaptic function.)
##  [output]
##   +"p_out": (1D numpy array shape (dim+dof))
##           List of the optimized parameters of ShinGLMCC/GLMCC
##            a(t) + [JR,JL] (last two elements)
##             (dof: 0 for without synaptic function,
##                    2 for with synaptic function.)
##             (dim: Dimensionality of a(t), typically set to 101.)
##   +"logpost":  (float) Log-posterior with the estimated parameters.
##   +"loglikep": (float) Log-likelihood for right-half side data.
##   +"loglikem": (float) Log-likelihood for left-half side data.
##   +"diff":  (float) Improvement in negative log-posterior
##                          at the last update step of the LM method.
##   +"s":     (int) Number of iterations using the LM method
##                          at the last update step.
##   +"cLM":   (int) A parameter of LM method at the last update step.
##               cLM controls the balance between the steepest descent method
##               and the Gauss-Newton method in the optimization algorithm.

def ShinGLMCC(mode, p_in, hsK, hsJ, fn, beta, dof):

    ## Estimate the parameters of GLMCC/ShinGLMCC
    ##          to maximize the posterior probability
    ##          by Levenberg-Marquard method
    
    p_out, logpost, loglikep, loglikem, diff, s, cLM = _my_LM_optimizer(
        mode,p_in,hsK,hsJ,fn,beta,dof)

    return p_out, logpost, loglikep, loglikem, diff, s, cLM


## Joblib2Para() -------------------------------------------
## Aggregate the joblib output and convert them into a parameter table format
##  [input]
##   +"jout": (list) List containing the outputs of ShinGLMCC/GLMCC.
##    {
##     [0]p_out: (1D numpy array shape (dim+dof))
##           List of the optimized parameters of ShinGLMCC/GLMCC
##            a(t) + [JR,JL] (last two elements)
##             (dof: 0 for without synaptic function,
##                    2 for with synaptic function.)
##             (dim: Dimensionality of a(t), typically set to 101.)
##     [1]logpost:  (float) Log-posterior with the estimated parameters.
##     [2]loglikep: (float) Log-likelihood for right-half side data.
##     [3]loglikem: (float) Log-likelihood for left-half side data.
##     [4]diff:     (float) Improvement in negative log-posterior
##                          at the last update step of the LM method.
##     [5]s:     (int) Number of iterations using the LM method
##                          at the last update step.
##     [6]cLM:   (int) A parameter of LM method at the last update step.
##                cLM controls the balance between the steepest descent method
##                 and the Gauss-Newton method in the optimization algorithm.
##    }
##   +"lscls":  (2D list of ints, (N(N-1)/2,2) )
##               List of neuron ID pairs [["ref","tar"],....],
##                  <lenght> N(N-1)/2  (N= Total number of neurons.)
##                      (for each Neuron ID pair (ref ID<tar ID)
##   +"cols": (Pandas Index) Column index
##                where discrete time points for the parameter a(t) are stored.
##  [output]
##   +Parameters: The same as col2glm().

def Joblib2Para(jout, lscls, cols):

    ## lsta: 2D List for parameters a(t) + [refidx,taridx]
    ## lstl: 2D List for the other parameters
    ##     [refidx,taridx,JR(plus),JL(minus),logpost,
    ##                     loglikep,loglikem, diff, s, cLM]

    lsta = []
    lstl = []

    for n in range(len(lscls)):
        lsta += [jout[n][0][:-2].tolist()+lscls[n]]
        lstl += [lscls[n]+jout[n][0][-2:].tolist()+
                [jout[n][1],jout[n][2],jout[n][3],jout[n][4],jout[n][5],jout[n][6]]]


    ## dfart: (Pandas DataFrame) Parameters a(t) + [refID,tarID] (ref<tar)
    ## dfatr: (Pandas DataFrame) Parameters a(t) + [refID,tarID] (ref>tar)

    npa = np.array(lsta)[:,:-2]
    npa = np.hstack((npa,np.zeros((npa.shape[0],2))))
    dfart = pd.DataFrame(lsta,columns=cols.tolist()+["ref","tar"])
    dfatr = dfart.copy()
    dfatr.columns = cols.tolist()[::-1]+["tar","ref"]
 
    ## dfa:  (Pandas DataFrame) Parameters a(t) + [refID,tarID] (all pairs)

    dfat = pd.concat([dfart,dfatr])
    dfa = dfat.sort_values(["ref","tar"]).reset_index(drop=True)


    ## dflrt: (Pandas DataFrame)
    ##          [refID,tarID] + Other parameters than a(t) (ref<tar)    
    ## dfltr: (Pandas DataFrame)
    ##          [refID,tarID] + Other parameters than a(t) (ref>tar)
    
    dflrt = pd.DataFrame(lstl)
    dfltr = dflrt.copy()
    dflrt.columns = ["ref","tar","J","Jm",
                   "logpost","loglike","loglikem","diff","s","cLM"]
    dfltr.columns = ["tar","ref","Jm","J",
                   "logpost","loglikem","loglike","diff","s","cLM"]

    ## dfl: (Pandas DataFrame)
    ##          [refID,tarID] + Other parameters than a(t) (all pairs)
   
    dflt = pd.concat([dflrt,dfltr])
    dfl = dflt.sort_values(["ref","tar"]).reset_index(drop=True)
    del dfl["loglikem"], dfl["Jm"]

    return npa, dfa, dfl


## cor2glm() -------------------------------------------
## Call a Levenberg-Marquard optimizer
##  to estimate synaptic connectivity between two neurons.
##  [input]
##   +"mode": (str) Specify the estimation method.
##                ("Shin" for ShinGLMCC, "GLM" for GLMCC)
##   +"dfccj": (Pandas DataFrame)
##             Cross-correlation histogram table with resolution "J"= 0.1ms
##   +"dfcck": (Pandas DataFrame)
##             Cross-correlation histogram table with resolution "K"= 1ms
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##                   Correlation histogram values
##                   at Time lags -WINHALF_MS to WINHALF_MS. 
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)
##   +"dof": (int) Additional degree of freedom of GLMCC/ShinGLMCC parameters.
##            (0 for without synaptic function, 2 for with synaptic function.)
##   +"d": (int) Delay parameter of synaptic function [ms]
##   +"t": (int) Decay time constant of synaptic function [ms]
##   -- The following parameter is not required for dof=0 --
##   +"npa":  (2D numpy array of floats,(N(N-1)/2,dim+2))
##            Initial values of parameters [a(t),JR,JL].
##            (N= Total number of neurons.)
##            (dim: Dimensionality of a(t), typically set to 101.)
##    {
##     *column: @Exclude the last two elements: a(t)
##              @Last two elements: JR, JL
##     *row: for each Neuron ID pair (reference ID, target ID) (ref<tar)
##    }
##  [output]
##   +"npa":  (2D numpy array of floats,(N(N-1)/2,dim+2))
##              Optimized values of parameters [a(t),JR,JL].
##            (N= Total number of neurons.)
##            (dim: Dimensionality of a(t), typically set to 101.)
##    {
##     *column: @Exclude the last two elements: a(t)
##              @Last two elements: JR, JL
##     *row: for each Neuron ID pair (reference ID, target ID) (ref<tar)
##    }
##   +"dfa": (Pandas DataFrame)
##    {
##     *-WINHALF_MS to WINHALF_MS: (float)
##                   Optimised values of parameter a(t)
##                   at Time lags -WINHALF_MS to WINHALF_MS. 
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)
##   +"dfl": (Pandas DataFrame)
##    {
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"J":   (float) Optimised values of parameter JR
##                      (Synapric weight for the right side.)
##     *"logpost": (float) Log-posterior across all ranges
##                                 with the estimated parameters.
##     *"loglike": (float) Log-likelihood for right-half side data.
##     *"diff":  (float) Improvement in negative log-posterior
##                          at the last update step of the LM method.
##     *"s":     (int) number of iterations using the LM method
##                          at the last update step.
##     *"cLM":   (int) A parameter of LM method at the last update step.
##        -- The following two columns are not included in dfma0. --
##     *"delay": (int) Delay parameter of synaptic function. [ms]
##     *"tau":   (int) Decay time constant of synaptic function. [ms]
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)

def all2glm(mode,dfccj,dfcck,dof,d=1,t=2,*args):

    ## Extract data for GLMCC estimation -------------------
    w = WINHALF_MS["C"]
    dfj = dfccj.T[:-2].query("-@w<=index<=@w").T.reset_index(drop=True)
    dfk = dfcck.T[:-2].query("-@w<=index<=@w").T.reset_index(drop=True)

    dfk.iloc[0,50]=int((dfk.iloc[0,49]+dfk.iloc[0,51])*0.5)

    ## Get synaptic_function data
    ##   [ fR, fL, (fR + fL) ] tuple of synaptic functions (1D numpy arrays)
    fnj = get_synapticfunc(DELTA_MS["J"],w,d,t)

    ## Set initial values of parameter a(t) and JR, JL
    npa = set_initpar(np.array(dfk))

    ## set parameter beta = 2/gamma
    beta = BETA["a"+mode]

    p_out, logpost, loglikep, loglikem, diff, s, cLM = ShinGLMCC(mode,npa[0],np.array(dfk.T[0]),np.array(dfj.T[0]),fnj,beta,dof)

    print(loglikep,loglikem,loglikep+loglikem)
    
    return np.exp(p_out[:-2]),dfk
   

## lik2best() -----------------------------------------------
## Find the best parameter set and calculate the p-values for two methods
##  [input]
##   +"dfm0": (Pandas DataFrame)
##         Optimised value tabels without synaptic function.
##    {
##     * 0 to WINHALF_MS: (float) Optimised values of parameter a(t)
##                                     at Time lags 0 to WINHALF_MS.
##         !! Note that only the right-half data was extracted.
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"logpost": (float) Log-posterior across all ranges
##                                 with the estimated parameters.
##     *"loglike": (float) Log-likelihood for right-half side data.
##    }
##       Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)
##   +"dfm": (Pandas DataFrame)
##         Optimised value table over all (delay, tau) pairs
##                                              with synaptic function.
##    {
##     * 0 to WINHALF_MS: (float) Optimised values of parameter a(t)
##                                     at Time lags 0 to WINHALF_MS.
##         !! Note that only the right-half data was extracted.
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"J":   (float) Optimised values of parameter JR
##     *"logpost": (float) Log-posterior across all ranges
##                                 with the estimated parameters.
##     *"loglike": (float) Log-likelihood for right-half side data.
##                      (Synapric weight for the right side.)
##     *"delay": (int) Delay parameter of synaptic function. [ms]
##     *"tau":   (int) Decay time constant of synaptic function. [ms]
##    }
##     Rows: N(N-1)*tN*dN, all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)
##            (tN*dN= Number of (tau,delay) pairs.)
##   +"threshold": (float) Significance threshold.
##  [output]
##   +"dfbs": (Pandas DataFrame)
##       Optimal parameter set (a(t), J) achieved the maximum log-likelihood
##                                         over all (delay, tau) pairs.
##    {
##     * 0 to WINHALF_MS: (float) Optimised values of parameter a(t)
##                                  at Time lags 0 to WINHALF_MS.
##         !! Note that only the right-half data was extracted.
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"J":   (float) Optimised values of parameter JR
##                      (Synapric weight for the right side.)
##     *"logpost": (float) Log-posterior across all ranges
##                                 with the estimated parameters.
##     *"loglike": (float) Log-likelihood for right-half side data.
##     *"delay": (int) Delay parameter of synaptic function. [ms]
##     *"tau":   (int) Decay time constant of synaptic function. [ms]
##     *"alpha": (float) P-value for the presence of putative connection.
##     *"ext":   (int) Putative excitatory connection indicator.
##     *"inh":   (int) Putative inhibitory connection indicator.
##                                 (1 if significant, 0 otherwise)
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##           (N= Total number of neurons.)

def lik2best(dfm0,dfm,threshold):

    idx = dfm.groupby(["ref","tar"])["loglike"].idxmax()
    #idx = dfm.groupby(["ref","tar"])["logpost"].idxmax()
    dfb = dfm.loc[idx].reset_index(drop=True)

    dfha = dfb[_get_rightcc_cols(dfm)]
    dfst0 = dfm0[_get_string_cols(dfm0)].rename(columns={"loglike":"loglike0"})
    dfst = pd.merge(dfb[_get_string_cols(dfm)],dfst0,on=["ref","tar"])

    dfst["loglikediff"] = dfst["loglike"]-dfst["loglike0"]
    dfst["alpha"] = chi2.sf(2*dfst["loglikediff"],df=2)
    dfst["ext"] =(dfst["J"]>0)*(dfst["alpha"]<threshold)*1
    dfst["inh"] =(dfst["J"]<0)*(dfst["alpha"]<threshold)*1
    dfst = dfst[["ref","tar","J","delay","tau","alpha","ext","inh"]]

    dfzrt = dfb[[0.0,"ref","tar"]]
    dfztr = dfzrt.copy()
    dfzrt.columns = ["zerort","ref","tar"]
    dfztr.columns = ["zerotr","tar","ref"]
    dfz = pd.merge(dfzrt,dfztr,on=["ref","tar"])
    dfz[0.0] = (dfz["zerort"]+dfz["zerotr"])*0.5
    dfha = dfha.copy()    ## avoid SettingWithCopyWarning
    dfha[0.0] = dfz[0.0]

    dfbs = pd.concat([dfha,dfst],axis=1)

    return dfbs



## figdata() -----------------------------------------------
## Generate data for plotting from the estimated model parameters.
##  [input]
##   +"dfbs": (Pandas DataFrame)
##       Optimal parameter set (a(t), J) achieved the maximum log-likelihood
##                                         over all (delay, tau) pairs.
##    {
##     * 0 to WINHALF_MS: (float) Optimised values of parameter a(t)
##                                  at Time lags 0 to WINHALF_MS.
##         !! Note that only the right-half data was extracted.
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"J":   (float) Optimised values of parameter JR
##                      (Synapric weight for the right side.)
##     *"logpost": (float) Log-posterior across all ranges
##                                 with the estimated parameters.
##     *"loglike": (float) Log-likelihood for right-half side data.
##     *"delay": (int) Delay parameter of synaptic function. [ms]
##     *"tau":   (int) Decay time constant of synaptic function. [ms]
##     *"alpha": (float) P-value for the presence of putative connection.
##     *"ext":   (int) Putative excitatory connection indicator.
##     *"inh":   (int) Putative inhibitory connection indicator.
##                                 (1 if significant, 0 otherwise)
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##           (N= Total number of neurons.)
##   +"dfa0": (Pandas DataFrame)
##         Optimised value tabels without synaptic function.
##    {
##     * 0 to WINHALF_MS: (float) Optimised values of parameter a(t)
##                                     at Time lags 0 to WINHALF_MS.
##         !! Note that only the right-half data was extracted.
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##     *"logpost": (float) Log-posterior across all ranges
##                                 with the estimated parameters.
##     *"loglike": (float) Log-likelihood for right-half side data.
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##            (N= Total number of neurons.)
##  [output]
##   +"dfJ": (Pandas DataFrame)
##     Data for plotting from the estimated model parameters.
##    {
##     * -WINHALF_MS to WINHALF_MS: (float)
##                 Estimated cross-correlation function
##                  by using the optimized parameters
##                  at Time lags -WINHALF_MS to WINHALF_MS.
##     *"ref": (int) Reference Neuron ID.
##     *"tar": (int) Target Neuron ID.
##    }
##     Rows: N(N-1), all neuron pairs, auto-indexed.
##           (N= Total number of neurons.)

def figdata(dfbs,dfa0):
    
    wK = int(WINHALF_MS["C"]/DELTA_MS["K"])
    wJ = int(WINHALF_MS["C"]/DELTA_MS["J"])

    dfcls = dfbs[["ref","tar"]]
    
    dfa = dfbs.iloc[:,:wK+1]
    dfa0 = dfa0.iloc[:,wK:-2]
    dfbs["syn"] = (dfbs["ext"]+dfbs["inh"])*dfbs["J"]

    xK = np.linspace(0, WINHALF_MS["C"], wK+1)
    xJ = np.linspace(0, WINHALF_MS["C"], wJ+1)

    lsJ = joblib.Parallel(n_jobs=PROCESS_NUM)(
        [joblib.delayed(_eachfigdata)(
            xJ, xK, wJ, dfa.iloc[n],dfa0.iloc[n],
            dfbs["syn"][n],dfbs["delay"][n],dfbs["tau"][n])
         for n in range(len(dfbs))])

    dffrt = pd.DataFrame(lsJ,columns=xJ)
    dfftr = dffrt.copy().drop(columns=0.0)
    dfftr.columns=-xJ[1:]
    dfJrt = pd.concat([dffrt,dfcls],axis=1)
    dfJtr = (pd.concat([dfftr.iloc[:,::-1],dfcls],axis=1)
             .rename(columns={"ref":"tar","tar":"ref"}))
    dfJ = (pd.merge(dfJtr,dfJrt,on=["ref","tar"])
           .sort_values(by=["ref","tar"]).reset_index(drop=True))
    dfJ = pd.concat([dfJ.drop(columns=["ref","tar"]),dfcls],axis=1)
    
    return dfJ
