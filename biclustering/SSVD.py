import numpy as np
import math

def thresh(z,delta):
    return np.sign(z)*(np.abs(z) >= delta)*(np.abs(z)-delta)

def ssvd(X, gamu = 2, gamv = 2, merr = 10**(-4), niter = 100):
    # initial values
    U, s, V = np.linalg.svd(X)
    u0 = U.T[0]
    v0 = V.T[0]
    
    n = X.shape[0]
    d = X.shape[1]
    ud = 1
    vd = 1
    iters = 0
    SST = np.sum(X*X)
    while (ud > merr or vd > merr):
        iters = iters +1
        # Updating v
        z =  X.T @ u0
        winv = np.abs(z)**gamv
        sigsq = np.abs(SST - np.sum(z*z))/(n*d-d)
        cand = z*winv
        delt = np.sort(np.append(np.abs(cand),0))
        delt_uniq = np.unique(delt)
        Bv = np.ones(len(delt_uniq)-1)*float("inf")
        ind = np.where(winv>10^(-8))
        cand1 = cand[ind]
        winv1 = winv[ind]
        for i in range(len(Bv)):
            temp2 = thresh(cand1,delta = delt_uniq[i])
            temp2 = temp2/winv1
            temp3 = np.zeros(d)
            temp3[ind] = temp2
            Bv[i] = np.sum((X - u0[:,None] @ temp3[None,:])**2)/sigsq + np.sum(temp2!=0)*math.log(n*d)
        Iv = min(np.where(Bv== np.min(Bv)))
        th = delt_uniq[Iv]
        temp2 = thresh(cand1,delta = th)
        temp2 = temp2/winv1
        v1 = np.zeros(d)
        v1[ind] = temp2
        v1 = v1/((np.sum(v1*v1))**0.5) #v_new
        
        # Updating u
        z = X @ v1
        winu = np.abs(z)**gamu
        sigsq = np.abs(SST - np.sum(z*z))/(n*d-n)
        cand = z*winu
        delt = np.sort(np.append(np.abs(cand),0))
        delt_uniq = np.unique(delt)
        Bu = np.ones(len(delt_uniq)-1)*float("inf")
        ind = np.where(winu > 10^(-8))
        cand1 = cand[ind]
        winu1 = winu[ind]
        for i in range(len(Bu)):
            temp2 = thresh(cand1,delta = delt_uniq[i])
            temp2 = temp2/winu1
            temp3 = np.zeros(n)
            temp3[ind] = temp2
            Bu[i] = np.sum((X - temp3[:,None] @ v1[None,:])**2)/sigsq + np.sum(temp2!=0)*math.log(n*d)
        Iu = min(np.where(Bu==np.min(Bu)))
        th = delt_uniq[Iu]
        temp2 = thresh(cand1,delta = th)
        temp2 = temp2/winu1
        u1 = np.zeros(n)
        u1[ind] =  temp2
        u1 = u1/((np.sum(u1*u1))**0.5)
        
        
        ud = np.sum((u0-u1)*(u0-u1))**0.5
        vd = np.sum((v0-v1)*(v0-v1))**0.5

        if iters > niter :
            print("Fail to converge! Increase the niter!")
            break
        
        u0 = u1
        v0 = v1

    s = u1[None, :] @ X @ v1[:, None]
    return u1, v1, s, iters
