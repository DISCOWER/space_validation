import numpy as np
import scipy as sp
from scipy import special, optimize

def bernstein_poly(i,n,t):
    return special.comb(n, i) * ( t**i ) * (1 - t)**(n-i)

def eval_bezier(cps, N=100):
    # if cps has three dimensions we consider the parallel bezier
    # for reachability
    if len(cps.shape) == 3:
        vals = np.zeros((cps.shape[0],N,cps.shape[2]))
        for i in range(cps.shape[2]):
            vals[:,:,i] = eval_bezier(cps[:,:,i],N)
        return vals
    else:
        ndim = cps.shape[0]
        n = cps.shape[1]-1

        ts = np.linspace(0,1,N)
        vals = np.zeros((ndim,N))
        for i in range(n+1):
            vals += np.outer(bernstein_poly(i,n,ts),cps[:,i]).T
        return vals