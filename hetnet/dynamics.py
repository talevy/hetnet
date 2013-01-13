from numpy import sqrt, linspace
from igraph import *
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from pylab import plot, axis, show, savefig
import matplotlib.pyplot as plt
from math import pow

# Globals for dynamics

beta = None
n = 0
s = [0]*n
gamma = [0]*n
theta = [0]*n
m = 0
inic = [0]*n
t = linspace(0, 5., 5)

def hill(x, b, th):
    x_m = pow(x,m)
    theta_m = pow(th, m)
    if b > 0:
        return x_m/(x_m + theta_m)
    elif b < 0:
        return 1/(1 + x_m/theta_m)
    else:  
        return 0

def f(x,time):
    """ Compute the derivate of 'z' at time 'time'.
        'z' is a list of four elements.
    """
    ret = []
    for i in xrange(n):
        summ=sum([abs(beta[i,j])*hill(x[j],beta[i,j],theta[0]) for j in xrange(n)])
        xd = s[i] - gamma[i]*x[i] + summ
        ret.append(xd)
    return ret

def runsim(_beta, _n, _s, _gamma, _theta, _m, _inic, _t):
    global beta, n, s, gamma, theta, m, inic, t
    beta = _beta
    n = _n
    s = _s
    gamma = _gamma
    theta = _theta
    m = _m
    inic = _inic
    t = _t
    
    res = odeint(f, inic, t)
    return res

def plot(res, t, filepath=None):
    plt.figure()
    for i in xrange(n):
        plt.plot(t, res[:,i], label=("x%d"%i))
        plt.xlabel('time')
        plt.ylabel('Concentration')
        plt.title('Model')
        plt.legend(loc=0)
        if filepath:
            savefig(filepath)
    show()


