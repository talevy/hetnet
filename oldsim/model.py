from numpy import sqrt, linspace
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from pylab import plot, axis, show, savefig
import matplotlib.pyplot as plt
from math import pow

## globals ##
#beta
#n
#s
#inic

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
        xd = s[i] - gamma[i]*x[i] + \
            sum([abs(beta[i,j])*hill(x[j],beta[i,j],theta[0]) \
                     for j in xrange(n)])
        ret.append(xd)
    return ret


def plot(res, t, filepath):
    plt.figure()
    for i in xrange(n):
        plt.plot(t, res[:,i], label=("x%d"%i))
    plt.xlabel('time')
    plt.ylabel('Concentration')
    plt.title('Model')
    #plt.legend(loc=0)
    savefig(filepath)
    show()

