from numpy import sqrt, linspace
from igraph import *
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from pylab import plot, axis, show, savefig
import matplotlib.pyplot as plt
from math import pow
import pickle
import parse_urchin_net as parse

# The derivative function.
# dy/dt = func(y, t)

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
        xd = s[i] - gamma[i]*x[i] + sum([abs(beta[i,j])*hill(x[j],beta[i,j],theta[0]) for j in xrange(n)])
        ret.append(xd)
    return ret

# Plot the results


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


pklf = open("urchin.pkl", "rb")
beta = pickle.load(pklf)


n = len(beta)
s = [0.2]*n
gamma = [0.9]*n
theta = [1.5]*n
m = 5
inic = [1]*n

def run_sim(adjmat, filename):
    # params set
    global beta, n, s, gamma, theta, m, inic
    beta = adjmat
    n = len(beta)
    s = [0.2]*n
    gamma = [0.9]*n
    theta = [1.5]*n
    m = 5
    inic = [1]*n
    # sim set
    beta *= 2
    t = linspace(0, 5., 5)
    res = odeint(f, inic, t)
    plot(res, t, filename)
    return res



def main():
    net = parse.get_net(gvd = "annotated_sea_urchin_net.gvd")
    global beta
    # Times to evaluate the ODEs. 800 times from 0 to 100 (inclusive).
    t = linspace(0, 10.,500)
    # Define the initial conditions for each of the four ODEs
    # Compute the ODE
    beta *= 2
    res = odeint(f, inic, t)
    plot(res, t, "/home/tal/Desktop/web/urchin.png")
    urchingraph = Graph.Adjacency(beta.tolist())
    deg = urchingraph.degree()
    # get node names with final value
    val = dict()
    for i in xrange(n):
        val[net[i]['name']]  = res[-1,i]
    vals = sorted(val.iteritems(), key=operator.itemgetter(1))
    for i in vals:
        print i
    #beta = 2*np.matrix(Graph.Barabasi(73, 160, directed=True, outpref=True).get_adjacency().data)
    #beta = 2*np.matrix(Graph.Degree_Sequence(deg).get_adjacency().data)
    #res = odeint(f, inic, t)
    #plot(res, t, "/home/tal/Desktop/web/urchinsim.png")


    
if __name__ == '__main__':
    main()
