from numpy import sqrt, linspace
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from pylab import plot, axis, show, savefig
import matplotlib.pyplot as plt
from math import pow
import random
from igraph import *
import sys

from urchinsim import *

# build network
# build child
# merge

def rewire(g):
    # with prob q, swap edge
    edgelist = g.get_edgelist()
    (i,j) = a = random.choice(edgelist)
    (k,l) = b = random.choice(edgelist)
    g.delete_edges([a,b])
    g.add_edges([(i,l), (k,j)])

def hybrid(a, b, r=0.5):
    n = len(a.vs)
    m = len(a.es)
    h = Graph(n=n, directed=True)
    ae = set(a.get_edgelist())
    be = set(b.get_edgelist())
    # add intercection
    inter = list(ae & be)
    symdiff = list(ae ^ be)
    h.add_edges(inter)
    samplesize = min(m - len(inter), len(symdiff))
    sample = random.sample(symdiff, int(len(symdiff)*r)) 
    h.add_edges(list(symdiff))
    return h

def vmax(a,b):
    return np.array([max(a[i],b[i]) for i in xrange(len(a))])

def run_env(A, B, H, signs):
    beta_H = np.matrix(H.get_adjacency().data)
    beta_A = np.matrix(A.get_adjacency().data)
    beta_B = np.matrix(B.get_adjacency().data)
    # set signs
    for i in xrange(len(A.vs)):
        for j in xrange(len(A.vs)):
            beta_H[i,j]*= signs[i,j]
            beta_A[i,j]*= signs[i,j]
            beta_B[i,j]*= signs[i,j]
    resH = run_sim(beta_H, "plots/hybrid.png")[-1]
    resA = run_sim(beta_A, "plots/parentA.png")[-1]
    resB = run_sim(beta_B, "plots/parentB.png")[-1]
    wiggle = 0.25
    diff_A_B = sum((resA - resB) > wiggle*vmax(resA, resB))
    diff_H_A = sum((resH - resA) > wiggle*vmax(resH, resA))
    diff_H_B = sum((resH - resB) > wiggle*vmax(resH, resB))
    return [diff_A_B, diff_H_A, diff_H_B]

def sign(x):
    return 1 if x > 0.5 else 0

def main():
    # diff series
    diff_AB = []
    diff_HA = []
    diff_HB = []
    # sea urchin
    pklf = open("urchin.pkl", "rb")
    beta = pickle.load(pklf)
    pklf = open("urchin_signs.pkl", 'rb')
    signs = pickle.load(pklf)
    # random signs
    #signs = np.array(map(sign, np.random.rand(73,73).ravel())).reshape(73,73)
    urchingraph = Graph.Adjacency(beta.tolist())    
    # get root network
    # set G to er
    #G = Graph.Erdos_Renyi(n, p)
    # set G to sea urchin network
    # set G to random graph with sequence like urchin
    indegree = urchingraph.degree(type='in')
    outdegree = urchingraph.degree(type='out')
    aa = len(indegree)
    #G = Graph.Degree_Sequence(random.sample(outdegree, aa), random.sample(indegree,aa))
    G = urchingraph
    A = G.copy()
    B = G.copy()
    # rewire
    n = len(G.vs)
    r = 20
    print "edges", len(G.es)
    for i in xrange(r):
        H = hybrid(A, B, float(i)/(r-1))
        for j in xrange(50):
            rewire(B)
        
        (resAB, resHA, resHB) = run_env(A, B, H, signs)
        
        # update diff series
        diff_AB.append(resAB)
        diff_HA.append(resHA)
        diff_HB.append(resHB)
    if len(sys.argv)>1:
        # save to file
        f = open(sys.argv[1], 'w')
        f.write(','.join(map(str, diff_AB)))
        f.write('\n')
        f.write(','.join(map(str, diff_HA)))
        f.write('\n')
        f.write(','.join(map(str, diff_HB)))
        f.close()
    else:
        print diff_AB
        print diff_HA
        print diff_HB




if __name__ == '__main__':
    main()
