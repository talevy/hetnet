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
from parse_urchin_net import *

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

def hybrid(a, b, q=0.5):
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
    sample = random.sample(symdiff, samplesize) 
    h.add_edges(list(symdiff))
    return h

def vmax(a,b):
    return np.array([max(a[i],b[i]) for i in xrange(len(a))])

def run_env(A, B, H, signs):
    beta_H = np.matrix(H.get_adjacency().data)
    beta_A = np.matrix(A.get_adjacency().data)
    beta_B = np.matrix(B.get_adjacency().data)
    num_edges = float(len(H.es))
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
    diff_A_B = (resA - resB) > wiggle*vmax(resA, resB)
    diff_B_A = (resB - resA) > wiggle*vmax(resA, resB)
    diff_H_A = (resH - resA) > wiggle*vmax(resH, resA)
    diff_A_H = (resA - resH) > wiggle*vmax(resH, resA)
    diff_H_B = (resH - resB) > wiggle*vmax(resH, resB)
    diff_B_H = (resB - resH) > wiggle*vmax(resH, resB)
    return [sum(diff_A_B), sum(diff_B_A), sum(diff_H_A), sum(diff_A_H), sum(diff_H_B), sum(diff_B_H)]

def sign(x):
    return 1 if x > 0.5 else 0


def remove_edges(g, n = 1, cluster=None):
    ''' '''
    if cluster:
        pass
    else:
        for i in xrange(n):
            edgelist = g.get_edgelist()
            if edgelist:
                a = random.choice(edgelist)
                g.delete_edges([a])
            else:
                return

def run_attack_sim(g):
    runs = 12
    beta = np.matrix(g.get_adjacency().data)
    for i in xrange(runs):
        remove_edges(g, n=3)
        res = run_sim(beta, "plots/attack%i.png" % i)[-1]
        print res

def run_attack_diff_sim(G):
    pklf = open("urchin_signs.pkl", 'rb')
    signs = pickle.load(pklf)
   
    A = G.copy()
    B = G.copy()
    for j in xrange(20):
        rewire(B)
    H = hybrid(A, B)
    res = (resAB, resBA, resHA, resAH, resHB, resBH) = run_env(A, B, H, signs)
    return res

def run_many(G, num_changes = 10):
    # diff series
    diff_AB = []
    diff_BA = []
    diff_HA = []
    diff_AH = []
    diff_HB = []
    diff_BH = []
    for i in xrange(num_changes):
        (resAB, resBA, resHA, resAH, resHB, resBH) = run_attack_diff_sim(G)
        diff_AB.append(resAB)
        diff_BA.append(resBA)
        diff_HA.append(resHA)
        diff_AH.append(resAH)
        diff_HB.append(resHB)
        diff_BH.append(resBH)
        remove_edges(G, n=3)
    return (diff_AB, diff_BA, diff_HA, diff_AH, diff_HB, diff_BH)

def plotdiff(x, y, filepath):
    ''' run through set of files and average them '''
    plt.figure()
    labels = ['A > B','B > A', 'H > A', 'A > H',  'H > B', 'B > H']
    for i in xrange(len(y)):
        plt.plot(x,y[i], label=(labels[i]))
    plt.xlabel('delta edges')
    plt.ylabel('# genes differentially expressed')
    plt.title('Model')
    plt.legend(loc=0)
    savefig(filepath)

def main():
    # sea urchin
    pklf = open("urchin.pkl", "rb")
    beta = pickle.load(pklf)
    pklf = open("urchin_signs.pkl", 'rb')
    signs = pickle.load(pklf)
    urchingraph = Graph.Adjacency(beta.tolist())    

    indegree = urchingraph.degree(type='in')
    outdegree = urchingraph.degree(type='out')
    aa = len(indegree)
    #G = Graph.Degree_Sequence(random.sample(outdegree, aa), random.sample(indegree,aa))

    G = urchingraph
    # add cluster info
    net = get_net(gvd = "annotated_sea_urchin_net.gvd")
    for i, node in enumerate(net):
        G.vs[i]['clusters'] = node['clusters']

    # attack fraction of edges in G (3 at a time)
    #run_attack_sim(G)

    # run differential comparision with attacks
    (diffAB, diffBA, diffHA, diffAH, diffHB, diffBH) = y = run_many(G, num_changes=10)
    if len(sys.argv)>1:
        # save to file
        f = open(sys.argv[1], 'w')
        f.write(','.join(map(str, diffAB)))
        f.write('\n')
        f.write(','.join(map(str, diffBA)))
        f.write('\n')
        f.write(','.join(map(str, diffHA)))
        f.write('\n')
        f.write(','.join(map(str, diffAH)))
        f.write('\n')
        f.write(','.join(map(str, diffHB)))
        f.write('\n')
        f.write(','.join(map(str, diffBH)))
        f.close()
    else:
        print diffAB
        print diffBA
        print diffHA
        print diffAH
        print diffHB
        print diffBH

if __name__ == '__main__':
    main()
