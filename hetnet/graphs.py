import random
from igraph import *
from utils import weighted_choice
import numpy as np


def ModGraph(n, k):
    G = Graph(n=n, directed=True)
    clusters = [[] for i in xrange(k)]
    cluster_index = []
    # add vertex to cluster
    for i in xrange(k):
        clusters[i] += [i]
        cluster_index.append(i)
    for i in xrange(k,n):
        choice = random.choice(range(k))
        clusters[choice] += [i]
        cluster_index.append(choice)
    # within clusters, set dense connections
    for i in xrange(k):
        clus = clusters[i]
        numc = len(clus)
        m = int((1)*numc*(numc-1)/4)
        for j in xrange(m):
            u, v = random.choice(clus), random.choice(clus)
            G.add_edges((u,v))
    # between clusters, set sparse connections
    m = int((1)*n)
    for i in xrange(m):
        c1, c2 = random.choice(clusters), random.choice(clusters)
        u, v = random.choice(c1), random.choice(c2)
        if (u,v) not in G.es:
            G.add_edges((u,v))
    return G, cluster_index


def run(n, k):
    g, cluster_index = ModGraph(n, k)
    vclus = g.community_edge_betweenness().as_clustering()
    return g.modularity(cluster_index)



def BATFGraph(m0 = 10, m = 1):
    # initialize graph with m0 vertices
    g = Graph(n = m0, directed=True)
    for v in xrange(m0):
        # PA step
        weights = g.vs.degree()
        u = weighted_choice(weights)
        g.add_edges((v,u))
        # TF step
        neighbors_u = g.neighbors(u)
        if neighbors_u:
            w = random.choice(neighbors_u)
            g.add_edges((v,w))
    return g


if __name__ == '__main__':
    g = BATFGraph(m0=1000)
    print g.vs.degree()

