from igraph import *
import random
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


import pylab
if __name__ == '__main__':
    n = 100
    res = []
    ran = range(2,n)
    for k in ran:
        r = np.mean([run(n, k) for i in xrange(500)])
        res.append(r)
        print k, r
    pylab.plot(ran, res)
    pylab.xlabel('# of clusters')
    pylab.ylabel('Q-value (Modularity)')
    pylab.title('Q-graph stuff')
    pylab.grid(True)
    pylab.savefig('qgraph.png')
    pylab.show()


