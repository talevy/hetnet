import random
from igraph import *
from utils import weighted_choice

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

