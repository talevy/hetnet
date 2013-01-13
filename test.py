from hetnet import graphs, dynamics
import numpy as np
from numpy import linspace
from pylab import plot, axis, show, savefig
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.vq import kmeans2 as kmeans

colors= ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta']

def plot(communities, res, t, filepath=None):
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('Concentration')
    plt.title('Model')
    for i, c in enumerate(communities):
        for v in c:
            plt.plot(t, res[:,v], label=str(i), color=colors[i % 7])
    if filepath:
        savefig(filepath)
    show()


def sign(x, mu=0.9):
    return 1 if x > mu else -1



g = graphs.BATFGraph(m0=80)

communities = g.community_edge_betweenness()

beta = np.matrix(g.get_adjacency().data)

n = len(g.vs)
s = [0.9]*n
gamma = [1.2 ]*n
theta = [1.8]*n
m = 5
inic = [3]*n
t = linspace(0, 5., 3)


# set signs
signs = np.array(map(sign, np.random.rand(n,n).ravel())).reshape(n,n)
for i in xrange(n):
    for j in xrange(n):
        beta[i,j] *= signs[i,j]


res = dynamics.runsim(beta, n, s, gamma, theta, m, inic, t)

clusters = g.community_spinglass()
plot(clusters, res, t, 'dyno.png')



from hetnet.video import *
def animate(graph, clusters,  res, outfile):
    '''animate the dynamics on the network'''
    layout = graph.layout(layout='fr')
    encoder = MEncoderVideoEncoder(bbox=(600, 600), fps=2)
    encoder.lavcopts = "vcodec=mpeg4:mbd=2:vbitrate=1600:trell:keyint=2"
    # Generate frames in the animation one by one
    with encoder.encode_to(outfile):
        for i in xrange(len(res)):
            print res[i]
            # Run one step of the layout algorithm
            layout = graph.layout("graphopt", niter=1, seed=layout)
            # Add the clustering to the encoder
            encoder.add(clusters, layout=layout, mark_groups=True, margin=20,
                        vertex_size=[np.math.pow(r, 3) for r in res[i]])



#animate(g, clusters, res, "tess.avi")

