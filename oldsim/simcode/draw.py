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
import glob
from urchinsim import *

def plotdiff(x, y, filepath):
    ''' run through set of files and average them '''
    plt.figure()
    labels = ['A > B', 'B > A', 'H > A', 'A > H', 'H > B', 'B > H']
    for i in xrange(len(y)):
        plt.plot(x,y[i], label=(labels[i]))
    plt.xlabel('number of edges removed')
    plt.ylabel('# genes differentially expressed')
    plt.title('Model')
    plt.legend(loc=0)
    savefig(filepath)

def sanitize(diff):
    return map(int, diff.strip().split(','))


def get_diff(filename):
    f = open(filename, 'r')
    diffs = f.readlines()
    f.close()
    diffs = map(sanitize, diffs)
    return diffs

def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        print "add filename after"
        exit()
    files = glob.glob('%s[0-9]*.txt' % name)
    diffsum = None
    for f in files:
        diffs = np.array(get_diff(f))
        if diffsum != None:
            diffsum += diffs
        else:
            diffsum = diffs
    diffavg = diffsum/float(len(files))
    x = [3*i for i in xrange(len(diffavg[0]))]
    plotdiff(x, diffavg,'urchin-diff.png')


if __name__=='__main__':
    main()

