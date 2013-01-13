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
    labels = ['A > B', 'H > A', 'H > B']
    for i in xrange(len(y)):
        plt.plot(x,y[i], label=(labels[i]))
    plt.xlabel('delta ratio')
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
    plotdiff(range(len(diffavg[0])), diffavg,'web/urchinratiodiff.png')


if __name__=='__main__':
    main()

