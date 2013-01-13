from numpy import sqrt, linspace
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from pylab import plot, axis, show, savefig
import matplotlib.pyplot as plt
from math import pow

# Define the initial conditions for each of the four ODEs
inic = [1,1,1]

# Times to evaluate the ODEs. 800 times from 0 to 100 (inclusive).
t = linspace(0, 50.,1000)

# The derivative function.
# dy/dt = func(y, t)
n = 3
s = [0.2]*n
gamma = [1.0,0.7,0.3,1.2,1.5]
theta = [1.5]*n
m = 5
beta = np.matrix([[0,0,-2], [2,0,0], [0,2,0]])

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
    return [
        s[0] - gamma[0]*x[0] + sum([abs(beta[0,j])*hill(x[j],beta[0,j],theta[0]) for j in xrange(n)]),
        s[1] - gamma[1]*x[1] + sum([abs(beta[1,j])*hill(x[j],beta[1,j],theta[1]) for j in xrange(n)]),
        s[2] - gamma[2]*x[2] + sum([abs(beta[2,j])*hill(x[j],beta[2,j],theta[2]) for j in xrange(n)]),
        ]


# Compute the ODE
res = odeint(f, inic, t)
# Plot the results

plt.figure()
plt.plot(t, res[:,0], label="x1")
plt.plot(t, res[:,1], label="x2")
plt.plot(t, res[:,2], label="x3")
plt.xlabel('time')
plt.ylabel('Concentration')
plt.title('Model')
plt.legend(loc=0)

savefig('/home/tal/Desktop/ok.png')
show()
