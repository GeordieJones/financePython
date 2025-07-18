#
# monte carlo valuation of european call (pure python)
# mcs_pure_python.py
#

from time import time
from math import exp,sqrt,log
from random import gauss, seed

seed(20000)
t0 = time()

S0 = 100 # initial value
K = 105 # strike price
T = 1.0 # maturity
r = 0.05 # riskless short rate
sigma = 0.2 # volitility
M = 50 # number of steps or times checked
dt = T / M # length of time
I = 250000 # number of paths

# sim of I steps with M steps
S = []
for i in range(I):
    path = []
    for t in range(M+1):
        if t==0:
            path.append(S0)
        else:
            z = gauss(0.0,1.0)
            St = path[t-1]* exp((r -0.5 * sigma**2)*dt + sigma * sqrt(dt) *z)
            path.append(St)
    S.append(path)

#calc of monte carlo simulator
C0 = exp(-r * T) * sum([max(path[-1]-K,0)for path in S]) / I

#results
tpy = time() - t0
print ("European call options val: %7.3f" % C0)
print ("Duration in sec: %7.3f" % tpy)