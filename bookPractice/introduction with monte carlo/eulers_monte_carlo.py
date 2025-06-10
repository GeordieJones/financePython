#
# using logarithm in order to calculate the option
# eulers_monte_carlo.py
#

import math
from numpy import *
from time import time

random.seed(20000)
t0 = time()

#same perameters as the other monte carlo simulators
S0 = 100.; K = 105.; T= 1.0; r= 0.05; sigma = 0.2; M =50
dt = T/M; I =250000

#sim I paths m times
# if you only care about final values you can use sum instead of cumsum
S= S0 * exp(cumsum((r - 0.5 * sigma **2)*dt + sigma * math.sqrt(dt) * random.standard_normal((M+1, I)), axis = 0))

S[0] = S0

C0 = math.exp(-r * T) * sum(maximum(S[-1] - K, 0)) / I

tnp2 = time() -t0
print("valuation with eulars numpy: %7.3f" %C0)
print("time with eulars numpy: %7.3f" %tnp2)