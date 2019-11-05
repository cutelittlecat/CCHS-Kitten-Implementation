#!/usr/bin/env python

"""
    Copyright (c) 2013, Triad National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Triad National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from pycopycat import ObjectiveFunctionInterface, harmony_search_copycat
from math import pow
import random
import numpy as np
from scipy import signal
from scipy import linalg
#from scipy import optimize
import pandas as pd


class ObjectiveFunction(ObjectiveFunctionInterface):

    def __init__(self):
        # define all input parameters
        self._maximize = False  # do we maximize or minimize?
        self._max_imp = 10000  # maximum number of improvisations
        self._hms = 10  # harmony memory size
        self._hmcr = 0.99  # harmony memory considering rate
        self._par = 0.3  # pitch adjusting rate
        self._mpap = 0.01  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        self._random_seed = 8675309  # optional random seed for reproducible results
        # Copycat algorithm options
        self._fixed_par = False  # False if a linearly increasing par is used
        self._parmin = 0  # min pitch adjusting rate
        self._parmax = 1  # max pitch adjusting rate
        self._dynamic_bw = True  # True if BW is dynamically adjusted (only continuous variables)
        self._ngh = 3  # top harmonies to consider for copycat (mixing the best harmonies)
        self._max_iters_best = 40  # maximum number of iterations without changing the best harmony
        self._max_iters_worst = 20  # maximum number of iterations without changing the worst harmony
        
        # Variable definition
        self._lower_bounds = [0]*3
        self._upper_bounds = [10000]*3
        self._variable = [True]*3
        # Parameters for function definition (curve fitting)
        df = pd.read_csv('data.csv')
        self.t = df[['t']].values[:,0]
        self.u = df[['u1', 'u2']].values
        self.zTarget = df[['zT1', 'zT2']].values
        self.dt = self.t[1] - self.t[0]
        

    def get_fitness(self, vector):
        #return sum([x*x for x in vector])
        # Parameters for function definition (curve fitting)
        m1 = 20.0
        m2 = 10.0
        k1 = 2e6
        k2 = 1e6
        k3 = 5e5
        # function calculation
        c1 = vector[0]
        c2 = vector[1]
        c3 = vector[2]
        Mvib = np.asarray([[m1, 0.0], [0.0, m2]], dtype = float)
        Cvib = np.asarray([[c1+c2, -c2], [-c2, c2+c3]], dtype = float)
        Kvib = np.asarray([[k1+k2, -k2], [-k2, k2+k3]], dtype = float)
        n = Mvib.shape[0]
        I = np.eye(n)
        Z = np.zeros([n,n])
        Minv = linalg.pinv(Mvib)
        negMinvK = - np.matmul(Minv, Kvib)
        negMinvC = - np.matmul(Minv, Cvib)
        Ac = np.hstack((np.vstack((Z,negMinvK)), np.vstack((I,negMinvC))))
        Bc = np.vstack((Z,Minv))
        Cc = np.hstack((I,Z))
        Dc = Z.copy()
        systemC = (Ac, Bc, Cc, Dc)
        sD = signal.cont2discrete(systemC, self.dt)
        Ad = sD[0]
        Bd = sD[1]
        Cd = sD[2]
        Dd = sD[3]
        systemD = (Ad, Bd, Cd, Dd, self.dt)
        x0 = np.zeros((Ad.shape[1],), dtype = 'float32')
        output = signal.dlsim(systemD, u = self.u, t = self.t, x0 = x0)
        zScipy = output[1]
        #zErr is the objective function
        zErr = (np.sum(np.sqrt((self.zTarget-zScipy)**2)))/2002
        return zErr

    def get_value(self, i, index=None):
        """
            Values are returned uniformly at random in their entire range. Since both parameters are continuous, index can be ignored.
        """
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        # all variables are continuous
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_random_seed(self):
        return self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self, imp):
        if self._fixed_par:
            return self._par
        return (self._parmin + (self._parmax-self._parmin) / self._max_imp * imp)

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize

if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    num_processes = 1
    num_iterations = 1  # because random_seed is defined, there's no point in running this multiple times
    results = harmony_search_copycat(obj_fun, num_processes, num_iterations)
    print('Elapsed time: {}\n\nBest harmony: {}\n\n*****************\nBest fitness: {}'.format(results.elapsed_time, results.best_harmony, results.best_fitness))
