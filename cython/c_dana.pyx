# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX


# ---------------------------------------------------------------- Function ---
cdef class Function:
    cdef double call(self, double x) except *:
        return x


# --- Identity ---
cdef class Identity(Function):

    cdef double call(self, double x) except *:
        if x < 0.0: return 0.0
        return x


# --- Clamp ---
cdef class Clamp(Function):
    cdef public double min, max

    def __init__(self, double min=0, double max=1e9):
        self.min = min
        self.max = max

    cdef double call(self, double x) except *:
        if x < self.min: return self.min
        if x > self.max: return self.max
        return x


# --- Noise ---
cdef class UniformNoise(Function):
    cdef public double amount

    def __init__(self, double amount):
        self.amount = amount

    cdef double call(self, double x) except *:
        return x + self.amount*(rand()/float(RAND_MAX) - 0.5)


# --- Sigmoid ---
cdef class Sigmoid(Function):
    cdef public double Vmin, Vmax, Vh, Vc

    def __init__(self, Vmin=0.0, Vmax=20.0, Vh=16., Vc=3.0):
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Vh = Vh
        self.Vc = Vc

    cdef double call(self, double V) except *:
        return self.Vmin + (self.Vmax-self.Vmin)/(1.0+exp((self.Vh-V)/self.Vc))



# ------------------------------------------------------------------- Group ---
# Python group type (dtype)
dtype = [("V",  float),
         ("U",  float),
         ("Isyn",  float),
         ("Iext", float)]

# C group type (ctype)
cdef packed struct ctype:
    np.float64_t V
    np.float64_t U
    np.float64_t Isyn
    np.float64_t Iext


cdef class Group:
    """  """

    cdef double      _tau
    cdef double      _rest
    cdef double      _noise
    cdef double      _delta
    cdef ctype[:]    _units
    cdef Function    _activation
    cdef int         _history_index
    cdef double[:,:] _history

    def __init__(self, units, tau=0.01, rest=0.0, noise=0.0, activation = Identity()):
        self._tau = tau
        self._rest = rest
        self._noise = noise
        self._units = units
        self._delta = 0
        self._activation = activation
        self._history_index = 0
        self._history = np.zeros((1000000000, len(self._units)))

    property history:
        """ Activity history (firing rate) """
        def __get__(self):
            return np.asarray(self._history)

    property delta:
        """ Difference of activity between the first two maximum activites """
        def __get__(self):
            return self._delta

    property tau:
        """ Membrane time constant """
        def __get__(self):
            return self._tau
        def __set__(self, value):
            self._tau = value

    property rest:
        """ Membrane resting potential """
        def __get__(self):
            return self._rest
        def __set__(self, value):
            self._rest = value

    property V:
        """ Firing rate """
        def __get__(self):
            return np.asarray(self._units)["V"]
        def __set__(self, value):
            np.asarray(self._units)["V"] = value

    property U:
        """ Membrane potential """
        def __get__(self):
            return np.asarray(self._units)["U"]
        def __set__(self, value):
            np.asarray(self._units)["U"] = value

    property Isyn:
        """ Input current from external synapses """
        def __get__(self):
            return np.asarray(self._units)["Isyn"]
        def __set__(self, value):
            np.asarray(self._units)["Isyn"] = value

    property Iext:
        """ Input current from external sources """
        def __get__(self):
            return np.asarray(self._units)["Iext"]
        def __set__(self, value):
            np.asarray(self._units)["Iext"] = value

    def evaluate(self, double dt):
        """ Compute activities (Forward Euler method) """

        cdef int i
        cdef noise
        cdef ctype * unit
        cdef double max1=0, max2=0

        for i in range(len(self._units)):
            unit = & self._units[i]
            # Compute white noise
            noise = self._noise*(rand()/float(RAND_MAX)) - self._noise/2.0
            # Update membrane potential
            unit.U += dt/self._tau*(-unit.U + unit.Isyn + unit.Iext - self._rest )
            # Update firing rate
            # unit.V = self._activation.call(unit.U + noise)
            unit.V = self._activation.call(unit.U *(1 + noise))
            # Store firing rate activity
            self._history[self._history_index,i] = unit.V

            # Here we record the max activities to store their difference
            # This is used later to decide if a motor decision has been made
            if unit.V > max1:   max1 = unit.V
            elif unit.V > max2: max2 = unit.V

        self._delta = max1 - max2
        self._history_index +=1


    def reset(self):
        """ Reset all activities and history index """

        cdef int i
        self._history_index = 0
        for i in range(len(self._units)):
            self._units[i].V = 0
            self._units[i].U = 0
            self._units[i].Isyn = 0
            self._units[i].Iext = 0


    def __getitem__(self, key):
        return np.asarray(self._units)[key]


    def __setitem__(self, key, value):
        np.asarray(self._units)[key] = value



# --------------------------------------------------------------- Structure ---
cdef class Structure:
    cdef Group _mot
    cdef Group _cog
    cdef Group _smath1
    cdef Group _smath2

    def __init__(self, tau=0.01, rest=0, noise=0, activation=Identity(), n=4, n_sma=17):
        self._mot = Group(np.zeros(n,dtype=dtype), tau=tau, rest=rest,
                           noise=noise, activation=activation)
        self._cog = Group(np.zeros(n,dtype=dtype), tau=tau, rest=rest,
                         noise=noise, activation=activation)
        self._smath1 = Group(np.zeros(n_sma,dtype=dtype), tau=tau, rest=rest,
                         noise=noise, activation=activation)
        self._smath2 = Group(np.zeros(n_sma,dtype=dtype), tau=tau, rest=rest,
                         noise=noise, activation=activation)

    property mot:
        """ The motor group """
        def __get__(self):
            return self._mot

    property cog:
        """ The cognitive group """
        def __get__(self):
            return self._cog

    property smath1:
        """ The cognitive group """
        def __get__(self):
            return self._smath1

    property smath2:
        """ The cognitive group """
        def __get__(self):
            return self._smath2

    def evaluate(self, double dt):
        self._mot.evaluate(dt)
        self._cog.evaluate(dt)
        self._smath1.evaluate(dt)
        self._smath2.evaluate(dt)

    def reset(self):
        self._mot.reset()
        self._cog.reset()
        self._smath1.reset()
        self._smath2.reset()

# ---------------------------------------------------- AssociativeStructure ---
cdef class AssociativeStructure(Structure):
    cdef public Group _ass

    def __init__(self, tau=0.01, rest=0, noise=0, activation=Identity()):
        Structure.__init__(self, tau, rest, noise, activation)
        self._ass = Group(np.zeros(16,dtype=dtype), tau=tau, rest=rest,
                          noise=noise, activation=activation)

    def evaluate(self, double dt):
        Structure.evaluate(self, dt)
        self._ass.evaluate(dt)

    def reset(self):
        Structure.reset(self)
        self._ass.reset()

    property ass:
        """ The associative group """
        def __get__(self):
            return self._ass

# --------------------------------------------------------------- ArmStructure ---
cdef class ArmStructure:
    cdef Group _theta1
    cdef Group _theta2

    def __init__(self, tau=0.01, rest=0, noise=0, activation=Identity(), n = 17):
        self._theta1 = Group(np.zeros(n,dtype=dtype), tau=tau, rest=rest,
                           noise=noise, activation=activation)
        self._theta2 = Group(np.zeros(n,dtype=dtype), tau=tau, rest=rest,
                         noise=noise, activation=activation)

    property theta1:
        """ The motor group """
        def __get__(self):
            return self._theta1

    property theta2:
        """ The cognitive group """
        def __get__(self):
            return self._theta2

    def evaluate(self, double dt):
        self._theta1.evaluate(dt)
        self._theta2.evaluate(dt)

    def reset(self):
        self._theta1.reset()
        self._theta2.reset()




# ------------------------------------------------------------- Connections ---
cdef class Connection:
    cdef int       _active
    cdef double[:] _source
    cdef double[:] _target
    cdef double[:] _weights
    cdef double    _gain
    cdef int       _n
    cdef int       _narm
    cdef int       _nsma

    def __init__(self, source, target, weights, gain=1.0, n=4, narm=9, nsma=17):
        self._active = True
        self._gain = gain
        self._source = source
        self._target = target
        self._weights = weights
        self._n         = n
        self._narm      = narm
        self._nsma      = nsma

    def flush(self):
        cdef int i
        for i in range(self._target.shape[0]):
            self._target[i] = 0.0

    property gain:
        """Gain of the connection"""
        def __get__(self):
            return self._gain
        def __set__(self, value):
            self._gain = value

    property n:
        """Gain of the connection"""
        def __get__(self):
            return self._n
        def __set__(self, value):
            self._n = value

    property narm:
        """Gain of the connection"""
        def __get__(self):
            return self._narm
        def __set__(self, value):
            self._narm = value

    property nsma:
        """Gain of the connection"""
        def __get__(self):
            return self._nsma
        def __set__(self, value):
            self._nsma = value

    property active:
        """ Whether connection is active """
        def __get__(self):
            return self._active
        def __set__(self, value):
            self._active = value

    property source:
        """Source of the connection """
        def __get__(self):
            return np.asarray(self._source)

    property target:
        """Target of the connection (numpy array)"""
        def __get__(self):
            return np.asarray(self._target)

    property weights:
        """Weights matrix (numpy array)"""
        def __get__(self):
            return np.asarray(self._weights)
        def __set__(self, weights):
            self._weights = weights


# --- OneToOne---
cdef class OneToOne(Connection):
    def propagate(self):
        cdef int i
        if not self._active: return

        for i in range(self._target.shape[0]):
            self._target[i] += self._source[i] * self._weights[i] * self._gain


# --- OneToAll ---
cdef class OneToAll(Connection):
    def propagate(self):
        cdef int i,j
        if not self._active: return

        for i in range(4):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(4):
                self._target[j] += v

# --- AllToAll ---
#cdef class AllToAll(Connection):
#    def propagate(self):
#        cdef int i,j
#        cdef int s_size = self._source.shape[0]
#        cdef int t_size = self._target.shape[0]
#        if not self._active: return
#
#        for i in range(s_size):
#            v = 0
#            for j in range(t_size):
#                v += self._source[j] * self._weights[i+j*s_size]
#            self._target[i] += v * self._gain

# --- AllToAll ---
cdef class AllToAll(Connection):
    def propagate(self):
        cdef int i,j
        cdef int s_size = self._source.shape[0]
        cdef int t_size = self._target.shape[0]
        if not self._active: return

        for i in range(t_size):
            v = 0
            for j in range(s_size):
                v += self._source[j] * self._weights[i*t_size+j]
            self._target[i] += v * self._gain


# --- AssToMot ---
cdef class AssToMot(Connection):
    def propagate(self):
        cdef int i,j
        if not self._active: return

        for i in range(4):
            v = 0
            for j in range(4):
                v += self._source[i+j*4] * self._weights[i]
            self._target[i] += v * self._gain

# --- AssToCog ---
cdef class AssToCog(Connection):
    def propagate(self):
        cdef int i,j
        if not self._active: return

        for i in range(4):
            v = 0
            for j in range(4):
                v += self._source[j+i*4] * self._weights[i]
            self._target[i] += v * self._gain


# --- MotToAss ---
cdef class MotToAss(Connection):
    def propagate(self):
        cdef int i,j
        cdef double v
        if not self._active: return

        for i in range(4):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(4):
                self._target[j*4+i] += v

# --- CogToAss ---
cdef class CogToAss(Connection):
    def propagate(self):
        cdef int i,j
        cdef double v
        if not self._active: return

        for i in range(4):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(4):
                self._target[i*4+j] += v

# --- MotToPPC ---
cdef class MotToPPC(Connection):
    def propagate(self):
        cdef int i,j
        cdef double v
        if not self._active: return

        for i in range(self._n):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(self._narm):
                self._target[j*self._n+i] += v

# --- ARMtoPPC ---
cdef class ARMtoPPC(Connection):
    def propagate(self):
        cdef int i,j
        cdef double v
        if not self._active: return

        for i in range(self._narm ):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(self._n):
                self._target[i*self._n+j] += v

# --- SMAtoM1 ---
cdef class SMAtoM1(Connection):
    def propagate(self):
        cdef int i,j
        cdef double v
        if not self._active: return

        for i in range(self._nsma):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(self._narm):
                self._target[j*self._nsma+i] += v


# --- ARMtoM1 ---
cdef class ARMtoM1(Connection):
    def propagate(self):
        cdef int i,j
        cdef double v
        if not self._active: return

        for i in range(self._narm ):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(self._nsma):
                self._target[i*self._nsma+j] += v

# --- M1toARM ---
cdef class M1toARM(Connection):
    def propagate(self):
        cdef int i,j,k
        cdef double v
        if not self._active: return

        for k in range(self._narm):
            v = 0

            for i in range(self._narm):
                for j in range(self._nsma):
                   v += self._source[i*self._nsma+j] * self._weights[k*self._narm*self._nsma+i*self._nsma+j]

            self._target[k] += v * self._gain

# --- PPCtoSMA ---
cdef class PPCtoSMA(Connection):
    def propagate(self):
        cdef int i,j,k
        cdef double v
        if not self._active: return

        for k in range(self._nsma):
            v = 0

            for i in range(self._narm):
                for j in range(self._n):
                   v += self._source[i*self._n+j] * self._weights[k*self._narm*self._n+i*self._n+j]

            self._target[k] += v * self._gain

# --- PPCtoSTR ---
cdef class PPCtoSTR(Connection):
    def propagate(self):
        cdef int i,k
        if not self._active: return

        for k in range(self._nsma):
            for i in range(self._narm*self._n):
                self._target[k*self._narm*self._n + i] += self._source[i] * self._weights[k*self._narm*self._n +i] * self._gain


# --- SMAtoSTR ---
cdef class SMAtoSTR(Connection):
    def propagate(self):
        cdef int i,k
        if not self._active: return

        for k in range(self._nsma):
            for i in range(self._narm*self._n):
                   self._target[k*self._narm*self._n + i] += self._source[k] * self._weights[k*self._narm*self._n + i] * self._gain


# --- STRsmaToBG ---
cdef class STRsmaToBG(Connection):
    def propagate(self):
        cdef int i,j,k
        cdef double v
        if not self._active: return

        for k in range(self._nsma):
            v = 0

            for i in range(self._narm*self._n):
                   v += self._source[k*self._narm*self._n+i] * self._weights[k*self._narm*self._n+i]

            self._target[k] += v * self._gain