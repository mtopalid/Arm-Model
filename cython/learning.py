# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------

import numpy as np
import random
from trial import *
from parameters import *
import sys


def learning(task, ncues = 2, trial_n=0, learn=True, debugging=True, debugging_arm=False, debugging_learning = False, duration=duration):
    trial(task, ncues=ncues, trial_n=trial_n, learn=learn, debugging=debugging, debugging_arm=debugging_arm, duration=duration)

    return


def learning_trials(task, ncues = 2, trials=n_trials, learn=True, debugging=True, debug_simulation=False,
                    debugging_arm=False, debugging_arm_learning = False, debugging_learning = False, duration=duration):
    if debug_simulation:
        steps = trials / 10
        print '  Starting   ',

    for i in range(trials):

        learning(task, ncues = ncues, trial_n=i, learn=learn, debugging=debugging, debugging_arm=debugging_arm, duration=duration)

        if debug_simulation:
            if i % steps == 0:
                print '\b.',
                sys.stdout.flush()

    if debug_simulation:
        print '   Done!'
    if debugging_arm_learning:
        debug_arm_learning()

    return
