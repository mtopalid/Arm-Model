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


def learning_trials_single(task, ncues=2, trials=n_trials, learn=True, debugging=True, debug_simulation=False,
                           debugging_arm=False, debugging_arm_learning=False,
                           duration=duration, folder='', tr=0, save=False):
    if debug_simulation:
        steps = trials / 10
        print '  Starting   ',

    for i in range(tr, trials):

        trial(task, ncues=ncues, trial_n=i, learn=learn, debugging=debugging, debugging_arm=debugging_arm,
              duration=duration)

        if save:
            f = folder + '/Cues.npy'
            np.save(f, task.trials[i])
            f = folder + '/Records.npy'
            np.save(f, task.records[i])
            f = folder + '/Trial_Number.npy'
            np.save(f, i)

            if i % 1000 == 0:
                f = folder + '/Cues1000s.npy'
                np.save(f, task.trials[:i])
                f = folder + '/Records1000s.npy'
                np.save(f, task.records[:i])
                f = folder + '/Trial_Number1000s.npy'
                np.save(f, i)

        if debug_simulation:
            if i % steps == 0:
                print '\b.',
                sys.stdout.flush()
    if save:
        f = folder + '/RecordsALL.npy'
        np.save(f, task.records[:i])
    if debug_simulation:
        print '   Done!'
    if debugging_arm_learning:
        debug_arm_learning()

    return


def learning_trials_continuous(task, ncues=2, trials=n_trials, learn=True, debugging=True, debug_simulation=False,
                               debugging_arm=False, debugging_arm_learning=False, folder="", duration=duration, save=False):
    if debug_simulation:
        steps = trials / 10
        print '  Starting   ',

    for i in range(trials):

        trial_continuous(task, ncues=ncues, trial_n=i, learn=learn, debugging=debugging, debugging_arm=debugging_arm,
                         duration=duration)

        if save:
            f = folder + '/Cues.npy'
            np.save(f, task.trials[i])
            f = folder + '/Records.npy'
            np.save(f, task.records[i])
            f = folder + '/Trial_Number.npy'
            np.save(f, i)
            if i % 1000 == 0:
                f = folder + '/Cues' + "%03d" % (i + 1) + '.npy'
                np.save(f, task.trials[:i])
                f = folder + '/Records' + "%03d" % (i + 1) + '.npy'
                np.save(f, task.records[:i])
                f = folder + '/Trial_Number' + "%03d" % (i + 1) + '.npy'
                np.save(f, i)

        if debug_simulation:
            if i % steps == 0:
                print '\b.',
                sys.stdout.flush()

    if debug_simulation:
        print '   Done!'
    if debugging_arm_learning:
        debug_arm_learning()

    return
