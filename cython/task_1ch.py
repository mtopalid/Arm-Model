# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Nicolas P. Rougier, Meropi Topalidou
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Task A (Guthrie et al. (2013) protocol)
=======================================

Trials:

 - n trials with random uniform sampling of cues and positions
   Reward probabilities: A=1.00, B=0.33, C=0.66, D=0.00

"""

import numpy as np
from task import Task


class Task_1ch(Task):
    def setup(self, n=180):

        # Make sure count is a multiple of 6
        n = (n // 4) * 4
        self.build(n)

        # All combinations of cues or positions
        Z = np.array([0, 1, 2, 3])
        # Z = np.array([1, 1, 1, 1])

        # n//4 x all combinations of positions
        M = np.repeat(np.arange(4), n // 4)
        np.random.shuffle(M)
        mot = Z[M]
        np.random.shuffle(mot[:])

        # n//4 x all combinations of cues
        C = np.repeat(np.arange(4), n // 4)
        np.random.shuffle(C)
        cog = Z[C]
        np.random.shuffle(cog[:])

        for i in range(n):
            c = cog[i]
            m = mot[i]
            trial = self.trials[i]

            trial["cog"][c] += 1
            trial["mot"][m] += 1
            trial["ass"][m, c] += 1
            trial["rwd"][...] = 1.00, 1.00, 1.00, 1.00

    def process(self, trial, action, RT=0, debug=False):

        # Only the associative feature can provide (m1,c1) and (m2,c2)
        i = (trial["ass"].ravel().argsort())[-1]
        m, c = np.unravel_index(i, (4, 4))

        r = trial["rwd"][c]
        move = self.records[self.index]["move"]

        if debug:
            print "Trial %d" % (self.index + 1)
            print "  Action                : %d " % action
        if m == move:
            reward = np.random.uniform(0, 1) < trial["rwd"][c]
            self.records[self.index]["shape"] = c
            best = True
            if debug:
                print "  Move			        : %d" % (m)
                if best:
                    print "  Choice			    : %d" % (c)
                print "  Reward (p=%.2f)		: %d" % (trial["rwd"][c], reward)

        # Record action, best action (was it the best action), reward and RT
        self.records[self.index]["action"] = action
        self.records[self.index]["best"] = best
        self.records[self.index]["RTmove"] = RT
        self.records[self.index]["reward"] = reward

        if debug:
            P = self.records[:self.index + 1]["best"]
            print "  Mean performance		: %.1f %%" % np.array(P * 100).mean()
            R = self.records[:self.index + 1]["reward"]
            print "  Mean reward			: %.3f" % np.array(R).mean()
            n_moves = self.records[self.index]["moves"]
            print "  Number of moves       : %d" % n_moves
            rt = self.records[self.index]["RTmove"] - self.records[self.index]["RTmot"]
            print "  Response time	move    : %.3f ms" % np.array(rt)
            rt = self.records[:self.index + 1]["RTmove"]
            print "  Mean Response time	move: %.3f ms" % np.array(rt).mean()
            rt = self.records[:self.index + 1]["RTmot"]
            print "  Mean Response time	mot: %.3f ms" % np.array(rt).mean()
            rt = self.records[:self.index + 1]["RTcog"]
            print "  Mean Response time	cog: %.3f ms" % np.array(rt).mean()

        return reward, best
