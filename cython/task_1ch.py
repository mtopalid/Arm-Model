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
        n = (n // 81) * 81
        self.build(n)

        # All combinations of cues or positions
        Z = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
                      [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                      [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
                      [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
                      [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                      [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
                      [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8],
                      [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8],
                      [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8]])

        # n//81 x all combinations of positions
        M = np.repeat(np.arange(81), n // 81)
        np.random.shuffle(M)
        pos = Z[M]

        # Make sure count is a multiple of 4
        # n = (n // 4) * 4
        # self.build(n)

        # All combinations of cues or positions
        # Z = np.array([0, 1, 2, 3])
        Z = np.array([1, 1, 1, 1])

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
            p = pos[i]
            trial = self.trials[i]

            trial["initial_pos"][:] = p[:]
            trial["cog"][c] += 1
            trial["mot"][m] += 1
            trial["ass"][m, c] += 1
            trial["rwd"][...] = 1.00, 1.00, 1.00, 1.00

    def process(self, trial, action=0, RT=0, debug=False):

        # Only the associative feature can provide (m1,c1) and (m2,c2)
        # i = (trial["ass"].ravel().argsort())[-1]
        # m, c = np.unravel_index(i, (4, 4))
        #
        # move = self.records[self.index]["move"]
        # reward = np.random.uniform(0, 1) < trial["rwd"][c]
        # self.records[self.index]["shape"] = c

        if debug:
            print "Trial %d" % (self.index + 1)
        # print "  Action                : %d " % action
        # if m == move:
        #     best = True
        # else:
        #     best = False
        # Record action, best action (was it the best action), reward and RT
        self.records[self.index]["action"] = action
        # self.records[self.index]["best"] = best
        self.records[self.index]["RTmove"] = RT - self.records[self.index]["RTmot"]
        # self.records[self.index]["reward"] = reward

        if debug:
            print "  Initial position      :", self.trials[self.index]["initial_pos"]
            print "  Final position        :", self.records[self.index]["final_pos"]
            print "  Target position       :", self.records[self.index]["target_pos"]
            print "  Number of moves       :", self.records[self.index]["moves"]
            # print "  Move			        : %d" % (m)
            # print "  Choice			    : %d" % (c)
            print "  Reward     		    : %.2f" % (self.records[self.index]["reward"])
            print "  Performance		    : %d" % self.records[self.index]["best"]
            # print "  Reward (p=%.2f)		: %d" % (trial["rwd"][c], reward)
            rt = self.records[self.index]["RTmove"]
            print "  Response time	move    : %.3f ms" % np.array(rt)
            # print

        if 0:  # debug:
            P = self.records[:self.index + 1]["best"]
            print "  Mean performance		: %.1f %%" % np.array(P * 100).mean()
            R = self.records[:self.index + 1]["reward"]
            print "  Mean reward			: %.3f" % np.array(R).mean()
            # n_moves = self.records[self.index]["moves"]
            # print "  Number of moves       : %d" % n_moves
            # print "  Mean Response time move: %.3f ms" % np.array(rt).mean()
            # rt = self.records[:self.index + 1]["RTmot"]
            # print "  Mean Response time	mot: %.3f ms" % np.array(rt).mean()

            # return reward, best
