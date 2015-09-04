# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#				Nicolas Rougier  (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
from model import *
from display import *
from parameters import *


def trial(task, cues_pres=True, learn=True, debugging=False, trial_n=0, wholeFig=False):
    reset_activities()
    reset_history()
    ct = None
    cog_time = None
    time = None
    choice_made = False
    for i in xrange(0, 500):
        iterate(dt)
        if CTX.cog.delta > 20 and not ct:
            ct = 1
        if CTX.cog.delta > threshold and not cog_time:
            cog_time = i - 500
        if i == 200:
            ARM.theta1.Iext[5] = 7
    if cues_pres:
        set_trial(task, n=2, trial=trial_n)
    t=0
    for i in xrange(500, duration):
        iterate(dt)

        # if i == 500:
        #     CTX.pfcth1.Iext[7] = 7
            # SMA.theta1.Iext.reshape((9,17))[8,0] = 7

        posIExt = np.argmax(ARM.theta1.Iext)
        arm = np.argmax(ARM.theta1.V)
        if ARM.theta1.delta > 1. and arm != posIExt:
            # ARM.theta1.Iext = 0
            # ARM.theta1.Iext[pos] = 7
            # CTX.pfcth1.Iext = 0
            # CTX.pfcth1.Iext[8] = 7
            # posIExt = np.argmax(ARM.theta1.Iext)
            arm = np.argmax(ARM.theta1.V)
            ppc = np.argmax(PPC.theta1.V)
            pfc = np.argmax(PFC.theta1.V)
            sma = np.argmax(SMA.theta1.V)
            mot = np.argmax(CTX.mot.V)
            print "Motor CTX: ", mot
            print "PPC: %d, %d" %(ppc/n, ppc%n)
            print "PFC: ", pfc
            print "SMA: %d, %d" %(sma/n_pfc, sma%n_pfc)
            print "Arm: ", arm
            break


        # if i == 1500:
        #     CTX.pfcth1.Iext = 0
        #     CTX.pfcth1.Iext[13] = 7
        # if i == 5000:
        #     CTX.pfcth1.Iext = 0
        #     CTX.pfcth1.Iext[0] = 7

        if not choice_made:
            # Test if a decision has been made
            if CTX.cog.delta > threshold and not cog_time:
                cog_time = i - 500
                task.records["RTcog"][trial_n] = cog_time
            if CTX.mot.delta > decision_threshold and not time:
                time = (i - 500)

            if time and cog_time:
                process(task, learn=learn, debugging=debugging, trial=trial_n, RT=time)
                task.records["RTcog"][trial_n] = cog_time
                task.records["CueValues"][trial_n] = CUE["value"]
                # task.records["Wstr"][trial_n] = connections["CTX.cog -> STR.cog"].weights
                # task.records["Wcog"][trial_n] = connections["CTX.cog -> CTX.ass"].weights
                # task.records["Wmot"][trial_n] = connections["CTX.mot -> CTX.ass"].weights
                if 0:  # ch[-1] is None:
                    mot_choice = np.argmax(CTX.mot.U)
                    cog_choice = np.argmax(CTX.cog.U)
                    print 'Wrong choice... \nMotor choice: %d\nCognitive choice: %d' % (mot_choice, cog_choice)
                    print Cue["mot"][:n], CUE["cog"][:n]

                if wholeFig:
                    choice_made = True
                else:
                    return time
    time = 2500

    if debugging:
        print 'Trial Failed!'
        print 'NoMove trial: ', trial_n

    return time
