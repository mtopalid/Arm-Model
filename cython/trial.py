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
    target = None
    t1 = 0
    t2 = 0
    pos = [4,4]
    for i in xrange(0, 500):
        iterate(dt)
        if CTX.cog.delta > 20 and not ct:
            ct = 1
        if CTX.cog.delta > threshold and not cog_time:
            cog_time = i - 500
        if i == 200:
            ARM.theta1.Iext[4] = 17
            ARM.theta2.Iext[4] = 17

    if cues_pres:
        set_trial(task, n=2, trial=trial_n)
    for i in xrange(500, duration):
        iterate(dt)

        # if i == 500:
        #     PPC.theta1.Iext[0] = 47
        #     PFC.theta1.Iext[10] = 47

        arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]
        # if  i == t+500 :
        #     PPC.theta1.Iext.reshape((n_arm,n))[4,0] = 40
        if  i == t1 + 3000 :
            arm[0] = np.argmax(ARM.theta1.V)
            ppc = np.argmax(PPC.theta1.V)
            pfc = np.argmax(PFC.theta1.V)
            sma = np.argmax(SMA.theta1.V)
            mot = buttons[np.argmax(CTX.mot.V),0]
            print "R Motor CTX: ", mot
            print "R PPC: (%d, %d)" %(ppc/n, ppc%n)
            print "R PFC: ", pfc
            print "R SMA: (%d, %d)" %(sma/n_pfc, sma%n_pfc)
            print "R Arm: ", arm[0]
            print
            reset_arm1_activities()
            t1 = i
        if  i == t2 + 3000 :
            arm[1] = np.argmax(ARM.theta2.V)
            ppc = np.argmax(PPC.theta2.V)
            pfc = np.argmax(PFC.theta2.V)
            sma = np.argmax(SMA.theta2.V)
            mot = buttons[np.argmax(CTX.mot.V),1]
            print "R Motor CTX: ", mot
            print "R PPC: (%d, %d)" %(ppc/n, ppc%n)
            print "R PFC: ", pfc
            print "R SMA: (%d, %d)" %(sma/n_pfc, sma%n_pfc)
            print "R Arm: ", arm[1]
            print
            reset_arm2_activities()
            t2 = i
        if arm[0] != pos[0] and ARM.theta1.delta > 0.5:
            print 'pos: ', pos, 'arm: ', arm
            pos[0] = arm[0]
            if target is not None:
                ppc = np.argmax(PPC.theta1.V)
                pfc = np.argmax(PFC.theta1.V)
                PFC_learning1(arm[0], ppc, pfc, target[0])
                arm[0] = np.argmax(ARM.theta1.V)
                sma = np.argmax(SMA.theta1.V)
                mot = buttons[np.argmax(CTX.mot.V),0]
                print "A Motor CTX: ", mot
                print "A PPC: (%d, %d)" %(ppc/n, ppc%n)
                print "A PFC: ", pfc
                print "A SMA: (%d, %d)" %(sma/n_pfc, sma%n_pfc)
                print "A Arm: ", arm[0]
                print
                if (arm == target).all():
                    return time
                else:
                    reset_arm1_activities()
                    t1 = i

        if arm[1] != pos[1] and ARM.theta2.delta > 0.5:
            print 'pos: ', pos, 'arm: ', arm
            pos[1] = arm[1]
            print 'i:', i, 'ARM.theta1.delta: ', ARM.theta1.delta, 'ARM.theta2.delta: ', ARM.theta2.delta
            if target is not None:
                ppc = np.argmax(PPC.theta2.V)
                pfc = np.argmax(PFC.theta2.V)
                PFC_learning2(arm[1], ppc, pfc, target[1])
                arm[1] = np.argmax(ARM.theta2.V)
                sma = np.argmax(SMA.theta2.V)
                mot = buttons[np.argmax(CTX.mot.V),1]
                print "A Motor CTX: ", mot
                print "A PPC: (%d, %d)" %(ppc/n, ppc%n)
                print "A PFC: ", pfc
                print "A SMA: (%d, %d)" %(sma/n_pfc, sma%n_pfc)
                print "A Arm: ", arm[1]
                print
                if (arm == target).all():
                    return time
                else:
                    reset_arm2_activities()
                    t2 = i

        if i == t1+200:
            ARM.theta1.Iext[pos[0]] = 17
        if i == t2+200:
            ARM.theta2.Iext[pos[1]] = 17


        if not choice_made:
            # Test if a decision has been made
            if CTX.cog.delta > threshold and not cog_time:
                cog_time = i - 500
                task.records["RTcog"][trial_n] = cog_time
            if CTX.mot.delta > decision_threshold and not time:
                time = (i - 500)

            if time and cog_time:
                cog_choice = np.argmax(CTX.cog.U)
                mot_choice = np.argmax(CTX.mot.U)
                process(task, mot_choice, learn=learn, debugging=debugging, trial=trial_n, RT=time)
                target = buttons[mot_choice,:]
                task.records["RTcog"][trial_n] = cog_time
                task.records["shape"][trial_n] = cog_choice
                task.records["CueValues"][trial_n] = CUE["value"]
                task.records["Wstr"][trial_n] = connections["CTX.cog -> STR.cog"].weights
                task.records["Wcog"][trial_n] = connections["CTX.cog -> CTX.ass"].weights
                task.records["Wmot"][trial_n] = connections["CTX.mot -> CTX.ass"].weights
                if 0:  # ch[-1] is None:
                    mot_choice = np.argmax(CTX.mot.U)
                    cog_choice = np.argmax(CTX.cog.U)
                    print 'Wrong choice... \nMotor choice: %d\nCognitive choice: %d' % (mot_choice, cog_choice)
                    print Cue["mot"][:n], CUE["cog"][:n]

                choice_made = True
                print 'choice: ', mot_choice
    time = duration

    if debugging:
        print 'Trial Failed!'
        print 'NoMove trial: ', trial_n

    return time
