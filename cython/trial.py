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


def trial(task, cues_pres=True, ncues=2, duration=duration, learn=True, debugging=True, debugging_arm=True, trial_n=0, wholeFig=False):
    reset_activities()
    reset_history()
    ct = None
    cog_time = None
    mot_time = None
    choice_made = False
    target = None
    t1 = 0
    t2 = 0
    pos = [4, 4]
    moves = 0
    for i in xrange(0, 500):
        iterate(dt)
        if CTX.cog.delta > 20 and not ct:
            ct = 1
        if CTX.cog.delta > decision_threshold and not cog_time:
            cog_time = i - 500
        if i == 200:
            ARM.theta1.Iext[4] = 17
            ARM.theta2.Iext[4] = 17

    if cues_pres:
        set_trial(task, n=ncues, trial=trial_n)
    for i in xrange(500, duration):
        iterate(dt)

        arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]

        if ((arm[0] != pos[0] and ARM.theta1.delta > 0.5) or (
                np.argmax(PFC.theta1.V) == 8 and PFC.theta1.delta > 5.) or i == t1 + 3000):
            pos[0] = arm[0]
            if target is not None:
                moves += 1
                if debugging_arm:#0:
                    debug_arm(theta=1)
                if (arm == target).all():
                    for j in range(n):
                        if (np.array(arm) == buttons[j, :]).all():
                            move = j
                            break
                    task.records["move"][trial_n] = move
                    task.records["PFCValues1"][trial_n] = PFC_value_th1
                    task.records["PPCValues1"][trial_n] = PPC_value_th1
                    task.records["PFCValues2"][trial_n] = PFC_value_th2
                    task.records["PPCValues2"][trial_n] = PPC_value_th2
                    task.records["Wppc_pfc1"][trial_n] = connections["PPC.theta1 -> PFC.theta1"].weights
                    task.records["Wpfc_str1"][trial_n] = connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights
                    task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights
                    task.records["Wppc_pfc2"][trial_n] = connections["PPC.theta2 -> PFC.theta2"].weights
                    task.records["Wpfc_str2"][trial_n] = connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights
                    task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights
                    task.records["moves"][trial_n] = moves

                    time = i  # - 500
                    task.process(task[trial_n], action=mot_choice, debug=debugging, RT=time - 500)
                    PFC_learning1(task.records["reward"][trial_n], np.argmax(PPC.theta1.V), np.argmax(PFC.theta1.V))
                    process(task, n=ncues, learn=learn, debugging=debugging, trial=trial_n)
                    # process(task, mot_choice, n=ncues, learn=learn, debugging=debugging, trial=trial_n, RT=time - 500)

                    # debug_arm_learning()
                    return time

                else:
                    # PFC_learning1(arm[0], np.argmax(PPC.theta1.V), np.argmax(PFC.theta1.V), target[0])
                    PFC_learning1(task.records["reward"][trial_n], np.argmax(PPC.theta1.V), np.argmax(PFC.theta1.V))
                    reset_arm1_activities()
                    t1 = i

        if ((arm[1] != pos[1] and ARM.theta2.delta > 0.5) or (
                np.argmax(PFC.theta2.V) == 8 and PFC.theta2.delta > 5.) or i == t2 + 3000):  # and choice_made
            pos[1] = arm[1]
            if target is not None:
                moves += 1
                if debugging_arm:#0:
                    debug_arm(theta=2)
                if (arm == target).all():
                    for j in range(n):
                        if (np.array(arm) == buttons[j, :]).all():
                            move = j
                            break
                    task.records["move"][trial_n] = move
                    task.records["PFCValues1"][trial_n] = PFC_value_th1
                    task.records["PPCValues1"][trial_n] = PPC_value_th1
                    task.records["PFCValues2"][trial_n] = PFC_value_th2
                    task.records["PPCValues2"][trial_n] = PPC_value_th2
                    task.records["Wppc_pfc1"][trial_n] = connections["PPC.theta1 -> PFC.theta1"].weights
                    task.records["Wpfc_str1"][trial_n] = connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights
                    task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights
                    task.records["Wppc_pfc2"][trial_n] = connections["PPC.theta2 -> PFC.theta2"].weights
                    task.records["Wpfc_str2"][trial_n] = connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights
                    task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights
                    task.records["moves"][trial_n] = moves

                    time = i  # - 500
                    task.process(task[trial_n], action=mot_choice, debug=debugging, RT=time - 500)
                    PFC_learning2(task.records["reward"][trial_n], np.argmax(PPC.theta2.V), np.argmax(PFC.theta2.V))
                    process(task, n=ncues, learn=learn, debugging=debugging, trial=trial_n)

                    return time
                else:

                    PFC_learning2(task.records["reward"][trial_n], np.argmax(PPC.theta2.V), np.argmax(PFC.theta2.V))
                    reset_arm2_activities()
                    t2 = i

        if i == t1 + 10:
            ARM.theta1.Iext[pos[0]] = 17
        if i == t2 + 10:
            ARM.theta2.Iext[pos[1]] = 17

        if not choice_made:
            # Test if a decision has been made
            if CTX.cog.delta > decision_threshold and not cog_time:
                cog_time = i - 500
                task.records["RTcog"][trial_n] = cog_time
            if CTX.mot.delta > decision_threshold and not mot_time:
                mot_time = i - 500
                task.records["RTmot"][trial_n] = mot_time

            if mot_time and cog_time:
                cog_choice = np.argmax(CTX.cog.U)
                mot_choice = np.argmax(CTX.mot.U)
                # process(task, mot_choice, learn=learn, debugging=debugging, trial=trial_n, RT=time)
                target = buttons[mot_choice, :]
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
                    print CUE["mot"][:n], CUE["cog"][:n]

                choice_made = True
                # print 'choice: ', mot_choice
    time = duration

    if debugging:
        print 'Trial Failed!'
        print 'NoMove trial: ', trial_n

    task.records["move"][trial_n] = 4
    task.records["PFCValues1"][trial_n] = PFC_value_th1
    task.records["PPCValues1"][trial_n] = PPC_value_th1
    task.records["PFCValues2"][trial_n] = PFC_value_th2
    task.records["PPCValues2"][trial_n] = PPC_value_th2
    task.records["Wppc_pfc1"][trial_n] = connections["PPC.theta1 -> PFC.theta1"].weights
    task.records["Wpfc_str1"][trial_n] = connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights
    task.records["Wppc_str1"][trial_n] = connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights
    task.records["Wppc_pfc2"][trial_n] = connections["PPC.theta2 -> PFC.theta2"].weights
    task.records["Wpfc_str2"][trial_n] = connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights
    task.records["Wppc_str2"][trial_n] = connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights
    task.records["moves"][trial_n] = moves

    return time
