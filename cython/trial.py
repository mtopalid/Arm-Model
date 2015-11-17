# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#				Nicolas Rougier  (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
from model import *
from kinematics import *
from parameters import *


def trial(task, cues_pres=True, ncues=2, duration=duration, learn=True, debugging=True, debugging_arm=True, trial_n=0,
          wholeFig=False):
    reset_activities()
    reset_history()
    # ct = None
    # cog_time = None
    mot_time = None
    choice_made = False
    target = None
    t1 = 0
    t2 = 0
    # pos = [4, 4]
    moves = 0
    for i in xrange(0, 500):
        iterate(dt)
        # if CTX.cog.delta > 20 and not ct:
        #     ct = 1
        # if CTX.cog.delta > decision_threshold and not cog_time:
        #     cog_time = i - 500
        if i == 200:
            # pos = np.random.randint(n_arm, size=2)
            # task.records["initial_pos"][trial_n] = pos

            pos = task.trials["initial_pos"][trial_n].copy()
            ARM.theta1.Iext[pos[0]] = 17
            ARM.theta2.Iext[pos[1]] = 17
            # ARM.theta1.Iext[4] = 17
            # ARM.theta2.Iext[4] = 17

    if cues_pres:
        m = set_trial(task, num=ncues, trial=trial_n)

    for i in xrange(500, duration):
        iterate(dt)

        if not choice_made:
            # Test if a decision has been made
            if CTX.mot.delta > decision_threshold and not mot_time:
                mot_time = i - 500
                task.records["RTmot"][trial_n] = mot_time
                mot_choice = np.argmax(CTX.mot.U)
                target = buttons[mot_choice, :]
                choice_made = True

        arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]

        if ((arm[0] != pos[0] and ARM.theta1.delta > 0.5) or (
                        np.argmax(
                            PFC.theta1.V) == 8 and PFC.theta1.delta > 5.)) and t1 == 0 and choice_made:  # or i == t1 + 3000):
            pos[0] = arm[0]
            moves += 1
            t1 = 1

        if ((arm[1] != pos[1] and ARM.theta2.delta > 0.5) or (
                        np.argmax(
                            PFC.theta2.V) == 8 and PFC.theta2.delta > 5.)) and t2 == 0 and choice_made:  # or i == t2 + 3000):  #
            pos[1] = arm[1]
            moves += 1
            t2 = 1

        if t1 == 1 and t2 == 1:

            task.records["final_pos"][trial_n] = pos
            task.records["target_pos"][trial_n] = target
            t = i  # - 500
            cor1 = coordinations(conver_degr2rad(task.trials["initial_pos"][trial_n][0]), conver_degr2rad(
                task.trials["initial_pos"][trial_n][1]))
            # tar = tar_pos(m)
            d_init = distance(target, cor1)
            cor2 = coordinations(conver_degr2rad(pos[0]), conver_degr2rad(pos[1]))
            d_final = distance(target, cor2)
            if d_final == 0.0:
                reward = 1
            elif d_final < d_init:
                reward = 0.5
            else:
                reward = 0
            # print tar
            # print cor1
            # print cor2
            if d_final == 0.0:
                task.records["move"][trial_n] = mot_choice
                task.records["best"][trial_n] = True
            else:
                task.records["move"][trial_n] = 4
                task.records["best"][trial_n] = False
            task.records["reward"][trial_n] = reward
            task.records["moves"][trial_n] = moves

            PFC_learning1(reward, np.argmax(PPC.theta1.V), np.argmax(PFC.theta1.V))
            PFC_learning2(reward, np.argmax(PPC.theta2.V), np.argmax(PFC.theta2.V))

            if debugging_arm:  # 0:
                print "Reward: ", reward
                debug_arm(theta=1)
                debug_arm(theta=2)

            task.process(task[trial_n], debug=debugging, RT=t - 500)
            print

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
            return t

    t = duration

    if debugging:
        print 'Trial Failed!'
        print 'NoMove trial: ', trial_n + 1

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

    return t


def trial_continuous_move(task, cues_pres=True, ncues=2, duration=duration, learn=True, debugging=True,
                          debugging_arm=True, trial_n=0, wholeFig=False):
    reset_activities()
    reset_history()
    ct = None
    cog_time = None
    mot_time = None
    choice_made = False
    target = None
    t1 = 0
    t2 = 0
    # pos = [4, 4]
    moves = 0
    for i in xrange(0, 500):
        iterate(dt)
        if CTX.cog.delta > 20 and not ct:
            ct = 1
        if CTX.cog.delta > decision_threshold and not cog_time:
            cog_time = i - 500
        if i == 200:
            # pos = np.random.randint(n_arm, size=2)
            # task.records["initial_pos"][trial_n] = pos

            pos = task.trials["initial_pos"][trial_n].copy()
            ARM.theta1.Iext[pos[0]] = 17
            ARM.theta2.Iext[pos[1]] = 17
            # ARM.theta1.Iext[4] = 17
            # ARM.theta2.Iext[4] = 17

    if cues_pres:
        m = set_trial(task, num=ncues, trial=trial_n)

    for i in xrange(500, duration):
        iterate(dt)

        if not choice_made:
            # Test if a decision has been made
            if CTX.mot.delta > decision_threshold and not mot_time:
                mot_time = i - 500
                task.records["RTmot"][trial_n] = mot_time
                mot_choice = np.argmax(CTX.mot.U)
                target = buttons[mot_choice, :]
                choice_made = True

        arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]

        if ((arm[0] != pos[0] and ARM.theta1.delta > 0.5) or (
                        np.argmax(PFC.theta1.V) == 8 and PFC.theta1.delta > 5.) or i == t1 + 3000) and choice_made:
            pos[0] = arm[0]
            moves += 1

            cor1 = coordinations(conver_degr2rad(task.trials["initial_pos"][trial_n][0]), conver_degr2rad(
                task.trials["initial_pos"][trial_n][1]))
            tar = tar_pos(m)
            d_init = distance(tar, cor1)
            cor2 = coordinations(conver_degr2rad(pos[0]), conver_degr2rad(pos[1]))
            d_final = distance(tar, cor2)
            if d_final == 0.0:
                reward = 1
            elif d_final < d_init:
                reward = 0.5
            else:
                reward = 0

            # task.records["reward"][trial_n] = reward
            PFC_learning1(reward, np.argmax(PPC.theta1.V), np.argmax(PFC.theta1.V))

            # if target is not None:
            if debugging_arm:  # 0:
                print "Reward:  ", reward
                debug_arm(theta=1)
            if (arm == target).all():
                for j in range(n):
                    if (np.array(arm) == buttons[j, :]).all():
                        move = j
                        break

                task.records["best"][trial_n] = True
                task.records["move"][trial_n] = move
                task.records["moves"][trial_n] = moves
                task.records["final_pos"][trial_n] = pos
                task.records["target_pos"][trial_n] = buttons[m, :]
                task.records["reward"][trial_n] = reward

                t = i  # - 500
                task.process(task[trial_n], action=mot_choice, debug=debugging, RT=t - 500)
                print
                # debug_arm_learning()

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

                return t

            else:
                reset_arm1_activities()
                t1 = i

        if ((arm[1] != pos[1] and ARM.theta2.delta > 0.5) or (
                        np.argmax(PFC.theta2.V) == 8 and PFC.theta2.delta > 5.) or i == t2 + 3000):  # and
            # choice_made
            pos[1] = arm[1]
            moves += 1

            cor1 = coordinations(conver_degr2rad(task.trials["initial_pos"][trial_n][0]), conver_degr2rad(
                task.trials["initial_pos"][trial_n][1]))
            tar = tar_pos(m)
            d_init = distance(tar, cor1)
            cor2 = coordinations(conver_degr2rad(pos[0]), conver_degr2rad(pos[1]))
            d_final = distance(tar, cor2)
            if d_final == 0.0:
                reward = 1
            elif d_final < d_init:
                reward = 0.5
            else:
                reward = 0

            PFC_learning2(reward, np.argmax(PPC.theta2.V), np.argmax(PFC.theta2.V))

            # if target is not None:
            if debugging_arm:  # 0:
                print "Reward:  ", reward
                debug_arm(theta=2)
            if (arm == target).all():
                for j in range(n):
                    if (np.array(arm) == buttons[j, :]).all():
                        move = j
                        break

                task.records["best"][trial_n] = True
                task.records["move"][trial_n] = move
                task.records["moves"][trial_n] = moves

                task.records["final_pos"][trial_n] = pos
                task.records["target_pos"][trial_n] = buttons[m, :]
                task.records["reward"][trial_n] = reward

                t = i  # - 500
                task.process(task[trial_n], action=mot_choice, debug=debugging, RT=t - 500)
                print
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

                return t

            else:
                reset_arm2_activities()
                t2 = i

        if i == t1 + 10:
            ARM.theta1.Iext[pos[0]] = 17
        if i == t2 + 10:
            ARM.theta2.Iext[pos[1]] = 17

    t = duration

    if 0:  # debugging:
        print 'Trial Failed!'
        print 'NoMove trial: ', trial_n + 1

    task.records["best"][trial_n] = False
    task.records["final_pos"][trial_n] = pos
    task.records["target_pos"][trial_n] = buttons[m, :]

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
    task.process(task[trial_n], action=mot_choice, debug=debugging, RT=t - 500)
    print

    return t - mot_time - 500
