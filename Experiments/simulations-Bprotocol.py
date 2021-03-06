# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)

# -----------------------------------------------------------------------------


# Simulations of protocol B.
# 	Habitual Condition (HC): The pair of pre-learned shapes is presented.
# 							  P1 = 75% and P2 = 25%
# 	Novel	 Condition (NC): An unknown pair of shapes is shown
# 							  P1 = 75% and P2 = 25%
# Learning phase: Two unknown shapes, with reward probabilities P1 = 75% and
# P2 = 25%, are shown to the system. This pair is used to HC.
# Testing Phase: Blocks of 10 trials HC and NC are mixed in each simulation.
# 				 Inactivation of connection between GPi and Thalamus and
#				 presentation of mixed blocks of HC and NC trials
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Include to the path files from cython folder
    temp = '../cython/'
    import sys

    sys.path.append(temp)

    import numpy as np
    import os

    # model file build the structures and initialize the model
    from model import *
    from learning import *
    from parameters import *
    from task_b import Task_B

    folder = '../Results/Learn_Positions'
    f = folder + '/Records.npy'
    temp = np.load(f)
    connections["PPC.theta1 -> PFC.theta1"].weights = temp["Wppc_pfc1"][-1]
    connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wpfc_str1"][-1]
    connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wppc_str1"][-1]
    connections["PPC.theta2 -> PFC.theta2"].weights = temp["Wppc_pfc2"][-1]
    connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wpfc_str2"][-1]
    connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wppc_str1"][-1]

    # Creation of folders to save the results
    folder = '../Results/B/Simulations'
    if not os.path.exists(folder):
        os.makedirs(folder)

    folderL = folder + '/Learning'
    if not os.path.exists(folderL):
        os.makedirs(folderL)
    folderTf = folder + '/Testing_fam'
    if not os.path.exists(folderTf):
        os.makedirs(folderTf)
    folderTuf = folder + '/Testing_unfam'
    if not os.path.exists(folderTuf):
        os.makedirs(folderTuf)
    folderTfnG = folder + '/Testing_fam_NoGPi'
    if not os.path.exists(folderTfnG):
        os.makedirs(folderTfnG)
    folderTufnG = folder + '/Testing_unfam_NoGPi'
    if not os.path.exists(folderTufnG):
        os.makedirs(folderTufnG)

    existing_learning = True
    for i in range(157,simulations):
        print 'Experiment: ', i + 1
        reset()

        # Formation of Habits or use of pre-learned weights
        if existing_learning:
            file = '../Results/B/Learning/Records' + "%03d" % (i + 1) + '.npy'
            temp = np.load(file)
            connections["CTX.cog -> CTX.ass"].weights = temp["Wcog"][-1, :]
            connections["CTX.mot -> CTX.ass"].weights = temp["Wmot"][-1, :]
            connections["CTX.cog -> STR.cog"].weights = temp["Wstr"][-1, :]
        else:
            print '-----------------Learning Phase----------------'

            # Be sure that the GPi-Thalamus connections are active
            connections["GPI.cog -> THL.cog"].active = True
            connections["GPI.mot -> THL.mot"].active = True

            # Define the shapes and the positions that we'll be used to each trial
            # n should be multiple of 6 because there are 6 valuable combinations of positions
            # The number of tasks are 2*n:
            # First n are formed only by the pair [0,1] that we want to learn
            # Last n are formed only by [2,3]
            task = Task_B(n=n_learning_trials)
            # In this phase, we want only the pair [0,1]
            task = task[:n_learning_trials]

            # Repeated trials with learning after each trial
            learning_trials(task, trials=n_learning_trials, debugging=False, debug_simulation=True)

            # Save results in files
            file = folderL + '/Cues' + '%03d' % (i + 1) + '.npy'
            np.save(file, task.trials)
            file = folderL + '/Records' + "%03d" % (i + 1) + '.npy'
            np.save(file, task.records)

            # Debugging information
            print "Mean performance of 30 last trials	: %.1f %%\n" % (
                np.array(task.records["best"][-30:]).mean() * 100)
            print "Mean RT								: %.1f ms\n" % (np.array(task.records["RTmot"][-30:]).mean())
            debug_learning(task.records["Wcog"][-1], task.records["Wmot"][-1], task.records["Wstr"][-1],
                           task.records["CueValues"][-1])

        print '\n--------Testing with GPi--------'
        # Be sure that the GPi-Thalamus connections are active
        connections["GPI.cog -> THL.cog"].active = True
        connections["GPI.mot -> THL.mot"].active = True

        task = Task_B(n=n_testing_trials)
        # Debugging information
        steps = n_testing_trials / 10
        print 'Starting   ',

        # Test changes between familiar and unfamiliar cues every 10 trials
        for j in range(n_testing_trials / 10):
            # familiar cues [0,1]
            t = task[j * 10:(j + 1) * 10]
            learning_trials(t, trials=10, debugging=False)

            # Debugging information
            print '\b.',
            sys.stdout.flush()
            # unfamiliar cues [2,3]
            t = task[n_testing_trials + j * 10:n_testing_trials + (j + 1) * 10]
            learning_trials(t, trials=10, debugging=False)

            # Debugging information
            print '\b.',
            sys.stdout.flush()


        # Debugging information
        print '   Done!'
        print '\n--------Familiar--------'
        print " Mean performance	: %.1f %%\n" % (np.array(task.records["best"][:n_testing_trials]).mean() * 100)
        print " Mean RT			: %.1f ms\n" % (np.array(task.records["RTmot"][:n_testing_trials]).mean())
        print
        print '--------UnFamiliar--------'
        print " Mean performance	: %.1f %%\n" % (np.array(task.records["best"][n_testing_trials:]).mean() * 100)
        print " Mean RT			: %.1f ms\n" % (np.array(task.records["RTmot"][n_testing_trials:]).mean())
        print

        # Save results in files
        file = folderTf + '/Cues' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.trials[:n_testing_trials])
        file = folderTf + '/Records' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.records[:n_testing_trials])
        file = folderTuf + '/Cues' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.trials[n_testing_trials:])
        file = folderTuf + '/Records' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.records[n_testing_trials:])

        print '\n\n-----------------Testing without GPi----------------'
        # Make GPI lesion
        connections["GPI.cog -> THL.cog"].active = False
        connections["GPI.mot -> THL.mot"].active = False

        # Initialization of weights and cue values to simulate unknown shapes
        connections["CTX.cog -> CTX.ass"].weights[2:] = weights(2, 0.00005)  # 0.5*np.ones(4)
        connections["CTX.cog -> STR.cog"].weights[2:] = weights(2)
        CUE["value"][2:] = 0.5

        task = Task_B(n=n_testing_trials)
        # Debugging information
        steps = n_testing_trials / 10
        print 'Starting   ',
        # Test changes between familiar and unfamiliar cues every 10 trials
        for j in range(n_testing_trials / 10):
            # familiar cues [0,1]
            t = task[j * 10:(j + 1) * 10]
            learning_trials(t, trials=10, debugging=False)

            # Debugging information
            print '\b.',
            sys.stdout.flush()

            # unfamiliar cues [2,3]
            t = task[n_testing_trials + j * 10:n_testing_trials + (j + 1) * 10]
            learning_trials(t, trials=10, debugging=False)
            # Debugging information
            print '\b.',
            sys.stdout.flush()


        # Debugging information
        print '   Done!'
        print '\n--------Familiar--------'
        print "Mean performance	: %.1f %%\n" % (np.array(task.records["best"][:n_testing_trials]).mean() * 100)
        print "Mean RT			: %.1f ms\n" % (np.array(task.records["RTmot"][:n_testing_trials]).mean())
        print
        print '--------UnFamiliar--------'
        print "Mean performance	: %.1f %%\n" % (np.array(task.records["best"][n_testing_trials:]).mean() * 100)
        print "Mean RT			: %.1f ms\n" % (np.array(task.records["RTmot"][n_testing_trials:]).mean())
        print

        # Save results in files
        file = folderTfnG + '/Cues' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.trials[:n_testing_trials])
        file = folderTfnG + '/Records' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.records[:n_testing_trials])
        file = folderTufnG + '/Cues' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.trials[n_testing_trials:])
        file = folderTufnG + '/Records' + "%03d" % (i + 1) + '.npy'
        np.save(file, task.records[n_testing_trials:])
