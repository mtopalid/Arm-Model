# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------

# Evolution of single trial with Guthrie protocol
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Include to the path files from cython folder
    t = '../cython/'
    import sys
    sys.path.append(t)
    # model file build the structures and initialize the model
    from model import *
    from trial import *
    from task_1ch import Task_1ch
    import os
    from learning import *

    folder = '../Results/Learn_Positions_-0_5'#'#+M1_learning
    if not os.path.exists(folder):
        os.makedirs(folder)

    # # Initialize the system
    task = Task_1ch(n=n_learning_positions_trials)
    #
    # Save the trials
    f = folder + '/Task.npy'
    np.save(f, task.trials)

    folder2 = folder + "/Backup"
    if not os.path.exists(folder2):
        os.makedirs(folder2)

    # f = folder + '/Task.npy'
    # task.trials = np.load(f)
    #
    # f = folder2 + '/Trial_Number1000s.npy'
    # tr = np.load(f)
    #
    # f = folder2 + '/Records1000s.npy'
    # task.records[:tr] = np.load(f)
    # # print "History of learning by moves: \n", temp["moves"]
    # connections["PPC.theta1 -> SMA.theta1"].weights = task.records["Wppc_sma1"][tr-1]
    # connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights = task.records["Wsma_str1"][tr-1]
    # connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights = task.records["Wppc_str1"][tr-1]
    # connections["PPC.theta2 -> SMA.theta2"].weights = task.records["Wppc_sma2"][tr-1]
    # connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights = task.records["Wsma_str2"][tr-1]
    # connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights = task.records["Wppc_str2"][tr-1]
    # connections["M1.theta1 -> M1.theta2"].weights = task.records["Wm1_1"][tr-1]
    # connections["M1.theta2 -> M1.theta1"].weights = task.records["Wm1_2"][tr-1]

    # Repeated trials with learning after each trial
    learning_trials_continuous(task, trials=n_learning_positions_trials, ncues=1, duration=duration_learning_positions,
                    debugging_arm_learning=False, folder=folder2, save=True)#, tr=tr-1)

    np.set_printoptions(threshold=3)
    P = task.records["best"]
    print "  Mean performance		: %.1f %%" % np.array(P * 100).mean()
    R = task.records["reward"]
    print "  Mean reward			: %.3f" % np.array(R).mean()
    # print "Moves:\n", task.records["moves"][:n_learning_positions_trials]

    f = folder + '/Records.npy'
    np.save(f, task.records)