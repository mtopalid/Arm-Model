# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------

# Testing learning for each model under Guthrie protocol
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    temp = '../cython/'
    import sys
    import os

    sys.path.append(temp)
    from model import *
    from display import *
    from learning import *
    from task_a import Task_A

    folder = '../Results/Learn_Positions'
    f = folder + '/Records.npy'
    temp = np.load(f)
    connections["PPC.theta1 -> PFC.theta1"].weights = temp["Wppc_pfc1"][-1]
    connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wpfc_str1"][-1]
    connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wppc_str1"][-1]
    connections["PPC.theta2 -> PFC.theta2"].weights = temp["Wppc_pfc2"][-1]
    connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wpfc_str2"][-1]
    connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wppc_str1"][-1]

    folder = '../Results/A/Learning_Cues/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    task = Task_A(n=n_trials)
    file = folder + 'task.npy'
    np.save(file,task.trials)

    learning_trials(task)

    file = folder +  'records.npy'
    np.save(file,task.records)
    print 'Mean performance of the 25 first trials: ', np.array(task.records["best"][:25]).mean()
    print 'Mean performance of the 25 last trials: ', np.array(task.records["best"][-25:]).mean()

    # histor = history()
    # arm1 = histor["ARM"]["theta1"]
    # arm2 = histor["ARM"]["theta2"]
    #
    # plt.figure()
    # plt.plot(arm1)
    # plt.title('Arm1')
    #
    # plt.figure()
    # plt.plot(arm2)
    # plt.title('Arm2')

    if 0: display_all(hist, duration)  # , "single-trial-all.pdf")
    if 0: display_ctx(hist, 3.0)
    if 0: display_all(hist, 3.0)  # , "single-trial-all.pdf")
