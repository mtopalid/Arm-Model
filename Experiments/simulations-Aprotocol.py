# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------

# Simulate number of experiments that is given in parameters.py of the different
# models. Each simulation is a number of trials under Guthrie protocol.
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
    from task_a import Task_A

    folder = '../Results/Learn_Positions'
    f = folder + '/Records.npy'
    temp = np.load(f)

    # Creation of folder to save the results
    folder = '../Results/A/Simulations'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(simulations):

        print 'Simulation: ', i + 1
        # Initialize the system
        reset()
        connections["PPC.theta1 -> PFC.theta1"].weights = temp["Wppc_pfc1"][-1]
        connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wpfc_str1"][-1]
        connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wppc_str1"][-1]
        connections["PPC.theta2 -> PFC.theta2"].weights = temp["Wppc_pfc2"][-1]
        connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wpfc_str2"][-1]
        connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wppc_str1"][-1]

        # Define the shapes and the positions that we'll be used to each trial
        # n should be multiple of 6 because there are 6 valuable combinations of shapes and positions
        task = Task_A(n=n_trials)

        # Repeated trials with learning after each trial
        learning_trials(task, debug_simulation = True, debugging=False)

        # Debugging information
        print "Mean performance of 30 last trials	: %.1f %%\n" %(np.array(task.records["best"][-30:]).mean()*100)
        debug_learning(task.records["Wcog"][-1], task.records["Wmot"][-1], task.records["Wstr"][-1], task.records["CueValues"][-1])

        # Save the results in files
        file = folder + '/Cues'  + "%03d" % (i+1) + '.npy'
        np.save(file,task.trials)
        file = folder + '/Records'  + "%03d" % (i+1) + '.npy'
        np.save(file,task.records)
        print




