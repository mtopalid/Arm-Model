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
    from task_1ch import Task_1ch

    # Creation of folder to save the results
    folder = '../Results/Learn_Positions'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(79,100):#simulations):

        print 'Simulation: ', i + 1
        # Initialize the system
        reset()

        # Define the shapes and the positions that we'll be used to each trial
        # n should be multiple of 6 because there are 6 valuable combinations of shapes and positions
        task = Task_1ch(n=n_learning_positions_trials)

        # Repeated trials with learning after each trial
        learning_trials(task, trials=n_learning_positions_trials, ncues=1, debug_simulation = True, debugging=False,
                        duration=duration_learning_positions, debugging_arm_learning=False)

        # Debugging information
        print "Moves needed to reach the positions:\n", task.records["moves"]

        # Save the results in files
        file = folder + '/Cues'  + "%03d" % (i+1) + '.npy'
        np.save(file,task.trials)
        file = folder + '/Records'  + "%03d" % (i+1) + '.npy'
        np.save(file,task.records)
        print




