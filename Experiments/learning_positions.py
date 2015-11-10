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
    temp = '../cython/'
    import sys

    sys.path.append(temp)
    # model file build the structures and initialize the model
    from model import *
    from display import *
    from trial import *
    from task_1ch import Task_1ch
    import os
    from learning import *

    folder = '../Results/Learn_Positions'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Initialize the system
    task = Task_1ch(n=n_learning_positions_trials)
    f = folder + '/Task.npy'
    #np.save(f,task.trials)

    # Repeated trials with learning after each trial
    learning_trials(task, trials=n_learning_positions_trials, ncues=1, duration=duration_learning_positions, debugging_arm_learning = True)
    print "Moves:\n", task.records["moves"][:n_learning_positions_trials]

    f = folder + '/Records.npy'
    #np.save(f,task.records)