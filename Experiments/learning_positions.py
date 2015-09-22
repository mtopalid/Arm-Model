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

    folder = '../Results/Learn_Positions'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 1 if there is presentation of cues else 0
    trials = 1200

    # Initialize the system
    task = Task_1ch(n=trials)
    for i in range(trials):
        print 'Experiment: ', i+1
        reset_activities()
        reset_history()
        # Define the shapes and the positions that we'll be used to each trial
        # n should be multiple of 6 because there are 6 valuable combinations of shapes and positions

        # Compute a single trial
        time = trial(task, ncues=1, wholeFig=True, trial_n=i)

    f = folder + '/Records.npy'
    np.save(f,task.records)