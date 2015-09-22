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
    from task_b import Task_B

    # 1 if there is presentation of cues else 0
    cues_pres = 1

    GPi_lesion = False



    # Define the shapes and the positions that we'll be used to each trial
    # n should be multiple of 6 because there are 6 valuable combinations of positions
    task = Task_B(n=6)

    folder = '../Results/Learn_Positions'
    f = folder + '/Records.npy'
    temp = np.load(f)
    connections["PPC.theta1 -> PFC.theta1"].weights = temp["Wppc_pfc1"][-1]
    connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wpfc_str1"][-1]
    connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wppc_str1"][-1]
    connections["PPC.theta2 -> PFC.theta2"].weights = temp["Wppc_pfc2"][-1]
    connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wpfc_str2"][-1]
    connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wppc_str1"][-1]


    # Make GPi lesion
    if GPi_lesion:
        connections["GPI.cog -> THL.cog"].active = False
        connections["GPI.mot -> THL.mot"].active = False
    # Compute a single trial
    time = trial(task, wholeFig=True, debugging=True)
    print "  Moves                 : ", task.records[0]["moves"]

    #retrieve the activity history of the structures
    histor = history()
    ctx = histor["CTX"]["mot"][:time]
    arm1 = histor["ARM"]["theta1"][:time]
    arm2 = histor["ARM"]["theta2"][:time]

    plt.figure()
    plt.plot(arm1)
    plt.title('Arm1')

    plt.figure()
    plt.plot(arm2)
    plt.title('Arm2')

    plt.figure()
    plt.plot(ctx)
    plt.title('CTX')

    plt.show()