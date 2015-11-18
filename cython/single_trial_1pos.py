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

temp = '../cython/'
import sys

sys.path.append(temp)
# model file build the structures and initialize the model
from model import *
from display import *
from trial import *
from task_1ch import Task_1ch


def single_trial():
    # 1 if there is presentation of cues else 0
    cues_pres = 1
    trials = 1
    # Define the shapes and the positions that we'll be used to each trial
    # n should be multiple of 6 because there are 6 valuable combinations of shapes and positions
    task = Task_1ch(n=81*4)

    # Compute a single trial
    time = trial(task, cues_pres=cues_pres, ncues=1, duration=duration_learning_positions,
                                 debugging=True, wholeFig=True) # trial_continuous_move #
    print "Moves        : ", task.records[0]["moves"]
    print "Time         : ", time

    # retrieve the activity history of the structures
    # histor = history()
    # mot = histor["CTX"]["mot"][:time]
    # pfc1 = histor["PFC"]["theta1"][:time]
    # pfc2 = histor["PFC"]["theta2"][:time]
    # sma1 = histor["SMA"]["theta1"][:time]
    # sma2 = histor["SMA"]["theta2"][:time]
    # arm1 = histor["ARM"]["theta1"][:time]
    # arm2 = histor["ARM"]["theta2"][:time]
    # ppc1 = histor["PPC"]["theta1"][:time]
    # ppc2 = histor["PPC"]["theta2"][:time]
    #
    # plt.figure()
    # plt.plot(mot)
    # plt.title('Mot')
    #
    # plt.figure()
    # plt.plot(arm1)
    # plt.title('Arm1')
    #
    # plt.figure()
    # plt.plot(arm2)
    # plt.title('Arm2')
    #
    # plt.figure()
    # plt.plot(sma1)
    # plt.title('SMA1')
    # plt.figure()
    # plt.plot(sma2)
    # plt.title('SMA2')
    #
    # plt.figure()
    # plt.plot(pfc1)
    # plt.title('PFC1')
    # plt.figure()
    # plt.plot(pfc2)
    # plt.title('PFC2')
    #
    # plt.figure()
    # plt.plot(ppc1)
    # plt.title('PPC1')
    #
    # plt.figure()
    # plt.plot(ppc2)
    # plt.title('PPC2')
    # plt.show()
    # Display cortical activity during the single trial
    if 0: display_ctx(histor, 3.0)  # , "single-trial.pdf")


if __name__ == "__main__":
    # Include to the path files from cython folder
    single_trial()
