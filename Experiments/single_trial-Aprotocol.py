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
    from task_a import Task_A

    # 1 if there is presentation of cues else 0
    cues_pres = 1

    for i in range(1):
        # Initialize the system
        reset()

        # Define the shapes and the positions that we'll be used to each trial
        # n should be multiple of 6 because there are 6 valuable combinations of shapes and positions
        task = Task_A(n=6)

        # Compute a single trial
        time = trial(task, cues_pres=cues_pres, wholeFig=True)

        # retrieve the activity history of the structures
        histor = history()
        pfc1 = histor["PFC"]["theta1"]
        pfc2 = histor["PFC"]["theta2"]
        sma1 = histor["SMA"]["theta1"]
        sma2 = histor["SMA"]["theta2"]
        arm1 = histor["ARM"]["theta1"]
        arm2 = histor["ARM"]["theta2"]
        ppc1 = histor["PPC"]["theta1"]
        ppc2 = histor["PPC"]["theta2"]

        # for i in range(n_arm):
        #     plt.figure()
        #     for j in range(n_pfc):
        #         plt.plot(sma[:,i*n_pfc+j], label='SMA [%s, %s]'%(i,j))
        #     plt.legend(loc='best')

        # plt.figure()
        # plt.plot(sma[:,8*17], label='SMA 8-0')
        # plt.plot(sma[:,0*17], label='SMA 0-0')
        # plt.legend(loc='best')

        plt.figure()
        plt.plot(arm1)
        plt.title('Arm1')

        plt.figure()
        plt.plot(arm2)
        plt.title('Arm2')

        plt.figure()
        plt.plot(sma1)
        plt.title('SMA1')
        plt.figure()
        plt.plot(sma2)
        plt.title('SMA2')

        plt.figure()
        plt.plot(pfc1)
        plt.title('PFC1')
        plt.figure()
        plt.plot(pfc2)
        plt.title('PFC2')

        plt.figure()
        plt.plot(ppc1)
        plt.title('PPC1')

        plt.figure()
        plt.plot(ppc2)
        plt.title('PPC2')
        plt.show()
        # Display cortical activity during the single trial
        if 0: display_ctx(histor, 3.0)  # , "single-trial.pdf")

        # Display activity from all structures during the single trial
        if 0: display_all(histor, 3.0)  # , "single-trial-all.pdf")
        # arm = [np.argmax(ARM.theta1.V), np.argmax(ARM.theta2.V)]
        # ppc = np.argmax(PPC.theta1.V)
        # pfc = np.argmax(PFC.theta1.V)
        # sma = np.argmax(SMA.theta1.V)
        # mot = buttons[np.argmax(CTX.mot.V),:]
        # print "Motor CTX: ", mot
        # print "PPC: %d, %d" %(ppc/n, ppc%n)
        # print "PFC: ", pfc
        # print "SMA: %d, %d" %(sma/n_pfc, sma%n_pfc)
        # print "Arm: ", arm
        # print
