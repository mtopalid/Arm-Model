# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------

from c_dana import *
from parameters import *

clamp = Clamp(min=0, max=1000)
sigmoid = Sigmoid(Vmin=Vmin, Vmax=Vmax, Vh=Vh, Vc=Vc)

# Build structures
CTX = AssociativeStructure(
    tau=tau, rest=CTX_rest, noise=Cortex_N, activation=clamp)
STR = AssociativeStructure(
    tau=tau, rest=STR_rest, noise=Striatum_N, activation=sigmoid)
STN = Structure(tau=tau, rest=STN_rest, noise=STN_N, activation=clamp)
GPE = Structure(tau=tau, rest=GPE_rest, noise=GPe_N, activation=clamp)
GPI = Structure(tau=tau, rest=GPI_rest, noise=GPi_N, activation=clamp)
THL = Structure(tau=tau, rest=THL_rest, noise=Thalamus_N, activation=clamp)

PPC = ArmStructure(tau=tau, rest=CTX_rest, noise=Cortex_N, activation=clamp, n=n_ppc)
PFC = ArmStructure(tau=tau, rest=PFC_rest, noise=Cortex_N, activation=clamp, n=n_pfc)
ARM = ArmStructure(tau=tau, rest=ARM_rest, noise=Cortex_N, activation=clamp, n=n_arm)
SMA = ArmStructure(tau=tau, rest=SMA_rest, noise=Cortex_N, activation=clamp, n=n_sma)
STR_PFC_PPC = ArmStructure(tau=tau, rest=STR_rest, noise=Striatum_N, activation=clamp, n=n_pfc * n_ppc)

structures = (CTX, STR, STN, GPE, GPI, THL, PPC, PFC, ARM, SMA, STR_PFC_PPC)
arm_structures = (PPC, PFC, SMA, STR_PFC_PPC, ARM)  #
BG_structures = (STR, STN, GPE, GPI, THL)
# Cue vector includes shapes, positions and the shapes' value used in reinforcement learning
CUE = np.zeros(4, dtype=[("mot", float),
                         ("cog", float),
                         ("value", float)])

# Initialization of the values
CUE["mot"] = 0, 1, 2, 3
CUE["cog"] = 0, 1, 2, 3
CUE["value"] = 0.5

PFC_value_th1 = 0.5 * np.ones(n_pfc * n_ppc)
PFC_value_th2 = 0.5 * np.ones(n_pfc * n_ppc)

PPC_value_th1 = 0.5 * np.ones(n_ppc * n_pfc)
PPC_value_th2 = 0.5 * np.ones(n_ppc * n_pfc)


# Add noise to weights
def weights(shape, s=0.005, initial=0.5):
    N = np.random.normal(initial, s, shape)
    N = np.minimum(np.maximum(N, 0.0), 1.0)
    return Wmin + (Wmax - Wmin) * N


def Wlateral(n):
    return (2 * np.eye(n) - np.ones((n, n))).ravel()


def Wsma2sma(n1, n2):
    n = n1 * n2
    W = np.zeros((n, n))
    for i in range(n1):
        for j in range(n2):
            W.reshape((n, n1, n2))[i * n2 + j, :, j] = -1
            # W.reshape((n,n1,n2))[i*n2+j,i,:]=-1
            W.reshape((n, n1, n2))[i * n2 + j, i, j] = 1

    return W


def Wsma2arm(n1=n_arm, n2=n_pfc):
    n_all = n1 * n2
    w = np.zeros((n1, n_all))
    for i in range(n1):
        w.reshape((n1, n1, n2))[i, i, :n2 / 2 - i] = 1
        w.reshape((n1, n1, n2))[i, i, n2 - i:] = 1
        # W.reshape((n1,n1,n2))[i,i,8] = 1
        for j in range(n1):
            w.reshape((n1, n1, n2))[i, j, n2 / 2 - j + i] = 1

    return w.reshape(n1 * n_all)


def Wppc2pfc(n1=n_pfc, n2=n_arm, n3=n):
    w = np.zeros((n1, n2, n3))
    for i in range(n1):
        if i < n1 / 2 + 1:
            w[i, n2 - 1 - i:, :] = 1
        else:
            w[i, :n2 - 1 - i, :] = 1

    return w.reshape(n1 * n2 * n3)


# np.set_printoptions(threshold='nan')
# print Wsma2arm(9,17).reshape((9,9,17))
# print Wsma2sma(9,17).reshape((9*17,9,17))
# print Wppc2pfc().reshape((17,9,4))

# Connectivity 
connections = {

    # PFC <-> BG
    "PFC.theta1 -> STN.pfcth1": OneToOne(CTX.pfcth1.V, STN.pfcth1.Isyn, np.ones(n_pfc)),
    "PFC.theta2 -> STN.pfcth2": OneToOne(CTX.pfcth2.V, STN.pfcth2.Isyn, np.ones(n_pfc)),

    "PFC.theta1 -> STR.pfcth1": OneToOne(PFC.theta1.V, STR.pfcth1.Isyn, 0.5 * np.ones(n_pfc)),  # plastic (RL)
    "PFC.theta2 -> STR.pfcth2": OneToOne(PFC.theta2.V, STR.pfcth2.Isyn, weights(n_pfc)),

    "PFC.theta1 -> STR_PFC_PPC.theta1": PFCtoSTR(PFC.theta1.V, STR_PFC_PPC.theta1.Isyn, weights(n_pfc * n_ppc)),
    # plastic (RL)
    "PFC.theta2 -> STR_PFC_PPC.theta2": PFCtoSTR(PFC.theta2.V, STR_PFC_PPC.theta2.Isyn, weights(n_pfc * n_ppc)),
    # plastic (RL)
    "PPC.theta1 -> STR_PFC_PPC.theta1": PPCtoSTR(PPC.theta1.V, STR_PFC_PPC.theta1.Isyn, 0.5 * np.ones(n_pfc * n_ppc)),
    # plastic (RL)
    "PPC.theta2 -> STR_PFC_PPC.theta2": PPCtoSTR(PPC.theta2.V, STR_PFC_PPC.theta2.Isyn, 0.5 * np.ones(n_pfc * n_ppc)),
    # plastic (RL)

    "STR_PFC_PPC.theta1 -> GPE.pfcth1": STRpfcToBG(STR_PFC_PPC.theta1.V, GPE.pfcth1.Isyn, np.ones(n_pfc * n_ppc)),
    "STR_PFC_PPC.theta2 -> GPE.pfcth2": STRpfcToBG(STR_PFC_PPC.theta2.V, GPE.pfcth2.Isyn, np.ones(n_pfc * n_ppc)),
    "STR_PFC_PPC.theta1 -> GPI.pfcth1": STRpfcToBG(STR_PFC_PPC.theta1.V, GPI.pfcth1.Isyn, np.ones(n_pfc * n_ppc)),
    "STR_PFC_PPC.theta2 -> GPI.pfcth2": STRpfcToBG(STR_PFC_PPC.theta2.V, GPI.pfcth2.Isyn, np.ones(n_pfc * n_ppc)),

    "STR.pfcth1 -> GPE.pfcth1": OneToOne(STR.pfcth1.V, GPE.pfcth1.Isyn, np.ones(n_pfc)),
    "STR.pfcth2 -> GPE.pfcth2": OneToOne(STR.pfcth2.V, GPE.pfcth2.Isyn, np.ones(n_pfc)),
    "GPE.pfcth1 -> STN.pfcth1": OneToOne(GPE.pfcth1.V, STN.pfcth1.Isyn, np.ones(n_pfc)),
    "GPE.pfcth2 -> STN.pfcth2": OneToOne(GPE.pfcth2.V, STN.pfcth2.Isyn, np.ones(n_pfc)),
    "STN.pfcth1 -> GPI.pfcth1": OneToAll(STN.pfcth1.V, GPI.pfcth1.Isyn, np.ones(n_pfc)),
    "STN.pfcth2 -> GPI.pfcth2": OneToAll(STN.pfcth2.V, GPI.pfcth2.Isyn, np.ones(n_pfc)),

    "STR.pfcth1 -> GPI.pfcth1": OneToOne(STR.pfcth1.V, GPI.pfcth1.Isyn, np.ones(n_pfc)),
    "STR.pfcth2 -> GPI.pfcth2": OneToOne(STR.pfcth2.V, GPI.pfcth2.Isyn, np.ones(n_pfc)),

    "GPI.pfcth1 -> THL.pfcth1": OneToOne(GPI.pfcth1.V, THL.pfcth1.Isyn, np.ones(n_pfc)),
    "GPI.pfcth2 -> THL.pfcth2": OneToOne(GPI.pfcth2.V, THL.pfcth2.Isyn, np.ones(n_pfc)),

    "THL.pfcth1 -> PFC.theta1": OneToOne(THL.pfcth1.V, PFC.theta1.Isyn, np.ones(n_pfc)),
    "THL.pfcth2 -> PFC.theta2": OneToOne(THL.pfcth2.V, PFC.theta2.Isyn, np.ones(n_pfc)),
    "PFC.theta1 -> THL.pfcth1": OneToOne(PFC.theta1.V, THL.pfcth1.Isyn, np.ones(n_pfc)),
    "PFC.theta2 -> THL.pfcth2": OneToOne(PFC.theta2.V, THL.pfcth2.Isyn, np.ones(n_pfc)),

    # Lateral connectivity

    "PPC.theta1 -> PPC.theta1": AllToAll(PPC.theta1.V, PPC.theta1.Isyn, Wlateral(n_ppc)),
    "PPC.theta2 -> PPC.theta2": AllToAll(PPC.theta2.V, PPC.theta2.Isyn, Wlateral(n_ppc)),

    "PFC.theta1 -> PFC.theta1": AllToAll(PFC.theta1.V, PFC.theta1.Isyn, Wlateral(n_pfc)),
    "PFC.theta2 -> PFC.theta2": AllToAll(PFC.theta2.V, PFC.theta2.Isyn, Wlateral(n_pfc)),

    "SMA.theta1 -> SMA.theta1": AllToAll(SMA.theta1.V, SMA.theta1.Isyn, Wlateral(n_sma)),
    "SMA.theta2 -> SMA.theta2": AllToAll(SMA.theta2.V, SMA.theta2.Isyn, Wlateral(n_sma)),

    "CTX.mot -> CTX.mot": AllToAll(CTX.mot.V, CTX.mot.Isyn, Wlateral(n)),

    # Input To PPC

    "CTX.mot -> PPC.theta1": MotToPPC(CTX.mot.V, PPC.theta1.Isyn, 0.5 * np.ones(n)),
    "CTX.mot -> PPC.theta2": MotToPPC(CTX.mot.V, PPC.theta2.Isyn, 0.5 * np.ones(n)),

    "ARM.theta1 -> PPC.theta1": ARMtoPPC(ARM.theta1.V, PPC.theta1.Isyn, 0.5 * np.ones(n_arm)),
    "ARM.theta2 -> PPC.theta2": ARMtoPPC(ARM.theta2.V, PPC.theta2.Isyn, 0.5 * np.ones(n_arm)),

    # Input To PFC
    "PPC.theta1 -> PFC.theta1": PPCtoPFC(PPC.theta1.V, PFC.theta1.Isyn, 0.5 * Wppc2pfc()),
    "PPC.theta2 -> PFC.theta2": PPCtoPFC(PPC.theta2.V, PFC.theta2.Isyn, 0.5 * Wppc2pfc()),

    # Input To ARM
    "SMA.theta1 -> ARM.theta1": SMAtoARM(SMA.theta1.V, ARM.theta1.Isyn, Wsma2arm(n_arm, n_pfc)),
    "SMA.theta2 -> ARM.theta2": SMAtoARM(SMA.theta2.V, ARM.theta2.Isyn, Wsma2arm(n_arm, n_pfc)),

    # Input To SMA
    "ARM.theta1 -> SMA.theta1": ARMtoSMA(ARM.theta1.V, SMA.theta1.Isyn, 0.5 * np.ones(n_arm)),
    "ARM.theta2 -> SMA.theta2": ARMtoSMA(ARM.theta2.V, SMA.theta2.Isyn, 0.5 * np.ones(n_arm)),

    "PFC.theta1 -> SMA.theta1": PFCtoSMA(PFC.theta1.V, SMA.theta1.Isyn, 0.5 * np.ones(n_pfc)),
    "PFC.theta2 -> SMA.theta2": PFCtoSMA(PFC.theta2.V, SMA.theta2.Isyn, 0.5 * np.ones(n_pfc)),

}
for name, gain in gains.items():
    connections[name].gain = gain


def set_trial(task, num=2, trial=0, protocol='Guthrie', familiar=True):
    if num == 1:
        temp = (task[trial]["ass"].ravel().argsort())[-1:]
        CUE["mot"][0], CUE["cog"][0] = np.unravel_index(temp, (4, 4))
    else:
        i1, i2 = (task[trial]["ass"].ravel().argsort())[-2:]
        CUE["mot"][0], CUE["cog"][0] = np.unravel_index(i1, (4, 4))
        CUE["mot"][1], CUE["cog"][1] = np.unravel_index(i2, (4, 4))

    CTX.mot.Iext = 0

    for i in range(num):
        c, m = CUE["cog"][i], CUE["mot"][i]

        CTX.mot.Iext[m] = 23 + np.random.uniform(-noise_cue / 2, noise_cue / 2)

def iterate(dt):
    # Flush connections
    for connection in connections.values():
        connection.flush()

    # Propagate activities
    for connection in connections.values():
        connection.propagate()

    # Compute new activities
    for structure in structures:
        structure.evaluate(dt)


def reset():
    CUE["mot"] = 0, 1, 2, 3
    CUE["cog"] = 0, 1, 2, 3
    CUE["value"] = 0.5
    reset_weights()
    reset_activities()
    reset_history()


def reset_weights():

    connections["PPC.theta1 -> PFC.theta1"].weights = 0.5 * Wppc2pfc()
    connections["PPC.theta2 -> PFC.theta2"].weights = 0.5 * Wppc2pfc()

    connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = weights(n_pfc * n_ppc)
    connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = weights(n_pfc * n_ppc)
    connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = 0.5 * np.ones(n_pfc * n_ppc)
    connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = 0.5 * np.ones(n_pfc * n_ppc)


def reset_activities():
    for structure in structures:
        structure.reset()


def reset_arm1_activities():
    for structure in arm_structures:
        structure.theta1.U = 0
        structure.theta1.V = 0
        structure.theta1.Isyn = 0
        structure.theta1.Iext = 0

    for structure in BG_structures:
        structure.pfcth1.U = 0
        structure.pfcth1.V = 0
        structure.pfcth1.Isyn = 0
        structure.pfcth1.Iext = 0


def reset_arm2_activities():
    for structure in arm_structures:
        structure.theta2.U = 0
        structure.theta2.V = 0
        structure.theta2.Isyn = 0
        structure.theta2.Iext = 0

    for structure in BG_structures:
        structure.pfcth2.U = 0
        structure.pfcth2.V = 0
        structure.pfcth2.Isyn = 0
        structure.pfcth2.Iext = 0


def history():
    histor = np.zeros(duration, dtype=dtype)
    histor["CTX"]["mot"] = CTX.mot.history[:duration]
    histor["THL"]["pfcth1"] = THL.pfcth1.history[:duration]
    histor["THL"]["pfcth2"] = THL.pfcth2.history[:duration]
    histor["CTX"]["pfcth1"] = CTX.pfcth1.history[:duration]
    histor["CTX"]["pfcth2"] = CTX.pfcth2.history[:duration]

    histor["PPC"]["theta1"] = PPC.theta1.history[:duration]
    histor["PPC"]["theta2"] = PPC.theta2.history[:duration]

    histor["PFC"]["theta1"] = PFC.theta1.history[:duration]
    histor["PFC"]["theta2"] = PFC.theta2.history[:duration]

    histor["STR_PFC_PPC"]["theta1"] = STR_PFC_PPC.theta1.history[:duration]
    histor["STR_PFC_PPC"]["theta2"] = STR_PFC_PPC.theta2.history[:duration]

    histor["SMA"]["theta1"] = SMA.theta1.history[:duration]
    histor["SMA"]["theta2"] = SMA.theta2.history[:duration]

    histor["ARM"]["theta1"] = ARM.theta1.history[:duration]
    histor["ARM"]["theta2"] = ARM.theta2.history[:duration]
    return histor


def reset_history():
    CTX.mot.history[:duration] = 0

    THL.pfcth1.history[:duration] = 0
    THL.pfcth2.history[:duration] = 0
    CTX.pfcth1.history[:duration] = 0
    CTX.pfcth2.history[:duration] = 0

    PPC.theta1.history[:duration] = 0
    PPC.theta2.history[:duration] = 0

    PFC.theta1.history[:duration] = 0
    PFC.theta2.history[:duration] = 0

    STR_PFC_PPC.theta1.history[:duration] = 0
    STR_PFC_PPC.theta2.history[:duration] = 0

    SMA.theta1.history[:duration] = 0
    SMA.theta2.history[:duration] = 0

    ARM.theta1.history[:duration] = 0
    ARM.theta2.history[:duration] = 0



def PFC_learning1(reward, ppc, pfc):
    # print "reward: ", reward
    # Compute prediction error
    error = reward - PFC_value_th1.reshape((n_pfc, n_ppc))[pfc, ppc]
    # Update cues values
    PFC_value_th1.reshape((n_pfc, n_ppc))[pfc, ppc] += error * alpha_CUE
    # PFC
    lrate = alpha_LTP  if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_PFC_PPC.theta1.V.reshape((n_pfc, n_ppc))[pfc, ppc]
    W = connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights
    W.reshape((n_pfc, n_ppc))[pfc, ppc] += dw * (Wmax - W.reshape((n_pfc, n_ppc))[pfc, ppc]) * \
                                           (W.reshape((n_pfc, n_ppc))[pfc, ppc] - Wmin)
    connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = W
    # print 'PFC1: %d   PPC1: %d  \nPFC->STR: ' % (pfc, ppc), W.reshape((n_pfc, n_ppc))[pfc, ppc]

    # Compute prediction error
    error = reward - PPC_value_th1.reshape((n_pfc, n_ppc))[pfc, ppc]
    # Update cues values
    PPC_value_th1.reshape((n_pfc, n_ppc))[pfc, ppc] += error * alpha_CUE
    # PPC
    lrate = alpha_LTP  if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_PFC_PPC.theta1.V.reshape((n_pfc, n_ppc))[pfc, ppc]
    W = connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights
    W.reshape((n_pfc, n_ppc))[pfc, ppc] += dw * (Wmax - W.reshape((n_pfc, n_ppc))[pfc, ppc]) * (
        W.reshape((n_pfc, n_ppc))[pfc, ppc] - Wmin)
    connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = W
    # print 'PPC->STR: ', W.reshape((n_pfc, n_ppc))[pfc, ppc]

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.theta1.V[ppc]
    W = connections["PPC.theta1 -> PFC.theta1"].weights
    W.reshape((n_pfc, n_ppc))[pfc, ppc] += dw * (Wmax - W.reshape((n_pfc, n_ppc))[pfc, ppc]) * (
        W.reshape((n_pfc, n_ppc))[pfc, ppc] - Wmin)
    connections["PPC.theta1 -> PFC.theta1"].weights = W
    # print 'PPC->PFC: ', W.reshape((n_pfc, n_ppc))[pfc, ppc]


def PFC_learning2(reward, ppc, pfc):
    # if arm_pos == target:
    #     reward = 1
    # else:
    #     reward = 0

    # print "reward: ", reward
    # Compute prediction error
    error = reward - PFC_value_th2.reshape((n_pfc, n_ppc))[pfc, ppc]
    # Update cues values
    PFC_value_th2.reshape((n_pfc, n_ppc))[pfc, ppc] += error * alpha_CUE
    # PFC
    lrate = alpha_LTP  if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_PFC_PPC.theta2.V.reshape((n_pfc, n_ppc))[pfc, ppc]
    W = connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights
    W.reshape((n_pfc, n_ppc))[pfc, ppc] += dw * (Wmax - W.reshape((n_pfc, n_ppc))[pfc, ppc]) * (
        W.reshape((n_pfc, n_ppc))[pfc, ppc] - Wmin)
    connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = W
    # print 'PFC2: %d   PPC2: %d  \nPFC->STR: ' % (pfc, ppc), W.reshape((n_pfc, n_ppc))[pfc, ppc]

    # Compute prediction error
    error = reward - PPC_value_th2.reshape((n_pfc, n_ppc))[pfc, ppc]
    # Update cues values
    PPC_value_th2.reshape((n_pfc, n_ppc))[pfc, ppc] += error * alpha_CUE
    # PPC
    lrate = alpha_LTP  if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_PFC_PPC.theta2.V.reshape((n_pfc, n_ppc))[pfc, ppc]
    W = connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights
    W.reshape((n_pfc, n_ppc))[pfc, ppc] += dw * (Wmax - W.reshape((n_pfc, n_ppc))[pfc, ppc]) * (
        W.reshape((n_pfc, n_ppc))[pfc, ppc] - Wmin)
    connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = W
    # print 'PPC->STR: ', W.reshape((n_pfc, n_ppc))[pfc, ppc]

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.theta2.V[ppc]
    W = connections["PPC.theta2 -> PFC.theta2"].weights
    W.reshape((n_pfc, n_ppc))[pfc, ppc] += dw * (Wmax - W.reshape((n_pfc, n_ppc))[pfc, ppc]) * (
        W.reshape((n_pfc, n_ppc))[pfc, ppc] - Wmin)
    connections["PPC.theta2 -> PFC.theta2"].weights = W
    # print 'PPC->PFC: ', W.reshape((n_pfc, n_ppc))[pfc, ppc]



def debug_arm(theta=1):
    if theta == 1:
        ppc = np.argmax(PPC.theta1.V)
        pfc = np.argmax(PFC.theta1.V)
        arm = np.argmax(ARM.theta1.V)
        sma = np.argmax(SMA.theta1.V)
        mot = buttons[np.argmax(CTX.mot.V), 0]
        print "Motor CTX: ", mot
        # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
        # print "PFC: ", pfc
        # print "SMA: (%d, %d)" % (sma / n_pfc, sma % n_pfc)
        print "Arm: ", arm
        print
    else:
        ppc = np.argmax(PPC.theta2.V)
        pfc = np.argmax(PFC.theta2.V)
        arm = np.argmax(ARM.theta2.V)
        sma = np.argmax(SMA.theta2.V)
        mot = buttons[np.argmax(CTX.mot.V), 1]
        print "Motor CTX: ", mot
        # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
        # print "PFC: ", pfc
        # print "SMA: (%d, %d)" % (sma / n_pfc, sma % n_pfc)
        print "Arm: ", arm
        print


def debug_arm_learning():
    print "  PFC Values	1		: ", PFC_value_th1
    print "  PPC Values	1		: ", PPC_value_th1
    print "  PPC -> PFC Weights 1: ", connections["PPC.theta1 -> PFC.theta1"].weights.reshape((n_pfc, n_ppc))
    print "  PFC -> STR Weights 1: ", connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights.reshape((n_pfc, n_ppc))
    print "  PPC -> STR Weights 1: ", connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights.reshape((n_pfc, n_ppc))
    print "  PFC Values	2		: ", PFC_value_th2
    print "  PPC Values	2		: ", PPC_value_th2
    print "  PPC -> PFC Weights 2: ", connections["PPC.theta2 -> PFC.theta2"].weights.reshape((n_pfc, n_ppc))
    print "  PFC -> STR Weights 2: ", connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights.reshape((n_pfc, n_ppc))
    print "  PPC -> STR Weights 2: ", connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights.reshape((n_pfc, n_ppc))
    print
