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
SMA = ArmStructure(tau=tau, rest=SMA_rest, noise=Cortex_N, activation=clamp, n=n_sma)
ARM = ArmStructure(tau=tau, rest=ARM_rest, noise=Cortex_N, activation=clamp, n=n_arm)
M1 = ArmStructure(tau=tau, rest=M1_rest, noise=Cortex_N, activation=clamp, n=n_m1)
STR_SMA_PPC = ArmStructure(tau=tau, rest=STR_rest, noise=Striatum_N, activation=clamp, n=n_sma * n_ppc)

structures = (CTX, STR, STN, GPE, GPI, THL, PPC, SMA, ARM, M1, STR_SMA_PPC)
arm_structures = (PPC, SMA, M1, STR_SMA_PPC, ARM)  #
BG_structures = (STR, STN, GPE, GPI, THL)
# Cue vector includes shapes, positions and the shapes' value used in reinforcement learning
CUE = np.zeros(4, dtype=[("mot", float),
                         ("cog", float),
                         ("value", float)])

# Initialization of the values
CUE["mot"] = 0, 1, 2, 3
CUE["cog"] = 0, 1, 2, 3
CUE["value"] = 0.5

SMA_value_th1 = 0.5 * np.ones(n_sma * n_ppc)
SMA_value_th2 = 0.5 * np.ones(n_sma * n_ppc)

PPC_value_th1 = 0.5 * np.ones(n_ppc * n_sma)
PPC_value_th2 = 0.5 * np.ones(n_ppc * n_sma)

# Add noise to weights
def weights(shape, s=0.005, initial=0.5):
    N = np.random.normal(initial, s, shape)
    N = np.minimum(np.maximum(N, 0.0), 1.0)
    return Wmin + (Wmax - Wmin) * N


def Wlateral(n):
    return (2 * np.eye(n) - np.ones((n, n))).ravel()


def Wm12m1(n1, n2):
    n = n1 * n2
    W = np.zeros((n, n))
    for i in range(n1):
        for j in range(n2):
            W.reshape((n, n1, n2))[i * n2 + j, :, j] = -1
            # W.reshape((n,n1,n2))[i*n2+j,i,:]=-1
            W.reshape((n, n1, n2))[i * n2 + j, i, j] = 1

    return W


def Wm12arm(n1=n_arm, n2=n_sma):
    n_all = n1 * n2
    w = np.zeros((n1, n_all))
    for i in range(n1):
        w.reshape((n1, n1, n2))[i, i, :n2 / 2 - i] = 1
        w.reshape((n1, n1, n2))[i, i, n2 - i:] = 1
        # W.reshape((n1,n1,n2))[i,i,8] = 1
        for j in range(n1):
            w.reshape((n1, n1, n2))[i, j, n2 / 2 - j + i] = 1

    return w.reshape(n1 * n_all)


def Wppc2sma(n1=n_sma, n2=n_arm, n3=n):
    w = np.zeros((n1, n2, n3))
    for i in range(n1):
        if i < n1 / 2 + 1:
            w[i, n2 - 1 - i:, :] = 1
        else:
            w[i, :n2 - 1 - i, :] = 1

    return w.reshape(n1 * n2 * n3)


# np.set_printoptions(threshold='nan')
# print Wm12arm(9,17).reshape((9,9,17))
# print Wm12m1(9,17).reshape((9*17,9,17))
# print Wppc2sma().reshape((17,9,4))

# Connectivity 
connections = {

    # SMA <-> BG
    "SMA.theta1 -> STN.smath1": OneToOne(CTX.smath1.V, STN.smath1.Isyn, np.ones(n_sma)),
    "SMA.theta2 -> STN.smath2": OneToOne(CTX.smath2.V, STN.smath2.Isyn, np.ones(n_sma)),

    "SMA.theta1 -> STR.smath1": OneToOne(SMA.theta1.V, STR.smath1.Isyn, 0.5 * np.ones(n_sma)),  # plastic (RL)
    "SMA.theta2 -> STR.smath2": OneToOne(SMA.theta2.V, STR.smath2.Isyn, weights(n_sma)),

    "SMA.theta1 -> STR_SMA_PPC.theta1": SMAtoSTR(SMA.theta1.V, STR_SMA_PPC.theta1.Isyn, weights(n_sma * n_ppc)),
    # plastic (RL)
    "SMA.theta2 -> STR_SMA_PPC.theta2": SMAtoSTR(SMA.theta2.V, STR_SMA_PPC.theta2.Isyn, weights(n_sma * n_ppc)),
    # plastic (RL)
    "PPC.theta1 -> STR_SMA_PPC.theta1": PPCtoSTR(PPC.theta1.V, STR_SMA_PPC.theta1.Isyn, 0.5 * np.ones(n_sma * n_ppc)),
    # plastic (RL)
    "PPC.theta2 -> STR_SMA_PPC.theta2": PPCtoSTR(PPC.theta2.V, STR_SMA_PPC.theta2.Isyn, 0.5 * np.ones(n_sma * n_ppc)),
    # plastic (RL)

    "STR_SMA_PPC.theta1 -> GPE.smath1": STRsmaToBG(STR_SMA_PPC.theta1.V, GPE.smath1.Isyn, np.ones(n_sma * n_ppc)),
    "STR_SMA_PPC.theta2 -> GPE.smath2": STRsmaToBG(STR_SMA_PPC.theta2.V, GPE.smath2.Isyn, np.ones(n_sma * n_ppc)),
    "STR_SMA_PPC.theta1 -> GPI.smath1": STRsmaToBG(STR_SMA_PPC.theta1.V, GPI.smath1.Isyn, np.ones(n_sma * n_ppc)),
    "STR_SMA_PPC.theta2 -> GPI.smath2": STRsmaToBG(STR_SMA_PPC.theta2.V, GPI.smath2.Isyn, np.ones(n_sma * n_ppc)),

    "STR.smath1 -> GPE.smath1": OneToOne(STR.smath1.V, GPE.smath1.Isyn, np.ones(n_sma)),
    "STR.smath2 -> GPE.smath2": OneToOne(STR.smath2.V, GPE.smath2.Isyn, np.ones(n_sma)),
    "GPE.smath1 -> STN.smath1": OneToOne(GPE.smath1.V, STN.smath1.Isyn, np.ones(n_sma)),
    "GPE.smath2 -> STN.smath2": OneToOne(GPE.smath2.V, STN.smath2.Isyn, np.ones(n_sma)),
    "STN.smath1 -> GPI.smath1": OneToAll(STN.smath1.V, GPI.smath1.Isyn, np.ones(n_sma)),
    "STN.smath2 -> GPI.smath2": OneToAll(STN.smath2.V, GPI.smath2.Isyn, np.ones(n_sma)),

    "STR.smath1 -> GPI.smath1": OneToOne(STR.smath1.V, GPI.smath1.Isyn, np.ones(n_sma)),
    "STR.smath2 -> GPI.smath2": OneToOne(STR.smath2.V, GPI.smath2.Isyn, np.ones(n_sma)),

    "GPI.smath1 -> THL.smath1": OneToOne(GPI.smath1.V, THL.smath1.Isyn, np.ones(n_sma)),
    "GPI.smath2 -> THL.smath2": OneToOne(GPI.smath2.V, THL.smath2.Isyn, np.ones(n_sma)),

    "THL.smath1 -> SMA.theta1": OneToOne(THL.smath1.V, SMA.theta1.Isyn, np.ones(n_sma)),
    "THL.smath2 -> SMA.theta2": OneToOne(THL.smath2.V, SMA.theta2.Isyn, np.ones(n_sma)),
    "SMA.theta1 -> THL.smath1": OneToOne(SMA.theta1.V, THL.smath1.Isyn, np.ones(n_sma)),
    "SMA.theta2 -> THL.smath2": OneToOne(SMA.theta2.V, THL.smath2.Isyn, np.ones(n_sma)),

    # Lateral connectivity

    "PPC.theta1 -> PPC.theta1": AllToAll(PPC.theta1.V, PPC.theta1.Isyn, Wlateral(n_ppc)),
    "PPC.theta2 -> PPC.theta2": AllToAll(PPC.theta2.V, PPC.theta2.Isyn, Wlateral(n_ppc)),

    "SMA.theta1 -> SMA.theta1": AllToAll(SMA.theta1.V, SMA.theta1.Isyn, Wlateral(n_sma)),
    "SMA.theta2 -> SMA.theta2": AllToAll(SMA.theta2.V, SMA.theta2.Isyn, Wlateral(n_sma)),

    "M1.theta1 -> M1.theta1": AllToAll(M1.theta1.V, M1.theta1.Isyn, Wlateral(n_m1)),
    "M1.theta2 -> M1.theta2": AllToAll(M1.theta2.V, M1.theta2.Isyn, Wlateral(n_m1)),

    "CTX.mot -> CTX.mot": AllToAll(CTX.mot.V, CTX.mot.Isyn, Wlateral(n)),

    # M1 between angles
    "M1.theta1 -> M1.theta2": AllToAll(M1.theta1.V, M1.theta2.Isyn, 0.01 * np.ones(n_m1*n_m1)),
    "M1.theta2 -> M1.theta1": AllToAll(M1.theta2.V, M1.theta1.Isyn, 0.01 * np.ones(n_m1*n_m1)),

    # Input To PPC

    "CTX.mot -> PPC.theta1": MotToPPC(CTX.mot.V, PPC.theta1.Isyn, 0.5 * np.ones(n)),
    "CTX.mot -> PPC.theta2": MotToPPC(CTX.mot.V, PPC.theta2.Isyn, 0.5 * np.ones(n)),

    "ARM.theta1 -> PPC.theta1": ARMtoPPC(ARM.theta1.V, PPC.theta1.Isyn, 0.5 * np.ones(n_arm)),
    "ARM.theta2 -> PPC.theta2": ARMtoPPC(ARM.theta2.V, PPC.theta2.Isyn, 0.5 * np.ones(n_arm)),

    # Input To SMA
    "PPC.theta1 -> SMA.theta1": PPCtoSMA(PPC.theta1.V, SMA.theta1.Isyn, 0.5 * Wppc2sma()),
    "PPC.theta2 -> SMA.theta2": PPCtoSMA(PPC.theta2.V, SMA.theta2.Isyn, 0.5 * Wppc2sma()),

    # Input To ARM
    "M1.theta1 -> ARM.theta1": M1toARM(M1.theta1.V, ARM.theta1.Isyn, Wm12arm(n_arm, n_sma)),
    "M1.theta2 -> ARM.theta2": M1toARM(M1.theta2.V, ARM.theta2.Isyn, Wm12arm(n_arm, n_sma)),

    # Input To M1
    "ARM.theta1 -> M1.theta1": ARMtoM1(ARM.theta1.V, M1.theta1.Isyn, 0.5 * np.ones(n_arm)),
    "ARM.theta2 -> M1.theta2": ARMtoM1(ARM.theta2.V, M1.theta2.Isyn, 0.5 * np.ones(n_arm)),

    "SMA.theta1 -> M1.theta1": SMAtoM1(SMA.theta1.V, M1.theta1.Isyn, 0.5 * np.ones(n_sma)),
    "SMA.theta2 -> M1.theta2": SMAtoM1(SMA.theta2.V, M1.theta2.Isyn, 0.5 * np.ones(n_sma)),

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
    return m


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
    connections["PPC.theta1 -> SMA.theta1"].weights = 0.5 * Wppc2sma()
    connections["PPC.theta2 -> SMA.theta2"].weights = 0.5 * Wppc2sma()

    # connections["M1.theta1 -> M1.theta2"].weights = 0.01 * np.ones(n_m1*n_m1)
    # connections["M1.theta2 -> M1.theta1"].weights = 0.01 * np.ones(n_m1*n_m1)

    connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights = weights(n_sma * n_ppc)
    connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights = weights(n_sma * n_ppc)
    connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights = 0.5 * np.ones(n_sma * n_ppc)
    connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights = 0.5 * np.ones(n_sma * n_ppc)


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
        structure.smath1.U = 0
        structure.smath1.V = 0
        structure.smath1.Isyn = 0
        structure.smath1.Iext = 0


def reset_arm2_activities():
    for structure in arm_structures:
        structure.theta2.U = 0
        structure.theta2.V = 0
        structure.theta2.Isyn = 0
        structure.theta2.Iext = 0

    for structure in BG_structures:
        structure.smath2.U = 0
        structure.smath2.V = 0
        structure.smath2.Isyn = 0
        structure.smath2.Iext = 0


def history():
    histor = np.zeros(duration, dtype=dtype)
    histor["CTX"]["mot"] = CTX.mot.history[:duration]
    histor["THL"]["smath1"] = THL.smath1.history[:duration]
    histor["THL"]["smath2"] = THL.smath2.history[:duration]
    histor["CTX"]["smath1"] = CTX.smath1.history[:duration]
    histor["CTX"]["smath2"] = CTX.smath2.history[:duration]

    histor["PPC"]["theta1"] = PPC.theta1.history[:duration]
    histor["PPC"]["theta2"] = PPC.theta2.history[:duration]

    histor["SMA"]["theta1"] = SMA.theta1.history[:duration]
    histor["SMA"]["theta2"] = SMA.theta2.history[:duration]

    histor["STR_SMA_PPC"]["theta1"] = STR_SMA_PPC.theta1.history[:duration]
    histor["STR_SMA_PPC"]["theta2"] = STR_SMA_PPC.theta2.history[:duration]

    histor["M1"]["theta1"] = M1.theta1.history[:duration]
    histor["M1"]["theta2"] = M1.theta2.history[:duration]

    histor["ARM"]["theta1"] = ARM.theta1.history[:duration]
    histor["ARM"]["theta2"] = ARM.theta2.history[:duration]
    return histor


def reset_history():
    CTX.mot.history[:duration] = 0

    THL.smath1.history[:duration] = 0
    THL.smath2.history[:duration] = 0
    CTX.smath1.history[:duration] = 0
    CTX.smath2.history[:duration] = 0

    PPC.theta1.history[:duration] = 0
    PPC.theta2.history[:duration] = 0

    SMA.theta1.history[:duration] = 0
    SMA.theta2.history[:duration] = 0

    STR_SMA_PPC.theta1.history[:duration] = 0
    STR_SMA_PPC.theta2.history[:duration] = 0

    M1.theta1.history[:duration] = 0
    M1.theta2.history[:duration] = 0

    ARM.theta1.history[:duration] = 0
    ARM.theta2.history[:duration] = 0


def M1_learning1(m1_th1, m1_th2, Wmax = 0.75, Wmin = 0.00):

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * M1.theta1.V[m1_th1]
    W = connections["M1.theta1 -> M1.theta2"].weights
    W.reshape((n_m1, n_m1))[m1_th1, m1_th2] += dw * (Wmax - W.reshape((n_m1, n_m1))[m1_th1, m1_th2]) * (
        W.reshape((n_m1, n_m1))[m1_th1, m1_th2] - Wmin)
    connections["M1.theta1 -> M1.theta2"].weights = W

def M1_learning2(m1_th1, m1_th2, Wmax = 0.75, Wmin = 0.00):
    dw = alpha_LTP_ctx * M1.theta2.V[m1_th2]
    W = connections["M1.theta2 -> M1.theta1"].weights
    W.reshape((n_m1, n_m1))[m1_th2, m1_th1] += dw * (Wmax - W.reshape((n_m1, n_m1))[m1_th2, m1_th1]) * (
        W.reshape((n_m1, n_m1))[m1_th2, m1_th1] - Wmin)
    connections["M1.theta1 -> M1.theta2"].weights = W


def SMA_learning1(reward, ppc, sma):
    # print "reward: ", reward
    # Compute prediction error
    error = reward - SMA_value_th1.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    SMA_value_th1.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # SMA
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.theta1.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * \
                                           (W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights = W
    # print 'SMA1: %d   PPC1: %d  \nSMA->STR: ' % (sma, ppc), W.reshape((n_sma, n_ppc))[sma, ppc]

    # Compute prediction error
    error = reward - PPC_value_th1.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    PPC_value_th1.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # PPC
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.theta1.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights = W
    # print 'PPC->STR: ', W.reshape((n_sma, n_ppc))[sma, ppc]

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.theta1.V[ppc]
    W = connections["PPC.theta1 -> SMA.theta1"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.theta1 -> SMA.theta1"].weights = W
    # print 'PPC->SMA: ', W.reshape((n_sma, n_ppc))[sma, ppc]


def SMA_learning2(reward, ppc, sma):
    # if arm_pos == target:
    #     reward = 1
    # else:
    #     reward = 0

    # print "reward: ", reward
    # Compute prediction error
    error = reward - SMA_value_th2.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    SMA_value_th2.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # SMA
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.theta2.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights = W
    # print 'SMA2: %d   PPC2: %d  \nSMA->STR: ' % (sma, ppc), W.reshape((n_sma, n_ppc))[sma, ppc]

    # Compute prediction error
    error = reward - PPC_value_th2.reshape((n_sma, n_ppc))[sma, ppc]
    # Update cues values
    PPC_value_th2.reshape((n_sma, n_ppc))[sma, ppc] += error * alpha_CUE
    # PPC
    lrate = alpha_LTP if error > 0 else alpha_LTD * 10
    dw = error * lrate * STR_SMA_PPC.theta2.V.reshape((n_sma, n_ppc))[sma, ppc]
    W = connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights = W
    # print 'PPC->STR: ', W.reshape((n_sma, n_ppc))[sma, ppc]

    # Hebbian cortical learning
    dw = alpha_LTP_ctx * PPC.theta2.V[ppc]
    W = connections["PPC.theta2 -> SMA.theta2"].weights
    W.reshape((n_sma, n_ppc))[sma, ppc] += dw * (Wmax - W.reshape((n_sma, n_ppc))[sma, ppc]) * (
        W.reshape((n_sma, n_ppc))[sma, ppc] - Wmin)
    connections["PPC.theta2 -> SMA.theta2"].weights = W
    # print 'PPC->SMA: ', W.reshape((n_sma, n_ppc))[sma, ppc]


# def debug_arm(theta=1):
#     if theta == 1:
#         ppc = np.argmax(PPC.theta1.V)
#         sma = np.argmax(SMA.theta1.V)
#         arm = np.argmax(ARM.theta1.V)
#         m1 = np.argmax(M1.theta1.V)
#         mot = buttons[np.argmax(CTX.mot.V), 0]
#         print "Motor CTX: ", mot
#         # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
#         # print "SMA: ", sma
#         # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
#         print "Arm: ", arm
#         print
#     else:
#         ppc = np.argmax(PPC.theta2.V)
#         sma = np.argmax(SMA.theta2.V)
#         arm = np.argmax(ARM.theta2.V)
#         m1 = np.argmax(M1.theta2.V)
#         mot = buttons[np.argmax(CTX.mot.V), 1]
#         print "Motor CTX: ", mot
#         # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
#         # print "SMA: ", sma
#         # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
#         print "Arm: ", arm
#         print

def debug_arm():
        ppc = np.argmax(PPC.theta1.V)
        sma = np.argmax(SMA.theta1.V)
        arm1 = np.argmax(ARM.theta1.V)
        m1 = np.argmax(M1.theta1.V)
        mot1 = buttons[np.argmax(CTX.mot.V), 0]
        # print "Motor CTX: ", mot
        # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
        # print "SMA: ", sma
        # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
        # print "Arm: ", arm
        ppc = np.argmax(PPC.theta2.V)
        sma = np.argmax(SMA.theta2.V)
        arm2 = np.argmax(ARM.theta2.V)
        m1 = np.argmax(M1.theta2.V)
        mot2 = buttons[np.argmax(CTX.mot.V), 1]
        print "Motor CTX: ", mot1, mot2
        # print "PPC: (%d, %d)" % (ppc / n, ppc % n)
        # print "SMA: ", sma
        # print "M1: (%d, %d)" % (m1 / n_sma, m1 % n_sma)
        print "Arm: ", arm1,arm2
        print


def debug_arm_learning():
    print "  SMA Values	1		: ", SMA_value_th1
    print "  PPC Values	1		: ", PPC_value_th1
    print "  PPC -> SMA Weights 1: ", connections["PPC.theta1 -> SMA.theta1"].weights.reshape((n_sma, n_ppc))
    print "  SMA -> STR Weights 1: ", connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights.reshape((n_sma, n_ppc))
    print "  PPC -> STR Weights 1: ", connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights.reshape((n_sma, n_ppc))
    print "  SMA Values	2		: ", SMA_value_th2
    print "  PPC Values	2		: ", PPC_value_th2
    print "  PPC -> SMA Weights 2: ", connections["PPC.theta2 -> SMA.theta2"].weights.reshape((n_sma, n_ppc))
    print "  SMA -> STR Weights 2: ", connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights.reshape((n_sma, n_ppc))
    print "  PPC -> STR Weights 2: ", connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights.reshape((n_sma, n_ppc))
    print
