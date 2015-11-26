# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
import numpy as np

# Population size
n = 4
n_sma = 17
n_arm = 9
n_m1 = n_sma * n_arm
n_ppc = n_arm * n
# Protocol A
n_trials = 480
# Protocol C
n_reverse_trials = 720  # 4800#
n_reverse_trials_Piron = 2400
# Protocol B, D
n_learning_trials = 4800  # 960 #720 #240 #
n_testing_trials = 240
# Learning Positions
n_learning_positions_trials = 81*40#81*81*20

simulations = 100

buttons = np.ones((n, 2))
buttons[0, :] = [4, 1]  # [90,75]
buttons[1, :] = [1, 6]  # [75,100]
buttons[2, :] = [4, 6]  # [90,100]
buttons[3, :] = [6, 1]  # [100,75]

angles = np.linspace(70, 110, num=9)

# --- Time ---
ms = 0.001
duration = int(9. / ms)
duration_learning_positions = int(64. / ms)
dt = 1 * ms
tau = 10 * ms

# --- Learning ---
a = 1.
alpha_CUE = 0.0025 * a  # 0.0005
alpha_LTP = 0.005 * a
alpha_LTD = 0.00375 * a
alpha_LTP_ctx = alpha_LTP ** 2 * a  # 0.000025

# --- Sigmoid ---
Vmin = 0
Vmax = 20
Vh = 16
Vc = 3

# --- Model ---
decision_threshold = 40
CTX_rest = -3.0
M1_rest = 3.0
SMA_rest = 27.0
ARM_rest = -30.0
STR_rest = 0.0
STN_rest = -10.0
GPE_rest = -10.0
GPI_rest = -10.0
THL_rest = -40.0

# Noise level (%)
Cortex_N = 0.01
Striatum_N = 0.01
STN_N = 0.01
GPi_N = 0.01
GPe_N = 0.01
Thalamus_N = 0.01

# --- Cues & Rewards ---
Value_cue = 7
noise_cue = 0.001

rewards_Guthrie = 3 / 3., 2 / 3., 1 / 3., 0 / 3.
rewards_Guthrie_reverse_all = 0 / 3., 1 / 3., 2 / 3., 3 / 3.
rewards_Guthrie_reverse_middle = 3 / 3., 1 / 3., 2 / 3., 0 / 3
rewards_Piron = 0.75, 0.25, 0.75, 0.25
rewards_Piron_reverse = 0.25, 0.75, 0.25, 0.75

# -- Weight ---
Wmin = 0.25
Wmax = 0.75

gains = {
    # SMA <-> BG
    "SMA.theta1 -> STN.smath1": +1.0,
    "SMA.theta2 -> STN.smath2": +1.0,

    "SMA.theta1 -> STR_SMA_PPC.theta1": +0.2,
    "SMA.theta2 -> STR_SMA_PPC.theta2": +0.2,
    "PPC.theta1 -> STR_SMA_PPC.theta1": +0.2,
    "PPC.theta2 -> STR_SMA_PPC.theta2": +0.2,

    "STR_SMA_PPC.theta1 -> GPE.smath1": -2.0,
    "STR_SMA_PPC.theta2 -> GPE.smath2": -2.0,
    "STR_SMA_PPC.theta1 -> GPI.smath1": -2.0,
    "STR_SMA_PPC.theta2 -> GPI.smath2": -2.0,

    # "STR_SMA_PPC.theta1 -> STR.smath1": +1.0,
    # "STR_SMA_PPC.theta2 -> STR.smath2": +1.0,

    "SMA.theta1 -> STR.smath1": +1.0,
    "SMA.theta2 -> STR.smath2": +1.0,

    "STR.smath1 -> GPE.smath1": -2.0,
    "STR.smath2 -> GPE.smath2": -2.0,

    "GPE.smath1 -> STN.smath1": -0.25,
    "GPE.smath2 -> STN.smath2": -0.25,

    "STN.smath1 -> GPI.smath1": +1.0,
    "STN.smath2 -> GPI.smath2": +2.0,

    "STR.smath1 -> GPI.smath1": -2.0,
    "STR.smath2 -> GPI.smath2": -2.0,

    "GPI.smath1 -> THL.smath1": -0.25,
    "GPI.smath2 -> THL.smath2": -0.25,

    "THL.smath1 -> SMA.theta1": +0.4,
    "THL.smath2 -> SMA.theta2": +0.4,
    "SMA.theta1 -> THL.smath1": +0.1,
    "SMA.theta2 -> THL.smath2": +0.1,

    # Lateral connectivity
    # "ARM.theta1 -> ARM.theta1": +0.5,
    # "ARM.theta2 -> ARM.theta2": +0.5,

    "PPC.theta1 -> PPC.theta1": +0.5,
    "PPC.theta2 -> PPC.theta2": +0.5,

    "M1.theta1 -> M1.theta1": +0.5,
    "M1.theta2 -> M1.theta2": +0.5,

    "SMA.theta1 -> SMA.theta1": +0.5,
    "SMA.theta2 -> SMA.theta2": +0.5,

    "CTX.mot -> CTX.mot": +0.5,

    # M1 between angles
    # "M1.theta1 -> M1.theta2": +0.5,
    # "M1.theta2 -> M1.theta1": +0.5,

    # Input To PPC
    "CTX.mot -> PPC.theta1": +0.3,
    "CTX.mot -> PPC.theta2": +0.3,

    "ARM.theta1 -> PPC.theta1": +0.3,
    "ARM.theta2 -> PPC.theta2": +0.3,

    # Input To SMA
    "PPC.theta1 -> SMA.theta1": +0.7,
    "PPC.theta2 -> SMA.theta2": +0.7,

    # Input To ARM
    "M1.theta1 -> ARM.theta1": +1.,
    "M1.theta2 -> ARM.theta2": +1.,

    # Input To M1
    "ARM.theta1 -> M1.theta1": +0.3,
    "ARM.theta2 -> M1.theta2": +0.3,

    "SMA.theta1 -> M1.theta1": +3.,
    "SMA.theta2 -> M1.theta2": +3.,

}

dtype = [("CTX", [("mot", float, n), ("cog", float, n), ("ass", float, 16), ("smath1", float, n_sma),
                  ("smath2", float, n_sma)]),
         ("STR", [("mot", float, n), ("cog", float, n), ("ass", float, n * n)]),
         ("GPE", [("mot", float, n), ("cog", float, n)]),
         ("GPI", [("mot", float, n), ("cog", float, n)]),
         ("THL", [("mot", float, n), ("cog", float, n), ("smath1", float, n_sma), ("smath2", float, n_sma)]),
         ("STN", [("mot", float, n), ("cog", float, n)]),
         ("PPC", [("theta1", float, n_ppc), ("theta2", float, n_ppc)]),
         ("SMA", [("theta1", float, n_sma), ("theta2", float, n_sma)]),
         ("STR_SMA_PPC", [("theta1", float, n_sma * n_ppc), ("theta2", float, n_sma * n_ppc)]),
         ("M1", [("theta1", float, n_m1), ("theta2", float, n_m1)]),
         ("ARM", [("theta1", float, n_arm), ("theta2", float, n_arm)])]
