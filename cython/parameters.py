# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Meropi Topalidou (Meropi.Topalidou@inria.fr)
#               Nicolas Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------


# Population size
n = 4
n_pfc = 17
n_arm = 9
n_sma = n_pfc * n_arm
n_ppc = n_arm * n
# Protocol A
n_trials = 480
# Protocol C
n_reverse_trials = 720  # 4800#
n_reverse_trials_Piron = 2400
# Protocol B, D
n_learning_trials = 4800 #960 #720 #240 #
n_testing_trials = 240

simulations = 250
threshold = 40

# --- Time ---
ms = 0.001
duration = int(7. / ms)
dt = 1 * ms
tau = 10 * ms

# --- Learning ---
alpha_CUE = 0.0025  # 0.0005
alpha_LTP = 0.005
alpha_LTD = 0.00375
alpha_LTP_ctx = alpha_LTP ** 2  # 0.000025

# --- Sigmoid ---
Vmin = 0
Vmax = 20
Vh = 16
Vc = 3

# --- Model ---
decision_threshold = 40
CTX_rest = -3.0
SMA_rest =  3.0
PFC_rest =  27.0
ARM_rest =  -7.0
STR_rest =  0.0
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
         # CTX <-> BG
         # "CTX.cog -> STR.cog": +1.0,
         # "CTX.mot -> STR.mot": +1.0,
         # "CTX.ass -> STR.ass": +1.0,
         # "CTX.cog -> STR.ass": +0.2,
         # "CTX.mot -> STR.ass": +0.2,
         #
         # "CTX.cog -> STN.cog": +1.0,
         # "CTX.mot -> STN.mot": +1.0,
         #
         # "STR.cog -> GPE.cog": -2.0,
         # "STR.mot -> GPE.mot": -2.0,
         # "STR.ass -> GPE.cog": -2.0,
         # "STR.ass -> GPE.mot": -2.0,
         # "GPE.cog -> STN.cog": -0.25,
         # "GPE.mot -> STN.mot": -0.25,
         # "STN.cog -> GPI.cog": +1.0,
         # "STN.mot -> GPI.mot": +1.0,
         #
         # "STR.cog -> GPI.cog": -2.0,
         # "STR.mot -> GPI.mot": -2.0,
         # "STR.ass -> GPI.cog": -2.0,
         # "STR.ass -> GPI.mot": -2.0,
         #
         # "GPI.cog -> THL.cog": -0.25,
         # "GPI.mot -> THL.mot": -0.25,

         # Lateral connectivity
         # "ARM.theta1 -> ARM.theta1": +0.5,
         # "ARM.theta2 -> ARM.theta2": +0.5,

         "PPC.theta1 -> PPC.theta1": +0.5,
         "PPC.theta2 -> PPC.theta2": +0.5,

         "SMA.theta1 -> SMA.theta1": +0.5,
         "SMA.theta2 -> SMA.theta2": +0.5,

         "PFC.theta1 -> PFC.theta1": +0.5,
         "PFC.theta2 -> PFC.theta2": +0.5,
    
         "CTX.mot -> CTX.mot": +0.5,
         "CTX.cog -> CTX.cog": +0.5,
         "CTX.ass -> CTX.ass": +0.5,
    
         # Input To PPC
         "CTX.mot -> PPC.theta1": +0.3,
         "CTX.mot -> PPC.theta2": +0.3,

         "ARM.theta1 -> PPC.theta1": +0.3,
         "ARM.theta2 -> PPC.theta2": +0.3,

         # Input To PFC
         "PPC.theta1 -> PFC.theta1": +0.3,
         "PPC.theta2 -> PFC.theta2": +0.3,

         # Input To ARM
         "SMA.theta1 -> ARM.theta1": +0.5,
         "SMA.theta2 -> ARM.theta2": +0.5,

         # Input To SMA
         "ARM.theta1 -> SMA.theta1": +0.5,
         "ARM.theta2 -> SMA.theta2": +0.5,

         "PFC.theta1 -> SMA.theta1": +0.5,
         "PFC.theta2 -> SMA.theta2": +0.5,
         
         
         # CTX - THL
         "THL.pfcth1 -> PFC.theta1": +0.4,
         "THL.pfcth2 -> PFC.theta2": +0.4,
         "PFC.theta1 -> THL.pfcth1": +0.1,
         "PFC.theta2 -> THL.pfcth2": +0.1,

         "THL.cog -> CTX.cog": +0.4,
         "THL.mot -> CTX.mot": +0.4,
         "CTX.cog -> THL.cog": +0.1,
         "CTX.mot -> THL.mot": +0.1,

         # Cortical Connectivity
         "CTX.cog -> CTX.ass": +0.01,
         "CTX.mot -> CTX.ass": +0.01,

         "CTX.ass -> CTX.cog": +0.01,
         "CTX.ass -> CTX.mot": +0.01,

         }

dtype = [("CTX", [("mot", float, n), ("cog", float, n), ("ass", float, 16), ("pfcth1", float, n_pfc), ("pfcth2", float, n_pfc)]),
         ("STR", [("mot", float, n), ("cog", float, n), ("ass", float, n*n)]),
         ("GPE", [("mot", float, n), ("cog", float, n)]),
         ("GPI", [("mot", float, n), ("cog", float, n)]),
         ("THL", [("mot", float, n), ("cog", float, n), ("pfcth1", float, n_pfc), ("pfcth2", float, n_pfc)]),
         ("STN", [("mot", float, n), ("cog", float, n)]),
         ("PPC", [("theta1", float, n_ppc), ("theta2", float, n_ppc)]),
         ("PFC", [("theta1", float, n_pfc), ("theta2", float, n_pfc)]),
         ("SMA", [("theta1", float, n_sma), ("theta2", float, n_sma)]),
         ("ARM", [("theta1", float, n_arm), ("theta2", float, n_arm)])]
