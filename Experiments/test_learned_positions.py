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
    import numpy as np
    path = '../cython/'
    import sys
    sys.path.append(path)
    from model import *
    from learning import *
    from parameters import *
    from single_trial_1pos import single_trial
    folder = '../Results/Learn_Positions_single_sim'
    file = folder + '/Records1200-64.npy'
    records = np.load(file)

    reset()
    PFC_value_th1 = records["PFCValues1"][-1] 
    PPC_value_th1 = records["PPCValues1"][-1]
    PFC_value_th2 = records["PFCValues2"][-1] 
    PPC_value_th2 = records["PPCValues2"][-1] 
    connections["PPC.theta1 -> PFC.theta1"].weights = records["Wppc_pfc1"][-1]
    connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = records["Wpfc_str1"][-1] 
    connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = records["Wppc_str1"][-1] 
    connections["PPC.theta2 -> PFC.theta2"].weights = records["Wppc_pfc2"][-1] 
    connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = records["Wpfc_str2"][-1] 
    connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = records["Wppc_str2"][-1]

    single_trial()

