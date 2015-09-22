#!/usr/bin/env python
import numpy as np
import os
import matplotlib.pyplot as plt
from parameters import *
import sys
from model import *
from display import *
from trial import *
from task_1ch import Task_1ch
path = '../cython/'
sys.path.append(path)


task = Task_1ch(n=4)

folder = '../Results/Learn_Positions'
# for i in range(4):
#     f = folder + '/moves' + "%03d" % (i + 1) + '.npy'
#     temp = np.load(f)
#     print temp

f = folder + '/Records.npy'
temp = np.load(f)
# print "History of learning by moves: \n", temp["moves"]
connections["PPC.theta1 -> PFC.theta1"].weights = temp["Wppc_pfc1"][-1]
connections["PFC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wpfc_str1"][-1]
connections["PPC.theta1 -> STR_PFC_PPC.theta1"].weights = temp["Wppc_str1"][-1]
connections["PPC.theta2 -> PFC.theta2"].weights = temp["Wppc_pfc2"][-1]
connections["PFC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wpfc_str2"][-1]
connections["PPC.theta2 -> STR_PFC_PPC.theta2"].weights = temp["Wppc_str1"][-1]
time = trial(task, ncues=1, wholeFig=True, debugging=True)
print " Moves needed to reach a position after learning: ", task.records["moves"][0]

histor = history()
ctx = histor["CTX"]["mot"][:time]
pfc1 = histor["PFC"]["theta1"][:time]
pfc2 = histor["PFC"]["theta2"][:time]
sma1 = histor["SMA"]["theta1"][:time]
sma2 = histor["SMA"]["theta2"][:time]
arm1 = histor["ARM"]["theta1"][:time]
arm2 = histor["ARM"]["theta2"][:time]
ppc1 = histor["PPC"]["theta1"][:time]
ppc2 = histor["PPC"]["theta2"][:time]


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

plt.figure()
plt.plot(ctx)
plt.title('CTX')
# plt.show()
# Display cortical activity during the single trial
if 0: display_ctx(histor, duration)  # , "single-trial.pdf")
