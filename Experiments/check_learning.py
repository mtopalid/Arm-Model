#!/usr/bin/env python
import sys
import os
path = '../cython/'
sys.path.append(path)
from display import *
from trial import *
from task_1ch import Task_1ch

# for i in range(4):
#     f = folder + '/moves' + "%03d" % (i + 1) + '.npy'
#     temp = np.load(f)
#     print temp

folder = '../Results/Learn_Positions/Backup'#+M1_learning#

f = folder + '/Records.npy'
temp = np.load(f)
# print "History of learning by moves: \n", temp["moves"]
connections["PPC.theta1 -> SMA.theta1"].weights = temp["Wppc_sma1"]
connections["SMA.theta1 -> STR_SMA_PPC.theta1"].weights = temp["Wsma_str1"]
connections["PPC.theta1 -> STR_SMA_PPC.theta1"].weights = temp["Wppc_str1"]
connections["PPC.theta2 -> SMA.theta2"].weights = temp["Wppc_sma2"]
connections["SMA.theta2 -> STR_SMA_PPC.theta2"].weights = temp["Wsma_str2"]
connections["PPC.theta2 -> STR_SMA_PPC.theta2"].weights = temp["Wppc_str2"]
# connections["M1.theta1 -> M1.theta2"].weights = temp["Wm1_1"]
# connections["M1.theta2 -> M1.theta1"].weights = temp["Wm1_2"]

task = Task_1ch(n=81 * 4)
for i in range(101):
    time = trial_continuous(task, trial_n=i, ncues=1, wholeFig=True, debugging=True, debugging_arm= False)
# print " Moves needed to reach a position after learning: ", task.records["moves"][0]

histor = history()
ctx = histor["CTX"]["mot"][:time]
sma1 = histor["SMA"]["theta1"][:time]
sma2 = histor["SMA"]["theta2"][:time]
m11 = histor["M1"]["theta1"][:time]
m12 = histor["M1"]["theta2"][:time]
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
plt.plot(m11)
plt.title('M11')
plt.figure()
plt.plot(m12)
plt.title('M12')

plt.figure()
plt.plot(sma1)
plt.title('SMA1')
plt.figure()
plt.plot(sma2)
plt.title('SMA2')

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
