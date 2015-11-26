import sys

t = '../cython/'
sys.path.append(t)
from kinematics import *


def compute_rewards(target = np.array([1,6])):

    rewards = np.zeros((9,9,9,9))
    angles = np.linspace(70, 110, num=9)

    for i in range(9):
        for j in range(9):

            initial_pos = np.array([i,j])

            for k in range(9):
                for l in range(9):

                    pos = np.array([k, l])
                    # Compute target angles
                    temp = np.array([angles[target[0]], angles[target[1]]])
                    # Compute target coordinations
                    cor_tar = coordinations(conver_degr2rad(temp[0]), conver_degr2rad(temp[1]))

                    # Compute initial position angles
                    temp = np.array([angles[initial_pos[0]], angles[initial_pos][1]])
                    # Compute initial position coordinations
                    cor = coordinations(conver_degr2rad(temp[0]), conver_degr2rad(temp[1]))
                    # Compute distance between initial position and target
                    d_init = distance(cor_tar, cor)

                    # Compute final position angles

                    temp = np.array([angles[pos[0]], angles[pos[1]]])
                    # Compute final position coordinations
                    cor = coordinations(conver_degr2rad(temp[0]), conver_degr2rad(temp[1]))
                    # Compute distance between initial position and target
                    d_final = distance(cor_tar, cor)

                    # Compute reward: 0.5 if it moved closer to the target
                    #                 1.0 if reached the target
                    #                 0.0 else

                    if d_final == 0.0:
                        rewards[i,j,k,l] = 1
                    elif d_final < d_init:
                        rewards[i,j,k,l] = 0.5
                    else:
                        rewards[i,j,k,l] = 0

    return rewards
#
# np.set_printoptions(precision=3, threshold='nan')
# rewards = compute_rewards()
# for i in range(9):
#     for j in range(9):
#                 print i,j
#                 print rewards[i,j]
def noise(shape, s=0.005, initial=0.5):
    N = np.random.normal(initial, s, shape)
    N = np.minimum(np.maximum(N, 0.0), 1.0)
    return 0.25 + (0.75 - 0.25) * N

positions = np.ones((9, 9))
# values = noise(81*81).reshape((9, 9, 9, 9))
values = noise(81).reshape((9, 9))
weights = noise(81*81).reshape((9, 9, 9, 9))

alpha_CUE = 0.0025  # 0.0005
alpha_LTP = 0.005
alpha_LTD = 0.00375
alpha_LTP_ctx = alpha_LTP ** 2  # 0.000025

buttons = np.ones((4, 2))
buttons[0, :] = [4, 1]  # [90,75]
buttons[1, :] = [1, 6]  # [75,100]
buttons[2, :] = [4, 6]  # [90,100]
buttons[3, :] = [6, 1]  # [100,75]

angles = np.linspace(70, 110, num=9)
initial_pos = np.array([4, 4])
target = buttons[1]
rewards = compute_rewards(target=target)
for i in range(70*81*81):
                    initial_pos = np.random.randint(0, 9, 2)
                    pos = np.random.randint(0, 9, 2)
    #
    # for i in range(9):
    #     for j in range(9):
    #
    #         initial_pos = np.array([i,j])
    #
    #         for k in range(9):
    #             for l in range(9):
    #
    #                 pos = np.array([k, l])
                    # Compute prediction error
                    # error = rewards[initial_pos[0], initial_pos[1], pos[0], pos[1]] - values[initial_pos[0], initial_pos[1],pos[0], pos[1]]
                    error = rewards[initial_pos[0], initial_pos[1], pos[0], pos[1]] - values[pos[0], pos[1]]
                    # Update cues values
                    # values[initial_pos[0], initial_pos[1],pos[0], pos[1]] += error * alpha_CUE
                    values[pos[0], pos[1]] += error * alpha_CUE
                    lrate = alpha_LTP if error > 0 else alpha_LTD
                    dw = error * lrate
                    W = weights[initial_pos[0], initial_pos[1], pos[0], pos[1]]
                    W += dw * (0.75 - W) * (W - 0.25)
                    weights[initial_pos[0], initial_pos[1], pos[0], pos[1]] = W

np.set_printoptions(precision=3, threshold='nan')

for i in range(9):
    for j in range(9):

        print i,j
        # print weights[i,j]
        # print rewards[i,j]
        # print values[i,j]

        # print np.argmax(values[i,j]), values[i,j].reshape(81)[np.argmax(values[i,j])]
        print np.argmax(weights[i,j]), weights[i,j].reshape(81)[np.argmax(weights[i,j])]
        print

print np.argmax(values), values.reshape(81)[np.argmax(values)]
print values