import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# loadtxt Weight hidden2 to output
Wh2_out = np.loadtxt("./Wh2_out.txt")
Wh2_out_1 = [Wh2_out[:,0], Wh2_out[:,1], Wh2_out[:,2]]
Wh2_out_2 = [Wh2_out[:,3], Wh2_out[:,4], Wh2_out[:,5]]
Wh2_out_3 = [Wh2_out[:,6], Wh2_out[:,7], Wh2_out[:,8]]

# loadtxt Weight hidden1 to hidden2
Wh1_h2 = np.loadtxt("./Wh1_h2.txt")
Wh1_h2_1 = [Wh1_h2[:,0], Wh1_h2[:,1], Wh1_h2[:,2]]
Wh1_h2_2 = [Wh1_h2[:,3], Wh1_h2[:,4], Wh1_h2[:,5]]
Wh1_h2_3 = [Wh1_h2[:,6], Wh1_h2[:,7], Wh1_h2[:,8]]
Wh1_h2_4 = [Wh1_h2[:,9], Wh1_h2[:,10], Wh1_h2[:,11]]
Wh1_h2_5 = [Wh1_h2[:,12], Wh1_h2[:,13], Wh1_h2[:,14]]

# loadtxt Weight input to hidden1
Win_h1 = np.loadtxt("./Win_h1.txt")
Win_h1_1 = [Win_h1[:,0], Win_h1[:,1], Win_h1[:,2], Win_h1[:,3], Win_h1[:,4]]
Win_h1_2 = [Win_h1[:,5], Win_h1[:,6], Win_h1[:,7], Win_h1[:,8], Win_h1[:,9]]
Win_h1_3 = [Win_h1[:,10], Win_h1[:,11], Win_h1[:,12], Win_h1[:,13], Win_h1[:,14]]


# draw Weight hidden2 to output with 3 figures
for i in range(1,4):
    plt.figure()
    plt.xlabel('Number of Learning')
    plt.title(f'Weight Hidden2 node {i} to Output nodes')
    for j in range(3):
        plt.plot(eval('Wh2_out_' + str(i))[j], label=f"Wh2_out_{i}{j+1}")
        plt.legend()


# draw Weight hidden1 to hidden2 with 5 figures
# for i in range(1,6):
#     plt.figure()
#     plt.xlabel('Number of Learning')
#     plt.title(f'Weight Hidden1 node {i} to Hidden2 nodes')
#     for j in range(3):
#         plt.plot(eval('Wh1_h2_' + str(i))[j], label=f"Wh1_h2_{i}{j+1}")
#         plt.legend()


# draw Weight input to hidden1 with 3 figures
# for i in range(1,4):
#     plt.figure()
#     plt.xlabel('Number of Learning')
#     plt.title(f'Weight Input node {i} to Hidden1 nodes')
#     for j in range(5):
#         plt.plot(eval('Win_h1_' + str(i))[j], label=f"Win_h1_{i}{j+1}")
#         plt.legend()


# find three Weights that affect the results the most

# All weights save in one list, Weight 0 ~ 38
Weight = []

for i in range(1,4):
    for j in range(5):
        Weight.append(eval('Win_h1_' + str(i))[j][-1])
        
for i in range(1,6):
    for j in range(3):
        Weight.append(eval('Wh1_h2_' + str(i))[j][-1])

for i in range(1,4):
    for j in range(3):
        Weight.append(eval('Wh2_out_' + str(i))[j][-1])
        


sort_Weight = sorted(Weight, key=abs, reverse=True)

Most_three_Weight = sort_Weight[:3]

print(Most_three_Weight)

# Wh2_out 3 to 3
Weight_1 = Weight.index(Most_three_Weight[0]) 
print(Weight_1)

# Wh1_h2 3 to 3
Weight_2 = Weight.index(Most_three_Weight[1])
print(Weight_2)

# Wh2_out 3 to 1
Weight_3 = Weight.index(Most_three_Weight[2])
print(Weight_3)

plt.show()