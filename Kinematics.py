# %%
# thump plot trial
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import os
import pandas as pd
import re
import math
import seaborn as sns

# reading coordinate files
# For A sample
thump1 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_1_Right_Hand_A_Raw.csv", header=None)
thump2 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_2_Right_Hand_A_Raw.csv", header=None)
thump3 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_3_Right_Hand_A_Raw.csv", header=None)
thump4 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_4_Right_Hand_A_Raw.csv", header=None)
index1 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_5_Right_Hand_A_Raw.csv", header=None)
index2 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_6_Right_Hand_A_Raw.csv", header=None)
index3 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_7_Right_Hand_A_Raw.csv", header=None)
index4 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_8_Right_Hand_A_Raw.csv", header=None)
mid1 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_9_Right_Hand_A_Raw.csv",
                   header=None)
mid2 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_10_Right_Hand_A_Raw.csv",
                   header=None)
mid3 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_11_Right_Hand_A_Raw.csv",
                   header=None)
mid4 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_12_Right_Hand_A_Raw.csv",
                   header=None)
ring1 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_13_Right_Hand_A_Raw.csv", header=None)
ring2 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_14_Right_Hand_A_Raw.csv", header=None)
ring3 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_15_Right_Hand_A_Raw.csv", header=None)
ring4 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_16_Right_Hand_A_Raw.csv", header=None)
pin1 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_17_Right_Hand_A_Raw.csv",
                   header=None)
pin2 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_18_Right_Hand_A_Raw.csv",
                   header=None)
pin3 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_19_Right_Hand_A_Raw.csv",
                   header=None)
pin4 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_20_Right_Hand_A_Raw.csv",
                   header=None)
origin = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_A_Results/LM_0_Right_Hand_A_Raw.csv", header=None)


# turning coordinates from datasets to numpy arrays


def coor_to_array(coordinate):
    T = []
    import numpy as np

    for i in range(0, coordinate.shape[1]):
        T.append(coordinate[i])

    T = np.array(T)
    return T


thump1 = coor_to_array(thump1)
thump2 = coor_to_array(thump2)
thump3 = coor_to_array(thump3)
thump4 = coor_to_array(thump4)
index1 = coor_to_array(index1)
index2 = coor_to_array(index2)
index3 = coor_to_array(index3)
index4 = coor_to_array(index4)
mid1 = coor_to_array(mid1)
mid2 = coor_to_array(mid2)
mid3 = coor_to_array(mid3)
mid4 = coor_to_array(mid4)
ring1 = coor_to_array(ring1)
ring2 = coor_to_array(ring2)
ring3 = coor_to_array(ring3)
ring4 = coor_to_array(ring4)
pin1 = coor_to_array(pin1)
pin2 = coor_to_array(pin2)
pin3 = coor_to_array(pin3)
pin4 = coor_to_array(pin4)
origin = coor_to_array(origin)

# Fixing wrist origin point at (0,0,0)
t1 = np.subtract(thump1, origin)
t2 = np.subtract(thump2, origin)
t3 = np.subtract(thump3, origin)
t4 = np.subtract(thump4, origin)
i1 = np.subtract(index1, origin)
i2 = np.subtract(index2, origin)
i3 = np.subtract(index3, origin)
i4 = np.subtract(index4, origin)
m1 = np.subtract(mid1, origin)
m2 = np.subtract(mid2, origin)
m3 = np.subtract(mid3, origin)
m4 = np.subtract(mid4, origin)
r1 = np.subtract(ring1, origin)
r2 = np.subtract(ring2, origin)
r3 = np.subtract(ring3, origin)
r4 = np.subtract(ring4, origin)
p1 = np.subtract(pin1, origin)
p2 = np.subtract(pin2, origin)
p3 = np.subtract(pin3, origin)
p4 = np.subtract(pin4, origin)
o = np.subtract(origin, origin)

# For O Sample

thumpo1 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_1_Right_Hand_O_Raw.csv", header=None)
thumpo2 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_2_Right_Hand_O_Raw.csv", header=None)
thumpo3 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_3_Right_Hand_O_Raw.csv", header=None)
thumpo4 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_4_Right_Hand_O_Raw.csv", header=None)
indexo1 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_5_Right_Hand_O_Raw.csv", header=None)
indexo2 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_6_Right_Hand_O_Raw.csv", header=None)
indexo3 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_7_Right_Hand_O_Raw.csv", header=None)
indexo4 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_8_Right_Hand_O_Raw.csv", header=None)
mido1 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_9_Right_Hand_O_Raw.csv",
                   header=None)
mido2 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_10_Right_Hand_O_Raw.csv",
                   header=None)
mido3 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_11_Right_Hand_O_Raw.csv",
                   header=None)
mido4 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_12_Right_Hand_O_Raw.csv",
                   header=None)
ringo1 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_13_Right_Hand_O_Raw.csv", header=None)
ringo2 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_14_Right_Hand_O_Raw.csv", header=None)
ringo3 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_15_Right_Hand_O_Raw.csv", header=None)
ringo4 = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_16_Right_Hand_O_Raw.csv", header=None)
pino1 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_17_Right_Hand_O_Raw.csv",
                   header=None)
pino2 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_18_Right_Hand_O_Raw.csv",
                   header=None)
pino3 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_19_Right_Hand_O_Raw.csv",
                header=None)
pino4 = pd.read_csv("C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_20_Right_Hand_O_Raw.csv",
                   header=None)
origino = pd.read_csv(
    "C:/Users/sseli/Downloads/Right_Hand (1)/Right_Hand/Right_Hand_O_Results/LM_0_Right_Hand_O_Raw.csv", header=None)


def coor_to_array(coordinate):  # calculate vx,vy,vz from coordinates and frame rate
    T = []
    import numpy as np

    for i in range(0, coordinate.shape[1]):
        T.append(coordinate[i])

    T = np.array(T)
    return T


thumpo1 = coor_to_array(thumpo1)
thumpo2 = coor_to_array(thumpo2)
thumpo3 = coor_to_array(thumpo3)
thumpo4 = coor_to_array(thumpo4)
indexo1 = coor_to_array(indexo1)
indexo2 = coor_to_array(indexo2)
indexo3 = coor_to_array(indexo3)
indexo4 = coor_to_array(indexo4)
mido1 = coor_to_array(mido1)
mido2 = coor_to_array(mido2)
mido3 = coor_to_array(mido3)
mido4 = coor_to_array(mido4)
ringo1 = coor_to_array(ringo1)
ringo2 = coor_to_array(ringo2)
ringo3 = coor_to_array(ringo3)
ringo4 = coor_to_array(ringo4)
pino1 = coor_to_array(pino1)
pino2 = coor_to_array(pino2)
pino3 = coor_to_array(pino3)
pino4 = coor_to_array(pino4)
origino = coor_to_array(origino)

t1o = np.subtract(thumpo1, origino)
t2o = np.subtract(thumpo2, origino)
t3o = np.subtract(thumpo3, origino)
t4o = np.subtract(thumpo4, origino)
i1o = np.subtract(indexo1, origino)
i2o = np.subtract(indexo2, origino)
i3o = np.subtract(indexo3, origino)
i4o = np.subtract(indexo4, origino)
m1o = np.subtract(mido1, origino)
m2o = np.subtract(mido2, origino)
m3o = np.subtract(mido3, origino)
m4o = np.subtract(mido4, origino)
r1o = np.subtract(ringo1, origino)
r2o = np.subtract(ringo2, origino)
r3o = np.subtract(ringo3, origino)
r4o = np.subtract(ringo4, origino)
p1o = np.subtract(pino1, origino)
p2o = np.subtract(pino2, origino)
p3o = np.subtract(pino3, origino)
p4o = np.subtract(pino4, origino)
oo = np.subtract(origino, origino)

# Plotting frames of hand before and after point fixing for comparison
ax = plt.axes(projection='3d')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(thump1), 90):
    x = []
    y = []
    z = []
    x.extend([origin[i][0], thump1[i][0], thump2[i][0], thump3[i][0], thump4[i][0]])
    y.extend([origin[i][1], thump1[i][1], thump2[i][1], thump3[i][1], thump4[i][1]])
    z.extend([origin[i][2], thump1[i][2], thump2[i][2], thump3[i][2], thump4[i][2]])
    ax.plot3D(x, y, z, color='red')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')

for i in range(0, len(thump1), 90):
    x = []
    y = []
    z = []
    x.extend([origin[i][0], index1[i][0], index2[i][0], index3[i][0], index4[i][0]])
    y.extend([origin[i][1], index1[i][1], index2[i][1], index3[i][1], index4[i][1]])
    z.extend([origin[i][2], index1[i][2], index2[i][2], index3[i][2], index4[i][2]])
    ax.plot3D(x, y, z, color='red')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 90):
    x = []
    y = []
    z = []
    x.extend([mid1[i][0], mid2[i][0], mid3[i][0], mid4[i][0]])
    y.extend([mid1[i][1], mid2[i][1], mid3[i][1], mid4[i][1]])
    z.extend([mid1[i][2], mid2[i][2], mid3[i][2], mid4[i][2]])
    ax.plot3D(x, y, z, color='red')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 90):
    x = []
    y = []
    z = []
    x.extend([ring1[i][0], ring2[i][0], ring3[i][0], ring4[i][0]])
    y.extend([ring1[i][1], ring2[i][1], ring3[i][1], ring4[i][1]])
    z.extend([ring1[i][2], ring2[i][2], ring3[i][2], ring4[i][2]])
    ax.plot3D(x, y, z, color='red')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 90):
    x = []
    y = []
    z = []
    x.extend([origin[i][0], pin1[i][0], pin2[i][0], pin3[i][0], pin4[i][0]])
    y.extend([origin[i][1], pin1[i][1], pin2[i][1], pin3[i][1], pin4[i][1]])
    z.extend([origin[i][2], pin1[i][2], pin2[i][2], pin3[i][2], pin4[i][2]])
    ax.plot3D(x, y, z, color='red')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')

for i in range(0, len(thump1), 9):
    x = []
    y = []
    z = []
    x.extend([index1[i][0], mid1[i][0], ring1[i][0], pin1[i][0]])
    y.extend([index1[i][1], mid1[i][1], ring1[i][1], pin1[i][1]])
    z.extend([index1[i][2], mid1[i][2], ring1[i][2], pin1[i][2]])
    ax.plot3D(x, y, z, color='green')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')

for i in range(0, len(thump1), 95):
    x = []
    y = []
    z = []
    x.extend([o[i][0], t1[i][0], t2[i][0], t3[i][0], t4[i][0]])
    y.extend([o[i][1], t1[i][1], t2[i][1], t3[i][1], t4[i][1]])
    z.extend([o[i][2], t1[i][2], t2[i][2], t3[i][2], t4[i][2]])
    ax.plot3D(x, y, z, color='blue')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 95):
    x = []
    y = []
    z = []
    x.extend([o[i][0], i1[i][0], i2[i][0], i3[i][0], i4[i][0]])
    y.extend([o[i][1], i1[i][1], i2[i][1], i3[i][1], i4[i][1]])
    z.extend([o[i][2], i1[i][2], i2[i][2], i3[i][2], i4[i][2]])
    ax.plot3D(x, y, z, color='blue')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 95):
    x = []
    y = []
    z = []
    x.extend([m1[i][0], m2[i][0], m3[i][0], m4[i][0]])
    y.extend([m1[i][1], m2[i][1], m3[i][1], m4[i][1]])
    z.extend([m1[i][2], m2[i][2], m3[i][2], m4[i][2]])
    ax.plot3D(x, y, z, color='blue')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 95):
    x = []
    y = []
    z = []
    x.extend([r1[i][0], r2[i][0], r3[i][0], r4[i][0]])
    y.extend([r1[i][1], r2[i][1], r3[i][1], r4[i][1]])
    z.extend([r1[i][2], r2[i][2], r3[i][2], r4[i][2]])
    # x = np.negative(x)
    # y = np.negative(y)
    # z = np.negative(z)

    ax.plot3D(x, y, z, color='blue')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 95):
    x = []
    y = []
    z = []
    x.extend([o[i][0], p1[i][0], p2[i][0], p3[i][0], p4[i][0]])
    y.extend([o[i][1], p1[i][1], p2[i][1], p3[i][1], p4[i][1]])
    z.extend([o[i][2], p1[i][2], p2[i][2], p3[i][2], p4[i][2]])
    ax.plot3D(x, y, z, color='blue')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
for i in range(0, len(thump1), 95):
    x = []
    y = []
    z = []
    x.extend([i1[i][0], m1[i][0], r1[i][0], p1[i][0]])
    y.extend([i1[i][1], m1[i][1], r1[i][1], p1[i][1]])
    z.extend([i1[i][2], m1[i][2], r1[i][2], p1[i][2]])
    ax.plot3D(x, y, z, color='blue')
    ax.scatter3D(x, y, z, c=z, cmap='hsv')
ax.view_init(120, 80)
plt.show()

# Comparing finger movement before and after the correction using dice coefficient


def dice_coef2(coor1, coor2):
    y_true = []
    y_pred = []
    l = coor1.shape[1]
    for i in range(0, l):
        y_true.append(coor1[i])
        y_pred.append(coor2[i])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    intersection = (np.logical_and(y_pred, y_true))
    union = np.sum(y_pred) + np.sum(y_true)
    inter = np.sum(intersection)

    dice = (2 * inter) / union

    return dice / 6


t4 = np.negative(t4)
print(dice_coef2(thump4, t4))
print(cv2.matchShapes(thump4, t4, 1, 0.0))

# Kinematic data analysis


def linear_velocity_xyz(coordinate,f_rate) : #calculate vx,vy,vz from coordinates and frame rate
    T = []
    import numpy as np

    for i in range (0, coordinate.shape[1]) :
        T.append(coordinate[i])

    T = np.array(T)
    distance = np.diff(T, axis=0)
    f_rate = 24
    velocity = distance*f_rate # time = 1/frame rate
    return velocity



def angular_velocity_xy(coordinate,f_rate) :#calculate angular velocity in xy plane from coordinates and frame rate
    T = []
    for i in range (0, coordinate.shape[1]) :
        T.append(coordinate[i])

    T = np.array(T)
    theta_x = []
    for i in T:
        theta_x.append (math.atan2(i[1], i[0]))
    theta_x = np.array(theta_x)
    ang_v = (np.diff(theta_x))*f_rate
    return ang_v


def linear_velocity(coordinate, f_rate) :# calculate velocity vector magnitude
    T = []
    for i in range (0, coordinate.shape[1]) :
        T.append(coordinate[i])
    T = np.array(T)
    distance = np.diff(T, axis=0)
    f_rate = 24
    v = distance*f_rate
    velocity = []
    for i in v:
        velocity.append(math.sqrt(i[0]**2+ i[1]**2 + i[2]**2))
    return velocity


def linear_acc_xyz(coordinate,f_rate) : # Calculate linear x,y,z acceleration components
    T = []
    import numpy as np

    for i in range (0, coordinate.shape[1]) :
        T.append(coordinate[i])

    T = np.array(T)
    distance = np.diff(T, axis=0)
    f_rate = 24
    velocity = distance*f_rate
    A = np.diff(velocity, axis = 0)
    acceleration = A*f_rate
    return acceleration


def angular_acc_xy(coordinate, f_rate): # Calculate frame-by-frame angular acceleration in xy plane
    T = []
    for i in range (0, coordinate.shape[1]) :
        T.append(coordinate[i])

    T = np.array(T)
    theta_x = []
    for i in T:
        theta_x.append (math.atan2(i[1], i[0]))
    theta_x = np.array(theta_x)
    ang_v = (np.diff(theta_x))*f_rate
    m = np.diff(ang_v, axis =0)
    ang_a = m*f_rate
    return ang_a


def linear_acc(coordinate, f_rate) :# calculate acceleration vector magnitude
    T = []
    for i in range (0, coordinate.shape[1]) :
        T.append(coordinate[i])
    T = np.array(T)
    distance = np.diff(T, axis=0)
    f_rate = 24
    v = distance*f_rate
    velocity = []
    for i in v:
        velocity.append(math.sqrt(i[0]**2+ i[1]**2 + i[2]**2))
    velocity = np.array(velocity)
    acc = (np.diff(velocity))*f_rate
    return acc

# Plotting results
# Linear Velocity
velocityA1 = [linear_velocity(t4, 24), linear_velocity(t4o, 24)]
sns.boxplot(data=velocityA1).set(title = 'Thump')
velocityA2 = [linear_velocity(i4, 24), linear_velocity(i4o, 24)]
sns.boxplot(data=velocityA2).set(title = 'Index')
velocityA3 = [linear_velocity(m4, 24), linear_velocity(m4o, 24)]
sns.boxplot(data=velocityA3).set(title = 'Middle')
velocityA4 = [linear_velocity(r4, 24), linear_velocity(r4o, 24)]
sns.boxplot(data=velocityA4).set(title = 'Ring')
velocityA5 = [linear_velocity(p4, 24), linear_velocity(p4o, 24)]
sns.boxplot(data=velocityA5).set(title = 'Little')

# Angular Velocity
ang_velocityA1 = [angular_velocity_xy(t4, 24), angular_velocity_xy(t4o, 24)]
sns.boxplot(data=ang_velocityA1).set(title = 'Thump')
ang_velocityA2 = [angular_velocity_xy(i4, 24), angular_velocity_xy(i4o, 24)]
sns.boxplot(data=ang_velocityA2).set(title = 'Index')
ang_velocityA3 = [angular_velocity_xy(m4, 24), angular_velocity_xy(m4o, 24)]
sns.boxplot(data=ang_velocityA3).set(title = 'Middle')
ang_velocityA4 = [angular_velocity_xy(r4, 24), angular_velocity_xy(r4o, 24)]
sns.boxplot(data=ang_velocityA4).set(title = 'Ring')
ang_velocityA5 = [angular_velocity_xy(p4, 24), angular_velocity_xy(p4o, 24)]
sns.boxplot(data=ang_velocityA5).set(title = 'Little')

# acceleration difference
accA1 = [np.diff(linear_acc(t4, 24)), np.diff(linear_acc(t4o, 24))]
sns.boxplot(data=accA1).set(title = 'Thump')
accA2 = [np.diff(linear_acc(i4, 24)), np.diff(linear_acc(i4o, 24))]
sns.boxplot(data=accA2).set(title = 'Index')
accA3 = [np.diff(linear_acc(m4, 24)), np.diff(linear_acc(m4o, 24))]
sns.boxplot(data=accA3).set(title = 'Middle')
accA4 = [np.diff(linear_acc(r4, 24)), np.diff(linear_acc(r4o, 24))]
sns.boxplot(data=accA4).set(title = 'Ring')
accA5 = [np.diff(linear_acc(p4, 24)), np.diff(linear_acc(p4o, 24))]
sns.boxplot(data=accA5).set(title = 'Little')

