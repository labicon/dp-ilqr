import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
pwd = os.getcwd()
print(pwd)
data_baseline = np.load('drone_sim_data.npy')
# print(data_baseline)

def func(num, dataSet, dotsEgo, dotsOpp, dotsOpp2=None):
# def func(num, dataSet, dotsEgo):
    # NOTE: there is no .set_data() for 3 dim data...
    dotsEgo.set_data([dataSet[0][num]], [dataSet[1][num]])
    dotsEgo.set_3d_properties([dataSet[2][num]], 'z')

    dotsOpp.set_data([dataSet[3][num]], [dataSet[4][num]])
    dotsOpp.set_3d_properties([dataSet[5][num]], 'z')

    # dotsOpp2.set_data([dataSet[6][num]], [dataSet[7][num]])
    # dotsOpp2.set_3d_properties([dataSet[8][num]], 'z')

    return dotsEgo
 
 
# THE DATA POINTS
dataSet = data_baseline
# print(dataSet)
dataSet = np.array(dataSet).T
# print(dataSet.shape)
numDataPoints = 50
 
# GET SOME MATPLOTLIB OBJECTS
fig = plt.figure()
ax = Axes3D(fig)
dotsEgo = ax.plot(dataSet[0][0], dataSet[1][0], dataSet[2][0], 'go')[0] # For scatter plot
x=np.linspace(0, 45, numDataPoints)
track_inner = ax.plot(x, np.sin(0.2*x) + 1, np.zeros(numDataPoints), 'r')[0]
track_outer = ax.plot(x, np.sin(0.2*x) + 3, np.zeros(numDataPoints), 'r')[0]
dotsOpp = ax.plot(dataSet[3][0], dataSet[4][0], dataSet[5][0], 'bo')[0]
# dotsOpp2 = ax.plot(dataSet[6][0], dataSet[7][0], dataSet[8][0], 'co')[0]
# NOTE: Can't pass empty arrays into 3d version of plot() 
# AXES PROPERTIES]
ax.set_xlim3d([-1, 45])
ax.set_ylim3d([-1, 5])
ax.set_zlim3d([-5, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
 
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, dotsEgo, dotsOpp), interval=50)
# line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, dotsEgo), interval=50)
line_ani.save('two_drone_sine_test.gif')
plt.show()