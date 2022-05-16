import numpy as np
from matplotlib import pyplot as plt

# function
# objective function
def objective(x):
	return x[0]**2 + x[0] * x[1] + x[1]**2
 
# gradient for the objective function
def gradient(x):
	return np.array([2 * x[0] + x[1], x[0] + 2 * x[1]])
 

def visualize(objective, point, end):

  # define range
  r_min, r_max = -10.0, 10.0
  # prepare inputs
  inputs =[np.arange(r_min, r_max, 0.1),np.arange(r_min, r_max, 0.1)]
  # plot inputs vs objective
  ax = plt.axes(projection ='3d')
  x=inputs[0]
  y=inputs[1]
  target=[]
  for i in range(len(x)):
      target.append(objective([x[i],y[i]]))
  ax.plot3D(x, y, target, 'green')

  ax.scatter3D(point[0], point[1], objective([point[0],point[1]]), color=['blue'])
  ax.scatter3D(end[0], end[1], objective([end[0],end[1]]),color=['red'])
  plt.show()
