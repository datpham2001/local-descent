import numpy as np
from matplotlib import pyplot as plt

# function
def objective(x):
  return np.sin(x) + x**2

def gradient(x):
  return np.cos(x) + 2*x

def visualize(objective, point, end):
  
  # define range
  r_min, r_max = -10.0, 20.0
  # prepare inputs
  inputs = np.arange(r_min, r_max, 0.1)
  # compute targets
  targets = [objective(x) for x in inputs]
  # plot inputs vs objective
  plt.plot(inputs, targets, '--', label='objective')
  # plot start and end of the search
  plt.plot([point], [objective(point)], 's', color='g')
  plt.plot([end], [objective(end)], 's', color='r')
  plt.legend()
  plt.show()