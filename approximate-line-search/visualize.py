import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def visualize(objective, point, end):
  
  
    # define range
    r_min, r_max = -3,3
    # prepare inputs
    inputs =[np.arange(r_min, r_max, 0.1),np.arange(r_min+1, r_max+1, 0.1)] 
    # compute targets
    # targets = [objective(x) for x in inputs]
    # plot inputs vs objective
    ax = plt.axes(projection ='3d')
    x=inputs[0]
    y=inputs[1]
    z=[]
    for i in range(len(x)):
        print(f'x={x[i]} y={y[i]} z={objective([x[i],y[i]])}')
        # ax.scatter3D(x[i],y[i],objective([x[i],y[i]]), 'green')
        z.append(objective([x[i],y[i]]))
    ax.plot3D(x, y, z, 'green')

    ax.scatter3D(point[0], point[1], objective([point[0],point[1]]), color=['blue'])
    ax.scatter3D(end[0], end[1], objective([end[0],end[1]]),color=['red'] )

    print(point)

    print(end)
    plt.show()