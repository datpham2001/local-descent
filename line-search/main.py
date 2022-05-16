from methods import *
from definitions import *

if __name__ == '__main__':
  
  #Initialize start point and direction
  point= np.array([1,1])
  
  direction = np.array([-2, -2])

  # print the initial conditions
  print('start={}, direction={}'.format(point, direction))
  method_armijo = Method(objective, gradient, point, direction)
  
  # Get the result of the armijo method
  method_armijo.Armijo()
  
  method_W1 = Method(objective, gradient, point, direction)
  method_W1.Wolfe1()
  #method.Wolfe2()

  