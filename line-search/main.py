from methods import *
from definitions import *

if __name__ == '__main__':
  
  #Initialize start point and direction
  point= np.array([1,1])
  direction = np.array([-2, -2])

  # print the initial conditions
  print('\nstart={}, direction={}'.format(point, direction))
  
  #initial method object
  method = Method(objective, gradient, point, direction)
  
  # Get the result of the armijo method
  print("\n===============\nARMIJO METHOD\n")
  method.Armijo()
  
  # Get the result of the Wolfe 1 condition
  print("\n===============\nWOLFE 1 CONDITION\n")
  method.Wolfe1()
  
  # Get the result of line search using bracket minimum
  print("\n===============\nBRACKET MINIMUM\n")
  method.bracket_minimum()

  