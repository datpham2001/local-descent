from methods import *
from definitions import *

if __name__ == '__main__':
  
  '''
    Armijo method 
  '''
  
  #Definitions
  point = -10.0  #starting point
  direction = 100 #direction
  method = Method(objective, gradient, point, direction)
  
  # Get the result of the armijo method
  method.Armijo()

  