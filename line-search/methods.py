from utils import *
from definitions import *

class Method():
  def __init__(self, objective_init, gradient_init, point_init, direction_init):
    self.objective = objective_init
    self.gradient = gradient_init
    self.point = point_init
    self.direction = direction_init
  
  def print_out(self):
    print(f'Alpha: {self.alpha}')
    print(f'Function evaluations: {self.result[1]}')
    # define objective function minima
    end = self.point + self.alpha * self.direction
    # evaluate objective function minima
    print(f'f(end) = f({end}) = {objective(end)}')
    #visualize 
    visualize(self.objective, self.point, end)
    
  def Armijo(self):
    self.result = line_search_armijo(self.objective, self.point, self.direction, self.gradient(self.point))
    self.alpha = self.result[0]
    
    #print and visualize the result
    self.print_out()
    
    
  def Wolfe1(self):
    self.result = line_search_wolfe1(self.objective, self.gradient, self.point, self.direction)
    self.alpha = self.result[0]
    
    #print and visualize the result
    self.print_out()
  
  def bracket_minimum(self):
    self.result = line_search(self.objective, self.point, self.direction)
    self.alpha = self.result[0]
    
    #print and visualize the result
    self.print_out()