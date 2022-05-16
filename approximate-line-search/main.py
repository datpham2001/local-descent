from method import *
from visualize import visualize
# do something
import numpy as np 

#function
def f_example_4_2(x):
    return x[0]**2 + x[0] * x[1] + x[1]**2

def f_example_4_2_grad(x):
    return np.array([2 * x[0] + x[1], x[0] + 2 * x[1]])


# alpha_first = backtracking_line_search(f_example_4_2, f_example_4_2_grad, np.array([1, 2]), np.array([-1, -1]), 10)
# print(f"[INFO] Step size alpha: {alpha_first:.4f} when applying the first Wolfe condition")

# alpha_both = backtracking_line_search_two(f_example_4_2, f_example_4_2_grad, np.array([1, 2]), np.array([-1, -1]), 10)
# print(f"[INFO] Step size alpha: {alpha_both:.4f} when applying the first and the second Wolfe condition")


alpha_strong = strong_backtracking(f_example_4_2, f_example_4_2_grad, np.array([1, 2]), np.array([-1, -1]), 10)
print(f"[INFO] Step size alpha: {alpha_strong:.4f} when applying the strong Wolfe condition")

point=np.array([1,2])
direction=np.array([-1,-1])
end=point+alpha_strong*direction
        
visualize(f_example_4_2,point,end)