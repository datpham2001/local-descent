from method import *
from func_optimization import *
from visualize import visualize

x0 = np.array([-1.0, 0.0])
# chạy trust region method
x_opt, pts_path, rads_path = trust_region_descent(rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array([-1.0, 0.0]), 10)
print(f"[INFO] Optimal trust region is: central point ({x_opt[0]:.4f},{x_opt[1]:.4f}) with radius {rads_path[-1]:.4f}")
# trực quan
visualize(rosenbrock, x0, pts_path, rads_path)