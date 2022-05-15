import time
import numpy as np
import cvxpy as cp
import scipy.optimize as opt
import matplotlib.pyplot as plt

from tqdm import tqdm
def solve_trust_region_subproblem(f_grad, H, x0, delta):
    x = cp.Variable(x0.shape)
    objective = cp.Minimize((x - x0).T @ f_grad(x0) + cp.quad_form(x - x0, H(x0)) * 0.5)
    constraints = [cp.norm(x - x0) <= delta]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value, problem.value

def trust_region_descent(f, f_grad, H, x, k_max, eta1=0.4, eta2=0.6, gamma1=0.5, gamma2=1.05, delta=1.0):
    pts, rads = [], []
    
    y = f(x)
    for k in tqdm(range(k_max)):
        x_, y_ = solve_trust_region_subproblem(f_grad, H, x, delta)
        r = (y - f(x_)) / (y - y_)
        if r < eta1:
            delta *= gamma1
        else:
            x, y = x_, y_
            if r > eta2:
                delta *= gamma2
        pts.append(x_)
        rads.append(delta)
    return x_, pts, rads

def rosenbrock(x, b=5):
    return (x[0] - 1)**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_grad(x, b=5):
    return np.array([2 * (x[0] - 1) - 4 * b * (x[1] - x[0]**2) * x[0], 2 * b * (x[1] - x[0]**2)])

def rosenbrock_hess(x, b=5):
    return np.array([[2 - 4 * b * x[1] + 12 * b * x[0]**2, -4 * b * x[0]]\
                     ,[-4 * b * x[0], 2 * b]])

x0 = np.array([-1.0, 0.0])
x_opt, pts_path, rads_path = trust_region_descent(rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array([-1.0, 0.0]), 10)
print(f"[INFO] Optimal trust region is: central point ({x_opt[0]:.4f},{x_opt[1]:.4f}) with radius {rads_path[-1]:.4f}")

fig, ax = plt.subplots()
fig.set_size_inches(10, 12)

x = np.linspace(-2, 2, 800)
y = np.linspace(-2, 2, 600)
z = rosenbrock(np.meshgrid(x, y))

ax.contour(x, y, z, levels=np.logspace(-1, 3, 10), alpha=0.2)
ax.plot(*x0, marker='o', markersize=5, color='r')
ax.annotate('Inital point (x0)', xy=x0, xytext=(-1.0, 0.5), arrowprops={'arrowstyle': '->', 'color': 'red'})

for point, radius in zip(pts_path, rads_path):
    ax.plot(*point, marker='o', markersize=5, color='b')
    trust_region = plt.Circle(point, radius, color='b', fill=False)
    ax.add_artist(trust_region)
ax.plot(*pts_path[-1], marker='o', markersize=5, color='g')
ax.annotate('Optimal point (x*)', xy=pts_path[-1], xytext=(1.0, 1.2), arrowprops={'arrowstyle': '->', 'color': 'green'})
ax.set_title("Contour plot of the Rosenbrock function")

plt.show()