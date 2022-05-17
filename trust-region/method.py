from utils import*

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