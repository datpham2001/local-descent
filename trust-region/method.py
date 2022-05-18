from utils import*

def solve_trust_region_subproblem(f_grad, H, x0, delta):
    x = cp.Variable(x0.shape)
    #tối tiểu
    objective = cp.Minimize((x - x0).T @ f_grad(x0) + cp.quad_form(x - x0, H(x0)) * 0.5) 
    # ràng buộc
    constraints = [cp.norm(x - x0) <= delta]
    # giải
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value, problem.value

def trust_region_descent(f, f_grad, H, x, k_max, eta1=0.4, eta2=0.6, gamma1=0.5, gamma2=1.05, delta=1.0):
    pts, rads = [], []
    
    y = f(x)
    for k in tqdm(range(k_max)):
        # tối tiểu cái f^ để thu được bước x' tiếp
        # và giá trị của f^(x'), giá trị này thì lưu vào y_
        x_, y_ = solve_trust_region_subproblem(f_grad, H, x, delta)     
        r = (y - f(x_)) / (y - y_)  # tính cái tỷ số

        # cập nhật vùng tin cậy dựa theo cái tỷ số đó
        if r < eta1:  
            # dưới ngưỡng thì thhu hẹp bán kính lại
            delta *= gamma1
        else:
            # ngược lại, bước x' dự đoán được chấp thuận, cập nhật nó
            x, y = x_, y_
            if r > eta2: # mở rộng cái vùng tin cậy lên
                delta *= gamma2
        pts.append(x_)
        rads.append(delta)
    return x_, pts, rads