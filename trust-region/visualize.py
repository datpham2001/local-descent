from utils import*
def visualize(rosenbrock, x0, pts_path, rads_path):
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