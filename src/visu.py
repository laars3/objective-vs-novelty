import numpy as np, pyvista as pv

def render(grid):

    filled=np.argwhere(grid==1)
    x_base= (grid.shape[0]//2)

    # hud stuffs
    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.background_color='white'

    cube=pv.Cube()

    points= pv.PolyData(filled.astype(float))
    points["role"] = np.sign(filled[:,0] - x_base)
    cubes = points.glyph(geom=cube, scale=False, orient=False)

    plotter.add_mesh(cubes,scalars="role", cmap="coolwarm",show_edges=True, edge_color='black', line_width=1, show_scalar_bar=False)

    plotter.show()

if __name__ == "__main__":
    from env import Environment
    env=Environment((24,8,12))
    render(env.best_possible_bridge(40))