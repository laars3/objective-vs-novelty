from pathlib import Path
import numpy as np, pyvista as pv, json

visu_file= Path("runs/novelty_bridge_s16.json")

def load_grid(path, shape=(24,8,12)):
    with open(path) as f:
        blocks=json.load(f)["best_blocks"]

    grid=np.zeros(shape, dtype=np.int8)

    for x,y,z in blocks:
        grid[x,y,z] =1

    return grid

def render(grid, plotter):

    filled=np.argwhere(grid==1)
    x_base= (grid.shape[0]//2)

    cube=pv.Cube()

    points= pv.PolyData(filled.astype(float))
    points["role"] = np.sign(filled[:,0] - x_base)
    cubes = points.glyph(geom=cube, scale=False, orient=False)

    plotter.add_mesh(cubes,scalars="role", cmap="coolwarm",show_edges=True, edge_color='black', line_width=1, show_scalar_bar=False)

    nx,ny,nz = grid.shape

    box = pv.Box(bounds=(-0.5, nx - 0.5,-0.5, ny-0.5, -0.5, nz-0.5))

    plotter.add_mesh(box, style="wireframe", color="grey")

if __name__ == "__main__":
    from env import Environment
    env=Environment((24,8,12))

    ref=env.best_possible_bridge(40)

    plotter = pv.Plotter(shape=(1,2))

    # hud stuffs
    plotter.show_axes()
    plotter.background_color = 'white'

    plotter.subplot(0,0)
    render(ref, plotter)

    plotter.subplot(0,1)
    render(load_grid(visu_file),plotter)

    plotter.link_views()
    plotter.show()
    print("success")