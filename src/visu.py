import numpy as np, pyvista as pv, random

def render(grid):

    filled=np.argwhere(grid==1)
    plotter = pv.Plotter()
    plotter.background_color='white'
    for x,y,z in filled:
        cube=pv.Cube(center=(x,y,z))
        random_color=(random.random(), random.random(), random.random())
        plotter.add_mesh(cube, color=random_color,show_edges=True, edge_color='black', line_width=1)

    plotter.show()
