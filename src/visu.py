import numpy as np, pyvista as pv, random
from pyvista import outline_algorithm

def render(grid):

    filled=np.argwhere(grid==1)
    plotter = pv.Plotter()
    plotter.background_color='white'
    for x,y,z in filled:
        cube=pv.Cube(center=(x,y,z))
        random_color=(random.random(), random.random(), random.random())
        plotter.add_mesh(cube, color=random_color,show_edges=True, edge_color='black', line_width=1)

    plotter.show()

# if __name__ == "__main__":
#     grid = np.zeros((8,8,8),dtype=np.int8)
#     grid[4,4,0]=1
#     grid[4,4,1]=1
#     grid[4,4,2]=1
#     render(grid)