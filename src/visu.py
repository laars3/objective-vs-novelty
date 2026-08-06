import numpy as np, pyvista as pv

def render(grid):

    filled=np.argwhere(grid==1)
    plotter = pv.Plotter()

    for x,y,z in filled:
        cube=pv.Cube(center=(x,y,z))
        plotter.add_mesh(cube)

    plotter.show()

if __name__ == "__main__":
    grid = np.zeros((8,8,8),dtype=np.int8)
    grid[4,4,0]=1
    grid[4,4,1]=1
    grid[4,4,2]=1
    render(grid)