from pathlib import Path
import numpy as np, pyvista as pv, json

visu_file= Path("runs/novelty_bridge_s16.json")

def load_grid(path, shape=(24,8,12)):
    with open(path) as f:
        blocks=json.load(f)["best_blocks"]

    return blocks_to_grid(blocks, shape)

def blocks_to_grid(blocks, shape=(24,8,12)):
    grid=np.zeros(shape, dtype=np.int8)

    for x,y,z in blocks:
        grid[x,y,z] =1

    return grid

def gify(frames_path, gif_path, ref):
    with open(frames_path) as f:
        data= json.load(f)

    blocks = data["gen_blocks"]
    best = data["gen_best"]

    gifplot = pv.Plotter(shape=(1,2), off_screen=True, window_size=(1200,450))
    gifplot.background_color="white"

    gifplot.subplot(0,0)
    render(ref, gifplot)
    gifplot.add_text("reference, reach 11", font="arial", font_size=10, color="black")

    gifplot.link_views()

    gifplot.open_gif(gif_path, fps=10)

    for i in range(len(blocks)):
        gifplot.subplot(0,1)
        render(blocks_to_grid(blocks[i]), gifplot)
        gifplot.add_text(f"gen {i}, reach {best[i]}", font_size=10, font="arial", color="black", name="label")
        gifplot.write_frame()

    gifplot.close()

def still(names, ref, out_path):
    plotter = pv.Plotter(shape=(2,3),off_screen=True,window_size=(1200,800))
    plotter.background_color="white"

    plotter.subplot(0,0)

    render(ref, plotter)
    plotter.add_text("reference, reach 11", font="arial", font_size=10, color="black")
    plotter.link_views()

    for i in range(len(names)):
        pos = i + 1
        plotter.subplot(pos//3, pos % 3)

        runs_path = f"runs/{names[i]}.json"
        with open(runs_path) as f:
            data =json.load(f)

        render(blocks_to_grid(data["best_blocks"]), plotter)

        plotter.add_text(f"{names[i]}, reach {data['best_so_far'][-1]}", font="arial", font_size=10, color="black")

    plotter.screenshot(out_path)
    plotter.close()


def render(grid, plotter):

    filled=np.argwhere(grid==1)
    x_base= (grid.shape[0]//2)

    cube=pv.Cube()

    points= pv.PolyData(filled.astype(float))
    points["role"] = np.sign(filled[:,0] - x_base)
    cubes = points.glyph(geom=cube, scale=False, orient=False)

    plotter.add_mesh(cubes,scalars="role", cmap="coolwarm",show_edges=True, edge_color='black', line_width=1, show_scalar_bar=False, name="structure", clim=[-1,1])

    nx,ny,nz = grid.shape

    box = pv.Box(bounds=(-0.5, nx - 0.5,-0.5, ny-0.5, -0.5, nz-0.5))

    plotter.add_mesh(box, style="wireframe", color="grey", name="box")

if __name__ == "__main__":
    from env import Environment
    import os

    os.makedirs("gifs", exist_ok=True, )
    os.makedirs("stills", exist_ok=True)

    env=Environment((24,8,12))

    gif_names= ("objectivelocal_bridge_s20", "objectivenolean_bridge_s2", "objective_bridge_s23", "novelty_bridge_s22", "blend_bridge_s28")

    ref=env.best_possible_bridge(40)

    # for gif in gif_names:
    #     gify("frames/"+ gif+".json", "gifs/" + gif + ".gif", ref)

    still(gif_names,ref, "stills/final_strucs.png")
    print("success")