from evolution import Evolution
from fitness import bridge_score, tower_score, bc, bc_split
from agent import RandomAgent, Agent, LocalAgent, LocalAgentNoCOM
import numpy as np, time, json, os


score_f = "bridge" # possible score functions (bridge, tower)
tasks = {"tower": tower_score, "bridge":bridge_score}


run_config = dict(steps=40, sigma=0.03, generations=200)

arms = [("objectivenolean", LocalAgentNoCOM, "objective", bc)]

def config_names(config, run_config): # swaps functions and classes for their names so json can save them
    names = {}
    for k, v in {**config, **run_config}.items():
        names[k] = getattr(v, "__name__", v)
    return names

def run_save(evolution, config, run_config, name):
    path = f"runs/{name}_{score_f}_s{config['seed']}.json"

    # json can't save np nums so wrap int() and float() here
    gen_best = []
    for x in evolution.gen_best:
        gen_best.append(int(x))

    best_so_far = []
    for x in evolution.best_so_far:
        best_so_far.append(int(x))

    gen_mean = []
    for x in evolution.gen_mean:
        gen_mean.append(float(x))

    result = {
        "config": config_names(config, run_config),
        "gen_best": gen_best,
        "best_so_far": best_so_far,
        "gen_mean": gen_mean,
        "best_blocks": np.argwhere(evolution.best_grid == 1).tolist(),
        "archive_size": len(evolution.archive.vectors),
        "rho_min": float(evolution.archive.rho_min),
    }

    with open(path, "w") as f:
        json.dump(result,f, indent=2)


for name, agent_class, selection, bc_func in arms:
    for seed in range(10):
        path=f"runs/{name}_{score_f}_s{seed}.json"
        if os.path.exists(path):
            continue

        config = dict(grid_shape=(24, 8, 12), population_size=50, score_func=tasks[score_f], seed=seed,agent_class=agent_class, selection=selection, bc_func=bc_func)
        print("Config: ", config_names(config, run_config))
        t = time.perf_counter()

        evolution = Evolution(**config)
        evolution.run(**run_config)
        run_save(evolution, config, run_config, name)

        print(name, seed, "best", evolution.best, "mean", round(evolution.gen_mean[-1], 2), "time",round(time.perf_counter() - t))


