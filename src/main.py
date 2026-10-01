from evolution import Evolution
from fitness import bridge_score
from agent import RandomAgent, Agent
import numpy as np, time
import json, os


run_config = dict(steps=40, sigma=0.002, generations=200)

arms = [("objective", Agent, "objective"), ("novelty", Agent, "novelty"), ("blend", Agent, "blend"), ("random", RandomAgent, "random")]

def run_save(evolution, config, run_config, name):
    path = f"runs/{name}_bridge_s{config['seed']}.json"

    # json can't save np nums so wrap int() here
    result = {"config": {k: getattr(v, "__name__", v) for k, v in {**config, **run_config}.items()},"gen_best": [int(x) for x in evolution.gen_best], "best_so_far": [int(x) for x in evolution.best_so_far],
    "gen_mean": [float(x) for x in evolution.gen_mean],"best_blocks": np.argwhere(evolution.best_grid == 1).tolist(),"archive_size": len(evolution.archive.vectors), "rho_min": float(evolution.archive.rho_min),}

    with open(path, "w") as f:
        json.dump(result,f, indent=2)


for name, agent_class, selection in arms:
    for seed in range(10):
        path=f"runs/{name}_bridge_s{seed}.json"
        if os.path.exists(path):
            continue

        config = dict(grid_shape=(24, 8, 12), population_size=50, score_func=bridge_score, seed=seed,agent_class=agent_class, selection=selection)
        print("Config: ", {k: getattr(v, "__name__", v) for k,v in {**config, **run_config}.items()})
        t = time.perf_counter()

        evolution = Evolution(**config)
        evolution.run(**run_config)
        run_save(evolution, config, run_config, name)

        print(name, seed, "best", evolution.best, "mean", round(evolution.gen_mean[-1], 2), "time",round(time.perf_counter() - t))


