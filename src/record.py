from evolution import Evolution
from fitness import bridge_score, bc, tower_score
from agent import Agent, LocalAgent, LocalAgentNoCOM
import json, os

# runs
runs = [("objectivelocal", LocalAgent, "objective", 20, 0.03),("objectivenolean", LocalAgentNoCOM, "objective", 2, 0.03),
("objective", Agent, "objective", 23, 0.002),("novelty", Agent, "novelty", 22, 0.002),("blend", Agent, "blend", 28, 0.002),]

# make dir
os.makedirs("frames", exist_ok=True)

# search set
score_f = "bridge"
tasks = {"tower": tower_score, "bridge":bridge_score}

# config
for name, agent_class, selection, seed, sigma in runs:
    config = dict(grid_shape=(24, 8, 12), population_size=50, score_func=tasks[score_f], seed=seed,agent_class=agent_class, selection=selection, bc_func=bc)
    ev = Evolution(**config)
    ev.run(steps=40, sigma=sigma, generations=200)

    with open(f"runs/{name}_{score_f}_s{seed}.json") as f:
        saved =json.load(f)
    print(name, seed, "match", saved["gen_best"] == ev.gen_best)

    gen_best = []
    for x in ev.gen_best:
        gen_best.append(int(x))

    with open(f"frames/{name}_{score_f}_s{seed}.json", "w") as f:
        json.dump({"gen_best": gen_best, "gen_blocks": ev.gen_blocks}, f)





