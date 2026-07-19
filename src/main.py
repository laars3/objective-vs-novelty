from evolution import Evolution

evolution = Evolution(grid_size=8, population_size=10)
evolution.run(grid_size=10, steps=20)

print("done")
for i, score in enumerate(evolution.scores):
    print(f"Agent {i}: {score}")