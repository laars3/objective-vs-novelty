# TODO / Working Plan

The working tracker: what is done, what is next, and the milestone plan. The
scientific writeup lives in [README.md](README.md); this file steers the build.

## Where things stand

Done and running:

- `src/env.py` : grid, base cell, support rule, balance rule, `valid_move()` mask.
  Loop-invariant work hoisted out of the triple loop, now scales `N^3` instead of
  `N^6` (37x faster at N=32, verified against a reference implementation).
- `src/agent.py` : MLP policy (`N^3 -> 128 -> N^3`), invalid cells masked, argmax
  placement, plus `growth(sigma)` for Gaussian weight mutation.
- `src/fitness.py` : `tower_score(grid)` = `max z`. `reach_score(grid)` =
  `max x - x_base`, the deceptive objective.
- `src/visu.py` : `render(grid)` draws a structure as outlined voxel cubes.
- `src/evolution.py` : full objective-driven loop (build, score, keep top half,
  breed via deepcopy + mutation, elitism). Takes `score_func` at construction, so
  a run targets either task without editing the file. Best score per generation
  logged to `self.record`.
- `src/main.py` : runs a population and renders a built structure.

Not started:

- Novelty search: BC vector, archive, k-NN scoring. Only the `objective` arm
  exists; `random`, `novelty`, `blend` are later.
- Seeding for reproducibility.
- Run logging. Every run currently vanishes on exit, and `self.grids` is
  overwritten each generation.
- `archive.py`, `tests/`, `configs/`, `paper/` : not created.

## Open question blocking everything

**Does the search actually work?** The most recent runs produced a flat `record`
(e.g. `[7, 7, 7, ..., 7]`). Flat is not "learned instantly": `record[0]` is the
best of a randomly initialized population before any mutation, the agent is
deterministic, and elitism copies the best agent forward unchanged, so a lucky
initial draw pins the line. Flat means no mutated child ever beat generation 0.

At population 10, 10 generations, and ~444k weights, that is expected. Raise
population toward 100 and generations toward 100+ and confirm the record climbs
before trusting any downstream number.

## Immediate next steps

1. Raise population and generations; confirm `record` rises on the tower task.
2. Add seeding, so runs are reproducible and comparable.
3. Add placement logging to CSV (`generation, agent_id, step, x, y, z`). Log
   placements, not grids: the ordered list reconstructs the grid and also drives
   build playback later. Prerequisite for the H1 plot.
4. **M3 proper:** run the objective arm on the reach task, record where it
   plateaus, hand-construct a balanced-arm reference solution, and compare. The
   gap between them is H1.

## Performance work

- [ ] **Optimize `valid_move()`.** It scans all `N^3` cells per placement and
  accounts for roughly 98% of runtime at target settings, far more than the
  network forward passes. The support rule means only cells face-adjacent to an
  existing block can ever be valid, so the true candidate set is bounded by the
  structure size (a few hundred cells at `B = 40`) rather than the grid size
  (4,096 at N = 16). Building candidates from the neighbors of filled cells
  instead of sweeping the grid is what makes the full 80-run sweep an overnight
  job. Do this before the sweep, not before the search is shown to work.

## Decisions to confirm

- **Ground-level extension.** `env.py` allows building along the ground plane
  outward from the base, since the support rule has no `z` restriction. An
  earlier README draft claimed ground cells outside the base were not buildable.
  The code's behavior is now what the README documents. Deception survives either
  way (counterweights are still required), but decide deliberately.
- **Grid size and budget.** README specifies N = 16 and B = 40. `main.py`
  currently runs `grid_size=12, steps=20`. Align once the search is working.
- **Symmetric objective ceilings.** Tower tops out at `N-1`, reach at about
  `N/2 - 1`, so tower always has roughly twice the range. Not required, since all
  comparisons are within-task, but a non-cubic grid (wide in x, short in z) would
  equalize them. Costs a refactor of `Agent` (`size**3`), `env.py` (one `size`
  for all axes), and `unravel_index` in `evolution.py`. Deferred.

## Milestones

Each milestone ends with something visible or testable.

- [x] **M0 : Environment rules.** Support rule, balance rule, `valid_move()` mask.
  Rules done; pytest cases still missing, including "greedy +x placement becomes
  invalid without counterweight" (the deception mechanism as a unit test).
- [x] **M1 : One agent builds; viewer shows it.** Masking in `agent.py`, PyVista
  rendering in `visu.py`.
- [x] **M2 : Evolution works on the tower task.** Scoring, selection, mutation,
  elitism. Fitness climbed 4 -> 7 and saturated. Still to add: seeding and a
  3-seed repeat, deferred to when seeding lands.
- [ ] **M3 : Reach task + H1 check.** Reach scoring implemented. Remaining: run
  the objective arm, hand-construct the reference solution, confirm the plateau.
  *Deliverable: the H1 plot, objective arm versus reference line.*
- [ ] **M4 : Novelty search.** BC vector, archive, k-NN novelty; `novelty` and
  `blend` arms; coverage metric. *Deliverable: coverage-over-time plot.*
- [ ] **M5 : Full 2x4 experiment + statistics.** 10+ seeds per condition;
  Mann-Whitney and effect sizes; the headline interaction figure.
  *Deliverable: the H2/H3 result, whatever it says.*
- [ ] **M6 : Explorer polish.** Build playback from logged placements, generation
  slider, side-by-side arms, BC scatter animation, GIF export.
- [ ] **M7 : Paper + repro pack.** LaTeX writeup; `runs/` to figures script;
  README results section with the headline figure.

## Target repository structure

Bold = exists today.

```
src/
  env.py          # grid, support + balance rules, valid-move mask   [done, needs optimizing]
  agent.py        # MLP policy: grid -> masked logits -> placement    [done]
  evolution.py    # generation loop: build, score, select, mutate     [done, objective arm]
  fitness.py      # tower & reach objectives; BC vector; novelty      [objectives done]
  archive.py      # novelty archive (add-above-threshold, k-NN)       [todo]
  visu.py         # PyVista explorer, build playback, GIF export      [render() done]
  main.py         # run one experiment arm from a config file         [runs + renders]
configs/          # one YAML per condition                            [todo]
tests/            # pytest: env invariants, BC math, novelty math     [todo]
runs/             # saved results (gitignored except an example run)  [todo]
paper/            # the LaTeX writeup                                 [todo]
```

## Stretch goals

- **MAP-Elites arm** (Mouret and Clune, 2015): keep the best structure per BC
  cell, the quality-diversity method this line of work grew into.
- **BC ablation**: rerun H2 with a deliberately impoverished BC (`[block_count]`
  only) to demonstrate the known BC-sensitivity critique.
- **Behavioral strategy analysis**: cluster each arm's final structures and
  characterize build-order patterns. What did each selection pressure learn?
