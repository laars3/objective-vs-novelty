# TODO / Working Plan

This is the working tracker: what is done, what is next, and the milestone plan.
The scientific writeup lives in [README.md](README.md); this file is just for
steering the build.

## Where things stand

Done and running:

- `src/env.py` -- grid, base cell, support rule, balance rule, `valid_move()` mask.
- `src/agent.py` -- MLP policy (`N^3 -> 128 -> N^3`), invalid cells masked, argmax placement.
- `src/evolution.py` -- generation loop that builds one structure per agent.
- `src/main.py` -- smoke test; runs end to end, prints placeholder scores.

Not started yet:

- `src/visu.py` -- file exists but is empty (this is the current task).
- `fitness.py`, `archive.py`, `tests/`, `configs/`, `paper/` -- not created.
- Real scoring: `evolution.py` still hard-codes every score to `0.0`.

## Known issues to fix

- `evolution.py` `run()` takes a `grid_size` argument and uses it for
  `np.unravel_index`, but the agents were built with the size stored on the
  Evolution object. `main.py` calls `run(grid_size=10, ...)` while the population
  was built with `grid_size=8`. These must be the same value. Drop the `grid_size`
  parameter from `run()` and use `self.grid_size` everywhere so they can never
  disagree.

## Immediate next steps

1. Write `render(grid)` in `src/visu.py` -- take a numpy grid, show it as voxels
   in a PyVista window. This closes M1.
2. Fix the `grid_size` mismatch above.
3. Start `fitness.py` with the tower objective (`max(zi)`), then wire it into
   `evolution.py` so scores stop being `0.0`. This opens M2.

## Milestones

Each milestone ends with something visible or testable.

- [ ] **M0 -- Environment rules + tests.** Support rule, balance rule,
  `valid_move()` mask in `env.py` -- *rules done*; pytest cases still missing,
  including "greedy +x placement becomes invalid without counterweight" (the
  deception mechanism as a unit test). *Deliverable: green tests.*
- [ ] **M1 -- One agent builds; viewer shows it.** Masking done in `agent.py`;
  the PyVista viewer (`visu.py`) is the remaining piece.
  *Deliverable: screenshot of a real, rule-obeying structure.*
- [ ] **M2 -- Evolution works on the tower task.** Real scoring, selection,
  mutation, elitism (scores are currently hard-coded to 0). Tower first -- if
  fitness doesn't climb on the easy task, the loop is broken, not the theory.
  *Deliverable: rising fitness curve, 3 seeds.*
- [ ] **M3 -- Cantilever task + H1 check.** Implement reach scoring; run the
  objective arm; hand-construct a reference balanced-arm solution; confirm the
  plateau. *Deliverable: the H1 plot -- objective arm vs. reference line.*
- [ ] **M4 -- Novelty search.** BC vector, archive, k-NN novelty; `novelty` and
  `blend` arms; coverage metric. *Deliverable: coverage-over-time plot.*
- [ ] **M5 -- Full 2x4 experiment + statistics.** 10+ seeds per condition;
  Mann-Whitney + effect sizes; the headline interaction figure.
  *Deliverable: the H2/H3 result, whatever it says.*
- [ ] **M6 -- Explorer polish.** Slider, side-by-side, BC scatter animation, GIFs.
- [ ] **M7 -- Paper + repro pack.** Writeup; `runs/` to figures script; README
  results section with the headline figure.

## Target repository structure

What the finished repo should look like (bold = exists today):

```
src/
  env.py          # grid, base cell, support + balance rules, valid-move mask   [done]
  agent.py        # MLP policy: grid -> masked logits -> placement              [done]
  evolution.py    # generation loop: build, score, select, mutate, elitism      [partial: no scoring]
  fitness.py      # cantilever & tower objectives; BC vector; novelty score     [todo]
  archive.py      # novelty archive (add-above-threshold, k-NN queries)         [todo]
  visu.py         # PyVista/Qt explorer, BC scatter, GIF export                 [empty file]
  main.py         # run one experiment arm from a config file                   [smoke test only]
configs/          # one YAML per condition (task, arm, seeds, hyperparams)      [todo]
tests/            # pytest: env invariants, BC math, novelty math               [todo]
runs/             # saved results (gitignored except an example run)            [todo]
paper/            # the writeup                                                 [todo]
```

## Stretch goals

- **MAP-Elites arm** (Mouret & Clune, 2015): keep the best structure per BC
  cell -- the quality-diversity method this line of work grew into.
- **BC ablation**: rerun H2 with a deliberately impoverished BC
  (`[block_count]` only) to demonstrate the known BC-sensitivity critique.
- **Behavioral strategy analysis**: cluster each arm's final structures and
  characterize build-order patterns -- "what did each selection pressure
  learn?", answered at the behavior level.
