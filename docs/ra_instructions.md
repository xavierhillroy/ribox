# LGP Symbolic Regression — Validation Run & Analysis

This document walks through (1) running the LGP engine sweep on the DRAC
cluster via SLURM, (2) running a matched baseline in Python, and (3)
analysing both into a single comparison table plus convergence plots. The goal
is a **sanity-level validation**: showing the LGP engine ("d1") reaches roughly
the same solution quality as an established genetic-programming implementation
on standard 1-D symbolic-regression benchmarks. It is *not* a claim that d1
beats or loses to any particular tool — see "Framing the result" at the end.

NOTE THE REFERENCE CODE IS AI GEN in this doc! Its a starting point to give you an idea but be vigilent and make sure it works! THEY ARE JUST GUIDELINES YOU MAY HAVE TO REWORK THEM TO GIVE ME THE DELIVERABLES I AM LOOKING FOR!
---

## 0. Background (read first if new to symbolic regression)

### What symbolic regression (SR) is

Most regression methods assume a fixed model form and fit its coefficients —
linear regression fits `y = ax + b`, a neural net fits fixed-architecture
weights. **Symbolic regression instead searches for the *form of the equation
itself*.** Given data points `(x, y)`, it tries to discover a closed-form
expression — e.g. `y = x³ + x² + x` — that explains the data, including the
structure (which operations, in what arrangement), not just numeric parameters.

This is powerful (the output is a human-readable formula) but hard: the space
of possible expressions is enormous and discrete. The dominant classical
approach is **genetic programming (GP)** — an evolutionary algorithm that
maintains a *population* of candidate expressions, evaluates each on the data
(its *fitness*), and repeatedly selects the fitter ones to breed (crossover)
and perturb (mutation), over many *generations*. Over time the population, on
average, gets better at fitting the data.

The engine we're validating ("d1") is a **linear GP (LGP)** implementation:
instead of representing programs as trees (the classic GP encoding), it
represents them as short sequences of register-machine instructions. The two
encodings are different but solve the same problem. gplearn, our baseline, is a
*tree*-GP. Both are GP; the comparison is GP-to-GP, which is what makes it fair.

### Why R²

We score how well a candidate expression fits the data using **R² (coefficient
of determination)**: `R² = 1 − SS_res/SS_tot`, where `SS_res = Σ(y − ŷ)²` is
the model's squared error and `SS_tot = Σ(y − ȳ)²` is the variance of the
target. Intuitively, R² answers "how much better is this model than just
predicting the average?" `R² = 1.0` is a perfect fit; `R² = 0` is no better
than the mean; negative means *worse* than the mean. Because it's normalised by
the target's variance, R² is comparable across targets of different scales —
unlike raw error, where "MSE = 0.5" means different things on different targets.

### The Nguyen benchmarks

**Nguyen** is the standard suite of symbolic-regression test functions
(Uy et al., 2011), and the most commonly used benchmark in the modern SR
literature (DSR, gplearn examples, transformer-based SR papers, etc.). Each
Nguyen target is a known closed-form equation, so we can check whether the
engine *recovered the true function*, not just fit the points. We use the
1-D subset (one input variable `x`):

| Target   | Equation                          | Notes                     |
|----------|-----------------------------------|---------------------------|
| Nguyen-1 | x³ + x² + x                       | easiest; `+ ×` only       |
| Nguyen-2 | x⁴ + x³ + x² + x                  | polynomial                |
| Nguyen-3 | x⁵ + x⁴ + x³ + x² + x             | polynomial                |
| Nguyen-4 | x⁶ + x⁵ + x⁴ + x³ + x² + x        | polynomial (hardest poly) |
| Nguyen-5 | sin(x²)·cos(x) − 1                | needs sin, cos            |
| Nguyen-6 | sin(x) + sin(x + x²)              | needs sin                 |

All sampled over `x ∈ [-1, 1]`, the standard Nguyen domain. Because these are
exactly recoverable, a working SR engine should hit R² ≈ 1.0 on the easy ones
(Nguyen-1, -2) on most runs, with success degrading gracefully on the harder
ones. That degradation pattern *is* the validation signal.

The datasets already exist in `datasets/` (`<target>_train.csv`,
`<target>_test.csv`). **Do not regenerate them** — both d1 and the Python
baseline must read the identical files.

---

## 1. Running the LGP sweep on DRAC (SLURM)

### 1.1 Build

From the project root on the cluster:

```bash
module load gcc           # or whatever provides a C++20-capable g++ on DRAC
make clean && make lgp_run
```

Confirm it built: `ls -l lgp_run` should show the binary.

### 1.2 One-cell local sanity check (do this first)

Before submitting ~150 jobs, run a single cell interactively and confirm the
numbers look sane. **Note the seven arguments** — the last three are the
result row, the best-program dump, and the per-generation history trace:

```bash
mkdir -p results programs history logs
./lgp_run datasets/nguyen1_train.csv datasets/nguyen1_test.csv nguyen1 0 \
          results/nguyen1_seed0.csv programs/nguyen1_seed0.txt \
          history/nguyen1_seed0.csv
cat results/nguyen1_seed0.csv
head history/nguyen1_seed0.csv
```

Expected: `nguyen1` is `x³+x²+x` (the easiest target). Train R² should be
**near 1.0**, and the history file should show `best_fitness` climbing toward
~1.0 across generations. If either looks wrong, stop and report back — running
the full sweep would just produce 150 bad files.

### 1.3 The sweep

The sweep is **6 targets × 25 seeds = 150 independent jobs**, dispatched as a
SLURM job array. Each job is one (target, seed) cell and writes one result row,
one program dump, and one history trace. The seed varies the *search* (GP is
stochastic, so we need many runs to get a stable recovery rate); the dataset is
fixed and shared.

`sweep.sh` should already be present in the project root. It should read:

```bash
#!/bin/bash
#SBATCH --job-name=lgp_sweep
#SBATCH --array=0-149              # 6 targets × 25 seeds = 150 cells (0..149)
#SBATCH --time=00:15:00            # generous; each cell is seconds-to-minutes
#SBATCH --mem=512M                 # runs are tiny
#SBATCH --cpus-per-task=1          # engine is single-threaded
#SBATCH --output=logs/%A_%a.out    # %A = array id, %a = task id
#SBATCH --account=def-YOURPI       # <-- set your DRAC allocation account

set -euo pipefail

targets=(nguyen1 nguyen2 nguyen3 nguyen4 nguyen5 nguyen6)
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

n_seeds=${#seeds[@]}
t_idx=$(( SLURM_ARRAY_TASK_ID / n_seeds ))   # -> target
s_idx=$(( SLURM_ARRAY_TASK_ID % n_seeds ))   # -> seed

target=${targets[$t_idx]}
seed=${seeds[$s_idx]}

mkdir -p results programs history logs
echo "cell ${SLURM_ARRAY_TASK_ID}: target=${target} seed=${seed} on $(hostname)"

./lgp_run \
    datasets/${target}_train.csv \
    datasets/${target}_test.csv \
    ${target} \
    ${seed} \
    results/${target}_seed${seed}.csv \
    programs/${target}_seed${seed}.txt \
    history/${target}_seed${seed}.csv
```

**Before submitting:** set `--account=def-YOURPI` to the correct DRAC
allocation, and `sbatch` from the project root (paths are relative to the
submission directory).

Submit:

```bash
sbatch sweep.sh
```

Monitor:

```bash
squeue -u $USER                                  # queued/running cells
sacct -j <JOBID> --format=JobID,State,ExitCode   # check for FAILED cells
```

### 1.4 Collect results

When the array finishes you should have **150 files each** in `results/`,
`programs/`, and `history/`. Verify, then build the merged tables.

**Results** (each file is one row, no header — `cat` is fine):

```bash
ls results/*.csv | wc -l        # expect 150 (LGP only, before gplearn)
echo "target,tool,run_seed,train_r2,test_r2,best_length" > all_results.csv
cat results/*.csv >> all_results.csv
```

**History** — careful: each history file *has its own header* (it's a
standalone multi-row trace, openable on its own for debugging one run). So you
can't plain `cat` them — strip the duplicate headers:

```bash
ls history/*.csv | wc -l        # expect 150
awk 'FNR==1 && NR!=1 {next} {print}' history/*.csv > all_history.csv
# keeps the first file's header, drops every subsequent file's header line
```

If any count is short, a cell failed silently — cross-check `sacct` for FAILED
tasks, read the matching `logs/<jobid>_<task>.out`, and re-run just those cells.

---

## 2. The Python baseline (gplearn)

We run the **same 6 targets × 25 seeds** through gplearn, a standard pure-Python
GP implementation, and append its rows to the **same schema** so everything
merges into one table.

### 2.1 Parameter matching — IMPORTANT

Match d1's parameters wherever the concept exists in both systems. Where it
doesn't (LGP-specific mutation operators), leave gplearn at its tree-GP
defaults — do **not** invent values.

| Parameter            | Set to                  | Source / reason                                   |
|----------------------|-------------------------|---------------------------------------------------|
| Population size      | **1000**                | matches d1 (`POPULATION_SIZE`)                    |
| Generations          | **1000**                | matches d1 (`MAX_GENERATIONS`)                    |
| Tournament size      | **3**                   | matches d1 (`TOURNAMENT_SIZE`)                    |
| Function set         | `add, sub, mul, div, sin, cos, gt, lt` | matches the d1 ISA ops these targets use   |
| Fitness metric       | R² (reported at end)    | matches d1                                        |
| Max program size     | see note (try ~50 nodes)| LGP length ≠ tree depth; not 1:1                  |
| Elitism              | not directly settable   | see note — gplearn's regressor has no elitism knob|
| Crossover / mutation rates | **gplearn defaults**  | LGP rates are operator-specific, no tree analogue |

Notes:
- **Function set:** d1's ISA exposes `+ − × ÷ sin cos ` (it also has two
  comparison ops, LT/GT, but none of Nguyen-1..-6 use them). Give gplearn
  exactly `add, sub, mul, div, sin, cos` so both search the same space.
  **ADD Comparison operators if they are available in thre**. Use gplearn's **protected**
  division (its default), since d1 also guards division.
- **Elitism:** d1 preserves ~10% elites (`ELITES = 0.1`). gplearn's
  `SymbolicRegressor` does **not** expose an elitism parameter ( I THINK DOUBLE CHECK) — there's no
  direct "keep top N" knob. So this can't be matched exactly; **note it as a
  known mismatch** in the writeup rather than faking a value.
- **Program size:** gplearn limits programs via `init_depth` / parsimony, not a
  hard node count. If you want to approximate d1's 50-instruction cap, set
  parsimony pressure or an `init_depth` that keeps trees roughly that size, but
  there is no exact equivalent. Document whatever you choose.
- **Where a parameter has no d1 equivalent and no clear default**, fall back to
  the Operon paper value (Burlacu et al., GECCO 2020, Table 4 — at the end of
  this doc). But keep population=1000 and generations=1000 regardless. OR atleast try to make it as close as possible. The entire point is a smell test 

### 2.2 The script

Save as `run_baseline.py` in the project root.

```python
#!/usr/bin/env python3
"""
gplearn baseline on the same 1-D Nguyen datasets the LGP engine used.
Writes results/gplearn_<target>_seed<seed>.csv (one row each, no header),
matching the C++ schema: target,tool,run_seed,train_r2,test_r2,best_length
"""
import os
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

TARGETS = ["nguyen1", "nguyen2", "nguyen3", "nguyen4", "nguyen5", "nguyen6"]
SEEDS   = list(range(25))

# Match d1 where the concept exists; gplearn defaults elsewhere (see 2.1).
POP_SIZE     = 1000
GENERATIONS  = 1000
TOURNAMENT   = 3
FUNCTION_SET = ("add", "sub", "mul", "div", "sin", "cos")  # = d1 ISA ops used

os.makedirs("results", exist_ok=True)


def r2(y, yhat):
    """Coefficient of determination — matches d1's evaluator."""
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def program_length(est):
    """gplearn program length = number of nodes in the expression tree."""
    return int(est._program.length_)


for target in TARGETS:
    train = pd.read_csv(f"datasets/{target}_train.csv")
    test  = pd.read_csv(f"datasets/{target}_test.csv")
    Xtr, ytr = train[["x"]].values, train["y"].values
    Xte, yte = test[["x"]].values,  test["y"].values

    for seed in SEEDS:
        est = SymbolicRegressor(
            population_size=POP_SIZE,
            generations=GENERATIONS,
            tournament_size=TOURNAMENT,
            function_set=FUNCTION_SET,
            metric="mse",        # gplearn optimises MSE internally;
            random_state=seed,   # we compute & report R² ourselves below
            n_jobs=1,
            verbose=0,
            # crossover / mutation rates: gplearn defaults (no LGP analogue)
            # no elitism knob exists on SymbolicRegressor (see doc 2.1)
        )
        est.fit(Xtr, ytr)

        train_r2 = r2(ytr, est.predict(Xtr))
        test_r2  = r2(yte, est.predict(Xte))
        length   = program_length(est)

        out = f"results/gplearn_{target}_seed{seed}.csv"
        with open(out, "w") as f:
            f.write(f"{target},gplearn,{seed},{train_r2},{test_r2},{length}\n")
        print(f"{target} seed {seed}: train_r2={train_r2:.4f} test_r2={test_r2:.4f}")
```

Install and run:

```bash
python -m venv venv && source venv/bin/activate
pip install gplearn numpy pandas
python run_baseline.py
```

> **Runtime note:** 1000 gen × 1000 pop in pure-Python gplearn is much slower
> than the C++ engine — expect minutes per run, so ~1–4 hours for all 150.
> Easiest fix: run `run_baseline.py` as a **single SLURM job** (not an array —
> one job that loops all 150 internally), e.g. a script with
> `#SBATCH --time=06:00:00 --mem=2G --cpus-per-task=1` that just calls
> `python run_baseline.py`. Don't reduce population below 1000 — keep the
> budget matched.

After it runs, re-build the merged results table so it contains **both** tools:

```bash
echo "target,tool,run_seed,train_r2,test_r2,best_length" > all_results.csv
cat results/*.csv >> all_results.csv     # picks up both lgp_* and gplearn_* files
```
ALSO CReate a history file for GP learn same as before. HISTORY RESULTS MAY BE OVERRIDDEN BY THIS RUN. I THINK THEY MAY BE... I have created a scafolding but I havent checked over everything. I will trust you to do this. YOu might have to save the files carefully before so we dont override everything 
### 2.3 A state-of-the-art ceiling: PySR

If there's time, also run **PySR** (modern, Julia-backed SR) as a *ceiling
reference*, treated differently from gplearn:

- PySR has **constant optimisation**, which d1 and gplearn lack. It will likely
  score higher, especially on Nguyen-5 (which has a `−1` constant). That's
  expected — PySR is a ceiling, not a fair matched peer.
- Same datasets, same R² columns, `tool=pysr`. `pip install pysr` (bootstraps
  Julia on first run); same loop as gplearn, set `niterations` to a comparable
  budget and `random_state=seed`.

If time is short, **skip PySR** — gplearn alone is a sufficient matched baseline.
create all the same outputs csvs 

---

## 3. Analysis (single Python file)

Put everything below into one file, `analysis.py`. Run it after both
`all_results.csv` and `all_history.csv` exist. It produces the summary table
and **three** plot types: the test-R² box plot, the recovery-rate bar chart (),
and the **convergence curves** I want this convergence curve compared against one for the other baselines. Make it nice overall and easy to read (Up to you if seperate curves) I dontt think the code is doing it properly(no its not) cause it assumes you create one based on one engine ours (I want one for each engine) (the key GP deliverable — see below).

```python
#!/usr/bin/env python3
"""Analysis of LGP vs baseline(s) on Nguyen 1-D benchmarks."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Recovery threshold: a run "recovered" the target if test R² >= this.
# Chosen BEFORE looking at results. 0.99 is standard for exact-recoverable
# polynomial/trig targets. State this threshold in the writeup.
RECOVERY_THRESHOLD = 0.99

# ---------------------------------------------------------------------------
# PART A — final-quality analysis (from all_results.csv)
# ---------------------------------------------------------------------------
df = pd.read_csv("all_results.csv")

def iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)

summary = (
    df.groupby(["target", "tool"])
      .agg(
          median_test_r2=("test_r2", "median"),
          iqr_test_r2=("test_r2", iqr),
          median_train_r2=("train_r2", "median"),
          recovery_rate=("test_r2", lambda s: (s >= RECOVERY_THRESHOLD).mean()),
          median_length=("best_length", "median"),
          n_runs=("test_r2", "size"),
      )
      .reset_index()
)
summary.to_csv("summary_table.csv", index=False)
print(summary.to_string(index=False))

# Plot 1: box plot of test R² per target, grouped by tool.
targets = sorted(df["target"].unique())
tools   = sorted(df["tool"].unique())
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.8 / len(tools)
for ti, target in enumerate(targets):
    for tj, tool in enumerate(tools):
        vals = df[(df.target == target) & (df.tool == tool)]["test_r2"]
        pos = ti + (tj - (len(tools) - 1) / 2) * width
        ax.boxplot(vals, positions=[pos], widths=width * 0.9)
ax.set_xticks(range(len(targets)))
ax.set_xticklabels(targets, rotation=30)
ax.axhline(RECOVERY_THRESHOLD, ls="--", c="grey", lw=1,
           label=f"recovery threshold ({RECOVERY_THRESHOLD})")
ax.set_ylabel("Test R²")
ax.set_title("Test R² by target and tool (boxes = 25 seeds)")
ax.legend()
plt.tight_layout()
plt.savefig("test_r2_boxplot.png", dpi=150)

# Plot 2: recovery-rate bar chart per tool.
piv = summary.pivot(index="target", columns="tool", values="recovery_rate")
piv.plot(kind="bar", figsize=(10, 5))
plt.ylabel(f"Recovery rate (test R² ≥ {RECOVERY_THRESHOLD})")
plt.title("Recovery rate by target and tool")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("recovery_rate.png", dpi=150)

# ---------------------------------------------------------------------------
# PART B — convergence analysis (from all_history.csv) — the key GP plot
# ---------------------------------------------------------------------------
# The classic GP convergence curve: best fitness vs generation. It shows
# whether the SEARCH is behaving (climbing then plateauing) and explains the
# final-quality plots — a low-recovery target either plateaus early (stuck) or
# never converges (too hard / too few generations).

#NOT DOING IT PROPERLYU
hist = pd.read_csv("all_history.csv")

for target in sorted(hist["target"].unique()):
    sub = hist[hist.target == target]
    g = sub.groupby("generation")["best_fitness"]
    med = g.median()
    lo  = g.quantile(0.25)
    hi  = g.quantile(0.75)
    plt.figure(figsize=(8, 4))
    plt.plot(med.index, med.values, label="median best fitness")
    plt.fill_between(med.index, lo.values, hi.values, alpha=0.2,
                     label="IQR over 25 seeds")
    plt.xlabel("generation")
    plt.ylabel("best fitness (R²)")
    plt.ylim(0, 1.02)
    plt.title(f"Convergence — {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"convergence_{target}.png", dpi=150)

# Optional Plot: program length (bloat) over generations, same data.
for target in sorted(hist["target"].unique()):
    sub = hist[hist.target == target]
    g = sub.groupby("generation")["mean_length"].median()
    plt.figure(figsize=(8, 4))
    plt.plot(g.index, g.values)
    plt.xlabel("generation")
    plt.ylabel("median mean program length")
    plt.title(f"Program length over time — {target}")
    plt.tight_layout()
    plt.savefig(f"bloat_{target}.png", dpi=150)

print("\nWrote: summary_table.csv, test_r2_boxplot.png, recovery_rate.png,")
print("       convergence_<target>.png, bloat_<target>.png")
```

---

## 4. What to report

Produce a short writeup (1–2 pages) containing:

1. **The summary table** (`summary_table.csv`): per target, per tool — median
   test R² ± IQR, recovery rate, median program length, number of runs. The
   headline deliverable, in the spirit of Operon paper Table 5 (median ± IQR
   over repeated runs).
2. **The plots:**
   - test-R² box plot (final quality, seed spread)
   - recovery-rate bar chart (reliability)
   - **convergence curves, one per target — this is a required GP deliverable.**
     Median best-fitness trajectory with an IQR band over the 25 seeds. They
     show the search is *working* (monotone climb to a plateau near 1.0 on easy
     targets) and *explain* the final-quality plots: if a target has low
     recovery, the convergence curve tells you whether it plateaued early
     (stuck in a local optimum) or simply needed more generations.
   - optionally, program-length-over-time (bloat) plots.
3. **Per-target commentary:** for any target where d1's recovery rate is much
   lower than the baseline, open the `programs/<target>_seed*.txt` dumps and
   note *what* d1 found — right shape with wrong constants, or structurally off?
   The program dumps are why we saved them; a low R² alone doesn't explain a
   failure. The convergence curve + the program dump together tell the story. (DO THIS roughly- dont spend too much time only if something is clearly wrong, rmb to trace by r0)
4. **The exact configuration per tool** (population, generations, tournament
   size, elitism, function set, fitness metric) and **every place the two
   configs differ** (LGP length cap vs. gplearn tree depth; tournament size 3
   vs. Operon's 5; LGP elitism vs. gplearn having no elitism knob; LGP mutation
   operators having no tree-GP analogue). Honesty about mismatches matters more
   than a clean-looking table.
5. **Seeds and reproducibility:** dataset seeds (fixed, shared), the 25 run
   seeds, and the recovery threshold — all stated explicitly.

### Framing the result (read before writing conclusions)

This is a **validation / sanity check**, not a competitive benchmark. The
correct claim is: *"On standard 1-D Nguyen benchmarks under a matched
population/generation budget, the LGP engine reaches solution quality and
recovery rates comparable to an established GP implementation (gplearn), and its
convergence behaviour is consistent with a correctly functioning evolutionary
search."*

Do **not** write "d1 beats X" or "d1 is worse than X." The two systems use
different encodings and operators, neither has constant optimisation, and the
comparison isn't controlled tightly enough to rank them. A gap in either
direction is informative ("d1 is in the right ballpark") but not a ranking.

If PySR was run, note separately that it is a SOTA tool *with* constant
optimisation and so sits above both GP systems by design — a ceiling, not a peer.

---

## 5. Reference: Operon paper parameters (Burlacu et al., GECCO 2020, Table 4)

Use only as a fallback when a parameter has no d1 equivalent and no clear
library default. **Population and generations stay at 1000 regardless.**

| Parameter            | Operon value                        |
|----------------------|-------------------------------------|
| Function set         | +, −, ×, ÷, sin, cos, exp, log      |
| Terminal set         | constant, weight · variable         |
| Tree limits          | 10 levels, 50 nodes                 |
| Tree initialisation  | Balanced tree creator               |
| Population size      | 1000                                |
| Generations          | 1000                                |
| Parent selection     | Tournament, group size 5            |
| Crossover probability| 100%                                |
| Crossover operator   | Subtree crossover                   |
| Mutation probability | 25%                                 |
| Mutation operator    | Single-point mutation               |
| Fitness function     | R² correlation with the target      |

Note: Operon's function set includes `exp` and `log`, which d1's current ISA
does not. None of Nguyen-1..-6 need `exp`/`log`, so this doesn't affect these
experiments — but if the target list is extended to Nguyen-7 (which uses
`log`), both d1 and the baseline would need those primitives added, on both
sides, before comparing.

---

## 6. Build reference: Makefile

The Makefile builds three binaries: `lgp_run` (the sweep driver — the one you
need), `lgp_test` (engine test harness), and `gen_datasets` (regenerates the
CSVs — **you should not need this**, the datasets are fixed and shared). The
`lgp_run` driver takes **seven** arguments (it gained the `history_csv` output
for convergence tracking); no Makefile change was needed for that. The full
Makefile is reproduced here for reference (already in repo)

```makefile
# ---- Configuration ----
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -O2
DEPFLAGS := -MMD -MP

# ---- Files ----
# Core engine objects shared by the binaries.
CORE_SRCS := LGPEngine.cpp Interpreter.cpp Evaluator.cpp Dataset.cpp
CORE_OBJS := $(CORE_SRCS:.cpp=.o)

# Each binary has its own entry-point translation unit.
TEST_OBJS := $(CORE_OBJS) test_bed.o
RUN_OBJS  := $(CORE_OBJS) main.o
GEN_OBJS  := Dataset.o gen_datasets.o      # generator only needs Dataset

# All objects, for dependency includes and clean.
ALL_OBJS := $(CORE_OBJS) test_bed.o main.o gen_datasets.o
DEPS     := $(ALL_OBJS:.o=.d)

TEST_TARGET := lgp_test
RUN_TARGET  := lgp_run
GEN_TARGET  := gen_datasets

# ---- Targets ----
all: $(TEST_TARGET) $(RUN_TARGET) $(GEN_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) -o $(TEST_TARGET)

$(RUN_TARGET): $(RUN_OBJS)
	$(CXX) $(CXXFLAGS) $(RUN_OBJS) -o $(RUN_TARGET)

$(GEN_TARGET): $(GEN_OBJS)
	$(CXX) $(CXXFLAGS) $(GEN_OBJS) -o $(GEN_TARGET)

# Pattern rule: compile any .cpp into a .o with header-dependency tracking.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(ALL_OBJS) $(DEPS) $(TEST_TARGET) $(RUN_TARGET) $(GEN_TARGET)

-include $(DEPS)

.PHONY: all clean
```

For the validation run you only need `make lgp_run`. Use `make` (or `make all`)
to build everything. `make clean` removes all binaries, objects, and dependency
files.