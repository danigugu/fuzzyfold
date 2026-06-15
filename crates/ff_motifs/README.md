# ff_motifs

Cotranscriptional folding simulation and motif-occupancy scoring for RNA sequence design.

`ff_motifs` answers a central question in RNA design: **does a given RNA sequence fold into the intended secondary structure as it is being transcribed, nucleotide by nucleotide?** It runs kinetic Monte Carlo simulations of RNA folding during transcription and checks, at defined transcription positions, whether the growing RNA molecule is in the correct structural state (a *motif*).

## Background

RNA secondary structure can depend critically on the *order* in which the sequence is synthesised. A sequence may fold into the thermodynamic minimum correctly yet fail *cotranscriptionally* because an unwanted structure forms during transcription and kinetically traps the molecule. `ff_motifs` evaluates this by simulating the SSA (Stochastic Simulation Algorithm) on the folding landscape at each transcription step, then asking what fraction of simulated trajectories are in the intended structural state at each checkpoint.

---

## Concepts

### Domain architecture and `dl_seq`

RNA designs are described as sequences of named functional domains. `dl_seq` is a whitespace-separated list of domain names:

```
s1 L1 s2 T1 s2* L1* s1* T2
```

Domain naming conventions:
- **`s`** — stem/duplex domains (e.g. `s1`, `s2`, `s2*` is the complement of `s2`)
- **`L`** — loop domains that will be part of a hairpin or pseudoknot arm (e.g. `L1`, `L1*`)
- **`T`** — transcription checkpoint domains; after the last nucleotide of each `T`-domain the simulation pauses to record motif occupancy
- `*` suffix — complement of the corresponding domain; same length

A `dom_length_dict` maps each base domain name (without `*`) to its nucleotide length.

### ACFP (Abstract Cotranscriptional Folding Path)

The ACFP string encodes the *intended* secondary structure shape at each transcription stage. It has one whitespace-separated entry per `T`-domain (i.e. one per segment of `dl_seq` ending with a `T` domain). Each entry grows by one character per new segment and uses dot-bracket notation *over segments*, not nucleotides:

```
. .. .() (())
```

Here there are four segments. Reading left to right:
- After segment 1 (`.`): all segments unpaired
- After segment 2 (`..`): both segments unpaired
- After segment 3 (`..()`): segments 2 and 3 pair with each other; segment 1 is free
- After segment 4 (`(())`): segments 1 and 4 pair; segments 2 and 3 pair inside

`L`-domains translate the segment-level bracket notation into nucleotide-level dot-bracket; `s` and `T` domains are always shown as unpaired (`.`) in the resulting structure.

### Motifs

A **motif** is a named partial structural constraint. It specifies what the RNA *should* look like at a given transcription position, without prescribing every position:

- `(` / `)` — this nucleotide must be paired with its partner
- `x` — this nucleotide must be unpaired
- `.` — this position is unconstrained (either state is acceptable)

A motif can be *shorter* than the sequence; only the specified prefix is checked. An optional **Hamming distance** tolerance allows fuzzy matching.

The special motif name **`Unassigned`** is automatically added to every registry and matches whenever no named motif matches.

### Simulation parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `params` | `"rna_default"` | Thermodynamic parameter set: `"rna_default"` (RNA Turner 2004), `"rna_extended"`, or `"dna"` (DNA Mathews 2004) |
| `celsius` | `37.0` | Temperature in °C |
| `k0` | `1e5` | Attempt frequency (s⁻¹); sets the overall timescale of structural transitions |
| `k3ws` | `0.0` | Three-way shift rate constant; `0.0` disables three-way shifts |
| `k4ws` | `0.0` | Four-way shift rate constant; `0.0` disables four-way shifts |
| `t_ext` | `0.02` | Time window per transcription step (s); how long the molecule folds before the next nucleotide is added |
| `t_end` | `0.02` | Time to simulate at full sequence length (s) |
| `num_sims` | — | Number of independent SSA trajectories to run in parallel |
| `threshold` | — | Minimum fraction of simulations that must be in the expected motif at a checkpoint to continue; simulations failing this are cut short (see [Checkpoint logic](#checkpoint-logic)) |
| `num_workers` | `0` | Thread count for Rayon: `0` = all available cores, `1` = serial (safe when called from OpenMP), `N` = exactly N threads |

---

## Input formats

### Motif file / string

Used by `simulate_timecourse`. Can be supplied as a file path or as a raw string. Format:

```
<RNA sequence>
>motif_name_1 <allowed_distance>
<constraint string>
>motif_name_2 <allowed_distance>
<constraint string>
...
```

- The first non-empty line must be the RNA sequence in standard IUPAC notation. It must match the sequence passed to the simulation call.
- Each motif block starts with a `>` header line: the name followed by a non-negative integer Hamming distance.
- The next line is the constraint string using `(`, `)`, `x`, `.`.
- Blank lines and lines starting with `#` are ignored.

**Example:**

```
GGGAAACCC
>stem_formed 0
(((xxx)))
>open 1
.........
```

`"stem_formed"` matches any structure where positions 0–2 are open and positions 6–8 are close, and positions 3–5 are unpaired. `"open"` matches any structure within Hamming distance 1 of fully unpaired.

When `ff_motifs` builds the motif string automatically from an ACFP target (via `CotransConfig`), it uses `x` for positions that must be in a specific paired/unpaired state and `.` elsewhere, with distance `0`.

### `check_positions`

A mapping from **nucleotide index (0-based)** to a list of **motif names** to check at that position:

```python
check_positions = {
    20: ["motif_1"],          # check motif_1 at position 20
    41: ["motif_1", "motif_2"],  # check both at position 41
}
```

At each position in this dict, the current structure (from each simulation trajectory) is tested against the listed motifs. Positions not in this dict are skipped entirely.

### `checkpoints` (optional)

A mapping from **nucleotide index** to a dict of **motif name → threshold fraction**:

```python
checkpoints = {
    20: {"motif_1": 0.6},   # at position 20, at least 60% of sims must match motif_1
    41: {"motif_2": 0.5},   # at position 41, at least 50% must match motif_2
}
```

If the threshold is not met, the simulation returns early with a partial result. Useful for fast rejection of bad sequences during design optimisation.

---

## Output formats

### Timecourse result (Python)

`simulate_timecourse` and `simulate_timecourse_checkpoints` return a list of dicts, one per `check_position`, sorted by nucleotide index:

```python
[
    {
        "nucleotide": 20,
        "motif_counts": {
            "motif_1":    73,   # 73 out of num_sims trajectories matched motif_1
            "Unassigned": 27,   # 27 trajectories did not match any named motif
        }
    },
    {
        "nucleotide": 41,
        "motif_counts": {
            "motif_2":    55,
            "Unassigned": 45,
        }
    },
]
```

- `"nucleotide"`: 0-based index in the sequence where the check was performed.
- `"motif_counts"`: counts across all `num_sims` trajectories. Multiple named motifs can match the same structure, so counts within one position do not necessarily sum to `num_sims`.
- `"Unassigned"` appears when no named motif matched.

### `cotrans_score` (Python / C)

A scalar in **[0, 1]**, lower is better:

```
score = (pathlength - Σ avg_occupancy_per_T_domain) / pathlength
```

- `pathlength` = number of checkpoint T-domains.
- `avg_occupancy_per_T_domain` = average fraction of simulations matching any named (non-Unassigned) motif, averaged over all nucleotide positions within that T-domain.
- **0.0** — perfect cotranscriptional folding: every simulation matches the expected motif at every position along the entire path.
- **1.0** — complete failure: no simulation matched the expected motif at the first checkpoint.

### C FFI output

`sim_timecourse` and `sim_timecourse_checkpoints` write into caller-allocated flat arrays:

- `out_nucleotides[i]` — nucleotide index of the i-th timepoint
- `out_counts[i]` — total named-motif hits at that timepoint (sum across all named motifs, excluding `"Unassigned"`)
- Return value: number of timepoints written, or `-1` on error.

`sim_score` returns the `cotrans_score` as a `double`, or `-1.0` on error.

---

## Build isolation (Python vs C/C++)

Both the Python extension and the C/C++ shared library are compiled from the
same Rust crate, but with **different feature flags**:

| Consumer | Feature flag | Output |
|----------|-------------|--------|
| Python (maturin) | `--features python` | `target/release/libmotifs.so` + installs into the active virtualenv |
| C / C++ (cargo) | `--no-default-features` | `target_motifs_ffi/release/libmotifs.so` |

The two builds use **separate Cargo target directories** (`target/` and
`target_motifs_ffi/`) so they never overwrite each other.  If they shared the same
directory, a maturin build would leave `libmotifs.so` linked against CPython,
and the C++ binary would immediately crash with an `undefined symbol:
PyExc_TypeError` error on the next invocation.

**Build order** (both can be done independently, in either order):

```bash
# Python extension — installs into the active conda / virtualenv
conda activate fuzzyfold
maturin develop --manifest-path crates/ff_motifs/Cargo.toml --features python --release

# C++ shared library + header + binary — uses target_motifs_ffi/ to stay isolated
cd /path/to/SamplingDesign
make main          # runs: CARGO_TARGET_DIR=…/target_motifs_ffi cargo build --release --no-default-features
                   #        cbindgen → ff_motifs_ffi.h
                   #        g++ → bin/main
```

The `make main` target in `SamplingDesign/Makefile` already sets
`CARGO_TARGET_DIR` correctly; you never need to set it manually.

---

## Usage

### Python

Build with the `python` feature (requires [maturin](https://github.com/PyO3/maturin)):

```bash
conda activate <your-env>
maturin develop --manifest-path crates/ff_motifs/Cargo.toml --features python --release
```

#### Quick start — `simulate_timecourse`

```python
from motifs import MotifCheckSimulator

sim = MotifCheckSimulator(
    params="rna_default",
    celsius=37.0,
    k0=1e5,
)

motif_str = """\
GCGAAAGCG
>hairpin 0
(((xxx)))
"""

results = sim.simulate_timecourse(
    sequence="GCGAAAGCG",
    motifs=motif_str,
    check_positions={8: ["hairpin"]},
    t_ext=0.02,
    t_end=0.02,
    num_sims=200,
    num_workers=0,
)

for entry in results:
    pos = entry["nucleotide"]
    counts = entry["motif_counts"]
    frac = counts.get("hairpin", 0) / 200
    print(f"pos {pos}: hairpin occupancy = {frac:.2f}")
```

`motifs` can also be a file path string — if the string resolves to an existing file, it is read from disk; otherwise it is parsed as a literal motif string.

#### Using `simulate_timecourse_checkpoints`

```python
results = sim.simulate_timecourse_checkpoints(
    sequence="GCGAAAGCG",
    motifs=motif_str,
    check_positions={8: ["hairpin"]},
    checkpoints={8: {"hairpin": 0.5}},   # require ≥50% occupancy to continue
    t_ext=0.02,
    t_end=0.02,
    num_sims=200,
)
```

The returned list may be shorter than expected if a checkpoint threshold is not met.

#### ACFP-based workflow with `PyCotransConfig`

`PyCotransConfig` precomputes the ACFP parse once (motif strings, check positions, checkpoints) so the same config object can be reused efficiently across many sequence evaluations — the typical pattern in sequence design optimisation.

```python
from motifs import MotifCheckSimulator, PyCotransConfig

sim = MotifCheckSimulator(params="rna_default", celsius=37.0, k0=1e5)

config = PyCotransConfig(
    acfp=". .. .() (())",
    dl_seq="s3 L2 s4 T0 s1 L1 s2 T1 s2* L1* s1* T2 s4* L2* s3* T3",
    dom_length_dict={
        "s1": 5, "s2": 3, "s3": 5, "s4": 3,
        "L1": 10, "L2": 10,
        "T0": 3, "T1": 3, "T2": 5, "T3": 5,
    },
    t_ext=0.02,
    t_end=0.02,
    num_sims=100,
    threshold=0.6,
    num_workers=4,
)

# Inspect the precomputed path
print(config.nl_path)        # list of dot-bracket strings, one per segment
print(config.check_positions)  # {nucleotide: [motif_names]}

# Evaluate a sequence
score = sim.cotrans_score(sequence, config)
print(f"score = {score:.4f}")  # lower is better

# Or get the full timecourse
results = sim.simulate_timecourse_checkpoints(
    sequence=sequence,
    motifs=config.motifs,
    check_positions=config.check_positions,
    checkpoints=config.checkpoints,
    t_ext=config.t_ext,
    t_end=config.t_end,
    num_sims=config.num_sims,
)
```

#### `PyCotransConfig` properties

| Property | Type | Description |
|----------|------|-------------|
| `nl_path` | `list[str]` | Dot-bracket target structures, one per transcription segment |
| `motifs` | `str` | Auto-generated motif file string (ready to pass to `simulate_timecourse`) |
| `check_positions` | `dict[int, list[str]]` | Nucleotide positions to sample and which motifs to check |
| `checkpoints` | `dict[int, dict[str, float]]` | Per-T-domain early-exit thresholds |
| `dl_seq` | `str` | The domain layout sequence as given |
| `t_ext` | `float` | Extension time window (s) |
| `t_end` | `float` | Final simulation time (s) |
| `num_sims` | `int` | Number of trajectories |
| `threshold` | `float` | Checkpoint threshold fraction |
| `num_workers` | `int` | Rayon thread count |

---

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
motifs = { path = "path/to/crates/ff_motifs" }
```

```rust
use std::sync::Arc;
use ahash::HashMap;
use motifs::timecourse::{Simulator, CotransConfig};

let sim = Simulator::new("rna_default", 37.0, 1e5, 0.0, 0.0)?;

let mut dom_length_dict = HashMap::default();
dom_length_dict.insert("s1".to_string(), 5);
dom_length_dict.insert("L1".to_string(), 10);
dom_length_dict.insert("T1".to_string(), 3);

let config = CotransConfig::from_target(
    ". ()",         // acfp
    "s1 L1 T1",     // dl_seq
    dom_length_dict,
    0.02,           // t_ext
    0.02,           // t_end
    100,            // num_sims
    0.6,            // threshold
    0,              // num_workers (all cores)
)?;

let score = sim.cotrans_score(&sequence, &config)?;
```

---

### C / C++

The crate builds a `cdylib` and exposes a C-compatible API (header: `src/motifs.h`, generated at build time by [cbindgen](https://github.com/mozilla/cbindgen)).

```c
#include "motifs.h"

// 1. Create simulator
SimHandle *sim = sim_create("rna_default", 37.0, 1e5, 0.0, 0.0);

// 2. Create config (precomputed from ACFP)
const char *keys[]   = { "s1", "L1", "T1" };
size_t      vals[]   = {  5,    10,   3   };
ConfigHandle *cfg = config_create(
    ". ()",         // acfp
    "s1 L1 T1",     // dl_seq
    keys, vals, 3,  // domain length dict (parallel arrays)
    0.02,           // t_ext
    0.02,           // t_end
    100,            // num_sims
    0.6,            // threshold
    1               // num_workers = 1 (serial, safe inside OpenMP)
);

// 3. Score a sequence
double score = sim_score(sim, cfg, "GGGAAACCC...");

// 4. Get a full timecourse
size_t nucleotides[1024], counts[1024];
int n = sim_timecourse(sim, cfg, "GGGAAACCC...",
                       nucleotides, counts, 1024);
for (int i = 0; i < n; i++)
    printf("pos %zu: %zu named-motif hits\n", nucleotides[i], counts[i]);

// 5. Inspect the precomputed path
size_t path_len = config_nl_path_len(cfg);
char buf[512];
for (size_t i = 0; i < path_len; i++) {
    config_nl_path_entry(cfg, i, buf, sizeof(buf));
    printf("segment %zu: %s\n", i, buf);
}

// 6. Clean up
config_destroy(cfg);
sim_destroy(sim);
```

> **Thread safety**: `sim_score`, `sim_timecourse`, and `sim_timecourse_checkpoints` are read-only on the handles and are safe to call concurrently from multiple threads. Set `num_workers = 1` when calling from within an OpenMP parallel region to prevent nested thread pool conflicts.

---

## Checkpoint logic

The checkpoint mechanism provides an efficient early-exit for sequence design loops where most candidate sequences fail early in the cotranscriptional path.

For each checkpoint T-domain (in order):
1. Run all simulations from the current start structures through the segment's time windows.
2. At the T-domain's last nucleotide, count the fraction of simulations matching each expected motif.
3. If any motif's occupancy is below its threshold, **stop** and return the partial result.
4. Otherwise, record the final structure of each simulation as the start structure for the next segment and continue.

This means a sequence that folds badly at the first checkpoint costs almost nothing to evaluate, while a sequence that folds well all the way through is fully evaluated.

---

## `MotifRegistry` and motif matching (Rust internals)

`MotifRegistry<E>` (in `ff_kinetics::motifs`) is the core matching engine:

- Loaded from a sequence + motif file via `insert_from_reader` or `insert_from_file`.
- Each `Motif` internally stores a `ConstrPosMap` (from `ff_structure`) — a hash map of only the *specified* positions mapping each to `Pair(partner)` or `Unpaired`.
- `classify_specific_motifs(&structure, motif_names)` converts the structure to a `PairTable` and checks only the listed motifs in O(constrained positions) time. Returns the names of all matching motifs, or `["Unassigned"]` if none match.
- The registry always contains an `"Unassigned"` catch-all that matches everything when no named motif does.
