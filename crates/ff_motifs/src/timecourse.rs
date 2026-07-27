use ahash::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use ff_energy::{NucleotideVec, ViennaRNA};
use ff_energy::parameters::{RNA_TURNER_2004, RNA_EXTENDED, DNA_MATHEWS_2004};
use ff_kinetics::{SSA, Arrhenius, LoopNeighbors, MotifRegistry, shift_policy}; //Walker
use ff_structure::{DotBracketVec, PairTable};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

pub struct TimecourseResults {
    pub motif_match_table:    HashMap<usize, HashMap<String, usize>>,
    /// Sum of motif distances over all trajectories per check position.
    /// Keyed the same way as motif_match_table: position → motif_name → sum_of_distances.
    pub motif_distance_table: HashMap<usize, HashMap<String, usize>>,
    pub checkpoint_structures: Vec<String>,
}


// ---------------------------------------------------------------------------
// Precomputed config — computed once per design run, reused every evaluation
// ---------------------------------------------------------------------------

pub struct CotransConfig {
    pub motifs:          String,
    pub check_positions: HashMap<usize, Vec<String>>,
    pub checkpoints:     HashMap<usize, HashMap<String, f64>>,
    pub nl_path:         Vec<String>,
    pub dl_seq:          String,
    pub dom_length_dict: HashMap<String, usize>,
    pub t_ext:           f64,
    pub t_end:           f64,
    pub num_sims:        usize,
    pub threshold:       f64,
    pub num_workers:     usize,   // 0 = use Rayon default (all cores)
    /// Per-checkpoint weights (one per segment/T-domain). Empty = all 1.0.
    pub weights:         Vec<f64>,
    /// Scoring objective: "occupancy" (default) or "distance".
    pub objective:       String,
    /// Precomputed maximum possible distance per motif (motif_name → max_dist).
    pub max_distances:   HashMap<String, usize>,
}

impl CotransConfig {
    pub fn from_target(
        acfp:            &str,
        dl_seq:          &str,
        dom_length_dict: HashMap<String, usize>,
        t_ext:           f64,
        t_end:           f64,
        num_sims:        usize,
        threshold:       f64,
        num_workers:     usize,
    ) -> Result<Self, String> {
        let nl_path         = compute_nl_path(acfp, dl_seq, &dom_length_dict)?;
        let motifs          = compute_motifs(&nl_path)?;
        let check_positions = compute_check_positions(dl_seq, &dom_length_dict)?;
        let checkpoints     = compute_checkpoints(dl_seq, &dom_length_dict, threshold)?;

        let max_distances = compute_max_distances(&nl_path);

        Ok(Self {
            motifs,
            check_positions,
            checkpoints,
            nl_path,
            dl_seq: dl_seq.to_string(),
            dom_length_dict,
            t_ext,
            t_end,
            num_sims,
            threshold,
            num_workers,
            weights:       Vec::new(),
            objective:     "occupancy".to_string(),
            max_distances,
        })
    }
}

/// Maximum possible distance for a motif constraint string.
/// '(' contributes 2 (wrong or absent pair), 'x' contributes 1 (position should be unpaired).
/// ')' and '.' are not stored in ConstrPosMap and contribute 0.
fn compute_max_distances(nl_path: &[String]) -> HashMap<String, usize> {
    nl_path.iter().enumerate().map(|(i, s)| {
        let name = format!("motif_{}", i + 1);
        let max_dist: usize = s.chars().map(|c| match c {
            '(' => 2,
            'x' => 1,
            _   => 0,
        }).sum();
        (name, max_dist)
    }).collect()
}

// ---------------------------------------------------------------------------
// Utility functions — all take rna_struct, return precomputed data
// ---------------------------------------------------------------------------

fn compute_motifs(nl_path: &[String]) -> Result<String, String> {
    if nl_path.is_empty() {
        return Err("nl_path is empty".into());
    }

    let mut motif_str = String::new();

    for (i, structure) in nl_path.iter().enumerate() {
        motif_str.push_str(&format!(">motif_{} 0\n", i + 1));
        motif_str.push_str(structure);
        motif_str.push('\n');
    }

    Ok(motif_str)
}

fn compute_check_positions(
    dl_seq:          &str,
    dom_length_dict: &HashMap<String, usize>,
) -> Result<HashMap<usize, Vec<String>>, String> {
    let segments = parse_dl_seq(dl_seq);
    let mut check_positions: HashMap<usize, Vec<String>> = HashMap::default();
    let mut nucleotide_pos = 0usize;

    for (seg_idx, segment) in segments.iter().enumerate() {
        let motif_name = format!("motif_{}", seg_idx + 1);

        for domain in segment {
            let base   = get_base_name(domain);
            let length = dom_length_dict.get(base).ok_or_else(|| {
                format!("domain '{}' not found in dom_length_dict", base)
            })?;

            let is_t = base.starts_with('T') &&
                       base[1..].parse::<usize>().is_ok();

            if is_t {
                // map every nucleotide in this T-domain to the motif
                for pos in nucleotide_pos..nucleotide_pos + length {
                    check_positions
                        .entry(pos)
                        .or_insert_with(Vec::new)
                        .push(motif_name.clone());
                }
            }

            nucleotide_pos += length;
        }
    }

    Ok(check_positions)
}

fn compute_checkpoints(
    dl_seq:          &str,
    dom_length_dict: &HashMap<String, usize>,
    threshold:       f64,
) -> Result<HashMap<usize, HashMap<String, f64>>, String> {
    let segments = parse_dl_seq(dl_seq);
    let mut checkpoints: HashMap<usize, HashMap<String, f64>> = HashMap::default();
    let mut nucleotide_pos = 0usize;

    for (seg_idx, segment) in segments.iter().enumerate() {
        let motif_name = format!("motif_{}", seg_idx + 1);

        for domain in segment {
            let base   = get_base_name(domain);
            let length = dom_length_dict.get(base).ok_or_else(|| {
                format!("domain '{}' not found in dom_length_dict", base)
            })?;

            let is_t = base.starts_with('T') &&
                       base[1..].parse::<usize>().is_ok();

            if is_t {
                // last nucleotide of this T-domain
                let last_pos = nucleotide_pos + length - 1;
                let mut motif_threshold = HashMap::default();
                motif_threshold.insert(motif_name.clone(), threshold);
                checkpoints.insert(last_pos, motif_threshold);
            }

            nucleotide_pos += length;
        }
    }

    Ok(checkpoints)
}

fn parse_dl_seq(dl_seq: &str) -> Vec<Vec<String>> {
    // Split dl_seq into segments, each ending with a T-domain
    let domains: Vec<String> = dl_seq.split_whitespace()
        .map(|s| s.to_string())
        .collect();

    let mut segments: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for domain in domains {
        let is_t = domain.starts_with('T') && 
                   domain[1..].parse::<usize>().is_ok();
        current.push(domain);
        if is_t {
            segments.push(current.clone());
            current.clear();
        }
    }
    segments
}

fn get_base_name(domain: &str) -> &str {
    // "s3*" -> "s3", "L1*" -> "L1", "s3" -> "s3"
    domain.trim_end_matches('*')
}

fn is_l_domain(domain: &str) -> bool {
    let base = get_base_name(domain);
    base.starts_with('L') && base[1..].parse::<usize>().is_ok()
}

pub fn compute_nl_path(
    acfp:            &str,
    dl_seq:          &str,
    dom_length_dict: &HashMap<String, usize>,
) -> Result<Vec<String>, String> {
    let segments     = parse_dl_seq(dl_seq);
    let acfp_entries: Vec<&str> = acfp.split_whitespace().collect();

    if acfp_entries.len() != segments.len() {
        return Err(format!(
            "acfp has {} entries but dl_seq has {} segments",
            acfp_entries.len(), segments.len()
        ));
    }

    let mut nl_path: Vec<String> = Vec::new();

    for (entry_idx, entry) in acfp_entries.iter().enumerate() {
        // entry covers segments 0..=entry_idx
        // each character in entry corresponds to one segment
        let chars: Vec<char> = entry.chars().collect();

        if chars.len() != entry_idx + 1 {
            return Err(format!(
                "acfp entry '{}' has {} chars but should cover {} segments",
                entry, chars.len(), entry_idx + 1
            ));
        }

        // Find all paired segment index pairs: '(' matches with ')'
        // using a stack to match them
        let mut pair_map: HashMap<usize, usize> = HashMap::default();
        let mut stack: Vec<usize> = Vec::new();
        for (i, &c) in chars.iter().enumerate() {
            match c {
                '(' => stack.push(i),
                ')' => {
                    let j = stack.pop().ok_or_else(|| {
                        format!("unmatched ')' at position {} in '{}'", i, entry)
                    })?;
                    pair_map.insert(j, i);  // j pairs with i
                    pair_map.insert(i, j);  // i pairs with j
                }
                '.' => {}
                other => return Err(format!("unexpected char '{}' in acfp", other)),
            }
        }
        if !stack.is_empty() {
            return Err(format!("unmatched '(' in acfp entry '{}'", entry));
        }

        // Now build the nucleotide-level dot-bracket string
        let mut result = String::new();

        for (seg_idx, segment) in segments[..=entry_idx].iter().enumerate() {
            let seg_char = chars[seg_idx];

            for domain in segment {
                let base = get_base_name(domain);
                let length = dom_length_dict.get(base).ok_or_else(|| {
                    format!("domain '{}' not found in dom_length_dict", base)
                })?;

                if is_l_domain(domain) {
                    match seg_char {
                        '.' => {
                            // unpaired L domain
                            result.push_str(&"x".repeat(*length));
                        }
                        '(' => {
                            // L domain opens — use '('
                            result.push_str(&"(".repeat(*length));
                        }
                        ')' => {
                            // L domain closes — use ')'
                            result.push_str(&")".repeat(*length));
                        }
                        other => return Err(format!("unexpected char '{}'", other)),
                    }
                } else {
                    // s and T domains are always '.'
                    result.push_str(&".".repeat(*length));
                }
            }
        }

        nl_path.push(result);
    }

    Ok(nl_path)
}


// ---------------------------------------------------------------------------
// Simulator — owns shared immutable state, all logic as methods
// ---------------------------------------------------------------------------

pub struct Simulator {
    pub energy_model: Arc<ViennaRNA>,
    pub rate_model:   Arc<Arrhenius>,
    pub is_rna:       bool,
}

impl Simulator {
    pub fn new(
        params:  &str,
        celsius: f64,
        k0:      f64,
        k3ws:    f64,
        k4ws:    f64,
    ) -> Result<Self, String> {
        if k0 < 0.0 || k3ws < 0.0 || k4ws < 0.0 {
            return Err("rate constants must be non-negative".into());
        }
        let mut is_rna = true;
        let thermo = match params {
            "rna_default"  => &RNA_TURNER_2004,
            "rna_extended" => &RNA_EXTENDED,
            "dna"          => { is_rna = false; &DNA_MATHEWS_2004 }
            other => return Err(format!(
                "unknown parameter set '{}'; valid options: \
                 'rna_default', 'rna_extended', 'dna'", other
            )),
        };
        Ok(Self {
            energy_model: Arc::new(ViennaRNA::from_thermo_params(thermo, celsius)),
            rate_model:   Arc::new(Arrhenius::new(celsius, k0, Some(k3ws), Some(k4ws))),
            is_rna,
        })
    }

    // -----------------------------------------------------------------------
    // Full timecourse — single start structure, all sims identical
    // -----------------------------------------------------------------------

    pub fn simulate_timecourse(
        &self,
        sequence:       &str,
        motifs:         &str,
        check_positions: HashMap<usize, Vec<String>>,
        start:          Option<&str>,
        t_ext:          Option<f64>,
        t_end:          f64,
        num_sims:       usize,
        num_workers:     usize,   // 0 = default
    ) -> Result<TimecourseResults, String> {
        let sequence  = Arc::new(
            NucleotideVec::try_from_rna(sequence).map_err(|e| e.to_string())?
        );
        let start_pt  = parse_start(start)?;
        let start_len = start.map(|s| s.len()).unwrap_or(1);
        let times     = build_times(sequence.len(), start_len, t_ext, t_end);
        let registry  = build_motif_registry(&sequence, &self.energy_model, motifs)?;
        let starts    = vec![start_pt; num_sims];

        let per_sim = build_parallel_runs(
            Arc::clone(&sequence),
            starts,
            Arc::clone(&self.energy_model),
            Arc::clone(&self.rate_model),
            times,
            registry,
            Arc::new(check_positions),
            0,
            false,
            num_workers,
        );

        Ok(merge_results(per_sim))
    }

    // -----------------------------------------------------------------------
    // Checkpoint timecourse — segments with early-exit
    // -----------------------------------------------------------------------

    pub fn simulate_timecourse_checkpoints(
        &self,
        sequence:        &str,
        motifs:          &str,
        check_positions: HashMap<usize, Vec<String>>,
        checkpoints:     HashMap<usize, HashMap<String, f64>>,
        objective:       &str,
        max_distances:   &HashMap<String, usize>,
        start:           Option<&str>,
        t_ext:           Option<f64>,
        t_end:           f64,
        num_sims:        usize,
        num_workers:     usize,
    ) -> Result<TimecourseResults, String> {
        let sequence  = Arc::new(
            NucleotideVec::try_from_rna(sequence).map_err(|e| e.to_string())?
        );
        let start_pt  = parse_start(start)?;
        let start_len = start.map(|s| s.len()).unwrap_or(1);
        let times     = build_times(sequence.len(), start_len, t_ext, t_end);
        let registry  = build_motif_registry(&sequence, &self.energy_model, motifs)?;
        let check_positions = Arc::new(check_positions);
        let checkpoints     = Arc::new(checkpoints);

        let mut cp_indices: Vec<usize> = checkpoints.keys().copied().collect();
        cp_indices.sort_unstable();

        let mut accumulated = TimecourseResults {
            motif_match_table:    HashMap::default(),
            motif_distance_table: HashMap::default(),
            checkpoint_structures: Vec::new(),
        };

        let mut seg_start      = 0usize;
        let mut current_starts = vec![start_pt; num_sims];

        for (seg_idx, &cp) in cp_indices.iter().enumerate() {
            if cp >= times.len() { continue; }

            let seg_times = Arc::new(times[seg_start..=cp].to_vec());
            let is_last   = seg_idx == cp_indices.len() - 1;
            let owned     = std::mem::take(&mut current_starts);

            let per_sim = build_parallel_runs(
                Arc::clone(&sequence),
                owned,
                Arc::clone(&self.energy_model),
                Arc::clone(&self.rate_model),
                seg_times,
                Arc::clone(&registry),
                Arc::clone(&check_positions),
                seg_start,
                !is_last,
                num_workers,
            );

            let next_starts: Option<Vec<PairTable>> = if !is_last {
                Some(per_sim.iter().map(|r| {
                    let s  = r.checkpoint_structures.last()
                               .expect("sim produced no checkpoint structure");
                    let db = DotBracketVec::try_from(s.as_str())
                               .expect("invalid dot-bracket from sim");
                    PairTable::try_from(&db).expect("invalid pair table from sim")
                }).collect())
            } else {
                None
            };

            let seg = merge_results(per_sim);
            for (nuc, counts) in seg.motif_match_table {
                accumulated.motif_match_table.insert(nuc, counts);
            }
            for (nuc, dists) in seg.motif_distance_table {
                accumulated.motif_distance_table.insert(nuc, dists);
            }

            if let Some(thresholds) = checkpoints.get(&cp) {
                let all_pass = thresholds.iter().all(|(name, &threshold)| {
                    if threshold <= 0.0 { return true; }
                    match objective {
                        "distance" => {
                            let max_dist = max_distances.get(name).copied().unwrap_or(0);
                            if max_dist == 0 { return true; }
                            let dist_sum = accumulated.motif_distance_table
                                .get(&cp)
                                .and_then(|m| m.get(name))
                                .copied()
                                .unwrap_or(0);
                            let score = dist_sum as f64
                                / (num_sims as f64 * max_dist as f64);
                            score <= threshold
                        }
                        _ => {
                            // occupancy: threshold is upper bound on badness (1 - occ)
                            let count = accumulated.motif_match_table
                                .get(&cp)
                                .and_then(|m| m.get(name))
                                .copied()
                                .unwrap_or(0);
                            let badness = 1.0 - count as f64 / num_sims as f64;
                            badness <= threshold
                        }
                    }
                });
                if !all_pass { return Ok(accumulated); }
            }

            if let Some(s) = next_starts { current_starts = s; }
            seg_start = cp + 1;
        }

        Ok(accumulated)
    }

    // -----------------------------------------------------------------------
    // Scalar objective for C++ gradient descent
    // Returns fraction of sims reaching rna_struct, negated (lower = better)
    // -----------------------------------------------------------------------

    pub fn cotrans_score(
        &self,
        sequence: &str,
        config:   &CotransConfig,
    ) -> Result<f64, String> {
        let results = self.simulate_timecourse_checkpoints(
            sequence,
            &config.motifs,
            config.check_positions.clone(),
            config.checkpoints.clone(),
            &config.objective,
            &config.max_distances,
            None,
            Some(config.t_ext),
            config.t_end,
            config.num_sims,
            config.num_workers,
        )?;
        let (total, _) = compute_checkpoint_scores(&results, config);
        Ok(total.clamp(0.0, 1.0))
    }

    /// Like cotrans_score but also returns per-checkpoint scores.
    /// Returns (total_score, vec![d1, d2, ..., dn]) where each di is in [0,1].
    pub fn cotrans_score_vec(
        &self,
        sequence: &str,
        config:   &CotransConfig,
    ) -> Result<(f64, Vec<f64>), String> {
        let results = self.simulate_timecourse_checkpoints(
            sequence,
            &config.motifs,
            config.check_positions.clone(),
            config.checkpoints.clone(),
            &config.objective,
            &config.max_distances,
            None,
            Some(config.t_ext),
            config.t_end,
            config.num_sims,
            config.num_workers,
        )?;
        let (total, per_cp) = compute_checkpoint_scores(&results, config);
        Ok((total.clamp(0.0, 1.0), per_cp))
    }

    /// Compute both distance and occupancy scores from a single simulation pass.
    /// Uses distance-based early exit. Returns (dist_score, occ_score), each in [0, 1].
    pub fn cotrans_score_both(
        &self,
        sequence: &str,
        config:   &CotransConfig,
    ) -> Result<(f64, f64), String> {
        let results = self.simulate_timecourse_checkpoints(
            sequence,
            &config.motifs,
            config.check_positions.clone(),
            config.checkpoints.clone(),
            "distance",
            &config.max_distances,
            None,
            Some(config.t_ext),
            config.t_end,
            config.num_sims,
            config.num_workers,
        )?;
        let (dist, _) = compute_checkpoint_scores_for_obj(&results, config, "distance");
        let (occ,  _) = compute_checkpoint_scores_for_obj(&results, config, "occupancy");
        Ok((dist.clamp(0.0, 1.0), occ.clamp(0.0, 1.0)))
    }
}

// ---------------------------------------------------------------------------
// Shared per-checkpoint scoring helper
// Returns (total_weighted_score, per_checkpoint_normalized_scores)
// ---------------------------------------------------------------------------

fn compute_checkpoint_scores(
    results: &TimecourseResults,
    config:  &CotransConfig,
) -> (f64, Vec<f64>) {
    compute_checkpoint_scores_for_obj(results, config, &config.objective)
}

/// Like compute_checkpoint_scores but with an explicit objective override.
/// Allows computing both distance and occupancy from the same TimecourseResults
/// without re-running the simulation.
fn compute_checkpoint_scores_for_obj(
    results:   &TimecourseResults,
    config:    &CotransConfig,
    objective: &str,
) -> (f64, Vec<f64>) {
    // Build T-domain groups from ALL expected check positions (not just observed)
    // so that non-reached checkpoints (early exit) get worst-case penalty.
    let mut all_positions: Vec<usize> = config.check_positions.keys().cloned().collect();
    all_positions.sort_unstable();

    let mut t_domain_groups: Vec<(Vec<usize>, String)> = Vec::new();
    let mut current_group:   Vec<usize>                = Vec::new();

    for &pos in &all_positions {
        if current_group.is_empty() || pos == *current_group.last().unwrap() + 1 {
            current_group.push(pos);
        } else {
            let motif_name = config.check_positions[&current_group[0]][0].clone();
            t_domain_groups.push((current_group.clone(), motif_name));
            current_group = vec![pos];
        }
    }
    if !current_group.is_empty() {
        let motif_name = config.check_positions[&current_group[0]][0].clone();
        t_domain_groups.push((current_group, motif_name));
    }

    let n_groups = t_domain_groups.len();
    let weights: Vec<f64> = if config.weights.is_empty() {
        vec![1.0; n_groups]
    } else {
        let mut w = config.weights.clone();
        w.resize(n_groups, 1.0);
        w
    };
    let weight_sum: f64 = weights.iter().sum::<f64>().max(f64::EPSILON);

    let mut per_cp: Vec<f64> = Vec::with_capacity(n_groups);
    let mut total  = 0.0_f64;

    match objective {
        "distance" => {
            for (i, (group, motif_name)) in t_domain_groups.iter().enumerate() {
                let w        = weights[i];
                let max_dist = config.max_distances.get(motif_name).copied().unwrap_or(0);

                let norm_dist = if max_dist == 0 {
                    0.0
                } else {
                    let mut dist_sum  = 0usize;
                    let mut n_reached = 0usize;
                    for pos in group {
                        if let Some(dist_map) = results.motif_distance_table.get(pos) {
                            dist_sum  += dist_map.get(motif_name).copied().unwrap_or(0);
                            n_reached += 1;
                        }
                    }
                    if n_reached == 0 {
                        1.0
                    } else {
                        dist_sum as f64
                            / (n_reached as f64 * config.num_sims as f64 * max_dist as f64)
                    }
                };
                per_cp.push(norm_dist.clamp(0.0, 1.0));
                total += w * norm_dist;
            }
            (total / weight_sum, per_cp)
        }
        _ => {
            // "occupancy" (default)
            let mut total_occ = 0.0_f64;
            for (i, (group, motif_name)) in t_domain_groups.iter().enumerate() {
                let w = weights[i];
                let mut hits  = 0usize;
                let mut n_pos = 0usize;
                for pos in group {
                    if let Some(counts) = results.motif_match_table.get(pos) {
                        let h: usize = counts.iter()
                            .filter(|(name, _)| *name != "Unassigned")
                            .map(|(_, &c)| c)
                            .sum();
                        hits  += h;
                        n_pos += 1;
                    }
                }
                let cp_occ = if n_pos > 0 {
                    hits as f64 / (n_pos as f64 * config.num_sims as f64)
                } else {
                    0.0
                };
                per_cp.push((1.0 - cp_occ).clamp(0.0, 1.0));
                total_occ += w * cp_occ;
                let _ = motif_name;
            }
            (1.0 - total_occ / weight_sum, per_cp)
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_start(start: Option<&str>) -> Result<PairTable, String> {
    let s  = start.unwrap_or(".");
    let db = DotBracketVec::try_from(s).map_err(|e| e.to_string())?;
    PairTable::try_from(&db).map_err(|e| e.to_string())
}

fn build_times(
    seq_len:   usize,
    start_len: usize,
    t_ext:     Option<f64>,
    t_end:     f64,
) -> Arc<Vec<f64>> {
    Arc::new(if let Some(dt) = t_ext {
        let mut v = vec![dt; seq_len - start_len];
        v.push(t_end);
        v
    } else {
        vec![t_end]
    })
}

fn build_motif_registry(
    sequence:     &Arc<NucleotideVec>,
    energy_model: &Arc<ViennaRNA>,
    motifs:       &str,
) -> Result<Arc<MotifRegistry<ViennaRNA>>, String> {
    let mut reg = MotifRegistry::from((Arc::clone(sequence), Arc::clone(energy_model)));
    let path = Path::new(motifs);
    if path.is_file() {
        reg.insert_from_file(&PathBuf::from(motifs))
           .map_err(|e| e.to_string())?;
    } else {
        // prepend the actual sequence so the registry accepts it
        let motifs_with_seq = format!("{}\n{}", sequence.to_string(), motifs);
        reg.insert_from_reader(Cursor::new(motifs_with_seq), "manual")
           .map_err(|e| e.to_string())?;
    }
    Ok(Arc::new(reg))
}

fn merge_results(results: Vec<TimecourseResults>) -> TimecourseResults {
    let mut merged_match: HashMap<usize, HashMap<String, usize>> = HashMap::default();
    let mut merged_dist:  HashMap<usize, HashMap<String, usize>> = HashMap::default();
    let mut checkpoint_structures = Vec::with_capacity(results.len());

    for r in results {
        for (tp, counts) in r.motif_match_table {
            let slot = merged_match.entry(tp).or_insert_with(HashMap::default);
            for (name, count) in counts {
                slot.entry(name).and_modify(|c| *c += count).or_insert(count);
            }
        }
        for (tp, dists) in r.motif_distance_table {
            let slot = merged_dist.entry(tp).or_insert_with(HashMap::default);
            for (name, dist) in dists {
                slot.entry(name).and_modify(|d| *d += dist).or_insert(dist);
            }
        }
        if let Some(s) = r.checkpoint_structures.into_iter().next() {
            checkpoint_structures.push(s);
        }
    }
    TimecourseResults {
        motif_match_table:    merged_match,
        motif_distance_table: merged_dist,
        checkpoint_structures,
    }
}

fn build_parallel_runs(
    sequence:              Arc<NucleotideVec>,
    starts:                Vec<PairTable>,
    energy_model:          Arc<ViennaRNA>,
    rate_model:            Arc<Arrhenius>,
    times:                 Arc<Vec<f64>>,
    motif_registry:        Arc<MotifRegistry<ViennaRNA>>,
    check_positions:       Arc<HashMap<usize, Vec<String>>>,
    nucleotide_offset:     usize,
    record_final_structure: bool,
    num_workers:           usize,   // 0 = Rayon default
) -> Vec<TimecourseResults> {
    let use_3ws = rate_model.k3ws().is_some();
    let use_4ws = rate_model.k4ws().is_some();

    let run = |start: PairTable| {
        SimulationRunner {
            sequence:              Arc::clone(&sequence),
            start,
            energy_model:          Arc::clone(&energy_model),
            rate_model:            Arc::clone(&rate_model),
            times:                 Arc::clone(&times),
            motif_registry:        Arc::clone(&motif_registry),
            check_positions:       Arc::clone(&check_positions),
            rng:                   SmallRng::from_os_rng(),
            nucleotide_offset,
            record_final_structure,
        }.run(use_3ws, use_4ws)
    };

    if num_workers == 1 {
        // fully serial — no Rayon at all, safe to call from OpenMP threads
        starts.into_iter().map(run).collect()
    } else if num_workers == 0 {
        // use Rayon global thread pool (all available cores)
        starts.into_par_iter().map(run).collect()
    } else {
        // build a scoped thread pool with exactly num_workers threads
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .expect("failed to build Rayon thread pool");
        pool.install(|| starts.into_par_iter().map(run).collect())
    }
}

// ---------------------------------------------------------------------------
// Per-simulation runner
// ---------------------------------------------------------------------------

struct SimulationRunner {
    sequence:              Arc<NucleotideVec>,
    start:                 PairTable,
    energy_model:          Arc<ViennaRNA>,
    rate_model:            Arc<Arrhenius>,
    times:                 Arc<Vec<f64>>,
    motif_registry:        Arc<MotifRegistry<ViennaRNA>>,
    check_positions:       Arc<HashMap<usize, Vec<String>>>,
    rng:                   SmallRng,
    nucleotide_offset:     usize,
    record_final_structure: bool,
}

impl SimulationRunner {
    fn run(mut self, use_3ws: bool, use_4ws: bool) -> TimecourseResults {
        let mut results = TimecourseResults {
            motif_match_table:    HashMap::default(),
            motif_distance_table: HashMap::default(),
            checkpoint_structures: Vec::new(),
        };

        macro_rules! run_with_policy {
            ($policy:expr) => {{
                let walker = LoopNeighbors::try_from((
                    Arc::clone(&self.sequence),
                    self.start.clone(),
                    Arc::clone(&self.energy_model),
                    $policy,
                )).expect("failed to build walker");

                let mut ssa    = SSA::from((walker, *self.rate_model));
                let offset     = self.nucleotide_offset;

                ssa.co_simulate_checked(&mut self.rng, &self.times[..], |w, local_nuc| {
                    let global_nuc = local_nuc + offset;
                    if let Some(names) = self.check_positions.get(&global_nuc) {
                        if let Ok(structure) = DotBracketVec::try_from(w.to_string().as_str()) {
                            let (classifications, distances) =
                                self.motif_registry.classify_and_distance(&structure, names);
                            for classification in classifications {
                                results.motif_match_table
                                    .entry(global_nuc)
                                    .or_insert_with(HashMap::default)
                                    .entry(classification)
                                    .and_modify(|c| *c += 1)
                                    .or_insert(1);
                            }
                            for (name, dist) in distances {
                                results.motif_distance_table
                                    .entry(global_nuc)
                                    .or_insert_with(HashMap::default)
                                    .entry(name)
                                    .and_modify(|d| *d += dist)
                                    .or_insert(dist);
                            }
                        }
                    }
                });

                if self.record_final_structure {
                    results.checkpoint_structures =
                        vec![ssa.current_structure().to_string()];
                }
            }};
        }

        match (use_3ws, use_4ws) {
            (false, false) => run_with_policy!(shift_policy::NoShift),
            (true,  false) => run_with_policy!(shift_policy::ThreeWayOnly),
            (false, true)  => run_with_policy!(shift_policy::FourWayOnly),
            (true,  true)  => run_with_policy!(shift_policy::ThreeAndFour),
        }

        results
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Shared test fixtures
    // ---------------------------------------------------------------------------

    fn example_dom_length_dict() -> HashMap<String, usize> {
        let mut d = HashMap::default();
        d.insert("s1".to_string(), 5);
        d.insert("s2".to_string(), 3);
        d.insert("s3".to_string(), 5);
        d.insert("s4".to_string(), 3);
        d.insert("L1".to_string(), 10);
        d.insert("L2".to_string(), 10);
        d.insert("T0".to_string(), 3);
        d.insert("T1".to_string(), 3);
        d.insert("T2".to_string(), 5);
        d.insert("T3".to_string(), 5);
        d
    }

    const EXAMPLE_DL_SEQ: &str =
        "s3 L2 s4 T0 s1 L1 s2 T1 s2* L1* s1* T2 s4* L2* s3* T3";

    const EXAMPLE_ACFP: &str = ". .. .() (())";

    fn print_separator(title: &str) {
        println!("\n{}", "=".repeat(60));
        println!("  {}", title);
        println!("{}", "=".repeat(60));
    }

    fn print_subsection(title: &str) {
        println!("\n  --- {} ---", title);
    }

    // ---------------------------------------------------------------------------
    // parse_dl_seq
    // ---------------------------------------------------------------------------

    #[test]
    fn test_parse_dl_seq_segment_count() {
        print_separator("TEST: parse_dl_seq segment count");
        println!("  Input dl_seq: {}", EXAMPLE_DL_SEQ);

        let segments = parse_dl_seq(EXAMPLE_DL_SEQ);
        println!("  Number of segments found: {}", segments.len());
        for (i, seg) in segments.iter().enumerate() {
            println!("    Segment {}: {:?}", i, seg);
        }

        assert_eq!(segments.len(), 4, "should have 4 segments");
        println!("  ✓ Correct: 4 segments found");
    }

    #[test]
    fn test_parse_dl_seq_segment_contents() {
        print_separator("TEST: parse_dl_seq segment contents");
        println!("  Input dl_seq: {}", EXAMPLE_DL_SEQ);

        let segments = parse_dl_seq(EXAMPLE_DL_SEQ);

        print_subsection("Checking segment 0");
        println!("    Expected: [s3, L2, s4, T0]");
        println!("    Got:      {:?}", segments[0]);
        assert_eq!(segments[0], vec!["s3", "L2", "s4", "T0"]);
        println!("    ✓ Match");

        print_subsection("Checking segment 1");
        println!("    Expected: [s1, L1, s2, T1]");
        println!("    Got:      {:?}", segments[1]);
        assert_eq!(segments[1], vec!["s1", "L1", "s2", "T1"]);
        println!("    ✓ Match");

        print_subsection("Checking segment 2");
        println!("    Expected: [s2*, L1*, s1*, T2]");
        println!("    Got:      {:?}", segments[2]);
        assert_eq!(segments[2], vec!["s2*", "L1*", "s1*", "T2"]);
        println!("    ✓ Match");

        print_subsection("Checking segment 3");
        println!("    Expected: [s4*, L2*, s3*, T3]");
        println!("    Got:      {:?}", segments[3]);
        assert_eq!(segments[3], vec!["s4*", "L2*", "s3*", "T3"]);
        println!("    ✓ Match");
    }

    #[test]
    fn test_parse_dl_seq_each_segment_ends_with_t() {
        print_separator("TEST: parse_dl_seq each segment ends with T-domain");

        let segments = parse_dl_seq(EXAMPLE_DL_SEQ);
        for (i, seg) in segments.iter().enumerate() {
            let last = seg.last().unwrap();
            let base = get_base_name(last);
            println!(
                "  Segment {}: last domain = '{}', base = '{}'",
                i, last, base
            );
            assert!(
                base.starts_with('T') && base[1..].parse::<usize>().is_ok(),
                "segment should end with T-domain, got '{}'", last
            );
            println!("    ✓ Ends with T-domain");
        }
    }

    // ---------------------------------------------------------------------------
    // get_base_name / is_l_domain
    // ---------------------------------------------------------------------------

    #[test]
    fn test_get_base_name() {
        print_separator("TEST: get_base_name");

        let cases = vec![
            ("s3",  "s3"),
            ("s3*", "s3"),
            ("L1*", "L1"),
            ("T0",  "T0"),
            ("s2*", "s2"),
        ];

        for (input, expected) in cases {
            let result = get_base_name(input);
            println!(
                "  get_base_name({:6}) = {:6}  expected: {:6}  {}",
                input, result, expected,
                if result == expected { "✓" } else { "✗" }
            );
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_is_l_domain() {
        print_separator("TEST: is_l_domain");

        let cases = vec![
            ("L1",   true),
            ("L2*",  true),
            ("s1",   false),
            ("s1*",  false),
            ("T0",   false),
            ("T1*",  false),
        ];

        for (input, expected) in cases {
            let result = is_l_domain(input);
            println!(
                "  is_l_domain({:6}) = {:5}  expected: {:5}  {}",
                input, result, expected,
                if result == expected { "✓" } else { "✗" }
            );
            assert_eq!(result, expected);
        }
    }

    // ---------------------------------------------------------------------------
    // compute_nl_path
    // ---------------------------------------------------------------------------

    #[test]
    fn test_nl_path_length() {
        print_separator("TEST: compute_nl_path entry count");
        println!("  acfp:   {}", EXAMPLE_ACFP);
        println!("  dl_seq: {}", EXAMPLE_DL_SEQ);

        let d = example_dom_length_dict();
        let nl = compute_nl_path(EXAMPLE_ACFP, EXAMPLE_DL_SEQ, &d).unwrap();

        println!("  Number of nl_path entries: {}", nl.len());
        assert_eq!(nl.len(), 4, "should have 4 entries");
        println!("  ✓ Correct: 4 entries");
    }

    #[test]
    fn test_nl_path_growing() {
        print_separator("TEST: compute_nl_path entries grow in length");

        let d = example_dom_length_dict();
        let nl = compute_nl_path(EXAMPLE_ACFP, EXAMPLE_DL_SEQ, &d).unwrap();

        println!("  Entry lengths:");
        for (i, s) in nl.iter().enumerate() {
            println!("    nl_path[{}]: {} chars", i, s.len());
        }

        for i in 1..nl.len() {
            assert!(
                nl[i].len() > nl[i-1].len(),
                "entry {} ({}) should be longer than entry {} ({})",
                i, nl[i].len(), i-1, nl[i-1].len()
            );
            println!("  ✓ entry {} longer than entry {}", i, i-1);
        }
    }

    #[test]
    fn test_nl_path_all_entries() {
        print_separator("TEST: compute_nl_path full output");
        println!("  acfp:   {}", EXAMPLE_ACFP);
        println!("  dl_seq: {}", EXAMPLE_DL_SEQ);

        let d = example_dom_length_dict();
        let nl = compute_nl_path(EXAMPLE_ACFP, EXAMPLE_DL_SEQ, &d).unwrap();

        println!("\n  Domain lengths:");
        let mut sorted_domains: Vec<(&String, &usize)> = d.iter().collect();
        sorted_domains.sort_by_key(|(k, _)| k.as_str());
        for (name, len) in &sorted_domains {
            println!("    {}: {}", name, len);
        }

        println!("\n  nl_path entries:");
        let acfp_entries: Vec<&str> = EXAMPLE_ACFP.split_whitespace().collect();
        for (i, (entry, acfp)) in nl.iter().zip(acfp_entries.iter()).enumerate() {
            println!("\n  [{i}] acfp entry: '{acfp}'");
            println!("       length: {}", entry.len());
            println!("       structure: {}", entry);

            // annotate which part belongs to which segment/domain
            let segments = parse_dl_seq(EXAMPLE_DL_SEQ);
            let mut pos = 0;
            let mut annotation = String::new();
            let mut domain_labels = String::new();
            for seg in &segments[..=i] {
                for domain in seg {
                    let base = get_base_name(domain);
                    if let Some(&len) = d.get(base) {
                        let chunk = &entry[pos..pos+len];
                        println!("         domain {:6}: pos {:3}-{:3} → '{}'",
                            domain, pos, pos+len-1, chunk);
                        pos += len;
                    }
                }
            }
        }
    }

    #[test]
    fn test_nl_path_entry0_all_dots_and_x() {
        print_separator("TEST: compute_nl_path entry 0 (segment 0 only, all unpaired)");

        let d = example_dom_length_dict();
        let nl = compute_nl_path(EXAMPLE_ACFP, EXAMPLE_DL_SEQ, &d).unwrap();
        let entry = &nl[0];

        println!("  acfp entry: '.'  → segment 0 unpaired");
        println!("  structure:  {}", entry);
        println!("  length:     {} (expected 21: s3=5 + L2=10 + s4=3 + T0=3)", entry.len());

        print_subsection("s3 (pos 0-4): should be '.'");
        println!("    got: '{}'", &entry[..5]);
        assert!(entry[..5].chars().all(|c| c == '.'), "s3 should be dots");
        println!("    ✓");

        print_subsection("L2 (pos 5-14): should be 'x' (unpaired loop)");
        println!("    got: '{}'", &entry[5..15]);
        assert!(entry[5..15].chars().all(|c| c == 'x'), "L2 should be x");
        println!("    ✓");

        print_subsection("s4 (pos 15-17): should be '.'");
        println!("    got: '{}'", &entry[15..18]);
        assert!(entry[15..18].chars().all(|c| c == '.'), "s4 should be dots");
        println!("    ✓");

        print_subsection("T0 (pos 18-20): should be '.'");
        println!("    got: '{}'", &entry[18..21]);
        assert!(entry[18..21].chars().all(|c| c == '.'), "T0 should be dots");
        println!("    ✓");

        assert_eq!(entry.len(), 21);
        println!("\n  ✓ Total length correct: 21");
    }

    #[test]
    fn test_nl_path_entry2_pairing() {
        print_separator("TEST: compute_nl_path entry 2 (segments 0-2, pairing)");

        let d = example_dom_length_dict();
        let nl = compute_nl_path(EXAMPLE_ACFP, EXAMPLE_DL_SEQ, &d).unwrap();
        let entry = &nl[2];

        println!("  acfp entry: '.()'");
        println!("    '.' → segment 0 unpaired");
        println!("    '(' → segment 1 L-domain opens");
        println!("    ')' → segment 2 L-domain closes");
        println!("  structure: {}", entry);
        println!("  length:    {} (expected 65)", entry.len());
        println!();
        println!("  Segment 0 (pos  0-20): s3(5) L2(10)x s4(3) T0(3)");
        println!("    s3  (0- 4): '{}'", &entry[0..5]);
        println!("    L2  (5-14): '{}'  ← should be 'x' (unpaired)", &entry[5..15]);
        println!("    s4 (15-17): '{}'", &entry[15..18]);
        println!("    T0 (18-20): '{}'", &entry[18..21]);
        println!();
        println!("  Segment 1 (pos 21-41): s1(5) L1(10)( s2(3) T1(3)");
        println!("    s1 (21-25): '{}'", &entry[21..26]);
        println!("    L1 (26-35): '{}'  ← should be '(' (opens)", &entry[26..36]);
        println!("    s2 (36-38): '{}'", &entry[36..39]);
        println!("    T1 (39-41): '{}'", &entry[39..42]);
        println!();
        println!("  Segment 2 (pos 42-64): s2*(3) L1*(10)) s1*(5) T2(5)");
        println!("    s2* (42-44): '{}'", &entry[42..45]);
        println!("    L1* (45-54): '{}'  ← should be ')' (closes)", &entry[45..55]);
        println!("    s1* (55-59): '{}'", &entry[55..60]);
        println!("    T2  (60-64): '{}'", &entry[60..65]);

        assert_eq!(entry.len(), 65, "entry 2 length should be 65, got {}", entry.len());
        assert!(entry[5..15].chars().all(|c| c == 'x'),  "seg0 L2 should be x");
        assert!(entry[26..36].chars().all(|c| c == '('), "seg1 L1 should be (");
        assert!(entry[45..55].chars().all(|c| c == ')'), "seg2 L1* should be )");
        println!("\n  ✓ All pairing checks passed");
    }

    #[test]
    fn test_nl_path_mismatched_parens_errors() {
        print_separator("TEST: compute_nl_path rejects mismatched parentheses");

        let d = example_dom_length_dict();
        let bad_acfp = ". .. .(( (())";
        println!("  Bad acfp: '{}'", bad_acfp);

        let result = compute_nl_path(bad_acfp, EXAMPLE_DL_SEQ, &d);
        match &result {
            Err(e) => println!("  ✓ Got expected error: {}", e),
            Ok(_)  => println!("  ✗ Should have errored but didn't"),
        }
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------------------
    // compute_motifs
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compute_motifs_format() {
        print_separator("TEST: compute_motifs output format");

        let d = example_dom_length_dict();
        let nl = compute_nl_path(EXAMPLE_ACFP, EXAMPLE_DL_SEQ, &d).unwrap();
        let motifs = compute_motifs(&nl).unwrap();
        let lines: Vec<&str> = motifs.lines().collect();

        println!("  Full motif string:\n");
        println!("{}", motifs);
        println!("  Total lines: {}", lines.len());

        // now: N motifs × 2 lines each (header + structure), no sequence line
        assert_eq!(lines.len(), 2 * nl.len());

        print_subsection("Headers and structures");
        for i in 0..nl.len() {
            let header    = lines[i * 2];
            let structure = lines[i * 2 + 1];
            println!("    motif_{}: header='{}' structure_len={}", i+1, header, structure.len());
            assert_eq!(header, format!(">motif_{} 0", i + 1));
            assert_eq!(structure, nl[i]);
            println!("      ✓");
        }

        println!("\n  ✓ All format checks passed");
    }

    // ---------------------------------------------------------------------------
    // compute_check_positions
    // ---------------------------------------------------------------------------

    #[test]
    fn test_check_positions_full() {
        print_separator("TEST: compute_check_positions");
        println!("  dl_seq: {}", EXAMPLE_DL_SEQ);

        let d = example_dom_length_dict();
        let cp = compute_check_positions(EXAMPLE_DL_SEQ, &d).unwrap();

        println!("\n  check_positions ({} entries):", cp.len());
        let mut sorted: Vec<(&usize, &Vec<String>)> = cp.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        for (pos, motifs) in &sorted {
            println!("    pos {:3}: {:?}", pos, motifs);
        }

        print_subsection("Checking T0 range (pos 18-20) → motif_1");
        for pos in 18..21 {
            let motifs = cp.get(&pos).unwrap();
            println!("    pos {}: {:?}  ✓", pos, motifs);
            assert_eq!(motifs, &vec!["motif_1".to_string()]);
        }

        print_subsection("Checking T1 range (pos 39-41) → motif_2");
        for pos in 39..42 {
            let motifs = cp.get(&pos).unwrap();
            println!("    pos {}: {:?}  ✓", pos, motifs);
            assert_eq!(motifs, &vec!["motif_2".to_string()]);
        }

        print_subsection("Checking non-T positions are absent");
        for pos in [0, 5, 15, 21, 26] {
            let present = cp.contains_key(&pos);
            println!("    pos {:3}: in check_positions = {}  {}",
                pos, present, if !present { "✓" } else { "✗ (should be absent)" });
            assert!(!present);
        }

        assert_eq!(cp.len(), 16, "T0(3)+T1(3)+T2(5)+T3(5)=16");
        println!("\n  ✓ Total count correct: 16 positions");
    }

    // ---------------------------------------------------------------------------
    // compute_checkpoints
    // ---------------------------------------------------------------------------

    #[test]
    fn test_checkpoints_full() {
        print_separator("TEST: compute_checkpoints");
        println!("  dl_seq:    {}", EXAMPLE_DL_SEQ);
        println!("  threshold: 0.6");

        let d = example_dom_length_dict();
        let ckpts = compute_checkpoints(EXAMPLE_DL_SEQ, &d, 0.6).unwrap();

        println!("\n  checkpoints ({} entries):", ckpts.len());
        let mut sorted: Vec<(&usize, &HashMap<String, f64>)> = ckpts.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        for (pos, motif_map) in &sorted {
            println!("    pos {:3}: {:?}", pos, motif_map);
        }

        print_subsection("Checking last pos of T0 (pos 20) → motif_1: 0.6");
        println!("    got: {:?}", ckpts.get(&20));
        assert!(ckpts.contains_key(&20));
        assert_eq!(ckpts[&20].get("motif_1").copied(), Some(0.6));
        println!("    ✓");

        print_subsection("Checking last pos of T1 (pos 41) → motif_2: 0.6");
        println!("    got: {:?}", ckpts.get(&41));
        assert!(ckpts.contains_key(&41));
        assert_eq!(ckpts[&41].get("motif_2").copied(), Some(0.6));
        println!("    ✓");

        print_subsection("Checking only last T-pos has checkpoint (not earlier T positions)");
        for pos in [18, 19, 39, 40] {
            let present = ckpts.contains_key(&pos);
            println!("    pos {:3}: in checkpoints = {}  {}",
                pos, present, if !present { "✓ (correctly absent)" } else { "✗" });
            assert!(!present);
        }

        assert_eq!(ckpts.len(), 4, "one checkpoint per segment");
        println!("\n  ✓ 4 checkpoints found (one per segment)");
    }

    #[test]
    fn test_checkpoints_threshold_values() {
        print_separator("TEST: compute_checkpoints threshold stored correctly");

        let d = example_dom_length_dict();
        for threshold in [0.0, 0.5, 0.6, 0.8, 1.0] {
            let ckpts = compute_checkpoints(EXAMPLE_DL_SEQ, &d, threshold).unwrap();
            let all_correct = ckpts.values()
                .all(|m| m.values().all(|&v| v == threshold));
            println!("  threshold {:.1}: all values correct = {}  {}",
                threshold, all_correct, if all_correct { "✓" } else { "✗" });
            assert!(all_correct);
        }
    }

    // ---------------------------------------------------------------------------
    // CotransConfig::from_target
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cotrans_config_from_target() {
        print_separator("TEST: CotransConfig::from_target");
        println!("  acfp:      {}", EXAMPLE_ACFP);
        println!("  dl_seq:    {}", EXAMPLE_DL_SEQ);
        println!("  t_ext:     0.02");
        println!("  t_end:     1.0");
        println!("  num_sims:  100");
        println!("  threshold: 0.6");

        let d = example_dom_length_dict();
        let config = CotransConfig::from_target(
            EXAMPLE_ACFP, EXAMPLE_DL_SEQ, d,
            0.02, 1.0, 100, 0.6, 0
        ).unwrap();

        print_subsection("nl_path");
        println!("    entries: {}", config.nl_path.len());
        for (i, s) in config.nl_path.iter().enumerate() {
            println!("    [{}] len={:3}  {}", i, s.len(), s);
        }
        assert_eq!(config.nl_path.len(), 4);
        println!("    ✓");

        print_subsection("checkpoints");
        println!("    entries: {}", config.checkpoints.len());
        let mut sorted: Vec<_> = config.checkpoints.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        for (pos, m) in &sorted {
            println!("    pos {:3}: {:?}", pos, m);
        }
        assert_eq!(config.checkpoints.len(), 4);
        println!("    ✓");

        print_subsection("check_positions");
        println!("    total positions tracked: {}", config.check_positions.len());
        assert_eq!(config.check_positions.len(), 16);
        println!("    ✓");

        print_subsection("motif string (first 3 lines)");
        for line in config.motifs.lines().take(3) {
            println!("    {}", line);
        }
        let full_len = config.nl_path.last().unwrap().len();
        assert!(config.motifs.starts_with(&"A".repeat(full_len)));
        println!("    ✓ starts with correct dummy sequence");

        print_subsection("scalar parameters");
        println!("    t_ext:    {} (expected 0.02)", config.t_ext);
        println!("    t_end:    {} (expected 1.0)",  config.t_end);
        println!("    num_sims: {} (expected 100)",  config.num_sims);
        println!("    threshold:{} (expected 0.6)",  config.threshold);
        assert_eq!(config.t_ext,     0.02);
        assert_eq!(config.t_end,     1.0);
        assert_eq!(config.num_sims,  100);
        assert_eq!(config.threshold, 0.6);
        println!("    ✓");

        println!("\n  ✓ All CotransConfig checks passed");
    }

    #[test]
    fn test_cotrans_config_missing_domain_errors() {
        print_separator("TEST: CotransConfig::from_target rejects missing domain");

        let mut d = example_dom_length_dict();
        d.remove("L1");
        println!("  Removed 'L1' from dom_length_dict");

        let result = CotransConfig::from_target(
            EXAMPLE_ACFP, EXAMPLE_DL_SEQ, d, 0.02, 1.0, 100, 0.6, 0
        );
        match &result {
            Err(e) => println!("  ✓ Got expected error: {}", e),
            Ok(_)  => println!("  ✗ Should have errored but didn't"),
        }
        assert!(result.is_err());
    }
}