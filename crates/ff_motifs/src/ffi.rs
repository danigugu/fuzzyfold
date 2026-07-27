use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use ahash::HashMap;
use rayon::prelude::*;

use crate::timecourse::{Simulator, CotransConfig};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

pub struct SimHandle {
    inner: Simulator,
}

pub struct ConfigHandle {
    inner: CotransConfig,
}

// ---------------------------------------------------------------------------
// SimHandle — create / destroy
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn sim_create(
    params:  *const c_char,
    celsius: f64,
    k0:      f64,
    k3ws:    f64,
    k4ws:    f64,
) -> *mut SimHandle {
    let params_str = unsafe {
        match CStr::from_ptr(params).to_str() {
            Ok(s)  => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    match Simulator::new(params_str, celsius, k0, k3ws, k4ws) {
        Ok(inner) => Box::into_raw(Box::new(SimHandle { inner })),
        Err(_)    => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn sim_destroy(handle: *mut SimHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

// ---------------------------------------------------------------------------
// ConfigHandle — create / destroy
// ---------------------------------------------------------------------------

/// Create a CotransConfig from the target structure.
/// dom_length_keys and dom_length_values are parallel arrays of length n_domains.
/// num_workers: 0 = Rayon default (all cores), 1 = serial (safe for OpenMP), N = N threads
#[no_mangle]
pub extern "C" fn config_create(
    acfp:              *const c_char,
    dl_seq:            *const c_char,
    dom_length_keys:   *const *const c_char,
    dom_length_values: *const usize,
    n_domains:         usize,
    t_ext:             f64,
    t_end:             f64,
    num_sims:          usize,
    threshold:         f64,
    num_workers:       usize,
) -> *mut ConfigHandle {
    let acfp_str = unsafe {
        match CStr::from_ptr(acfp).to_str() {
            Ok(s)  => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let dl_seq_str = unsafe {
        match CStr::from_ptr(dl_seq).to_str() {
            Ok(s)  => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let mut dom_length_dict: HashMap<String, usize> = HashMap::default();
    for i in 0..n_domains {
        let key = unsafe {
            match CStr::from_ptr(*dom_length_keys.add(i)).to_str() {
                Ok(s)  => s.to_owned(),
                Err(_) => return std::ptr::null_mut(),
            }
        };
        let value = unsafe { *dom_length_values.add(i) };
        dom_length_dict.insert(key, value);
    }

    match CotransConfig::from_target(
        acfp_str, dl_seq_str, dom_length_dict,
        t_ext, t_end, num_sims, threshold, num_workers,
    ) {
        Ok(inner) => Box::into_raw(Box::new(ConfigHandle { inner })),
        Err(e)    => {
            eprintln!("config_create error: {}", e);
            std::ptr::null_mut()
        },
    }
}

#[no_mangle]
pub extern "C" fn config_destroy(handle: *mut ConfigHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

// ---------------------------------------------------------------------------
// config_nl_path_len / config_nl_path_entry — inspect nl_path from C++
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn config_nl_path_len(handle: *mut ConfigHandle) -> usize {
    if handle.is_null() { return 0; }
    unsafe { (*handle).inner.nl_path.len() }
}

/// Copies nl_path[index] into out_buf (caller-allocated, size >= buf_len).
/// Returns bytes written (excluding null terminator), or 0 on error.
#[no_mangle]
pub extern "C" fn config_nl_path_entry(
    handle:  *mut ConfigHandle,
    index:   usize,
    out_buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if handle.is_null() || out_buf.is_null() || buf_len == 0 { return 0; }
    let config = unsafe { &(*handle).inner };
    if index >= config.nl_path.len() { return 0; }

    let entry = &config.nl_path[index];
    let bytes = entry.as_bytes();
    let n     = bytes.len().min(buf_len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out_buf, n);
        *out_buf.add(n) = 0;
    }
    n
}

// ---------------------------------------------------------------------------
// Result buffer — shared output format for timecourse functions
// Each timepoint writes one entry: (nucleotide_position, motif_name, count)
// motif_name is written into a caller-allocated flat char buffer
// ---------------------------------------------------------------------------

/// Full timecourse — no checkpoints, runs to t_end.
/// Writes per-timepoint total motif counts (excluding Unassigned) into flat arrays.
/// Returns number of timepoints written, or -1 on error.
/// out_nucleotides and out_counts must be pre-allocated with capacity >= max_timepoints.
#[no_mangle]
pub extern "C" fn sim_timecourse(
    sim_handle:      *mut SimHandle,
    config_handle:   *mut ConfigHandle,
    sequence:        *const c_char,
    out_nucleotides: *mut usize,
    out_counts:      *mut usize,
    max_timepoints:  usize,
) -> c_int {
    if sim_handle.is_null() || config_handle.is_null() || sequence.is_null() {
        return -1;
    }
    let seq = unsafe {
        match CStr::from_ptr(sequence).to_str() {
            Ok(s)  => s.to_owned(),
            Err(_) => return -1,
        }
    };

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    let results = match sim.simulate_timecourse(
        &seq,
        &config.motifs,
        config.check_positions.clone(),
        None,
        Some(config.t_ext),
        config.t_end,
        config.num_sims,
        config.num_workers,
    ) {
        Ok(r)  => r,
        Err(e) => {
            eprintln!("sim_timecourse error: {}", e);
            return -1;
        },
    };

    write_timecourse_results(&results, out_nucleotides, out_counts, max_timepoints)
}

/// Checkpoint timecourse — early-exits if occupancy falls below threshold.
/// Same output format as sim_timecourse.
/// Returns number of timepoints written (may be less than total if checkpoint failed),
/// or -1 on error.
#[no_mangle]
pub extern "C" fn sim_timecourse_checkpoints(
    sim_handle:      *mut SimHandle,
    config_handle:   *mut ConfigHandle,
    sequence:        *const c_char,
    out_nucleotides: *mut usize,
    out_counts:      *mut usize,
    max_timepoints:  usize,
) -> c_int {
    if sim_handle.is_null() || config_handle.is_null() || sequence.is_null() {
        return -1;
    }
    let seq = unsafe {
        match CStr::from_ptr(sequence).to_str() {
            Ok(s)  => s.to_owned(),
            Err(_) => return -1,
        }
    };

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    let results = match sim.simulate_timecourse_checkpoints(
        &seq,
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
    ) {
        Ok(r)  => r,
        Err(e) => {
            eprintln!("sim_timecourse_checkpoints error: {}", e);
            return -1;
        },
    };

    write_timecourse_results(&results, out_nucleotides, out_counts, max_timepoints)
}

/// Set per-checkpoint weights for a config.
/// weights must be a pointer to n_weights doubles, one per checkpoint segment (in order).
/// If fewer weights than checkpoints, remaining checkpoints use weight 1.0.
/// Pass null or n_weights=0 to reset to equal weights.
#[no_mangle]
pub extern "C" fn config_set_weights(
    handle:    *mut ConfigHandle,
    weights:   *const f64,
    n_weights: usize,
) {
    if handle.is_null() { return; }
    let config = unsafe { &mut (*handle).inner };
    if weights.is_null() || n_weights == 0 {
        config.weights = Vec::new();
    } else {
        config.weights = unsafe {
            std::slice::from_raw_parts(weights, n_weights).to_vec()
        };
    }
}

/// Set the scoring objective for a config.
/// Valid values: "occupancy" (default) or "distance".
#[no_mangle]
pub extern "C" fn config_set_objective(
    handle:    *mut ConfigHandle,
    objective: *const c_char,
) {
    if handle.is_null() || objective.is_null() { return; }
    let obj_str = unsafe {
        match CStr::from_ptr(objective).to_str() {
            Ok(s)  => s.to_owned(),
            Err(_) => return,
        }
    };
    unsafe { (*handle).inner.objective = obj_str; }
}

/// Scalar objective for gradient descent.
/// Dispatches to occupancy or distance scoring based on config.objective.
/// Returns score in [0, 1] (lower is better), or -1.0 on error.
#[no_mangle]
pub extern "C" fn sim_score(
    sim_handle:    *mut SimHandle,
    config_handle: *mut ConfigHandle,
    sequence:      *const c_char,
) -> f64 {
    if sim_handle.is_null() || config_handle.is_null() || sequence.is_null() {
        return -1.0;
    }
    let seq = unsafe {
        match CStr::from_ptr(sequence).to_str() {
            Ok(s)  => s.to_owned(),
            Err(_) => return -1.0,
        }
    };

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    match sim.cotrans_score(&seq, config) {
        Ok(score) => score,
        Err(e)    => {
            eprintln!("sim_score error: {}", e);
            -1.0
        },
    }
}

/// Objective + per-checkpoint scores for gradient descent with experience replay.
///
/// Writes into out_scores[0..n+1]:
///   out_scores[0]   = total weighted score (same as sim_score)
///   out_scores[1..] = per-checkpoint normalized distances/scores, one per checkpoint
///
/// Returns n+1 (total entries written), or -1 on error.
/// max_scores must be >= n_checkpoints + 1; values beyond n+1 are not written.
#[no_mangle]
pub extern "C" fn sim_score_vec(
    sim_handle:    *mut SimHandle,
    config_handle: *mut ConfigHandle,
    sequence:      *const c_char,
    out_scores:    *mut f64,
    max_scores:    usize,
) -> c_int {
    if sim_handle.is_null() || config_handle.is_null() || sequence.is_null() || out_scores.is_null() {
        return -1;
    }
    let seq = unsafe {
        match CStr::from_ptr(sequence).to_str() {
            Ok(s)  => s.to_owned(),
            Err(_) => return -1,
        }
    };

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    match sim.cotrans_score_vec(&seq, config) {
        Ok((total, per_cp)) => {
            let n_written = (per_cp.len() + 1).min(max_scores);
            if n_written == 0 { return 0; }
            unsafe { *out_scores = total; }
            for i in 0..n_written.saturating_sub(1) {
                if i < per_cp.len() {
                    unsafe { *out_scores.add(i + 1) = per_cp[i]; }
                }
            }
            n_written as c_int
        }
        Err(e) => {
            eprintln!("sim_score_vec error: {}", e);
            -1
        }
    }
}

/// Batch scalar scoring — evaluates n_seqs sequences in parallel using Rayon.
///
/// sequences: array of n_seqs C-string pointers (each a null-terminated RNA sequence).
/// out_scores: caller-allocated array of n_seqs doubles; receives the score for each
///             sequence (same semantics as sim_score: [0,1] lower-is-better, -1.0 on error).
///
/// Returns n_seqs on success, or -1 if either handle is null.
/// Individual sequences that fail to score write -1.0 into their out_scores slot.
#[no_mangle]
pub extern "C" fn sim_score_batch(
    sim_handle:    *mut SimHandle,
    config_handle: *mut ConfigHandle,
    sequences:     *const *const c_char,
    n_seqs:        usize,
    out_scores:    *mut f64,
) -> c_int {
    if sim_handle.is_null() || config_handle.is_null() || sequences.is_null() || out_scores.is_null() {
        return -1;
    }

    // Collect sequence strings upfront (single unsafe block, before spawning threads).
    let seqs: Vec<String> = (0..n_seqs).map(|i| {
        let ptr = unsafe { *sequences.add(i) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr).to_str().unwrap_or("").to_owned() }
        }
    }).collect();

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    // Score all sequences in parallel; collect into a Vec so we stay out of raw pointers.
    let scores: Vec<f64> = seqs.par_iter()
        .map(|seq| {
            if seq.is_empty() {
                return -1.0;
            }
            match sim.cotrans_score(seq, config) {
                Ok(score) => score,
                Err(e)    => {
                    eprintln!("sim_score_batch error: {}", e);
                    -1.0
                }
            }
        })
        .collect();

    // Write results back to the caller's buffer.
    for (i, &s) in scores.iter().enumerate() {
        unsafe { *out_scores.add(i) = s; }
    }

    n_seqs as c_int
}

/// Compute both distance and occupancy scores from a single simulation pass.
///
/// Uses distance-based early exit. Writes dist_score and occ_score to the
/// respective output pointers (each in [0, 1], lower is better).
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn sim_score_both(
    sim_handle:    *mut SimHandle,
    config_handle: *mut ConfigHandle,
    sequence:      *const c_char,
    out_dist:      *mut f64,
    out_occ:       *mut f64,
) -> c_int {
    if sim_handle.is_null() || config_handle.is_null() || sequence.is_null()
        || out_dist.is_null() || out_occ.is_null()
    {
        return -1;
    }
    let seq = unsafe {
        match CStr::from_ptr(sequence).to_str() {
            Ok(s)  => s.to_owned(),
            Err(_) => return -1,
        }
    };

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    match sim.cotrans_score_both(&seq, config) {
        Ok((dist, occ)) => {
            unsafe { *out_dist = dist; *out_occ = occ; }
            0
        }
        Err(e) => {
            eprintln!("sim_score_both error: {}", e);
            -1
        }
    }
}

/// Batch combined scoring — evaluates n_seqs sequences in parallel using Rayon.
///
/// sequences: array of n_seqs C-string pointers.
/// out_dist, out_occ: caller-allocated arrays of n_seqs doubles.
/// Each entry receives [0, 1] lower-is-better scores, or -1.0 on per-sequence error.
/// Returns n_seqs on success, -1 if handles are null.
#[no_mangle]
pub extern "C" fn sim_score_both_batch(
    sim_handle:    *mut SimHandle,
    config_handle: *mut ConfigHandle,
    sequences:     *const *const c_char,
    n_seqs:        usize,
    out_dist:      *mut f64,
    out_occ:       *mut f64,
) -> c_int {
    if sim_handle.is_null() || config_handle.is_null() || sequences.is_null()
        || out_dist.is_null() || out_occ.is_null()
    {
        return -1;
    }

    let seqs: Vec<String> = (0..n_seqs).map(|i| {
        let ptr = unsafe { *sequences.add(i) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr).to_str().unwrap_or("").to_owned() }
        }
    }).collect();

    let sim    = unsafe { &(*sim_handle).inner };
    let config = unsafe { &(*config_handle).inner };

    let scores: Vec<(f64, f64)> = seqs.par_iter()
        .map(|seq| {
            if seq.is_empty() { return (-1.0, -1.0); }
            match sim.cotrans_score_both(seq, config) {
                Ok((d, o)) => (d, o),
                Err(e) => {
                    eprintln!("sim_score_both_batch error: {}", e);
                    (-1.0, -1.0)
                }
            }
        })
        .collect();

    for (i, &(d, o)) in scores.iter().enumerate() {
        unsafe {
            *out_dist.add(i) = d;
            *out_occ.add(i)  = o;
        }
    }

    n_seqs as c_int
}

// ---------------------------------------------------------------------------
// Internal helper — shared by sim_timecourse and sim_timecourse_checkpoints
// ---------------------------------------------------------------------------

fn write_timecourse_results(
    results:         &crate::timecourse::TimecourseResults,
    out_nucleotides: *mut usize,
    out_counts:      *mut usize,
    max_timepoints:  usize,
) -> c_int {
    let mut timepoints: Vec<usize> =
        results.motif_match_table.keys().cloned().collect();
    timepoints.sort_unstable();

    let n = timepoints.len().min(max_timepoints);
    for (i, &tp) in timepoints[..n].iter().enumerate() {
        // sum only named motif hits, exclude Unassigned
        let total: usize = results.motif_match_table[&tp]
            .iter()
            .filter(|(name, _)| *name != "Unassigned")
            .map(|(_, &c)| c)
            .sum();
        unsafe {
            *out_nucleotides.add(i) = tp;
            *out_counts.add(i)      = total;
        }
    }
    n as c_int
}