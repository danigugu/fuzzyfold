use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use ahash::HashMap;

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

/// Scalar objective for gradient descent.
/// score = pathlength - sum(avg T-domain occupancies)
/// range: [0, pathlength], lower is better.
/// Returns -1.0 on error.
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