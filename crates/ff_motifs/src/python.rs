use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};
use ahash::HashMap;

use crate::timecourse::{Simulator, TimecourseResults, CotransConfig};

// ---------------------------------------------------------------------------
// Helper — convert TimecourseResults to Python list of dicts
// ---------------------------------------------------------------------------

fn to_python_list(py: Python<'_>, results: TimecourseResults) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    let mut timepoints: Vec<usize> = results.motif_match_table.keys().cloned().collect();
    timepoints.sort_unstable();

    for tp in timepoints {
        let counts     = &results.motif_match_table[&tp];
        let motif_dict = PyDict::new_bound(py);
        for (name, &count) in counts {
            motif_dict.set_item(name, count)?;
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("nucleotide", tp)?;
        dict.set_item("motif_counts", motif_dict)?;
        list.append(dict)?;
    }
    Ok(list.to_object(py))
}

// ---------------------------------------------------------------------------
// Python wrapper for CotransConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "CotransConfig")]
pub struct PyCotransConfig {
    pub inner: CotransConfig,
}

#[pymethods]
impl PyCotransConfig {
    #[new]
    #[pyo3(signature = (acfp, dl_seq, dom_length_dict, t_ext, t_end, num_sims, threshold, num_workers=0))]
    fn new(
        acfp:            &str,
        dl_seq:          &str,
        dom_length_dict: HashMap<String, usize>,
        t_ext:           f64,
        t_end:           f64,
        num_sims:        usize,
        threshold:       f64,
        num_workers:     usize,
    ) -> PyResult<Self> {
        let inner = CotransConfig::from_target(
            acfp, dl_seq, dom_length_dict, t_ext, t_end, num_sims, threshold, num_workers,
        ).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    // expose precomputed fields as read-only Python properties

    #[getter]
    fn nl_path(&self) -> Vec<String> {
        self.inner.nl_path.clone()
    }

    #[getter]
    fn motifs(&self) -> &str {
        &self.inner.motifs
    }

    #[getter]
    fn dl_seq(&self) -> &str {
        &self.inner.dl_seq
    }

    #[getter]
    fn t_ext(&self) -> f64 {
        self.inner.t_ext
    }

    #[getter]
    fn t_end(&self) -> f64 {
        self.inner.t_end
    }

    #[getter]
    fn num_sims(&self) -> usize {
        self.inner.num_sims
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    #[getter]
    fn check_positions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (pos, motifs) in &self.inner.check_positions {
            dict.set_item(pos, motifs.clone())?;
        }
        Ok(dict.to_object(py))
    }

    #[getter]
    fn checkpoints(&self, py: Python<'_>) -> PyResult<PyObject> {
        let outer = PyDict::new_bound(py);
        for (pos, motif_map) in &self.inner.checkpoints {
            let inner = PyDict::new_bound(py);
            for (name, &threshold) in motif_map {
                inner.set_item(name, threshold)?;
            }
            outer.set_item(pos, inner)?;
        }
        Ok(outer.to_object(py))
    }

    #[getter]
    fn num_workers (&self) -> usize {
        self.inner.num_workers 
    }

    fn __repr__(&self) -> String {
        format!(
            "CotransConfig(dl_seq='{}', t_ext={}, t_end={}, num_sims={}, threshold={}, segments={})",
            self.inner.dl_seq,
            self.inner.t_ext,
            self.inner.t_end,
            self.inner.num_sims,
            self.inner.threshold,
            self.inner.nl_path.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Python wrapper for Simulator
// ---------------------------------------------------------------------------

#[pyclass]
pub struct MotifCheckSimulator {
    inner: Simulator,
}

#[pymethods]
impl MotifCheckSimulator {
    #[new]
    #[pyo3(signature = (params="rna_default", celsius=37.0, k0=1e5, k3ws=0.0, k4ws=0.0))]
    fn new(
        params:  &str,
        celsius: f64,
        k0:      f64,
        k3ws:    f64,
        k4ws:    f64,
    ) -> PyResult<Self> {
        let inner = Simulator::new(params, celsius, k0, k3ws, k4ws)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    // -----------------------------------------------------------------------
    // simulate_timecourse
    // -----------------------------------------------------------------------

    #[pyo3(signature = (sequence, motifs, check_positions, start=None, t_ext=None, t_end=1.0, num_sims=100, num_workers=0))]
    fn simulate_timecourse(
        &self,
        py:              Python<'_>,
        sequence:        &str,
        motifs:          &str,
        check_positions: HashMap<usize, Vec<String>>,
        start:           Option<&str>,
        t_ext:           Option<f64>,
        t_end:           f64,
        num_sims:        usize,
        num_workers:     usize,
    ) -> PyResult<PyObject> {
        let results = py.allow_threads(|| {
            self.inner.simulate_timecourse(
                sequence, motifs, check_positions,
                start, t_ext, t_end, num_sims, num_workers,
            )
        }).map_err(PyValueError::new_err)?;

        to_python_list(py, results)
    }

    // -----------------------------------------------------------------------
    // simulate_timecourse_checkpoints
    // -----------------------------------------------------------------------

    #[pyo3(signature = (sequence, motifs, check_positions, checkpoints, start=None, t_ext=None, t_end=1.0, num_sims=100, num_workers=0))]
    fn simulate_timecourse_checkpoints(
        &self,
        py:              Python<'_>,
        sequence:        &str,
        motifs:          &str,
        check_positions: HashMap<usize, Vec<String>>,
        checkpoints:     HashMap<usize, HashMap<String, f64>>,
        start:           Option<&str>,
        t_ext:           Option<f64>,
        t_end:           f64,
        num_sims:        usize,
        num_workers:     usize,
    ) -> PyResult<PyObject> {
        let results = py.allow_threads(|| {
            self.inner.simulate_timecourse_checkpoints(
                sequence, motifs, check_positions, checkpoints,
                start, t_ext, t_end, num_sims, num_workers,
            )
        }).map_err(PyValueError::new_err)?;

        to_python_list(py, results)
    }

    // -----------------------------------------------------------------------
    // cotrans_score — takes a PyCotransConfig
    // -----------------------------------------------------------------------

    #[pyo3(signature = (sequence, config))]
    fn cotrans_score(
        &self,
        py:       Python<'_>,
        sequence: &str,
        config:   &PyCotransConfig,
    ) -> PyResult<f64> {
        py.allow_threads(|| {
            self.inner.cotrans_score(sequence, &config.inner)
        }).map_err(PyValueError::new_err)
    }

    // -----------------------------------------------------------------------
    // compute_nl_path — exposed as a standalone method for inspection
    // -----------------------------------------------------------------------

    #[pyo3(signature = (acfp, dl_seq, dom_length_dict))]
    fn compute_nl_path(
        &self,
        acfp:            &str,
        dl_seq:          &str,
        dom_length_dict: HashMap<String, usize>,
    ) -> PyResult<Vec<String>> {
        crate::timecourse::compute_nl_path(acfp, dl_seq, &dom_length_dict)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!("MotifCheckSimulator(is_rna={})", self.inner.is_rna)
    }
}