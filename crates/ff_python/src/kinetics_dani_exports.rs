//use ff_kinetics::Motif;
use ff_kinetics::MotifRegistry;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};

use core::num;
//use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use ff_structure::DotBracketVec;
use ff_structure::PairTable;
use ff_energy::NucleotideVec;
use ff_energy::ViennaRNA;
use ff_kinetics::SSA;
use ff_kinetics::shift_policy;
use ff_kinetics::Arrhenius;
use ff_kinetics::Walker;
use ff_kinetics::LoopNeighbors;
use ff_kinetics::timeline_motif::Timepoint;
use ff_kinetics::timeline_motif::Timeline;
use ff_energy::parameters::RNA_EXTENDED;
use ff_energy::parameters::RNA_TURNER_2004;
use ff_energy::parameters::DNA_MATHEWS_2004;

use rayon::prelude::*;

//TODO: support shifts, rename to arrhenius

#[pyclass]
pub struct Simulator {
    energy_model: Arc<ViennaRNA>,
    rate_model: Arrhenius,
    is_rna: bool,
}


#[pymethods]
impl Simulator {
    #[new]
    #[pyo3(signature = (
        params = "rna_default",
        celsius=37.0,
        k0=1e5,
        k3ws=0.0,
        k4ws=0.0,
    ))]
    fn new(
        params: &str,
        celsius: f64,
        k0: f64,
        k3ws: f64,
        k4ws: f64,
    ) -> PyResult<Self> {
        let mut is_rna = true;
        let thermo = match params {
            "rna_default" => &RNA_TURNER_2004,
            "rna_extended" => &RNA_EXTENDED,
            "dna" => {
                is_rna = false;
                &DNA_MATHEWS_2004
            },
            _ => {
                return Err(PyValueError::new_err(
                    format!(
                        "Unknown parameter set '{}'. \
                         Valid options are: 'rna_default', 'rna_extended', 'dna'.",
                        params
                    )
                ));
            }
        };

        if k0 < 0.0 || k3ws < 0.0 || k4ws < 0.0 {
            return Err(PyValueError::new_err(
                "Rate constants must be non-negative",
            ));
        }

        let energy_model = ViennaRNA::from_thermo_params(thermo, celsius);
        let rate_model = Arrhenius::new(
            celsius,
            k0,
            Some(k3ws),
            Some(k4ws),
        );

        Ok(Self {
            energy_model: Arc::new(energy_model),
            rate_model,
            is_rna,
        })
    }

    #[pyo3(signature = (
            sequence,
            start=None,
            t_ext=None,
            t_end=1.0,
    ))]
    fn simulate(
        &self,
        sequence: &str,
        start: Option<&str>,
        t_ext: Option<f64>,
        t_end: f64,
    ) -> PyResult<SimulationIterator> {

        let sequence = match self.is_rna {
            true => NucleotideVec::try_from_rna(sequence)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            false => NucleotideVec::try_from_dna(sequence)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };

        let start_db = match start {
            Some(s) => DotBracketVec::try_from(s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => DotBracketVec::try_from(".")
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };

        if start_db.len() < sequence.len() && t_ext.is_none() {
            return Err(PyValueError::new_err(
                    "t_ext must be provided when start is shorter than sequence",
            ));
        }

        let times = if let Some(dt) = t_ext {
            let mut v = vec![dt; sequence.len() - start_db.len()];
            v.push(t_end);
            v
        } else {
            vec![t_end]
        };

        let start_pt = PairTable::try_from(&start_db)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        match (self.rate_model.k3ws().is_some(), self.rate_model.k4ws().is_some()) {
            (false, false) => build_iterator(
                sequence,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                shift_policy::NoShift,
                SSAKind::NoShift,
            ),

            (true, false) => build_iterator(
                sequence,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                shift_policy::ThreeWayOnly,
                SSAKind::ThreeWayOnly,
            ),

            (false, true) => build_iterator(
                sequence,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                shift_policy::FourWayOnly,
                SSAKind::FourWayOnly,
            ),

            (true, true) => build_iterator(
                sequence,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                shift_policy::ThreeAndFour,
                SSAKind::ThreeAndFour,
            ),
        }
   }

   #[pyo3(signature = (
            sequence,
            motifs_file,
            start=None,
            t_ext=None,
            t_end=1.0,
    ))]
   fn simulate_to_target_motifs(
        &self,
        sequence: &str,
        motifs_file: &str,
        start: Option<&str>,
        t_ext: Option<f64>,
        t_end: f64,
    ) -> PyResult<SimulationIteratorMotifMatch> {

        let sequence_vec = match self.is_rna {
            true => NucleotideVec::try_from_rna(sequence)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            false => NucleotideVec::try_from_dna(sequence)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };

        let sequence_arc = Arc::new(sequence_vec);

        let start_db = match start {
            Some(s) => DotBracketVec::try_from(s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => DotBracketVec::try_from(".")
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };

        if start_db.len() < sequence_arc.len() && t_ext.is_none() {
            return Err(PyValueError::new_err(
                    "t_ext must be provided when start is shorter than sequence",
            ));
        }

        let file_path = PathBuf::from(&motifs_file);
        let mut motif_reg = MotifRegistry::from((Arc::clone(&sequence_arc), Arc::clone(&self.energy_model)));
        motif_reg.insert_from_file(&file_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let times = if let Some(dt) = t_ext {
            let mut v = vec![dt; sequence_arc.len() - start_db.len()];
            v.push(t_end);
            v
        } else {
            vec![t_end]
        };

        let start_pt = PairTable::try_from(&start_db)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Clone the inner NucleotideVec to avoid try_unwrap panic due to motif_reg holding a reference
        let seq_clone = (*sequence_arc).clone();

        match (self.rate_model.k3ws().is_some(), self.rate_model.k4ws().is_some()) {
            (false, false) => build_iterator_motif_match(
                seq_clone,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                motif_reg,
                shift_policy::NoShift,
                SSAKind::NoShift,
            ),

            (true, false) => build_iterator_motif_match(
                seq_clone,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                motif_reg,
                shift_policy::ThreeWayOnly,
                SSAKind::ThreeWayOnly,
            ),

            (false, true) => build_iterator_motif_match(
                seq_clone,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                motif_reg,
                shift_policy::FourWayOnly,
                SSAKind::FourWayOnly,
            ),

            (true, true) => build_iterator_motif_match(
                seq_clone,
                &start_pt,
                Arc::clone(&self.energy_model),
                self.rate_model,
                times,
                motif_reg,
                shift_policy::ThreeAndFour,
                SSAKind::ThreeAndFour,
            ),
        }
   }

   #[pyo3(signature = (
        sequence,
        motifs_file,
        start=None,
        t_ext=None,
        t_end=1.0,
        num_sims=100
    ))]
    fn simulate_ensemble_to_target_motifs(
        &self,
        sequence: &str,
        motifs_file: &str,
        start: Option<&str>,
        t_ext: Option<f64>,
        t_end: f64,
        num_sims: usize,
    ) -> PyResult<PyObject> {  // return PyObject instead of &PyList
        Python::with_gil(|py| {  // acquire GIL
            let sequence_vec = match self.is_rna {
                true => NucleotideVec::try_from_rna(sequence)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                false => NucleotideVec::try_from_dna(sequence)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            };

            let sequence_arc = Arc::new(sequence_vec);

            let start_db = match start {
                Some(s) => DotBracketVec::try_from(s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                None => DotBracketVec::try_from(".")
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            };

            if start_db.len() < sequence_arc.len() && t_ext.is_none() {
                return Err(PyValueError::new_err(
                    "t_ext must be provided when start is shorter than sequence",
                ));
            }

            let file_path = PathBuf::from(&motifs_file);
            let mut motif_reg = MotifRegistry::from((Arc::clone(&sequence_arc), Arc::clone(&self.energy_model)));
            motif_reg.insert_from_file(&file_path)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let times = if let Some(dt) = t_ext {
                let mut v = vec![dt; sequence_arc.len() - start_db.len()];
                v.push(t_end);
                v
            } else {
                vec![t_end]
            };

            let start_pt = PairTable::try_from(&start_db)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let seq_clone = (*sequence_arc).clone();
            let master_times: Vec<f64> = times
                .iter()
                .scan(0.0, |acc, &dt| {
                    *acc += dt;
                    Some(*acc)
                })
                .collect();
            let motif_reg = Arc::new(motif_reg);

            let init_struc = match start {
                Some(s) => DotBracketVec::try_from(s).unwrap(),
                None => DotBracketVec::try_from(".").unwrap(),
            };

            if !motif_reg
                .classify(&init_struc)
                .iter()
                .all(|&x| x == 0) {
                    let mut master = Timeline::new(master_times, Arc::clone(&motif_reg));
                    for i in 0..num_sims {
                        master.assign_structure(0, &init_struc);
                    }
                    return convert_timeline_python_friendly(py, master).map(|list| list.to_object(py))
                }

            let timelines: Vec<_> = match (self.rate_model.k3ws().is_some(), self.rate_model.k4ws().is_some()) {
                (false, false) => build_par_iterator_motif_match(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::NoShift,
                    SSAKind::NoShift,
                ).unwrap(),
                (true, false) => build_par_iterator_motif_match(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::ThreeWayOnly,
                    SSAKind::ThreeWayOnly,
                ).unwrap(),
                (false, true) => build_par_iterator_motif_match(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::FourWayOnly,
                    SSAKind::FourWayOnly,
                ).unwrap(),
                (true, true) => build_par_iterator_motif_match(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::ThreeAndFour,
                    SSAKind::ThreeAndFour,
                ).unwrap(),
            };

            let mut master = Timeline::new(master_times, Arc::clone(&motif_reg));
            for timeline in timelines {
                master.merge(timeline);
            }

            master.points.insert(0, Timepoint::new(0.0));
            let init_struc = match start {
                Some(s) => DotBracketVec::try_from(s).unwrap(),
                None => DotBracketVec::try_from(".").unwrap(),
            };
            for i in 0..num_sims {
                master.assign_structure(0, &init_struc);
            }
            return convert_timeline_python_friendly(py, master).map(|list| list.to_object(py))
        })
    }
}

fn convert_timeline_python_friendly(
    py: Python,
    timeline: Timeline<ViennaRNA>
) -> PyResult<&PyList> {
    let py_list = PyList::empty(py);

    for tp in timeline.points {
        let dict = PyDict::new(py);
        dict.set_item("time", tp.time)?;
        dict.set_item("counter", tp.counter)?;

        let ensemble = PyDict::new(py);
        for (k, v) in tp.ensemble.iter() {
            ensemble.set_item(k, v)?;
        }
        dict.set_item("ensemble", ensemble)?;
        py_list.append(dict)?;
    }

    Ok(py_list)
}

fn build_iterator<P>(
    seq: NucleotideVec,
    start_pt: &PairTable,
    energy_model: Arc<ViennaRNA>,
    rate_model: Arrhenius,
    times: Vec<f64>,
    policy: P,
    wrap: fn(SSA<LoopNeighbors<ViennaRNA, P>, Arrhenius>) -> SSAKind,
) -> PyResult<SimulationIterator>
where
    P: shift_policy::ShiftPolicy,
{
    let walker = LoopNeighbors::try_from((
        seq,
        start_pt,
        energy_model,
        policy,
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let ssa = wrap(SSA::from((walker, rate_model)));

    Ok(SimulationIterator {
        ssa,
        rng: SmallRng::from_os_rng(),
        times,
        elapsed: 0.0,
        finished: false,
    })
}


fn build_iterator_motif_match<P>(
    seq: NucleotideVec,
    start_pt: &PairTable,
    energy_model: Arc<ViennaRNA>,
    rate_model: Arrhenius,
    times: Vec<f64>,
    motif_registry: MotifRegistry<ViennaRNA>,
    policy: P,
    wrap: fn(SSA<LoopNeighbors<ViennaRNA, P>, Arrhenius>) -> SSAKind,
) -> PyResult<SimulationIteratorMotifMatch>
where
    P: shift_policy::ShiftPolicy,
{
    let walker = LoopNeighbors::try_from((
        seq,
        start_pt,
        energy_model,
        policy,
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let ssa = wrap(SSA::from((walker, rate_model)));


    Ok(SimulationIteratorMotifMatch {
        ssa,
        rng: SmallRng::from_os_rng(),
        times,
        elapsed: 0.0,
        finished: false,
        motif_registry: motif_registry,
    })
}

fn build_par_iterator_motif_match<P>(
    seq: NucleotideVec,
    start_pt: &PairTable,
    energy_model: Arc<ViennaRNA>,
    rate_model: Arrhenius,
    times: Vec<f64>,
    motif_registry: Arc<MotifRegistry<ViennaRNA>>,
    num_sims: usize,
    policy: P,
    wrap: fn(SSA<LoopNeighbors<ViennaRNA, P>, Arrhenius>) -> SSAKind,
) -> PyResult<Vec<Timeline<ViennaRNA>>>
where
    P: shift_policy::ShiftPolicy + Send + Sync + Clone + 'static,
{
    let walker = LoopNeighbors::try_from((
        seq,
        start_pt,
        energy_model,
        policy,
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let master_times: Vec<f64> = times
        .iter()
        .scan(0.0, |acc, &dt| {
            *acc += dt;
            Some(*acc)
        })
        .collect();

    Ok((0..num_sims)
        .into_par_iter()
        .map_init(
            move || (times.clone(), motif_registry.clone(), rate_model.clone()),
            move |(times, motif_registry, rate_model), _| {
                // let walker = LoopNeighbors::try_from((seq.clone(), start_pt, energy_model.clone(), policy.clone())).unwrap();
                let mut sim_res = SimulationEnsembleIteratorMotifMatch {
                    ssa: wrap(SSA::from((walker.clone(), rate_model.clone()))),
                    rng: SmallRng::from_os_rng(),
                    times: times.clone(),
                    elapsed: 0.0,
                    finished: false,
                    motif_registry: motif_registry.clone(),
                    timeline: Timeline::new(master_times.clone(), motif_registry.clone()),
                    t_idx: 0,
                };

                while let Some(_) = sim_res.next() {}

                sim_res.timeline
            }
        )
        .collect::<Vec<_>>())

}

enum SSAKind {
    NoShift(SSA<LoopNeighbors<ViennaRNA, shift_policy::NoShift>, Arrhenius>),
    ThreeWayOnly(SSA<LoopNeighbors<ViennaRNA, shift_policy::ThreeWayOnly>, Arrhenius>),
    FourWayOnly(SSA<LoopNeighbors<ViennaRNA, shift_policy::FourWayOnly>, Arrhenius>),
    ThreeAndFour(SSA<LoopNeighbors<ViennaRNA, shift_policy::ThreeAndFour>, Arrhenius>),
}

#[pyclass]
pub struct SimulationIterator {
    ssa: SSAKind,
    rng: SmallRng,
    times: Vec<f64>,
    elapsed: f64,
    finished: bool,
}

#[pymethods]
impl SimulationIterator {

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<Self>
    ) -> Option<(String, i32, f64, f64, f64)> {

        let this: &mut Self = &mut slf;

        if this.finished {
            return None;
        }

        let mut produced: Option<(String, i32, f64, f64, f64)> = None;

        let rng = &mut this.rng;
        let mut mytinc = 0.0;
        let mut first_pass = true;

        macro_rules! dispatch_ssa {
            ($ssa:expr) => {{
                $ssa.co_simulate(
                    rng,
                    &this.times,
                    |t, tinc, flux, w| {
                        if first_pass {
                            mytinc = tinc.min(this.times[0]);

                            produced = Some((
                                    w.to_string(),
                                    w.current_energy(),
                                    this.elapsed + t,
                                    mytinc,
                                    flux,
                            ));

                            this.elapsed += mytinc;
                            first_pass = false;
                            // advance the simulator to update the structure.
                            true
                        } else {
                            false
                        }
                    },
                    );
            }};
        }

        match &mut this.ssa {
            SSAKind::NoShift(ssa) => dispatch_ssa!(ssa),
            SSAKind::ThreeWayOnly(ssa) => dispatch_ssa!(ssa),
            SSAKind::FourWayOnly(ssa) => dispatch_ssa!(ssa),
            SSAKind::ThreeAndFour(ssa) => dispatch_ssa!(ssa),
        }

        if (this.times[0] - mytinc).abs() < f64::EPSILON {
            this.times.remove(0); 
            if this.times.is_empty() {
                this.finished = true;
            }
        } else {
            assert!(this.times[0] > mytinc);
            this.times[0] -= mytinc;
        }
        produced
    }
}

#[pyclass]
pub struct SimulationIteratorMotifMatch {
    ssa: SSAKind,
    rng: SmallRng,
    times: Vec<f64>,
    elapsed: f64,
    finished: bool,
    motif_registry: MotifRegistry<ViennaRNA>
}

#[pymethods]
impl SimulationIteratorMotifMatch {

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<Self>
    ) -> Option<(String, i32, f64, f64, f64)> {

        let this: &mut Self = &mut slf;

        if this.finished {
            return None;
        }

        let mut produced: Option<(String, i32, f64, f64, f64)> = None;

        let rng = &mut this.rng;
        let mut mytinc = 0.0;
        let mut first_pass = true;

        macro_rules! dispatch_ssa {
            ($ssa:expr) => {{
                $ssa.co_simulate(
                    rng,
                    &this.times,
                    |t, tinc, flux, w| {
                        if first_pass {
                            mytinc = tinc.min(this.times[0]);

                            produced = Some((
                                    w.to_string(),
                                    w.current_energy(),
                                    this.elapsed + t,
                                    mytinc,
                                    flux,
                            ));

                            this.elapsed += mytinc;
                            first_pass = false;
                            // advance the simulator to update the structure.
                            true
                        } else {
                            false
                        }
                    },
                    );
            }};
        }

        match &mut this.ssa {
            SSAKind::NoShift(ssa) => dispatch_ssa!(ssa),
            SSAKind::ThreeWayOnly(ssa) => dispatch_ssa!(ssa),
            SSAKind::FourWayOnly(ssa) => dispatch_ssa!(ssa),
            SSAKind::ThreeAndFour(ssa) => dispatch_ssa!(ssa),
        }
        
        let structure = produced.as_ref()
            .and_then(|(s, ..)| DotBracketVec::try_from(s.as_str()).ok())?;

        let motif_found = !this.motif_registry.classify(&structure).iter().all(|&x| x == 0);

        if (this.times[0] - mytinc).abs() < f64::EPSILON || motif_found {
            this.times.remove(0); 
            if this.times.is_empty() || motif_found {
                this.finished = true;
            }
        } else {
            assert!(this.times[0] > mytinc);
            this.times[0] -= mytinc;
        }
        produced
    }
}

#[pyclass]
pub struct SimulationEnsembleIteratorMotifMatch {
    ssa: SSAKind,
    rng: SmallRng,
    times: Vec<f64>,
    elapsed: f64,
    finished: bool,
    motif_registry: Arc<MotifRegistry<ViennaRNA>>,
    timeline: Timeline<ViennaRNA>,
    t_idx: usize
}

impl Iterator for SimulationEnsembleIteratorMotifMatch {
    type Item = (String, i32, f64, f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }


        let mut produced: Option<(String, i32, f64, f64, f64)> = None;

        let rng = &mut self.rng;
        let mut mytinc = 0.0;
        let mut first_pass = true;

        macro_rules! dispatch_ssa {
            ($ssa:expr) => {{
                $ssa.co_simulate(
                    rng,
                    &self.times,
                    |t, tinc, flux, w| {
                        if first_pass {
                            mytinc = tinc.min(self.times[0]);

                            produced = Some((
                                w.to_string(),
                                w.current_energy(),
                                self.elapsed + t,
                                mytinc,
                                flux,
                            ));

                            self.elapsed += mytinc;
                            first_pass = false;
                            true
                        } else {
                            false
                        }
                    },
                );
            }};
        }

        match &mut self.ssa {
            SSAKind::NoShift(ssa) => dispatch_ssa!(ssa),
            SSAKind::ThreeWayOnly(ssa) => dispatch_ssa!(ssa),
            SSAKind::FourWayOnly(ssa) => dispatch_ssa!(ssa),
            SSAKind::ThreeAndFour(ssa) => dispatch_ssa!(ssa),
        }

        let structure = produced
            .as_ref()
            .and_then(|(s, ..)| DotBracketVec::try_from(s.as_str()).ok())?;

        let motif_found = !self
            .motif_registry
            .classify(&structure)
            .iter()
            .all(|&x| x == 0);


        if (self.times[0] - mytinc).abs() < f64::EPSILON {
            self.timeline.assign_structure(self.t_idx, &structure);
            self.t_idx += 1;

            self.times.remove(0);
            if self.times.is_empty() || motif_found{
                self.finished = true;
            }
        } else {
            self.times[0] -= mytinc;
        }

        produced
    }
}

#[pymethods]
impl SimulationEnsembleIteratorMotifMatch {

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<(String, i32, f64, f64, f64)> {
        slf.next()
    }
}