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
use std::path::Path;
use std::io::Cursor;
use std::io::{self, Write};

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
        
        let path = Path::new(&motifs_file);

        let mut motif_reg = MotifRegistry::from((Arc::clone(&sequence_arc), Arc::clone(&self.energy_model)));

        if path.exists() && path.is_file() {
            // It's a valid file path
            let file_path = PathBuf::from(&motifs_file);
            motif_reg.insert_from_file(&file_path)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        } else {
            // It's not a file; treat 'motifs_file' as raw content or a string identifier
            motif_reg.insert_from_reader(Cursor::new(motifs_file), "manual").unwrap();
        }

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
    fn simulate_ensemble_timecourse_motifs(
        &self,
        sequence: &str,
        motifs_file: &str,
        start: Option<&str>,
        t_ext: Option<f64>,
        t_end: f64,
        num_sims: usize,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let sequence_vec = NucleotideVec::try_from_rna(sequence)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

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

            let mut motif_reg = MotifRegistry::from((Arc::clone(&sequence_arc), Arc::clone(&self.energy_model)));

            let path = Path::new(&motifs_file);

            if path.exists() && path.is_file() {
                // It's a valid file path
                let file_path = PathBuf::from(&motifs_file);
                motif_reg.insert_from_file(&file_path)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            } else {
                // It's not a file; treat 'motifs_file' as raw content or a string identifier
                motif_reg.insert_from_reader(Cursor::new(motifs_file), "manual").unwrap();
            }

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
            let master_timeline_times: Vec<f64> = times
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

            // If the given start structure already belongs to a motif
            //if !motif_reg
            //    .classify(&init_struc)
            //    .iter()
            //    .all(|&x| x == 0) {
            //        let mut master_timeline = Timeline::new(master_timeline_times, Arc::clone(&motif_reg));
            //        master_timeline.points.insert(0, Timepoint::new(0.0));
            //        
            //        for i in 0..num_sims {
            //            master_timeline.assign_structure(0, &init_struc);
            //        }
            //        return convert_timeline_python_friendly(py, master_timeline).map(|list| list.to_object(py))
            //    }

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

            let mut master_timeline = Timeline::new(master_timeline_times, Arc::clone(&motif_reg));
            for timeline in timelines {
                master_timeline.merge(timeline);
            }

            master_timeline.points.insert(0, Timepoint::new(0.0));

            for i in 0..num_sims {
                master_timeline.assign_structure(0, &init_struc);
            }
            return convert_timeline_python_friendly(py, master_timeline).map(|list| list.to_object(py))
        })
    }


    #[pyo3(signature = (
        sequence,
        motifs_file,
        t_pos,
        start=None,
        t_ext=None,
        t_end=1.0,
        num_sims=100
    ))]
    fn simulate_ensemble_check_motifs(
        &self,
        sequence: &str,
        motifs_file: &str,
        t_pos: Vec<usize>,
        start: Option<&str>,
        t_ext: Option<f64>,
        t_end: f64,
        num_sims: usize,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let sequence_vec = NucleotideVec::try_from_rna(sequence)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

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

            let mut motif_reg = MotifRegistry::from((Arc::clone(&sequence_arc), Arc::clone(&self.energy_model)));

            let path = Path::new(&motifs_file);

            if path.exists() && path.is_file() {
                // It's a valid file path
                let file_path = PathBuf::from(&motifs_file);
                motif_reg.insert_from_file(&file_path)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            } else {
                // It's not a file; treat 'motifs_file' as raw content or a string identifier
                motif_reg.insert_from_reader(Cursor::new(motifs_file), "manual").unwrap();
            }

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
            let master_timeline_times: Vec<f64> = times
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

            let check_results: Vec<_> = match (self.rate_model.k3ws().is_some(), self.rate_model.k4ws().is_some()) {
                (false, false) => build_par_iterator_motif_check(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::NoShift,
                    SSAKind::NoShift,
                    t_pos
                ).unwrap(),
                (true, false) => build_par_iterator_motif_check(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::ThreeWayOnly,
                    SSAKind::ThreeWayOnly,
                    t_pos
                ).unwrap(),
                (false, true) => build_par_iterator_motif_check(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::FourWayOnly,
                    SSAKind::FourWayOnly,
                    t_pos
                ).unwrap(),
                (true, true) => build_par_iterator_motif_check(
                    seq_clone,
                    &start_pt,
                    Arc::clone(&self.energy_model),
                    self.rate_model,
                    times,
                    Arc::clone(&motif_reg),
                    num_sims,
                    shift_policy::ThreeAndFour,
                    SSAKind::ThreeAndFour,
                    t_pos
                ).unwrap(),
            };

            let mut master_check_results = merge_check_results(&check_results);
            return convert_check_results_python_friendly(py, master_check_results).map(|list| list.to_object(py))
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

        if !tp.ensemble.contains_key(&0) {
            ensemble.set_item(0, 0)?;
        }

        dict.set_item("ensemble", ensemble)?;
        py_list.append(dict)?;
    }

    Ok(py_list)
}

fn convert_check_results_python_friendly(
    py: Python,
    results: CheckResults,
) -> PyResult<&PyList> {
    let py_list = PyList::empty(py);

    for (i, (&count, distances)) in results
        .num_success_sims_per_ts
        .iter()
        .zip(results.distances_per_ts.iter())
        .enumerate()
    {
        let dict = PyDict::new(py);

        dict.set_item("ts", i+1)?;
        dict.set_item("num_success", count)?;

        let dist_list = PyList::empty(py);
        for d in distances {
            dist_list.append(d)?;
        }

        dict.set_item("distances", dist_list)?;

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
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;

    let master_timeline_times: Vec<f64> = times
        .iter()
        .scan(0.0, |acc, &dt| {
            *acc += dt;
            Some(*acc)
        })
        .collect();

    let pool = ThreadPoolBuilder::new()
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let results = pool.install(|| {
        (0..num_sims)
            .into_par_iter()
            .map_init(
                {
                    let times = times.clone();
                    let motif_registry = motif_registry.clone();
                    let rate_model = rate_model.clone();
                    move || (times.clone(), motif_registry.clone(), rate_model.clone())
                },
                {
                    let seq = seq.clone();
                    let energy_model = energy_model.clone();
                    let policy = policy.clone();

                    move |(times, motif_registry, rate_model), _| {
                        let walker = LoopNeighbors::try_from((
                            seq.clone(),
                            start_pt,
                            energy_model.clone(),
                            policy.clone(),
                        ))
                        .expect("Failed to build walker");

                        let mut sim_res = SimulationEnsembleIteratorMotifMatch {
                            ssa: wrap(SSA::from((walker, rate_model.clone()))),
                            rng: SmallRng::from_os_rng(),
                            times: times.clone(),
                            elapsed: 0.0,
                            finished: false,
                            motif_registry: motif_registry.clone(),
                            timeline: Timeline::new(
                                master_timeline_times.clone(),
                                motif_registry.clone(),
                            ),
                            t_idx: 0,
                        };

                        sim_res.run();

                        sim_res.get_timeline()
                    }
                },
            )
            .collect::<Vec<_>>()
    });

    Ok(results)
}

fn build_par_iterator_motif_check<P>(
    seq: NucleotideVec,
    start_pt: &PairTable,
    energy_model: Arc<ViennaRNA>,
    rate_model: Arrhenius,
    times: Vec<f64>,
    motif_registry: Arc<MotifRegistry<ViennaRNA>>,
    num_sims: usize,
    policy: P,
    wrap: fn(SSA<LoopNeighbors<ViennaRNA, P>, Arrhenius>) -> SSAKind,
    t_pos: Vec<usize>
    ) -> PyResult<Vec<CheckResults>>
    where
    P: shift_policy::ShiftPolicy + Send + Sync + Clone + 'static,
    {
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;

    let master_timeline_times: Vec<f64> = times
        .iter()
        .scan(0.0, |acc, &dt| {
            *acc += dt;
            Some(*acc)
        })
        .collect();

    let pool = ThreadPoolBuilder::new()
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let results = pool.install(|| {
        (0..num_sims)
            .into_par_iter()
            .map_init(
                {
                    let times = times.clone();
                    let motif_registry = motif_registry.clone();
                    let rate_model = rate_model.clone();
                    let t_pos = t_pos.clone();
                    move || (times.clone(), motif_registry.clone(), rate_model.clone(), t_pos.clone())
                },
                {
                    let seq = seq.clone();
                    let energy_model = energy_model.clone();
                    let policy = policy.clone();

                    move |(times, motif_registry, rate_model, t_pos), _| {
                        let walker = LoopNeighbors::try_from((
                            seq.clone(),
                            start_pt,
                            energy_model.clone(),
                            policy.clone(),
                        ))
                        .expect("Failed to build walker");

                        let mut sim_res = SimulationEnsembleIteratorMotifCheck {
                            ssa: wrap(SSA::from((walker, rate_model.clone()))),
                            rng: SmallRng::from_os_rng(),
                            times: times.clone(),
                            elapsed: 0.0,
                            finished: false,
                            motif_registry: motif_registry.clone(),
                            check_results: CheckResults { num_success_sims_per_ts: Vec::new(), distances_per_ts: Vec::new() },
                            t_idx: 0,
                            t_pos: t_pos.clone(),
                            cur_in_t: false,
                            ts_counter: 0,
                            distances: Vec::new(),
                            wrong_path: true
                        };

                        sim_res.run();

                        sim_res.get_check_results()
                    }
                },
            )
            .collect::<Vec<_>>()
    });

    Ok(results)
}

pub struct CheckResults {
    pub num_success_sims_per_ts: Vec<usize>,
    pub distances_per_ts: Vec<Vec<usize>>,
}

pub fn merge_check_results(results: &[CheckResults]) -> CheckResults {
    assert!(!results.is_empty(), "Cannot merge empty results");

    let len = results[0].num_success_sims_per_ts.len();

    // Optional safety: ensure all have same shape
    for r in results {
        assert_eq!(r.num_success_sims_per_ts.len(), len);
        assert_eq!(r.distances_per_ts.len(), len);
    }

    let mut merged_counts = vec![0; len];
    let mut merged_distances = vec![Vec::new(); len];

    for r in results {
        for i in 0..len {
            merged_counts[i] += r.num_success_sims_per_ts[i];

            // flatten (append inner vectors)
            merged_distances[i].extend(&r.distances_per_ts[i]);
        }
    }

    CheckResults {
        num_success_sims_per_ts: merged_counts,
        distances_per_ts: merged_distances,
    }
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
        
        if (this.times[0] - mytinc).abs() < f64::EPSILON {
            let structure = produced.as_ref()
                .and_then(|(s, ..)| DotBracketVec::try_from(s.as_str()).ok())?;
            
            let motif_found = !this.motif_registry.classify(&structure).iter().all(|&x| x == 0);

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


impl SimulationEnsembleIteratorMotifMatch {
    pub fn run(&mut self) {
        while !self.finished {
            self.step();
        }
    }

    /// Returns the timeline after the simulation is done
    pub fn get_timeline(self) -> Timeline<ViennaRNA> {
        self.timeline
    }
}

impl SimulationEnsembleIteratorMotifMatch {
    /// Internal helper that performs a single simulation step
    fn step(&mut self) {
        if self.finished {
            return;
        }

        let mut produced: Option<(String, i32, f64, f64, f64)> = None;
        let rng = &mut self.rng;
        let mut mytinc = 0.0;
        let mut first_pass = true;

        {
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
        }

        // let mut motif_found = false;

        if (self.times[0] - mytinc).abs() < f64::EPSILON {

            // Extract the structure string if it exists
            if let Some((ref s, ..)) = produced {
                if let Ok(structure) = DotBracketVec::try_from(s.as_str()) {
                    self.timeline.assign_structure(self.t_idx, &structure);
                    self.t_idx += 1;
                    // Check motif length and classify
                    // if self.motif_registry.min_motif_length() <= &structure.len() {
                    //    let classifications = self.motif_registry.classify(&structure);
                    //    motif_found = !classifications.iter().all(|&x| x == 0);
                    //}
                }
            }

            self.times.remove(0);
            
            if self.times.is_empty() {
                self.finished = true;
            }
        } else {
            // Guard against underflow
            assert!(self.times[0] > mytinc);
            self.times[0] -= mytinc;
        }
    }
}

pub struct SimulationEnsembleIteratorMotifCheck {
    ssa: SSAKind,
    rng: SmallRng,
    times: Vec<f64>,
    elapsed: f64,
    finished: bool,
    motif_registry: Arc<MotifRegistry<ViennaRNA>>,
    check_results: CheckResults,
    t_idx: usize,
    t_pos: Vec<usize>,
    cur_in_t: bool,
    ts_counter: usize,
    distances: Vec<usize>,
    wrong_path: bool
}


impl SimulationEnsembleIteratorMotifCheck {
    pub fn run(&mut self) {
        while !self.finished {
            self.step();
        }
    }

    /// Returns the timeline after the simulation is done
    pub fn get_check_results(self) -> CheckResults {
        self.check_results
    }
}

impl SimulationEnsembleIteratorMotifCheck {
    /// Internal helper that performs a single simulation step
    fn step(&mut self) {
        if self.finished {
            return;
        }

        let mut produced: Option<(String, i32, f64, f64, f64)> = None;
        let rng = &mut self.rng;
        let mut mytinc = 0.0;
        let mut first_pass = true;

        {
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
        }

        if (self.times[0] - mytinc).abs() < f64::EPSILON {
            // println!("INSIDE");
            if self.t_idx == self.t_pos[0] {
                // println!("IN T DOMAIN");
                if !self.cur_in_t{
                    // println!("ENTERING T");
                    self.ts_counter += 1;
                    self.cur_in_t = true;
                    self.wrong_path = true;
                }
                if let Some((ref s, ..)) = produced {
                    if let Ok(structure) = DotBracketVec::try_from(s.as_str()) {   
                        self.distances.push(
                            self.motif_registry.motifs()[self.ts_counter].distance(&PairTable::try_from(&structure).unwrap())
                        );

                        if self.wrong_path {
                            let classifications = self.motif_registry.classify(&structure);
                            //println!("classifications: {}", classifications);
                            self.wrong_path = !classifications.contains(&self.ts_counter);
                            // println!("wrong path: {}", self.wrong_path);
                        }
                    }
                }
                self.t_pos.remove(0);
            }
            else {
                // self.timeline.remove_point(self.t_idx);

                if self.cur_in_t {
                    // println!("MOVING OUTSIDE OF T-DOMAIN");
                    self.cur_in_t = false;

                    let old_distances = std::mem::take(&mut self.distances);
                    self.check_results.distances_per_ts.push(old_distances);

                    if !self.wrong_path {
                        self.check_results.num_success_sims_per_ts.push(1);
                        println!("SUCCESS");
                    }
                    else {
                        self.check_results.num_success_sims_per_ts.push(0);
                        println!("FAIL");
                        self.finished = true;
                    }
                }
            }

            self.times.remove(0);
            self.t_idx += 1;
            
            if self.times.is_empty() {
                self.finished = true;

                let old_distances = std::mem::take(&mut self.distances);
                self.check_results.distances_per_ts.push(old_distances);

                if !self.wrong_path {
                    self.check_results.num_success_sims_per_ts.push(1);
                    println!("SUCCESS");
                    }
                else {
                    self.check_results.num_success_sims_per_ts.push(0);
                    println!("FAIL");
                    self.finished = true;
                }
            }

        } else {
            // Guard against underflow
            assert!(self.times[0] > mytinc);
            self.times[0] -= mytinc;
        }
    }
}