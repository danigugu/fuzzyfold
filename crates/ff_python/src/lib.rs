#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::types::PyModule;
use pyo3::prelude::*;

pub mod energy_exports;
pub mod kinetics_exports;
pub mod kinetics_dani_exports;

#[pymodule]
fn fuzzyfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<energy_exports::ViennaRNA>()?;
    m.add_class::<kinetics_dani_exports::Simulator>()?;
    Ok(())
}

