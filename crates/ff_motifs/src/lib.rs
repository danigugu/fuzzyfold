// lib.rs
pub mod timecourse;
pub mod ffi;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn motifs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::MotifCheckSimulator>()?;
    m.add_class::<python::PyCotransConfig>()?;
    Ok(())
}