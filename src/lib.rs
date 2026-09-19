use pyo3::prelude::*;
use std::sync::Arc;

pub mod py_frame;
pub mod py_types;
pub mod py_fusion;
pub mod py_pipeline;

use py_frame::PyFrame;
use py_types::{
    PyBBox, PyEvent, PyFaceDetection, PyGaze, PyHeadPose, PyObjectDetection, PySignals,
    PyViolation,
};
use py_fusion::PyFusionEngine;
use py_pipeline::PyPipeline;

/// Helper to create a synthetic raw RGB8 frame for zero-copy testing & benchmarks.
#[pyfunction]
#[pyo3(signature = (width, height, seq=1, r=0, g=0, b=0))]
fn create_synthetic_frame(
    width: u32,
    height: u32,
    seq: u64,
    r: u8,
    g: u8,
    b: u8,
) -> PyFrame {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for chunk in data.chunks_exact_mut(3) {
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }

    let frame = vigilo_core::Frame {
        data: Arc::from(data),
        width,
        height,
        seq,
        captured_at: std::time::Instant::now(),
    };

    PyFrame::new(frame)
}

/// Returns the execution provider name and whether hardware acceleration is enabled in this build.
#[pyfunction]
fn device_info() -> PyResult<(String, bool)> {
    #[cfg(feature = "gpu-directml")]
    return Ok(("DirectML".to_string(), true));
    #[cfg(feature = "gpu-cuda")]
    return Ok(("CUDA".to_string(), true));
    #[cfg(feature = "gpu-coreml")]
    return Ok(("CoreML".to_string(), true));
    #[cfg(not(any(feature = "gpu-directml", feature = "gpu-cuda", feature = "gpu-coreml")))]
    return Ok(("CPU".to_string(), false));
}

#[pymodule]
mod _core {
    use super::*;

    #[pymodule_export]
    use super::create_synthetic_frame;

    #[pymodule_export]
    use super::device_info;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", "1.0.0")?;
        m.add_class::<PyFrame>()?;
        m.add_class::<PyPipeline>()?;
        m.add_class::<PyFusionEngine>()?;
        m.add_class::<PyBBox>()?;
        m.add_class::<PyFaceDetection>()?;
        m.add_class::<PyHeadPose>()?;
        m.add_class::<PyGaze>()?;
        m.add_class::<PyObjectDetection>()?;
        m.add_class::<PySignals>()?;
        m.add_class::<PyViolation>()?;
        m.add_class::<PyEvent>()?;
        Ok(())
    }
}
