use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};

/// Zero-copy RGB8 Frame container supporting:
/// 1. NumPy Array Interface (`np.asarray(frame)`)
/// 2. Direct zero-copy MemoryView (`frame.memoryview()`)
/// 3. Direct NumPy conversion (`frame.to_numpy()`)
#[pyclass(name = "Frame")]
pub struct PyFrame {
    pub inner: vigilo_core::Frame,
}

impl PyFrame {
    pub fn new(inner: vigilo_core::Frame) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFrame {
    #[getter]
    pub fn width(&self) -> u32 {
        self.inner.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.inner.height
    }

    #[getter]
    pub fn seq(&self) -> u64 {
        self.inner.seq
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.inner.height as usize, self.inner.width as usize, 3)
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.inner.data.len()
    }

    #[getter]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Access raw frame bytes.
    pub fn as_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }

    /// Python memoryview over the raw RGB8 frame buffer.
    pub fn memoryview<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bytes = PyBytes::new(py, &self.inner.data);
        let builtins = py.import("builtins")?;
        builtins.call_method1("memoryview", (bytes,))
    }

    /// NumPy array interface for direct zero-copy views.
    #[getter]
    pub fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", (self.inner.height, self.inner.width, 3))?;
        dict.set_item("typestr", "|u1")?;
        
        let ptr = self.inner.data.as_ptr() as usize;
        dict.set_item("data", (ptr, true))?;
        dict.set_item("version", 3)?;
        dict.set_item("strides", (self.inner.width * 3, 3, 1))?;
        Ok(dict)
    }

    /// Convert zero-copy to NumPy ndarray via __array_interface__.
    pub fn to_numpy<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.call_method1("asarray", (slf,))
    }

    fn __repr__(&self) -> String {
        format!("Frame(seq={}, {}x{} RGB8, {} bytes)",
            self.inner.seq, self.inner.width, self.inner.height, self.inner.data.len())
    }
}
