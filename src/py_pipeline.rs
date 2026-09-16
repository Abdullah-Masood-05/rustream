use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use std::path::Path;

use vigilo_core::capture::SourceSpec;
use vigilo_core::config::Config;
use vigilo_core::pipeline::Detector;

use crate::py_frame::PyFrame;
use crate::py_types::{PyEvent, PySignals};

/// High-performance multi-modal capture and detection pipeline.
///
/// Manages background worker threads for frame capture (DirectShow camera, video file,
/// or directory replay), ONNX neural network inference (face, pose, gaze, objects, identity),
/// and deterministic temporal fusion.
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    detector: Option<Detector>,
    config: Config,
    is_started: bool,
}

#[pymethods]
impl PyPipeline {
    /// Create a new Pipeline with default configuration or from a TOML file.
    #[new]
    #[pyo3(signature = (config_path=None, models_dir=None))]
    pub fn new(config_path: Option<String>, models_dir: Option<String>) -> PyResult<Self> {
        let mut config = if let Some(path_str) = config_path {
            let path = Path::new(&path_str);
            let text = std::fs::read_to_string(path)
                .map_err(|e| PyIOError::new_err(format!("Could not read config file '{}': {}", path_str, e)))?;
            toml::from_str::<Config>(&text)
                .map_err(|e| PyValueError::new_err(format!("Invalid TOML config in '{}': {}", path_str, e)))?
        } else {
            Config::default()
        };

        // Auto-fill models from directory if provided, or default to "models"
        let m_dir = models_dir.as_deref().unwrap_or("models");
        if Path::new(m_dir).exists() {
            config.models.fill_missing_from_dir(m_dir);
        }

        let detector = Detector::builder()
            .config(config.clone())
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build detector: {}", e)))?;

        Ok(Self {
            detector: Some(detector),
            config,
            is_started: false,
        })
    }

    /// Start processing from a video source specification.
    ///
    /// Supported formats:
    /// - `"camera:0"` (default webcam device 0)
    /// - `"file:path/to/video.mp4"` (offline video playback)
    /// - `"dir:path/to/frames_folder"` (directory of image sequence frames)
    pub fn start(&mut self, source_spec: String) -> PyResult<()> {
        let detector = self.detector.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Pipeline detector is uninitialized or closed")
        })?;

        let spec = source_spec.parse::<SourceSpec>()
            .map_err(|e| PyValueError::new_err(format!("Invalid source spec '{}': {}", source_spec, e)))?;

        let source = spec.open(&self.config.capture, false)
            .map_err(|e| PyIOError::new_err(format!("Failed to open source '{}': {}", source_spec, e)))?;

        detector.start(source)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start pipeline: {}", e)))?;

        self.is_started = true;
        Ok(())
    }

    /// Check if the pipeline background capture and detection workers are running.
    pub fn is_running(&self) -> bool {
        self.detector.as_ref().map(|d| d.is_running()).unwrap_or(false)
    }

    /// Poll the most recently detected frame (zero-copy RGB8 Frame).
    ///
    /// Returns `None` if no frame has arrived yet.
    pub fn poll_frame(&self) -> Option<PyFrame> {
        let detector = self.detector.as_ref()?;
        let latest = detector.latest()?;
        Some(PyFrame::new((*latest.frame).clone()))
    }

    /// Poll the latest instantaneous detection signals without blocking.
    ///
    /// Returns `None` if no frame has been processed yet.
    pub fn snapshot(&self) -> Option<PySignals> {
        let detector = self.detector.as_ref()?;
        let latest = detector.latest()?;
        Some((&latest.signals).into())
    }

    /// Drain and return all new violation events generated since the last call.
    pub fn events(&self) -> Vec<PyEvent> {
        let detector = match self.detector.as_ref() {
            Some(d) => d,
            None => return Vec::new(),
        };

        detector.events().try_iter().map(|e| (&e).into()).collect()
    }

    /// Request face identity enrollment on the next frame with a detected face.
    pub fn enrol(&self) {
        if let Some(detector) = self.detector.as_ref() {
            detector.enrol();
        }
    }

    /// Check if a face reference has been successfully enrolled for identity checks.
    pub fn is_enrolled(&self) -> bool {
        self.detector.as_ref().map(|d| d.is_enrolled()).unwrap_or(false)
    }

    /// Stop all background threads and release hardware resources.
    pub fn stop(&mut self) {
        if let Some(detector) = self.detector.as_mut() {
            if self.is_started {
                detector.stop();
                self.is_started = false;
            }
        }
    }

    /// Hot-reload threshold and cadence configuration without restarting the pipeline.
    pub fn update_config(&mut self, toml_str: String) -> PyResult<()> {
        let detector = self.detector.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Pipeline detector is uninitialized")
        })?;

        let cfg: Config = toml::from_str(&toml_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid TOML config: {}", e)))?;

        detector.update_config(cfg.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update config: {}", e)))?;

        self.config = cfg;
        Ok(())
    }

    // Python context manager support
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    pub fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) {
        self.stop();
    }

    fn __repr__(&self) -> String {
        format!("Pipeline(running={})", self.is_running())
    }
}
