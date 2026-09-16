use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use std::path::Path;

use vigilo_core::config::Config;
use vigilo_core::fusion::FusionEngine;
use vigilo_core::types::Signals;

use crate::py_types::{PyEvent, PySignals};

/// Deterministic temporal stream fusion engine.
///
/// Converts noisy instantaneous per-frame signals into stable, debounced
/// violation events using hold timers, asymmetric enter/exit hysteresis bands,
/// and decaying score accumulators.
#[pyclass(name = "FusionEngine")]
pub struct PyFusionEngine {
    inner: FusionEngine,
    config: Config,
}

#[pymethods]
impl PyFusionEngine {
    /// Create a new FusionEngine with default configuration or from a TOML file.
    #[new]
    #[pyo3(signature = (config_path=None))]
    pub fn new(config_path: Option<String>) -> PyResult<Self> {
        let config = if let Some(path_str) = config_path {
            let path = Path::new(&path_str);
            let text = std::fs::read_to_string(path)
                .map_err(|e| PyIOError::new_err(format!("Could not read config file '{}': {}", path_str, e)))?;
            toml::from_str::<Config>(&text)
                .map_err(|e| PyValueError::new_err(format!("Invalid TOML config in '{}': {}", path_str, e)))?
        } else {
            Config::default()
        };

        let inner = FusionEngine::new(&config);
        Ok(Self { inner, config })
    }

    /// Advance the fusion engine by one frame of signals at discrete timestamp `t_ms`.
    ///
    /// Pure function: reads no clock, touches no I/O.
    #[pyo3(signature = (signals, t_ms=None))]
    pub fn step(&mut self, signals: &PySignals, t_ms: Option<u64>) -> Vec<PyEvent> {
        let timestamp = t_ms.unwrap_or(signals.t_ms);
        let core_signals = vigilo_core::Signals {
            seq: signals.seq,
            t_ms: timestamp,
            faces: signals.faces.iter().map(|f| vigilo_core::FaceDetection {
                bbox: vigilo_core::BBox { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
                score: f.score,
                keypoints: if f.landmarks.len() == 5 {
                    Some(vigilo_core::FaceKeypoints {
                        right_eye: f.landmarks[0],
                        left_eye: f.landmarks[1],
                        nose: f.landmarks[2],
                        right_mouth: f.landmarks[3],
                        left_mouth: f.landmarks[4],
                    })
                } else {
                    None
                },
            }).collect(),
            head_pose: signals.head_pose.as_ref().map(|p| vigilo_core::HeadPose {
                yaw_deg: p.yaw_deg,
                pitch_deg: p.pitch_deg,
                roll_deg: p.roll_deg,
            }),
            gaze: signals.gaze.as_ref().map(|g| vigilo_core::Gaze {
                yaw_rad: g.yaw_rad,
                pitch_rad: g.pitch_rad,
                eye_yaw_rad: g.eye_yaw_rad,
                eye_pitch_rad: g.eye_pitch_rad,
            }),
            objects: signals.objects.iter().map(|o| vigilo_core::ObjectDetection {
                class_id: o.class_id,
                label: o.label.clone(),
                score: o.score,
                bbox: vigilo_core::BBox { x: o.bbox.x, y: o.bbox.y, w: o.bbox.w, h: o.bbox.h },
            }).collect(),
            identity_match: signals.identity_match,
            eye_aspect: None,
            produced_by: vigilo_core::SignalCoverage::default(),
            debug_directions: None,
        };

        let events = self.inner.step(&core_signals, timestamp);
        events.iter().map(Into::into).collect()
    }

    /// Close everything still open at the end of a session, generating finished events.
    pub fn finish(&mut self, t_ms: u64) -> Vec<PyEvent> {
        let events = self.inner.finish(t_ms);
        events.iter().map(Into::into).collect()
    }

    /// List active violation kinds currently open on the engine.
    pub fn active(&self) -> Vec<String> {
        self.inner.active().iter().map(|k| format!("{:?}", k)).collect()
    }

    /// Deterministically replay a recorded session of JSONL signals from disk.
    ///
    /// Yields byte-identical event sequences across runs.
    pub fn replay(&mut self, jsonl_path: String) -> PyResult<Vec<PyEvent>> {
        let path = Path::new(&jsonl_path);
        let text = std::fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("Could not read session file '{}': {}", jsonl_path, e)))?;

        let mut events = Vec::new();
        let mut last_t = 0u64;

        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let s: Signals = serde_json::from_str(line)
                .map_err(|e| PyValueError::new_err(format!("{}:{}: invalid Signals JSON: {}", jsonl_path, i + 1, e)))?;
            last_t = s.t_ms;
            let evs = self.inner.step(&s, s.t_ms);
            events.extend(evs.iter().map(Into::into));
        }

        let finished = self.inner.finish(last_t);
        events.extend(finished.iter().map(Into::into));

        Ok(events)
    }

    /// Hot-reload threshold configuration from a TOML string without restarting.
    pub fn update_config(&mut self, toml_str: String) -> PyResult<()> {
        let cfg: Config = toml::from_str(&toml_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid TOML config: {}", e)))?;
        self.config = cfg.clone();
        self.inner = FusionEngine::new(&self.config);
        Ok(())
    }

    /// Reset all internal timers and state back to empty.
    pub fn reset(&mut self) {
        self.inner = FusionEngine::new(&self.config);
    }

    fn __repr__(&self) -> String {
        format!("FusionEngine(active_violations={:?})", self.active())
    }
}
