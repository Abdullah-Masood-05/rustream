use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Axis-aligned 2D bounding box in source frame pixel coordinates.
#[pyclass(name = "BBox", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBBox {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
    #[pyo3(get)]
    pub w: f32,
    #[pyo3(get)]
    pub h: f32,
}

#[pymethods]
impl PyBBox {
    #[new]
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    #[getter]
    pub fn area(&self) -> f32 {
        (self.w * self.h).max(0.0)
    }

    #[getter]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w * 0.5, self.y + self.h * 0.5)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("x", self.x)?;
        dict.set_item("y", self.y)?;
        dict.set_item("w", self.w)?;
        dict.set_item("h", self.h)?;
        dict.set_item("area", self.area())?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("BBox(x={:.1}, y={:.1}, w={:.1}, h={:.1})", self.x, self.y, self.w, self.h)
    }
}

impl From<vigilo_core::BBox> for PyBBox {
    fn from(b: vigilo_core::BBox) -> Self {
        Self { x: b.x, y: b.y, w: b.w, h: b.h }
    }
}

/// A detected face with confidence score and 5 facial landmarks.
#[pyclass(name = "FaceDetection", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFaceDetection {
    #[pyo3(get)]
    pub bbox: PyBBox,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub landmarks: Vec<(f32, f32)>,
}

#[pymethods]
impl PyFaceDetection {
    #[new]
    #[pyo3(signature = (bbox, score=1.0, landmarks=None))]
    pub fn new(bbox: PyBBox, score: f32, landmarks: Option<Vec<(f32, f32)>>) -> Self {
        Self {
            bbox,
            score,
            landmarks: landmarks.unwrap_or_default(),
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bbox", self.bbox.to_dict(py)?)?;
        dict.set_item("score", self.score)?;
        dict.set_item("landmarks", self.landmarks.clone())?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("FaceDetection(score={:.2}, bbox={:?})", self.score, self.bbox)
    }
}

impl From<&vigilo_core::FaceDetection> for PyFaceDetection {
    fn from(f: &vigilo_core::FaceDetection) -> Self {
        let landmarks = f.keypoints.map(|kp| vec![
            kp.right_eye,
            kp.left_eye,
            kp.nose,
            kp.right_mouth,
            kp.left_mouth,
        ]).unwrap_or_default();
        Self {
            bbox: f.bbox.into(),
            score: f.score,
            landmarks,
        }
    }
}

/// Head pose Euler angles in degrees (aerospace yaw-pitch-roll convention).
#[pyclass(name = "HeadPose", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHeadPose {
    #[pyo3(get)]
    pub yaw_deg: f32,
    #[pyo3(get)]
    pub pitch_deg: f32,
    #[pyo3(get)]
    pub roll_deg: f32,
}

#[pymethods]
impl PyHeadPose {
    #[new]
    pub fn new(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Self {
        Self { yaw_deg, pitch_deg, roll_deg }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("yaw_deg", self.yaw_deg)?;
        dict.set_item("pitch_deg", self.pitch_deg)?;
        dict.set_item("roll_deg", self.roll_deg)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("HeadPose(yaw={:.1}°, pitch={:.1}°, roll={:.1}°)", self.yaw_deg, self.pitch_deg, self.roll_deg)
    }
}

impl From<vigilo_core::HeadPose> for PyHeadPose {
    fn from(p: vigilo_core::HeadPose) -> Self {
        Self {
            yaw_deg: p.yaw_deg,
            pitch_deg: p.pitch_deg,
            roll_deg: p.roll_deg,
        }
    }
}

/// Gaze direction in radians (yaw and pitch) and relative eye-in-head deflection.
#[pyclass(name = "Gaze", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGaze {
    #[pyo3(get)]
    pub yaw_rad: f32,
    #[pyo3(get)]
    pub pitch_rad: f32,
    #[pyo3(get)]
    pub eye_yaw_rad: Option<f32>,
    #[pyo3(get)]
    pub eye_pitch_rad: Option<f32>,
}

#[pymethods]
impl PyGaze {
    #[new]
    #[pyo3(signature = (yaw_rad=0.0, pitch_rad=0.0, eye_yaw_rad=None, eye_pitch_rad=None))]
    pub fn new(yaw_rad: f32, pitch_rad: f32, eye_yaw_rad: Option<f32>, eye_pitch_rad: Option<f32>) -> Self {
        Self {
            yaw_rad,
            pitch_rad,
            eye_yaw_rad,
            eye_pitch_rad,
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("yaw_rad", self.yaw_rad)?;
        dict.set_item("pitch_rad", self.pitch_rad)?;
        dict.set_item("eye_yaw_rad", self.eye_yaw_rad)?;
        dict.set_item("eye_pitch_rad", self.eye_pitch_rad)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("Gaze(yaw={:.2}rad, pitch={:.2}rad, eye_yaw={:?}, eye_pitch={:?})",
            self.yaw_rad, self.pitch_rad, self.eye_yaw_rad, self.eye_pitch_rad)
    }
}

impl From<&vigilo_core::Gaze> for PyGaze {
    fn from(g: &vigilo_core::Gaze) -> Self {
        Self {
            yaw_rad: g.yaw_rad,
            pitch_rad: g.pitch_rad,
            eye_yaw_rad: g.eye_yaw_rad,
            eye_pitch_rad: g.eye_pitch_rad,
        }
    }
}

/// Detected prohibited object (phone, book, laptop).
#[pyclass(name = "ObjectDetection", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyObjectDetection {
    #[pyo3(get)]
    pub class_id: u32,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub bbox: PyBBox,
}

#[pymethods]
impl PyObjectDetection {
    #[new]
    #[pyo3(signature = (label, score, bbox, class_id=0))]
    pub fn new(label: String, score: f32, bbox: PyBBox, class_id: u32) -> Self {
        Self {
            class_id,
            label,
            score,
            bbox,
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("class_id", self.class_id)?;
        dict.set_item("label", &self.label)?;
        dict.set_item("score", self.score)?;
        dict.set_item("bbox", self.bbox.to_dict(py)?)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("ObjectDetection(label='{}', score={:.2}, bbox={:?})",
            self.label, self.score, self.bbox)
    }
}

impl From<&vigilo_core::ObjectDetection> for PyObjectDetection {
    fn from(o: &vigilo_core::ObjectDetection) -> Self {
        Self {
            class_id: o.class_id,
            label: o.label.clone(),
            score: o.score,
            bbox: o.bbox.into(),
        }
    }
}

/// Instantaneous snapshot of all models' outputs for one frame.
#[pyclass(name = "Signals", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySignals {
    #[pyo3(get)]
    pub seq: u64,
    #[pyo3(get)]
    pub t_ms: u64,
    #[pyo3(get)]
    pub faces: Vec<PyFaceDetection>,
    #[pyo3(get)]
    pub head_pose: Option<PyHeadPose>,
    #[pyo3(get)]
    pub gaze: Option<PyGaze>,
    #[pyo3(get)]
    pub objects: Vec<PyObjectDetection>,
    #[pyo3(get)]
    pub identity_match: Option<f32>,
}

#[pymethods]
impl PySignals {
    #[new]
    #[pyo3(signature = (seq=0, t_ms=0, faces=None, head_pose=None, gaze=None, objects=None, identity_match=None))]
    pub fn new(
        seq: u64,
        t_ms: u64,
        faces: Option<Vec<PyFaceDetection>>,
        head_pose: Option<PyHeadPose>,
        gaze: Option<PyGaze>,
        objects: Option<Vec<PyObjectDetection>>,
        identity_match: Option<f32>,
    ) -> Self {
        Self {
            seq,
            t_ms,
            faces: faces.unwrap_or_default(),
            head_pose,
            gaze,
            objects: objects.unwrap_or_default(),
            identity_match,
        }
    }

    #[getter]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("seq", self.seq)?;
        dict.set_item("t_ms", self.t_ms)?;
        dict.set_item("face_count", self.face_count())?;
        
        let py_faces: Vec<Bound<'py, PyDict>> = self.faces.iter()
            .map(|f| f.to_dict(py))
            .collect::<PyResult<_>>()?;
        dict.set_item("faces", py_faces)?;
        
        if let Some(p) = &self.head_pose {
            dict.set_item("head_pose", p.to_dict(py)?)?;
        } else {
            dict.set_item("head_pose", py.None())?;
        }

        if let Some(g) = &self.gaze {
            dict.set_item("gaze", g.to_dict(py)?)?;
        } else {
            dict.set_item("gaze", py.None())?;
        }

        let py_objects: Vec<Bound<'py, PyDict>> = self.objects.iter()
            .map(|o| o.to_dict(py))
            .collect::<PyResult<_>>()?;
        dict.set_item("objects", py_objects)?;
        dict.set_item("identity_match", self.identity_match)?;
        
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("Signals(seq={}, t_ms={}, faces={}, pose={:?}, gaze={:?}, objects={})",
            self.seq, self.t_ms, self.faces.len(), self.head_pose.is_some(), self.gaze.is_some(), self.objects.len())
    }
}

impl From<&vigilo_core::Signals> for PySignals {
    fn from(s: &vigilo_core::Signals) -> Self {
        Self {
            seq: s.seq,
            t_ms: s.t_ms,
            faces: s.faces.iter().map(Into::into).collect(),
            head_pose: s.head_pose.map(Into::into),
            gaze: s.gaze.as_ref().map(Into::into),
            objects: s.objects.iter().map(Into::into).collect(),
            identity_match: s.identity_match,
        }
    }
}

/// A temporal violation produced by fusion rules.
#[pyclass(name = "Violation", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyViolation {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub severity: String,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub t_start_ms: u64,
    #[pyo3(get)]
    pub t_end_ms: Option<u64>,
    #[pyo3(get)]
    pub subject: Option<String>,
}

#[pymethods]
impl PyViolation {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("kind", &self.kind)?;
        dict.set_item("severity", &self.severity)?;
        dict.set_item("confidence", self.confidence)?;
        dict.set_item("t_start_ms", self.t_start_ms)?;
        dict.set_item("t_end_ms", self.t_end_ms)?;
        dict.set_item("subject", &self.subject)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("Violation(kind='{}', severity='{}', conf={:.2}, start={}ms, end={:?})",
            self.kind, self.severity, self.confidence, self.t_start_ms, self.t_end_ms)
    }
}

impl From<&vigilo_core::Violation> for PyViolation {
    fn from(v: &vigilo_core::Violation) -> Self {
        let kind = serde_json::to_value(&v.kind)
            .ok()
            .and_then(|val| val.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| format!("{:?}", v.kind));
        let severity = format!("{:?}", v.severity);
        Self {
            kind,
            severity,
            confidence: v.confidence,
            t_start_ms: v.t_start_ms,
            t_end_ms: v.t_end_ms,
            subject: v.subject.clone(),
        }
    }
}

/// Discrete event generated by the pipeline or fusion engine.
#[pyclass(name = "Event", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyEvent {
    #[pyo3(get)]
    pub event_type: String,
    #[pyo3(get)]
    pub violation: Option<PyViolation>,
    #[pyo3(get)]
    pub message: Option<String>,
}

#[pymethods]
impl PyEvent {
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("event_type", &self.event_type)?;
        if let Some(v) = &self.violation {
            dict.set_item("violation", v.to_dict(py)?)?;
        } else {
            dict.set_item("violation", py.None())?;
        }
        dict.set_item("message", &self.message)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        match (&self.violation, &self.message) {
            (Some(v), _) => format!("Event({}: {:?})", self.event_type, v),
            (None, Some(m)) => format!("Event({}: {})", self.event_type, m),
            _ => format!("Event({})", self.event_type),
        }
    }
}

impl From<&vigilo_core::Event> for PyEvent {
    fn from(e: &vigilo_core::Event) -> Self {
        match e {
            vigilo_core::Event::ViolationStarted(v) => Self {
                event_type: "ViolationStarted".to_string(),
                violation: Some(v.into()),
                message: None,
            },
            vigilo_core::Event::ViolationEnded(v) => Self {
                event_type: "ViolationEnded".to_string(),
                violation: Some(v.into()),
                message: None,
            },
            vigilo_core::Event::CalibrationProgress { pct } => Self {
                event_type: "CalibrationProgress".to_string(),
                violation: None,
                message: Some(format!("pct={:.1}", pct)),
            },
            vigilo_core::Event::CalibrationComplete => Self {
                event_type: "CalibrationComplete".to_string(),
                violation: None,
                message: None,
            },
            vigilo_core::Event::Degraded(reason) => Self {
                event_type: "Degraded".to_string(),
                violation: None,
                message: Some(format!("{:?}", reason)),
            },
            vigilo_core::Event::Recovered => Self {
                event_type: "Recovered".to_string(),
                violation: None,
                message: None,
            },
        }
    }
}
