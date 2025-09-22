use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::io::Write;
use tempfile::NamedTempFile;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ExecutionResult {
    #[pyo3(get)]
    pub stdout: String,
    #[pyo3(get)]
    pub stderr: String,
    #[pyo3(get)]
    pub return_code: i32,
    #[pyo3(get)]
    pub execution_time_ms: u64,
    #[pyo3(get)]
    pub truncated: bool,
}

#[pymethods]
impl ExecutionResult {
    fn __repr__(&self) -> String {
        format!(
            "ExecutionResult(return_code={}, time_ms={}, truncated={})",
            self.return_code, self.execution_time_ms, self.truncated
        )
    }

    fn is_success(&self) -> bool {
        self.return_code == 0
    }
}

#[pyclass]
pub struct PythonSandbox {
    timeout_ms: u64,
    max_output_size: usize,
    allowed_imports: Vec<String>,
    blocked_imports: Vec<String>,
    max_memory_mb: usize,
    enable_network: bool,
}

#[pymethods]
impl PythonSandbox {
    #[new]
    #[pyo3(signature = (timeout_ms=30000, max_output_size=1048576, max_memory_mb=512))]
    fn new(timeout_ms: u64, max_output_size: usize, max_memory_mb: usize) -> Self {
        Self {
            timeout_ms,
            max_output_size,
            max_memory_mb,
            allowed_imports: vec![
                "math".to_string(),
                "json".to_string(),
                "datetime".to_string(),
                "collections".to_string(),
                "itertools".to_string(),
                "functools".to_string(),
                "re".to_string(),
                "random".to_string(),
                "statistics".to_string(),
                "decimal".to_string(),
                "fractions".to_string(),
                "hashlib".to_string(),
                "base64".to_string(),
                "urllib.parse".to_string(),
            ],
            blocked_imports: vec![
                "os".to_string(),
                "subprocess".to_string(),
                "socket".to_string(),
                "sys".to_string(),
                "__builtins__".to_string(),
                "eval".to_string(),
                "exec".to_string(),
                "compile".to_string(),
                "__import__".to_string(),
            ],
            enable_network: false,
        }
    }

    fn execute(&self, code: String) -> PyResult<ExecutionResult> {
        let start = std::time::Instant::now();

        // Create sandboxed code with import restrictions
        let sandboxed_code = self.create_sandboxed_code(&code)?;

        // Write code to temporary file
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        temp_file.write_all(sandboxed_code.as_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let temp_path = temp_file.path().to_str()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid path"))?;

        // Execute with resource limits
        let output = Command::new("python")
            .arg("-u")  // Unbuffered output
            .arg(temp_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        // Truncate output if needed
        let (stdout, stdout_truncated) = self.truncate_output(
            String::from_utf8_lossy(&output.stdout).to_string()
        );
        let (stderr, stderr_truncated) = self.truncate_output(
            String::from_utf8_lossy(&output.stderr).to_string()
        );

        Ok(ExecutionResult {
            stdout,
            stderr,
            return_code: output.status.code().unwrap_or(-1),
            execution_time_ms,
            truncated: stdout_truncated || stderr_truncated,
        })
    }

    // TODO: Re-enable when pyo3-asyncio is updated for PyO3 0.24+
    /*
    fn execute_async(&self, py: Python, code: String) -> PyResult<&PyAny> {
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let result = tokio::task::spawn_blocking(move || {
                let sandbox = PythonSandbox::new(30000, 1048576, 512);
                sandbox.execute(code)
            })
            .await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            result
        })
    }
    */

    fn validate_code(&self, code: &str) -> PyResult<bool> {
        // Check for dangerous patterns
        for blocked in &self.blocked_imports {
            if code.contains(blocked) {
                return Ok(false);
            }
        }

        // Check for suspicious patterns
        let dangerous_patterns = [
            "__class__",
            "__bases__",
            "__subclasses__",
            "__code__",
            "__globals__",
            "open(",
            "file(",
            "input(",
            "raw_input(",
        ];

        for pattern in dangerous_patterns {
            if code.contains(pattern) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn set_allowed_imports(&mut self, imports: Vec<String>) {
        self.allowed_imports = imports;
    }

    fn add_allowed_import(&mut self, import_name: String) {
        if !self.allowed_imports.contains(&import_name) {
            self.allowed_imports.push(import_name);
        }
    }

    fn set_timeout(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms;
    }

    fn set_max_memory(&mut self, max_memory_mb: usize) {
        self.max_memory_mb = max_memory_mb;
    }

    #[getter]
    fn get_allowed_imports(&self) -> Vec<String> {
        self.allowed_imports.clone()
    }

    #[getter]
    fn get_blocked_imports(&self) -> Vec<String> {
        self.blocked_imports.clone()
    }
}

impl PythonSandbox {
    fn create_sandboxed_code(&self, code: &str) -> PyResult<String> {
        // Validate code first
        if !self.validate_code(code)? {
            return Err(PyErr::new::<pyo3::exceptions::PySecurityError, _>(
                "Code contains potentially dangerous operations"
            ));
        }

        // Create import whitelist
        let import_whitelist = self.allowed_imports.iter()
            .map(|s| format!("'{}'", s))
            .collect::<Vec<_>>()
            .join(", ");

        // Wrap code in sandbox
        let sandboxed = format!(r#"
import sys
import resource
import signal

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
resource.setrlimit(resource.RLIMIT_AS, ({memory}, {memory}))

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout_seconds})

# Import restrictions
allowed_imports = [{imports}]
original_import = __builtins__.__import__

def restricted_import(name, *args, **kwargs):
    if name.split('.')[0] not in allowed_imports:
        raise ImportError(f"Import of '{{name}}' is not allowed")
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = restricted_import

# Disable dangerous builtins
dangerous_builtins = ['eval', 'exec', 'compile', 'open', 'file', 'input', '__import__']
for builtin in dangerous_builtins:
    if hasattr(__builtins__, builtin):
        delattr(__builtins__, builtin)

# User code execution
try:
{indented_code}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"#,
            timeout = self.timeout_ms / 1000,
            timeout_seconds = self.timeout_ms / 1000,
            memory = self.max_memory_mb * 1024 * 1024,
            imports = import_whitelist,
            indented_code = code.lines()
                .map(|line| format!("    {}", line))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(sandboxed)
    }

    fn truncate_output(&self, output: String) -> (String, bool) {
        if output.len() <= self.max_output_size {
            (output, false)
        } else {
            let truncated = format!(
                "{}... [truncated {} bytes]",
                &output[..self.max_output_size],
                output.len() - self.max_output_size
            );
            (truncated, true)
        }
    }
}

// Jupyter notebook cell execution
#[pyclass]
pub struct JupyterCell {
    #[pyo3(get)]
    pub cell_type: String,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl JupyterCell {
    #[new]
    fn new(cell_type: String, source: String) -> Self {
        Self {
            cell_type,
            source,
            metadata: HashMap::new(),
        }
    }

    fn execute(&self, sandbox: &PythonSandbox) -> PyResult<ExecutionResult> {
        if self.cell_type == "code" {
            sandbox.execute(self.source.clone())
        } else {
            Ok(ExecutionResult {
                stdout: self.source.clone(),
                stderr: String::new(),
                return_code: 0,
                execution_time_ms: 0,
                truncated: false,
            })
        }
    }
}

// Module registration
pub fn register_sandbox_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let sandbox_module = PyModule::new(py, "sandbox")?;
    sandbox_module.add_class::<PythonSandbox>()?;
    sandbox_module.add_class::<ExecutionResult>()?;
    sandbox_module.add_class::<JupyterCell>()?;
    parent_module.add_submodule(sandbox_module)?;
    Ok(())
}
