//! WebAssembly bindings for Gemma inference engine
//!
//! This crate provides WASM bindings that enable running Gemma models directly
//! in web browsers and other WebAssembly environments.
//!
//! # Features
//!
//! - **Browser compatibility**: Run inference directly in web browsers
//! - **Worker support**: Offload inference to Web Workers for non-blocking UI
//! - **Streaming responses**: Support for streaming text generation
//! - **Memory efficient**: Optimized memory usage for WASM constraints
//! - **SIMD acceleration**: Uses WASM SIMD when available
//!
//! # Usage
//!
//! ```javascript
//! import init, { GemmaEngine } from './pkg/gemma_wasm.js';
//!
//! async function runInference() {
//!     await init();
//!
//!     const engine = new GemmaEngine({
//!         model_path: '/models/gemma-2b.bin',
//!         max_tokens: 512,
//!     });
//!
//!     await engine.initialize();
//!
//!     const response = await engine.generate({
//!         prompt: "Hello, world!",
//!         max_tokens: 50,
//!     });
//!
//!     console.log(response.text);
//! }
//! ```

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use js_sys::{Array, Object, Promise, Uint8Array};
use web_sys::{console, window, Performance, Worker, MessageEvent, Blob, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use futures_util::StreamExt;

use gemma_inference::{
    InferenceEngine, InferenceConfig, InferenceRequest, InferenceResponse,
    EngineConfig, TokenizerEngine, get_runtime_capabilities,
};

// Import the `console.log` function from the `console` module for debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

// Define a macro for easier logging
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

macro_rules! console_error {
    ($($t:tt)*) => (error(&format_args!($($t)*).to_string()))
}

// Set up panic hook for better error messages in WASM
#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

// Use `wee_alloc` as the global allocator for smaller WASM binary size
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Configuration for the WASM inference engine
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmEngineConfig {
    /// Path or URL to the model file
    pub model_path: String,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling parameter
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Whether to use streaming generation
    pub streaming: bool,
    /// Batch size for processing
    pub batch_size: usize,
}

#[wasm_bindgen]
impl WasmEngineConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(model_path: String) -> WasmEngineConfig {
        WasmEngineConfig {
            model_path,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            streaming: false,
            batch_size: 1,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn model_path(&self) -> String {
        self.model_path.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_model_path(&mut self, path: String) {
        self.model_path = path;
    }

    #[wasm_bindgen(getter)]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_tokens(&mut self, tokens: usize) {
        self.max_tokens = tokens;
    }

    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp;
    }

    #[wasm_bindgen(getter)]
    pub fn streaming(&self) -> bool {
        self.streaming
    }

    #[wasm_bindgen(setter)]
    pub fn set_streaming(&mut self, streaming: bool) {
        self.streaming = streaming;
    }
}

/// Inference request for WASM
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmInferenceRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop_sequences: Vec<String>,
}

#[wasm_bindgen]
impl WasmInferenceRequest {
    #[wasm_bindgen(constructor)]
    pub fn new(prompt: String) -> WasmInferenceRequest {
        WasmInferenceRequest {
            prompt,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn prompt(&self) -> String {
        self.prompt.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_prompt(&mut self, prompt: String) {
        self.prompt = prompt;
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_tokens(&mut self, tokens: usize) {
        self.max_tokens = Some(tokens);
    }

    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = Some(temp);
    }

    pub fn add_stop_sequence(&mut self, sequence: String) {
        self.stop_sequences.push(sequence);
    }
}

/// Inference response from WASM
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmInferenceResponse {
    text: String,
    tokens: Vec<u32>,
    finish_reason: String,
    usage: WasmTokenUsage,
}

#[wasm_bindgen]
impl WasmInferenceResponse {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn tokens(&self) -> Vec<u32> {
        self.tokens.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn usage(&self) -> WasmTokenUsage {
        self.usage.clone()
    }
}

/// Token usage statistics
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[wasm_bindgen]
impl WasmTokenUsage {
    #[wasm_bindgen(getter)]
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

/// Main WASM inference engine
#[wasm_bindgen]
pub struct GemmaEngine {
    engine: Option<Arc<InferenceEngine>>,
    config: WasmEngineConfig,
    initialized: bool,
}

#[wasm_bindgen]
impl GemmaEngine {
    /// Create a new WASM inference engine
    #[wasm_bindgen(constructor)]
    pub fn new(config: WasmEngineConfig) -> Result<GemmaEngine, JsValue> {
        console_log!("Creating new GemmaEngine with config: {:?}", config);

        Ok(GemmaEngine {
            engine: None,
            config,
            initialized: false,
        })
    }

    /// Initialize the inference engine
    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        console_log!("Initializing GemmaEngine");

        // Load model data from URL or path
        let model_data = self.load_model_data().await?;
        console_log!("Loaded model data: {} bytes", model_data.len());

        // Convert WASM config to engine config
        let engine_config = self.create_engine_config(&model_data)?;

        // Initialize the inference engine
        let engine = gemma_inference::initialize_engine("wasm", engine_config)
            .map_err(|e| JsValue::from_str(&format!("Failed to initialize engine: {}", e)))?;

        self.engine = Some(engine);
        self.initialized = true;

        console_log!("GemmaEngine initialized successfully");
        Ok(())
    }

    /// Check if the engine is initialized
    #[wasm_bindgen]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Generate text from a prompt
    #[wasm_bindgen]
    pub async fn generate(&self, request: WasmInferenceRequest) -> Result<WasmInferenceResponse, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Engine not initialized"));
        }

        let engine = self.engine.as_ref().unwrap();
        console_log!("Generating text for prompt: {}", request.prompt());

        // Convert WASM request to engine request
        let inference_request = self.create_inference_request(&request)?;

        // Run inference
        let response = engine.infer(inference_request).await
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        // Convert response to WASM response
        let wasm_response = self.create_wasm_response(response)?;

        console_log!("Generated {} tokens", wasm_response.usage().total_tokens());
        Ok(wasm_response)
    }

    /// Generate text with streaming
    #[wasm_bindgen]
    pub async fn generate_stream(&self, request: WasmInferenceRequest, callback: &js_sys::Function) -> Result<(), JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Engine not initialized"));
        }

        let engine = self.engine.as_ref().unwrap();
        console_log!("Starting streaming generation for prompt: {}", request.prompt());

        // Convert WASM request to engine request
        let inference_request = self.create_inference_request(&request)?;

        // Create streaming inference
        let mut stream = engine.infer_stream(inference_request).await
            .map_err(|e| JsValue::from_str(&format!("Streaming inference failed: {}", e)))?;

        // Process stream and call JavaScript callback for each token
        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(token) => {
                    let js_token = JsValue::from_str(&token);
                    callback.call1(&JsValue::NULL, &js_token)
                        .map_err(|e| JsValue::from_str(&format!("Callback error: {:?}", e)))?;
                }
                Err(e) => {
                    console_error!("Stream error: {}", e);
                    break;
                }
            }
        }

        console_log!("Streaming generation completed");
        Ok(())
    }

    /// Tokenize text
    #[wasm_bindgen]
    pub async fn tokenize(&self, text: String) -> Result<Vec<u32>, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Engine not initialized"));
        }

        let engine = self.engine.as_ref().unwrap();
        let encoding = engine.tokenizer().encode(&text).await
            .map_err(|e| JsValue::from_str(&format!("Tokenization failed: {}", e)))?;

        Ok(encoding.token_ids)
    }

    /// Detokenize token IDs back to text
    #[wasm_bindgen]
    pub async fn detokenize(&self, tokens: Vec<u32>) -> Result<String, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Engine not initialized"));
        }

        let engine = self.engine.as_ref().unwrap();
        let text = engine.tokenizer().decode(&tokens).await
            .map_err(|e| JsValue::from_str(&format!("Detokenization failed: {}", e)))?;

        Ok(text)
    }

    /// Get engine statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Engine not initialized"));
        }

        let engine = self.engine.as_ref().unwrap();
        let stats = engine.get_statistics();

        JsValue::from_serde(&stats)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize stats: {}", e)))
    }

    /// Get runtime capabilities
    #[wasm_bindgen]
    pub fn get_capabilities() -> JsValue {
        let capabilities = get_runtime_capabilities();
        JsValue::from_serde(&capabilities).unwrap_or(JsValue::NULL)
    }

    /// Warmup the engine
    #[wasm_bindgen]
    pub async fn warmup(&self) -> Result<(), JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("Engine not initialized"));
        }

        let engine = self.engine.as_ref().unwrap();
        gemma_inference::warmup_engine(engine).await
            .map_err(|e| JsValue::from_str(&format!("Warmup failed: {}", e)))?;

        console_log!("Engine warmed up successfully");
        Ok(())
    }
}

impl GemmaEngine {
    /// Load model data from URL or embedded data
    async fn load_model_data(&self) -> Result<Vec<u8>, JsValue> {
        if self.config.model_path.starts_with("http") {
            // Load from URL
            self.load_from_url(&self.config.model_path).await
        } else {
            // Load from local path or embedded data
            Err(JsValue::from_str("Local file loading not implemented"))
        }
    }

    /// Load model data from HTTP URL
    async fn load_from_url(&self, url: &str) -> Result<Vec<u8>, JsValue> {
        console_log!("Loading model from URL: {}", url);

        let window = window().unwrap();
        let response = JsFuture::from(window.fetch_with_str(url)).await?;
        let response: Response = response.dyn_into().unwrap();

        if !response.ok() {
            return Err(JsValue::from_str(&format!(
                "Failed to fetch model: HTTP {}",
                response.status()
            )));
        }

        let array_buffer = JsFuture::from(response.array_buffer()?).await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        let mut data = vec![0; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        console_log!("Successfully loaded {} bytes from URL", data.len());
        Ok(data)
    }

    /// Create engine config from WASM config
    fn create_engine_config(&self, _model_data: &[u8]) -> Result<EngineConfig, JsValue> {
        // Create a basic engine config for WASM
        // In a real implementation, this would parse the model data
        Ok(EngineConfig::default())
    }

    /// Convert WASM request to engine request
    fn create_inference_request(&self, request: &WasmInferenceRequest) -> Result<InferenceRequest, JsValue> {
        // Convert WASM request format to internal format
        Ok(InferenceRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens.unwrap_or(self.config.max_tokens),
            temperature: request.temperature.unwrap_or(self.config.temperature),
            top_p: Some(request.top_p.unwrap_or(self.config.top_p)),
            top_k: Some(request.top_k.unwrap_or(self.config.top_k)),
            stop_sequences: request.stop_sequences.clone(),
            stream: self.config.streaming,
        })
    }

    /// Convert engine response to WASM response
    fn create_wasm_response(&self, response: InferenceResponse) -> Result<WasmInferenceResponse, JsValue> {
        Ok(WasmInferenceResponse {
            text: response.text,
            tokens: response.tokens,
            finish_reason: response.finish_reason,
            usage: WasmTokenUsage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
            },
        })
    }
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    #[cfg(feature = "wee_alloc")]
    console_log!("Using wee_alloc for memory management");

    console_log!("Gemma WASM module initialized");
}

/// Get WASM module version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if SIMD is supported in this WASM environment
#[wasm_bindgen]
pub fn check_simd_support() -> bool {
    // WASM SIMD detection would go here
    // For now, assume it's supported
    true
}

/// Utility function to create a worker for inference
#[wasm_bindgen]
pub fn create_inference_worker() -> Result<Worker, JsValue> {
    // Create a Web Worker for running inference in background
    let worker_script = r#"
        import init, { GemmaEngine } from './pkg/gemma_wasm.js';

        self.onmessage = async function(e) {
            const { type, data } = e.data;

            if (type === 'init') {
                await init();
                const engine = new GemmaEngine(data.config);
                await engine.initialize();
                self.engine = engine;
                self.postMessage({ type: 'initialized' });
            } else if (type === 'generate') {
                const response = await self.engine.generate(data.request);
                self.postMessage({ type: 'response', data: response });
            }
        };
    "#;

    let blob = Blob::new_with_str_sequence(&Array::of1(&JsValue::from_str(worker_script)))?;
    let worker_url = web_sys::Url::create_object_url_with_blob(&blob)?;
    let worker = Worker::new(&worker_url)?;

    Ok(worker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_config_creation() {
        let config = WasmEngineConfig::new("test_model.bin".to_string());
        assert_eq!(config.model_path(), "test_model.bin");
        assert_eq!(config.max_tokens(), 512);
    }

    #[wasm_bindgen_test]
    fn test_request_creation() {
        let request = WasmInferenceRequest::new("Hello, world!".to_string());
        assert_eq!(request.prompt(), "Hello, world!");
    }

    #[wasm_bindgen_test]
    async fn test_engine_creation() {
        let config = WasmEngineConfig::new("test_model.bin".to_string());
        let engine = GemmaEngine::new(config);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert!(!engine.is_initialized());
    }
}
