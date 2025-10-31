//! High-performance document processing module for Gemma extensions
//!
//! This module provides comprehensive document processing capabilities including:
//! - PDF extraction with pdf-extract crate
//! - DOCX parsing with docx-rs or zip archive parsing
//! - Markdown processing with pulldown-cmark
//! - CSV streaming with csv crate
//! - JSON streaming with serde_json
//!
//! All processing is memory-efficient with streaming support for large files
//! and includes proper error handling with PyO3 integration.

use crate::error::{GemmaError, GemmaResult};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;
use tracing::{info, warn};
use zip::ZipArchive;
use csv::ReaderBuilder;

impl From<GemmaError> for PyErr {
    fn from(err: GemmaError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError>(err.to_string())
    }
}


/// Supported document formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[pyclass]
pub enum DocumentFormat {
    Pdf,
    Docx,
    Markdown,
    Csv,
    Json,
    Text,
    Unknown,
}

#[pymethods]
impl DocumentFormat {
    #[new]
    fn new(format_str: &str) -> Self {
        match format_str.to_lowercase().as_str() {
            "pdf" => Self::Pdf,
            "docx" => Self::Docx,
            "md" | "markdown" => Self::Markdown,
            "csv" => Self::Csv,
            "json" => Self::Json,
            "txt" | "text" => Self::Text,
            _ => Self::Unknown,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("DocumentFormat::{:?}", self)
    }

    /// Detect format from file extension
    #[staticmethod]
    fn from_extension(path: &str) -> Self {
        match Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .as_deref()
        {
            Some("pdf") => Self::Pdf,
            Some("docx") => Self::Docx,
            Some("md") | Some("markdown") => Self::Markdown,
            Some("csv") => Self::Csv,
            Some("json") => Self::Json,
            Some("txt") | Some("text") => Self::Text,
            _ => Self::Unknown,
        }
    }
}

/// Document processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DocumentConfig {
    /// Maximum file size to process (in bytes)
    pub max_file_size: usize,
    /// Chunk size for streaming processing
    pub chunk_size: usize,
    /// Whether to preserve formatting
    pub preserve_formatting: bool,
    /// Whether to extract metadata
    pub extract_metadata: bool,
    /// CSV delimiter character
    pub csv_delimiter: u8,
    /// CSV quote character
    pub csv_quote: u8,
    /// Whether CSV has headers
    pub csv_has_headers: bool,
    /// Maximum number of CSV rows to process
    pub csv_max_rows: Option<usize>,
    /// JSON streaming mode
    pub json_streaming: bool,
}

#[pymethods]
impl DocumentConfig {
    #[new]
    #[pyo3(signature = (
        max_file_size = 100_000_000,
        chunk_size = 8192,
        preserve_formatting = true,
        extract_metadata = true,
        csv_delimiter = b',',
        csv_quote = b'"',
        csv_has_headers = true,
        csv_max_rows = None,
        json_streaming = true
    ))]
    fn new(
        max_file_size: usize,
        chunk_size: usize,
        preserve_formatting: bool,
        extract_metadata: bool,
        csv_delimiter: u8,
        csv_quote: u8,
        csv_has_headers: bool,
        csv_max_rows: Option<usize>,
        json_streaming: bool,
    ) -> Self {
        Self {
            max_file_size,
            chunk_size,
            preserve_formatting,
            extract_metadata,
            csv_delimiter,
            csv_quote,
            csv_has_headers,
            csv_max_rows,
            json_streaming,
        }
    }

    /// Create default configuration
    #[staticmethod]
    fn default() -> Self {
        Self::new(100_000_000, 8192, true, true, b',', b'"', true, None, true)
    }

    fn __str__(&self) -> String {
        format!("DocumentConfig(max_file_size={}, chunk_size={})",
                self.max_file_size, self.chunk_size)
    }
}

/// Document metadata extracted during processing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DocumentMetadata {
    /// File size in bytes
    pub file_size: u64,
    /// Document format
    pub format: DocumentFormat,
    /// Estimated word count
    pub word_count: usize,
    /// Character count
    pub char_count: usize,
    /// Page count (for PDFs)
    pub page_count: Option<usize>,
    /// Number of rows (for CSV)
    pub row_count: Option<usize>,
    /// Number of columns (for CSV)
    pub column_count: Option<usize>,
    /// Column names (for CSV)
    pub column_names: Option<Vec<String>>,
    /// JSON object count (for JSON)
    pub json_object_count: Option<usize>,
    /// Additional format-specific metadata
    pub extra: HashMap<String, String>,
}

#[pymethods]
impl DocumentMetadata {
    #[new]
    fn new() -> Self {
        Self {
            file_size: 0,
            format: DocumentFormat::Unknown,
            word_count: 0,
            char_count: 0,
            page_count: None,
            row_count: None,
            column_count: None,
            column_names: None,
            json_object_count: None,
            extra: HashMap::new(),
        }
    }

    fn __str__(&self) -> String {
        format!("DocumentMetadata(format={:?}, size={}, words={})",
                self.format, self.file_size, self.word_count)
    }

    /// Get extra metadata as Python dict
    fn get_extra(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in &self.extra {
            dict.set_item(key, value)?;
        }
        Ok(dict.to_object(py))
    }

    /// Set extra metadata from Python dict
    fn set_extra(&mut self, py: Python, extra: &PyDict) -> PyResult<()> {
        self.extra.clear();
        for (key, value) in extra {
            let key_str = key.extract::<String>()?;
            let value_str = value.extract::<String>()?;
            self.extra.insert(key_str, value_str);
        }
        Ok(())
    }
}

/// Document processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ProcessingResult {
    /// Extracted text content
    pub content: String,
    /// Document metadata
    pub metadata: DocumentMetadata,
    /// Processing errors or warnings
    pub warnings: Vec<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

#[pymethods]
impl ProcessingResult {
    #[new]
    fn new(content: String, metadata: DocumentMetadata) -> Self {
        Self {
            content,
            metadata,
            warnings: Vec::new(),
            processing_time_ms: 0,
        }
    }

    fn __str__(&self) -> String {
        format!("ProcessingResult(content_length={}, format={:?})",
                self.content.len(), self.metadata.format)
    }

    /// Get warnings as Python list
    fn get_warnings(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::new(py, &self.warnings);
        Ok(list.to_object(py))
    }

    /// Add a warning message
    fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }
}

/// Main document processor class
#[pyclass]
pub struct DocumentProcessor {
    config: DocumentConfig,
}

#[pymethods]
impl DocumentProcessor {
    #[new]
    fn new(config: Option<DocumentConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(DocumentConfig::default),
        }
    }

    /// Process a document from file path
    fn process_file(&self, path: &str) -> PyResult<ProcessingResult> {
        let start_time = std::time::Instant::now();

        // Check file exists and get metadata
        let file_path = Path::new(path);
        if !file_path.exists() {
            return Err(GemmaError::Io(format!("File not found: {}", path)).into());
        }

        let file_size = std::fs::metadata(file_path)
            .map_err(|e| GemmaError::Io(format!("Cannot read file metadata: {}", e)))?
            .len();

        if file_size > self.config.max_file_size as u64 {
            return Err(GemmaError::InvalidArgument(format!(
                "File too large: {} bytes > {} bytes",
                file_size, self.config.max_file_size
            )).into());
        }

        // Detect format
        let format = DocumentFormat::from_extension(path);

        // Process based on format
        let mut result = match format {
            DocumentFormat::Pdf => self.process_pdf_file(path)?,
            DocumentFormat::Docx => self.process_docx_file(path)?,
            DocumentFormat::Markdown => self.process_markdown_file(path)?,
            DocumentFormat::Csv => self.process_csv_file(path)?,
            DocumentFormat::Json => self.process_json_file(path)?,
            DocumentFormat::Text => self.process_text_file(path)?,
            DocumentFormat::Unknown => {
                // Try to detect content type by reading first few bytes
                match self.detect_and_process(path)? {
                    Some(result) => result,
                    None => return Err(GemmaError::DocumentParsing(
                        format!("Unknown document format: {}", path)
                    ).into()),
                }
            }
        };

        // Update metadata
        result.metadata.file_size = file_size;
        result.metadata.format = format;
        result.processing_time_ms = start_time.elapsed().as_millis() as u64;

        info!("Processed {} ({:?}) in {}ms", path, result.metadata.format, result.processing_time_ms);
        Ok(result)
    }

    /// Process a document from bytes
    fn process_bytes(&self, data: &PyBytes, format: DocumentFormat) -> PyResult<ProcessingResult> {
        let start_time = std::time::Instant::now();
        let bytes = data.as_bytes();

        if bytes.len() > self.config.max_file_size {
            return Err(GemmaError::InvalidArgument(format!(
                "Data too large: {} bytes > {} bytes",
                bytes.len(), self.config.max_file_size
            )).into());
        }

        let mut result = match format {
            DocumentFormat::Pdf => self.process_pdf_bytes(bytes)?,
            DocumentFormat::Docx => self.process_docx_bytes(bytes)?,
            DocumentFormat::Markdown => self.process_markdown_bytes(bytes)?,
            DocumentFormat::Csv => self.process_csv_bytes(bytes)?,
            DocumentFormat::Json => self.process_json_bytes(bytes)?,
            DocumentFormat::Text => self.process_text_bytes(bytes)?,
            DocumentFormat::Unknown => {
                return Err(GemmaError::DocumentParsing(
                    "Cannot process unknown format from bytes".to_string()
                ).into());
            }
        };

        result.metadata.file_size = bytes.len() as u64;
        result.metadata.format = format;
        result.processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Batch process multiple files
    fn process_batch(&self, paths: Vec<String>) -> PyResult<Vec<ProcessingResult>> {
        let mut results = Vec::new();

        for path in paths {
            match self.process_file(&path) {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Failed to process {}: {}", path, e);
                    let mut error_result = ProcessingResult::new(
                        String::new(),
                        DocumentMetadata::new()
                    );
                    error_result.add_warning(format!("Processing failed: {}", e));
                    results.push(error_result);
                }
            }
        }

        Ok(results)
    }

    /// Stream process large CSV files
    fn stream_csv(&self, py: Python, path: &str) -> PyResult<PyObject> {
        self.stream_csv_internal(py, path)
    }

    /// Stream process large JSON files
    fn stream_json(&self, py: Python, path: &str) -> PyResult<PyObject> {
        self.stream_json_internal(py, path)
    }

    /// Get supported formats
    #[staticmethod]
    fn supported_formats() -> Vec<String> {
        vec![
            "pdf".to_string(),
            "docx".to_string(),
            "markdown".to_string(),
            "csv".to_string(),
            "json".to_string(),
            "text".to_string(),
        ]
    }
}

impl DocumentProcessor {
    /// Process PDF file
    fn process_pdf_file(&self, path: &str) -> GemmaResult<ProcessingResult> {
        #[cfg(feature = "pdf-support")]
        {
            use pdf_extract::extract_text;

            let content = extract_text(path)
                .map_err(|e| GemmaError::DocumentParsing(format!("PDF parsing failed: {}", e)))?;

            let mut metadata = DocumentMetadata::new();
            metadata.char_count = content.len();
            metadata.word_count = count_words(&content);

            // Try to count pages by looking for form feeds or page breaks
            let page_count = content.matches('\u{000C}').count().max(1);
            metadata.page_count = Some(page_count);

            Ok(ProcessingResult::new(content, metadata))
        }
        #[cfg(not(feature = "pdf-support"))]
        {
            Err(GemmaError::NotImplemented("PDF support not enabled".to_string()))
        }
    }

    /// Process PDF from bytes
    fn process_pdf_bytes(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        #[cfg(feature = "pdf-support")]
        {
            // pdf-extract doesn't directly support byte arrays, so we'll need to write to temp file
            use std::io::Write;
            let mut temp_file = tempfile::NamedTempFile::new()
                .map_err(|e| GemmaError::Io(format!("Cannot create temp file: {}", e)))?;
            temp_file.write_all(bytes)
                .map_err(|e| GemmaError::Io(format!("Cannot write temp file: {}", e)))?;

            self.process_pdf_file(temp_file.path().to_str().unwrap())
        }
        #[cfg(not(feature = "pdf-support"))]
        {
            Err(GemmaError::NotImplemented("PDF support not enabled".to_string()))
        }
    }

    /// Process DOCX file
    fn process_docx_file(&self, path: &str) -> GemmaResult<ProcessingResult> {
        #[cfg(feature = "docx-support")]
        {
            use docx_rs::*;

            let data = std::fs::read(path)
                .map_err(|e| GemmaError::Io(format!("Cannot read DOCX file: {}", e)))?;

            self.process_docx_bytes(&data)
        }
        #[cfg(not(feature = "docx-support"))]
        {
            // Fallback to ZIP parsing
            self.process_docx_via_zip(path)
        }
    }

    /// Process DOCX from bytes
    fn process_docx_bytes(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        #[cfg(feature = "docx-support")]
        {
            use docx_rs::read_docx;

            let docx = read_docx(bytes)
                .map_err(|e| GemmaError::DocumentParsing(format!("DOCX parsing failed: {:?}", e)))?;

            let content = extract_text_from_docx(&docx);

            let mut metadata = DocumentMetadata::new();
            metadata.char_count = content.len();
            metadata.word_count = count_words(&content);

            Ok(ProcessingResult::new(content, metadata))
        }
        #[cfg(not(feature = "docx-support"))]
        {
            self.process_docx_bytes_via_zip(bytes)
        }
    }

    /// Process DOCX via ZIP parsing (fallback)
    fn process_docx_via_zip(&self, path: &str) -> GemmaResult<ProcessingResult> {
        let data = std::fs::read(path)
            .map_err(|e| GemmaError::Io(format!("Cannot read DOCX file: {}", e)))?;
        self.process_docx_bytes_via_zip(&data)
    }

    /// Process DOCX bytes via ZIP parsing
    fn process_docx_bytes_via_zip(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        use zip::ZipArchive;
        use std::io::Read;

        let cursor = Cursor::new(bytes);
        let mut archive = ZipArchive::new(cursor)
            .map_err(|e| GemmaError::DocumentParsing(format!("Invalid DOCX/ZIP file: {}", e)))?;

        // Extract document.xml which contains the main text
        let mut document_xml = String::new();
        if let Ok(mut file) = archive.by_name("word/document.xml") {
            file.read_to_string(&mut document_xml)
                .map_err(|e| GemmaError::Io(format!("Cannot read document.xml: {}", e)))?;
        } else {
            return Err(GemmaError::DocumentParsing("No document.xml found in DOCX".to_string()));
        }

        // Parse XML and extract text content
        let content = extract_text_from_xml(&document_xml)?;

        let mut metadata = DocumentMetadata::new();
        metadata.char_count = content.len();
        metadata.word_count = count_words(&content);

        Ok(ProcessingResult::new(content, metadata))
    }

    /// Process Markdown file
    fn process_markdown_file(&self, path: &str) -> GemmaResult<ProcessingResult> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| GemmaError::Io(format!("Cannot read markdown file: {}", e)))?;
        self.process_markdown_bytes(content.as_bytes())
    }

    /// Process Markdown from bytes
    fn process_markdown_bytes(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        use pulldown_cmark::{Parser, Options, html};

        let markdown = std::str::from_utf8(bytes)
            .map_err(|e| GemmaError::Utf8Error(e))?;

        let content = if self.config.preserve_formatting {
            // Keep markdown formatting
            markdown.to_string()
        } else {
            // Convert to plain text via HTML intermediate
            let mut options = Options::empty();
            options.insert(Options::ENABLE_STRIKETHROUGH);
            options.insert(Options::ENABLE_TABLES);
            options.insert(Options::ENABLE_FOOTNOTES);

            let parser = Parser::new_ext(markdown, options);
            let mut html_output = String::new();
            html::push_html(&mut html_output, parser);

            // Strip HTML tags to get plain text
            strip_html_tags(&html_output)
        };

        let mut metadata = DocumentMetadata::new();
        metadata.char_count = content.len();
        metadata.word_count = count_words(&content);

        Ok(ProcessingResult::new(content, metadata))
    }

    /// Process text file
    fn process_text_file(&self, path: &str) -> GemmaResult<ProcessingResult> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| GemmaError::Io(format!("Cannot read text file: {}", e)))?;
        self.process_text_bytes(content.as_bytes())
    }

    /// Process text from bytes
    fn process_text_bytes(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        let content = std::str::from_utf8(bytes)
            .map_err(|e| GemmaError::Utf8Error(e))?
            .to_string();

        let mut metadata = DocumentMetadata::new();
        metadata.char_count = content.len();
        metadata.word_count = count_words(&content);

        Ok(ProcessingResult::new(content, metadata))
    }

    /// Process CSV file
    fn process_csv_file(&self, path: &str) -> GemmaResult<ProcessingResult> {
        use csv::ReaderBuilder;

        let mut reader = ReaderBuilder::new()
            .delimiter(self.config.csv_delimiter)
            .quote(self.config.csv_quote)
            .has_headers(self.config.csv_has_headers)
            .from_path(path)
            .map_err(|e| GemmaError::DocumentParsing(format!("CSV parsing failed: {}", e)))?;

        self.process_csv_reader(&mut reader)
    }

    /// Process CSV from bytes
    fn process_csv_bytes(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        use csv::ReaderBuilder;

        let mut reader = ReaderBuilder::new()
            .delimiter(self.config.csv_delimiter)
            .quote(self.config.csv_quote)
            .has_headers(self.config.csv_has_headers)
            .from_reader(bytes);

        self.process_csv_reader(&mut reader)
    }

    /// Process CSV reader (common logic)
    fn process_csv_reader<R: Read>(&self, reader: &mut csv::Reader<R>) -> GemmaResult<ProcessingResult> {
        let mut content = String::new();
        let mut row_count = 0;
        let mut column_names = None;
        let mut column_count = 0;

        // Get headers if present
        if self.config.csv_has_headers {
            let headers = reader.headers()
                .map_err(|e| GemmaError::DocumentParsing(format!("Cannot read CSV headers: {}", e)))?;
            column_count = headers.len();
            column_names = Some(headers.iter().map(|s| s.to_string()).collect());

            // Add headers to content
            if self.config.preserve_formatting {
                content.push_str(&headers.iter().collect::<Vec<_>>().join(","));
                content.push('\n');
            }
        }

        // Process records
        for (i, result) in reader.records().enumerate() {
            if let Some(max_rows) = self.config.csv_max_rows {
                if i >= max_rows {
                    break;
                }
            }

            let record = result
                .map_err(|e| GemmaError::DocumentParsing(format!("CSV record error: {}", e)))?;

            if column_count == 0 {
                column_count = record.len();
            }

            if self.config.preserve_formatting {
                content.push_str(&record.iter().collect::<Vec<_>>().join(","));
                content.push('\n');
            } else {
                content.push_str(&record.iter().collect::<Vec<_>>().join(" "));
                content.push(' ');
            }

            row_count += 1;
        }

        let mut metadata = DocumentMetadata::new();
        metadata.char_count = content.len();
        metadata.word_count = count_words(&content);
        metadata.row_count = Some(row_count);
        metadata.column_count = Some(column_count);
        metadata.column_names = column_names;

        Ok(ProcessingResult::new(content, metadata))
    }

    /// Process JSON file
    fn process_json_file(&self, path: &str) -> GemmaResult<ProcessingResult> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| GemmaError::Io(format!("Cannot read JSON file: {}", e)))?;
        self.process_json_bytes(content.as_bytes())
    }

    /// Process JSON from bytes
    fn process_json_bytes(&self, bytes: &[u8]) -> GemmaResult<ProcessingResult> {
        let json_str = std::str::from_utf8(bytes)
            .map_err(|e| GemmaError::Utf8Error(e))?;

        if self.config.json_streaming {
            self.process_json_streaming(json_str)
        } else {
            self.process_json_complete(json_str)
        }
    }

    /// Process complete JSON (parse entire document)
    fn process_json_complete(&self, json_str: &str) -> GemmaResult<ProcessingResult> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| GemmaError::DocumentParsing(format!("JSON parsing failed: {}", e)))?;

        let content = if self.config.preserve_formatting {
            serde_json::to_string_pretty(&value)
                .map_err(|e| GemmaError::Serialization(format!("JSON serialization failed: {}", e)))?
        } else {
            extract_text_from_json(&value)
        };

        let object_count = count_json_objects(&value);

        let mut metadata = DocumentMetadata::new();
        metadata.char_count = content.len();
        metadata.word_count = count_words(&content);
        metadata.json_object_count = Some(object_count);

        Ok(ProcessingResult::new(content, metadata))
    }

    /// Process JSON in streaming mode (for large files)
    fn process_json_streaming(&self, json_str: &str) -> GemmaResult<ProcessingResult> {
        use serde_json::Deserializer;

        let mut content = String::new();
        let mut object_count = 0;

        let stream = Deserializer::from_str(json_str).into_iter::<serde_json::Value>();

        for value in stream {
            let value = value
                .map_err(|e| GemmaError::DocumentParsing(format!("JSON streaming error: {}", e)))?;

            if self.config.preserve_formatting {
                content.push_str(&serde_json::to_string_pretty(&value)
                    .map_err(|e| GemmaError::Serialization(format!("JSON serialization failed: {}", e)))?);
                content.push('\n');
            } else {
                content.push_str(&extract_text_from_json(&value));
                content.push(' ');
            }

            object_count += count_json_objects(&value);
        }

        let mut metadata = DocumentMetadata::new();
        metadata.char_count = content.len();
        metadata.word_count = count_words(&content);
        metadata.json_object_count = Some(object_count);

        Ok(ProcessingResult::new(content, metadata))
    }

    /// Stream CSV processing (returns iterator)
    fn stream_csv_internal(&self, _py: Python, path: &str) -> PyResult<PyObject> {
        // For now, return the full result
        // TODO: Implement proper Python iterator
        let result = self.process_csv_file(path)?;
        Ok(result.into_py(_py))
    }

    /// Stream JSON processing (returns iterator)
    fn stream_json_internal(&self, _py: Python, path: &str) -> PyResult<PyObject> {
        // For now, return the full result
        // TODO: Implement proper Python iterator
        let result = self.process_json_file(path)?;
        Ok(result.into_py(_py))
    }

    /// Detect document format and process
    fn detect_and_process(&self, path: &str) -> GemmaResult<Option<ProcessingResult>> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| GemmaError::Io(format!("Cannot open file: {}", e)))?;

        let mut buffer = [0u8; 512];
        let bytes_read = file.read(&mut buffer)
            .map_err(|e| GemmaError::Io(format!("Cannot read file: {}", e)))?;

        // PDF magic number
        if buffer.starts_with(b"%PDF") {
            return Ok(Some(self.process_pdf_file(path)?));
        }

        // ZIP/DOCX magic number
        if buffer.starts_with(b"PK\x03\x04") || buffer.starts_with(b"PK\x05\x06") {
            return Ok(Some(self.process_docx_file(path)?));
        }

        // Try to detect JSON
        let text_portion = String::from_utf8_lossy(&buffer[..bytes_read]);
        if text_portion.trim_start().starts_with('{') || text_portion.trim_start().starts_with('[') {
            return Ok(Some(self.process_json_file(path)?));
        }

        // Default to text
        Ok(Some(self.process_text_file(path)?))
    }
}

// Helper functions

/// Count words in text
fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Strip HTML tags from text
fn strip_html_tags(html: &str) -> String {
    use regex::Regex;
    let re = Regex::new(r"<[^>]*>").unwrap();
    re.replace_all(html, "").to_string()
}

/// Extract text from XML (basic implementation)
fn extract_text_from_xml(xml: &str) -> GemmaResult<String> {
    use regex::Regex;

    // Remove XML tags and extract text content
    let re = Regex::new(r"<[^>]*>").unwrap();
    let text = re.replace_all(xml, " ");

    // Clean up whitespace
    let re_whitespace = Regex::new(r"\s+").unwrap();
    let cleaned = re_whitespace.replace_all(&text, " ");

    Ok(cleaned.trim().to_string())
}

/// Extract text content from JSON value
fn extract_text_from_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            arr.iter()
                .map(extract_text_from_json)
                .collect::<Vec<_>>()
                .join(" ")
        }
        serde_json::Value::Object(obj) => {
            obj.values()
                .map(extract_text_from_json)
                .collect::<Vec<_>>()
                .join(" ")
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
    }
}

/// Count JSON objects recursively
fn count_json_objects(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(obj) => {
            1 + obj.values().map(count_json_objects).sum::<usize>()
        }
        serde_json::Value::Array(arr) => {
            arr.iter().map(count_json_objects).sum()
        }
        _ => 0,
    }
}

#[cfg(feature = "docx-support")]
fn extract_text_from_docx(docx: &docx_rs::Docx) -> String {
    // Extract text from DOCX document
    // This is a simplified implementation for the 0.4.x API
    docx.document
        .children
        .iter()
        .filter_map(|child| {
            match child {
                docx_rs::DocumentChild::Paragraph(p) => {
                    Some(p.children.iter()
                        .filter_map(|run_child| {
                            match run_child {
                                docx_rs::ParagraphChild::Run(run) => {
                                    Some(run.children.iter()
                                        .filter_map(|run_child| {
                                            match run_child {
                                                docx_rs::RunChild::Text(text) => Some(text.text.clone()),
                                                _ => None,
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join(""))
                                }
                                _ => None,
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" "))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Register the document processor module with Python
pub fn register_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "document_processor")?;

    module.add_class::<DocumentFormat>()?;
    module.add_class::<DocumentConfig>()?;
    module.add_class::<DocumentMetadata>()?;
    module.add_class::<ProcessingResult>()?;
    module.add_class::<DocumentProcessor>()?;

    parent_module.add_submodule(module)?;
    Ok(())
}

/// Standalone functions for Python binding

/// Process a single document file
#[pyfunction]
pub fn process_document(path: &str, config: Option<DocumentConfig>) -> PyResult<ProcessingResult> {
    let processor = DocumentProcessor::new(config);
    processor.process_file(path)
}

/// Batch process multiple documents
#[pyfunction]
pub fn process_documents_batch(paths: Vec<String>, config: Option<DocumentConfig>) -> PyResult<Vec<ProcessingResult>> {
    let processor = DocumentProcessor::new(config);
    processor.process_batch(paths)
}

/// Detect document format from file path
#[pyfunction]
pub fn detect_format(path: &str) -> DocumentFormat {
    DocumentFormat::from_extension(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_format_detection() {
        assert_eq!(DocumentFormat::from_extension("test.pdf"), DocumentFormat::Pdf);
        assert_eq!(DocumentFormat::from_extension("test.docx"), DocumentFormat::Docx);
        assert_eq!(DocumentFormat::from_extension("test.md"), DocumentFormat::Markdown);
        assert_eq!(DocumentFormat::from_extension("test.csv"), DocumentFormat::Csv);
        assert_eq!(DocumentFormat::from_extension("test.json"), DocumentFormat::Json);
        assert_eq!(DocumentFormat::from_extension("test.txt"), DocumentFormat::Text);
        assert_eq!(DocumentFormat::from_extension("test.unknown"), DocumentFormat::Unknown);
    }

    #[test]
    fn test_text_processing() {
        let processor = DocumentProcessor::new(None);
        let content = "Hello, world! This is a test document.";
        let result = processor.process_text_bytes(content.as_bytes()).unwrap();

        assert_eq!(result.content, content);
        assert_eq!(result.metadata.char_count, content.len());
        assert_eq!(result.metadata.word_count, 8);
        assert_eq!(result.metadata.format, DocumentFormat::Unknown); // Will be set by caller
    }

    #[test]
    fn test_csv_processing() {
        let processor = DocumentProcessor::new(None);
        let csv_content = "name,age,city\nJohn,30,NYC\nJane,25,LA\n";
        let result = processor.process_csv_bytes(csv_content.as_bytes()).unwrap();

        assert!(result.content.contains("name"));
        assert!(result.content.contains("John"));
        assert_eq!(result.metadata.row_count, Some(2));
        assert_eq!(result.metadata.column_count, Some(3));
        assert_eq!(result.metadata.column_names, Some(vec!["name".to_string(), "age".to_string(), "city".to_string()]));
    }

    #[test]
    fn test_json_processing() {
        let processor = DocumentProcessor::new(None);
        let json_content = r#"{"name": "John", "age": 30, "city": "NYC"}"#;
        let result = processor.process_json_bytes(json_content.as_bytes()).unwrap();

        assert!(result.content.contains("John"));
        assert!(result.content.contains("NYC"));
        assert_eq!(result.metadata.json_object_count, Some(1));
    }

    #[test]
    fn test_markdown_processing() {
        let processor = DocumentProcessor::new(None);
        let md_content = "# Hello\n\nThis is **bold** text.";
        let result = processor.process_markdown_bytes(md_content.as_bytes()).unwrap();

        // In preserve formatting mode, should keep markdown
        assert!(result.content.contains("# Hello"));
        assert!(result.content.contains("**bold**"));
    }

    #[test]
    fn test_file_processing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "Hello, world! This is a test.").unwrap();

        let processor = DocumentProcessor::new(None);
        let result = processor.process_file(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(result.content, "Hello, world! This is a test.");
        assert!(result.metadata.file_size > 0);
        assert!(result.processing_time_ms >= 0);
    }

    #[test]
    fn test_word_counting() {
        assert_eq!(count_words("Hello world"), 2);
        assert_eq!(count_words("  Hello    world  "), 2);
        assert_eq!(count_words(""), 0);
        assert_eq!(count_words("Single"), 1);
    }

    #[test]
    fn test_json_object_counting() {
        let json: serde_json::Value = serde_json::from_str(r#"
            {
                "user": {"name": "John", "age": 30},
                "items": [{"id": 1}, {"id": 2}]
            }
        "#).unwrap();

        assert_eq!(count_json_objects(&json), 4); // root + user + 2 items
    }
}