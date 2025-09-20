//! Server configuration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub server: ServerSettings,
    pub model: ModelSettings,
    pub inference: InferenceSettings,
    pub security: SecuritySettings,
    pub logging: LoggingSettings,
    pub enable_metrics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub timeout_seconds: u64,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    pub path: String,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSettings {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    pub rate_limit_enabled: bool,
    pub max_requests_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    pub level: String,
    pub format: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings {
                host: "127.0.0.1".to_string(),
                port: 8080,
                timeout_seconds: 30,
                max_connections: 1000,
            },
            model: ModelSettings {
                path: "model.bin".to_string(),
                format: "safetensors".to_string(),
            },
            inference: InferenceSettings {
                max_batch_size: 8,
                max_sequence_length: 2048,
            },
            security: SecuritySettings {
                rate_limit_enabled: true,
                max_requests_per_minute: 100,
            },
            logging: LoggingSettings {
                level: "info".to_string(),
                format: "json".to_string(),
            },
            enable_metrics: true,
        }
    }
}
