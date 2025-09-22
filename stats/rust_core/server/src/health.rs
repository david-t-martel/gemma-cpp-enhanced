//! Health check implementations

use serde::Serialize;

#[derive(Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub checks: Vec<HealthCheck>,
}

#[derive(Serialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
    pub message: Option<String>,
}

pub fn get_health_status() -> HealthStatus {
    HealthStatus {
        status: "healthy".to_string(),
        checks: vec![
            HealthCheck {
                name: "inference_engine".to_string(),
                status: "healthy".to_string(),
                message: None,
            }
        ],
    }
}
