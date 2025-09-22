//! Rate limiting middleware

use axum::{http::Request, middleware::Next, response::Response};
use crate::AppState;

pub async fn rate_limit_middleware<B>(
    state: axum::extract::State<AppState>,
    request: Request<B>,
    next: Next<B>,
) -> Response {
    // Stub implementation
    next.run(request).await
}
