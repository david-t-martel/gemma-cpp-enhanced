//! Authentication middleware

use axum::{http::Request, middleware::Next, response::Response};

pub async fn auth_middleware<B>(request: Request<B>, next: Next<B>) -> Response {
    // Stub implementation
    next.run(request).await
}
