"""Security headers middleware for enhanced protection.

This module provides comprehensive security headers following OWASP best practices.
"""

from typing import Dict
from typing import Optional

from fastapi import FastAPI
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    def __init__(self, app, csp_directives: dict[str, str] | None = None):
        """Initialize the security headers middleware.

        Args:
            app: The FastAPI application
            csp_directives: Custom Content-Security-Policy directives
        """
        super().__init__(app)
        self.csp_directives = csp_directives or self._get_default_csp()

    def _get_default_csp(self) -> dict[str, str]:
        """Get default Content-Security-Policy directives.

        Returns:
            Dictionary of CSP directives
        """
        return {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline' 'unsafe-eval'",  # Restrict in production
            "style-src": "'self' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' data:",
            "connect-src": "'self'",
            "media-src": "'self'",
            "object-src": "'none'",
            "frame-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "frame-ancestors": "'none'",
            "upgrade-insecure-requests": "",
        }

    def _build_csp_header(self) -> str:
        """Build the Content-Security-Policy header value.

        Returns:
            CSP header string
        """
        directives = []
        for key, value in self.csp_directives.items():
            if value:
                directives.append(f"{key} {value}")
            else:
                directives.append(key)
        return "; ".join(directives)

    async def dispatch(self, request: Request, call_next):
        """Add security headers to the response.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response with security headers
        """
        response = await call_next(request)

        # Content-Security-Policy
        response.headers["Content-Security-Policy"] = self._build_csp_header()

        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options
        response.headers["X-Frame-Options"] = "DENY"

        # X-XSS-Protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        )

        # Strict-Transport-Security (HSTS) - only for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Remove server header for security
        if "server" in response.headers:
            del response.headers["server"]

        # Add custom security header
        response.headers["X-Security-Policy"] = "strict"

        return response


def add_security_headers(
    app: FastAPI, csp_directives: dict[str, str] | None = None, production_mode: bool = False
) -> None:
    """Add security headers middleware to the application.

    Args:
        app: The FastAPI application
        csp_directives: Custom CSP directives
        production_mode: Whether running in production mode
    """
    if production_mode:
        # Stricter CSP for production
        csp = {
            "default-src": "'self'",
            "script-src": "'self'",  # No unsafe-inline in production
            "style-src": "'self'",
            "img-src": "'self' data: https:",
            "font-src": "'self'",
            "connect-src": "'self'",
            "media-src": "'none'",
            "object-src": "'none'",
            "frame-src": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "frame-ancestors": "'none'",
            "upgrade-insecure-requests": "",
            "block-all-mixed-content": "",
        }
    else:
        csp = csp_directives

    app.add_middleware(SecurityHeadersMiddleware, csp_directives=csp)


# OWASP Security Headers Checklist
SECURITY_HEADERS_CHECKLIST = {
    "Content-Security-Policy": "Prevents XSS, clickjacking, and other code injection attacks",
    "X-Content-Type-Options": "Prevents MIME type sniffing",
    "X-Frame-Options": "Prevents clickjacking attacks",
    "X-XSS-Protection": "Legacy XSS protection for older browsers",
    "Referrer-Policy": "Controls information sent in Referer header",
    "Permissions-Policy": "Controls browser features and APIs",
    "Strict-Transport-Security": "Forces HTTPS connections",
}


def get_security_headers_report(headers: dict[str, str]) -> dict[str, bool]:
    """Generate a security headers compliance report.

    Args:
        headers: Response headers dictionary

    Returns:
        Dictionary with header compliance status
    """
    report = {}
    for header in SECURITY_HEADERS_CHECKLIST:
        report[header] = header.lower() in [h.lower() for h in headers]
    return report
