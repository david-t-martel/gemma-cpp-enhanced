# Security Guidelines

## Private Key Management

### ❌ NEVER Commit Real Private Keys

Private keys, service account credentials, and API tokens must NEVER be committed to the repository, even in:
- Test files
- Example scripts
- Archived code
- Documentation

### ✅ Use Environment Variables

Store sensitive credentials in environment variables or `.env` files (which should be in `.gitignore`):

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Good - load from environment
api_key = os.getenv("API_KEY")
private_key = os.getenv("PRIVATE_KEY")
```

### ✅ Use Mock Values in Tests

When testing code that requires credentials, use clearly marked mock/placeholder values:

```python
# Good - clearly marked as mock data
MOCK_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\n[REDACTED - MOCK KEY CONTENT]\n-----END PRIVATE KEY-----"
PLACEHOLDER_API_KEY = "PLACEHOLDER_API_KEY_REPLACE_WITH_ACTUAL"
```

### ✅ Use Placeholder Templates

For configuration templates and examples, use obvious placeholders:

```json
{
  "private_key_id": "PLACEHOLDER_KEY_ID",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n[PLACEHOLDER - REPLACE WITH ACTUAL KEY DATA]\n-----END RSA PRIVATE KEY-----"
}
```

## Pre-commit Hooks

The project uses pre-commit hooks to detect and prevent accidental credential commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Credential Rotation

If credentials are accidentally exposed:

1. **Immediately rotate** the exposed credentials
2. **Revoke** the old credentials
3. **Audit** access logs for unauthorized usage
4. **Update** all systems using the credentials

## Environment Configuration

### Development
- Use `.env` file (gitignored)
- Load with `python-dotenv`
- Never hardcode credentials

### Production
- Use secret management services:
  - Google Secret Manager
  - AWS Secrets Manager
  - Azure Key Vault
  - HashiCorp Vault

### CI/CD
- Use GitHub Secrets or equivalent
- Limit secret access to necessary workflows
- Rotate CI/CD credentials regularly

## File Permissions

Ensure sensitive files have restricted permissions:

```bash
# Unix/Linux/Mac
chmod 600 ~/.config/gcp-service-account.json
chmod 600 .env

# Windows (PowerShell as Admin)
icacls "config\gcp-service-account.json" /inheritance:r /grant:r "%USERNAME%:F"
```

## Scanning for Secrets

Regular scanning for accidentally committed secrets:

```bash
# Using detect-secrets
detect-secrets scan --baseline .secrets.baseline

# Using git-secrets
git secrets --scan

# Using truffleHog
trufflehog git file://./
```

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do NOT** create a public issue
2. Contact the maintainers privately
3. Allow time for a fix before disclosure

## Compliance

This project follows security best practices:

- OWASP Top 10 awareness
- Principle of least privilege
- Defense in depth
- Regular dependency updates
- Security-first code reviews
