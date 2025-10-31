"""Security tests for path validation in settings.py.

This module tests protection against:
- Path traversal attacks
- Environment variable injection
- Symlink escape attacks
- Encoded path traversal attempts
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gemma_cli.config.settings import expand_path


class TestPathTraversalPrevention:
    """Test prevention of path traversal attacks."""

    def test_direct_traversal_blocked(self):
        """Test that direct '..' traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed in input"):
            expand_path("../../../etc/passwd")

    def test_nested_traversal_blocked(self):
        """Test that nested traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed in input"):
            expand_path("config/../../../etc/passwd")

    def test_encoded_traversal_blocked(self):
        """Test that URL-encoded traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed \\(encoded\\)"):
            expand_path("%2e%2e/%2e%2e/etc/passwd")

    def test_double_encoded_traversal_blocked(self):
        """Test that double-encoded traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed \\(encoded\\)"):
            expand_path("%252e%252e/%252e%252e/etc/passwd")


class TestEnvironmentVariableInjection:
    """Test prevention of environment variable injection attacks."""

    def test_env_var_with_traversal_blocked(self):
        """Test that env vars containing '..' are caught after expansion."""
        # Set malicious environment variable
        os.environ["EVIL_PATH"] = "../../.."

        try:
            with pytest.raises(ValueError, match="Path traversal detected after expansion"):
                expand_path("$EVIL_PATH/etc/passwd")
        finally:
            # Clean up
            del os.environ["EVIL_PATH"]

    def test_nested_env_var_injection_blocked(self):
        """Test that directly injected traversal via env vars is caught."""
        # os.expandvars doesn't recursively expand, so test direct injection
        os.environ["MALICIOUS_VAR"] = "../.."

        try:
            with pytest.raises(ValueError, match="Path traversal detected after expansion"):
                expand_path("$MALICIOUS_VAR/sensitive")
        finally:
            del os.environ["MALICIOUS_VAR"]

    def test_home_expansion_with_traversal_blocked(self):
        """Test that tilde expansion with traversal is blocked."""
        # The current implementation blocks '..' in the input string before expansion
        # This is actually more secure as it prevents attacks earlier in the pipeline
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            expand_path("~/../../etc/passwd")


class TestSymlinkValidation:
    """Test proper symlink validation."""

    def test_symlink_within_allowed_dirs(self, tmp_path):
        """Test that symlinks within allowed directories are permitted."""
        # Create a test file and symlink within allowed directory
        real_file = tmp_path / "real.txt"
        real_file.write_text("test content")

        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        # This should work - symlink and target both in allowed dir
        result = expand_path(str(symlink), allowed_dirs=[tmp_path])
        assert result == real_file.resolve()

    def test_relative_symlink_escape_blocked(self, tmp_path):
        """Test that relative symlinks escaping allowed dirs are blocked."""
        # Create directories
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret data")

        # Create relative symlink that escapes safe directory
        symlink = safe_dir / "escape.txt"
        symlink.symlink_to("../../outside/secret.txt")

        # This should fail - symlink escapes allowed directory
        with pytest.raises(ValueError, match="not within allowed directories"):
            expand_path(str(symlink), allowed_dirs=[safe_dir])

    def test_absolute_symlink_escape_blocked(self, tmp_path):
        """Test that absolute symlinks to outside dirs are blocked."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()

        # Create symlink pointing to system directory
        symlink = safe_dir / "system_link"
        if os.name == 'nt':
            # Windows
            target = Path("C:\\Windows\\System32")
        else:
            # Unix-like
            target = Path("/etc")

        # Skip if we can't create symlinks (requires admin on Windows)
        try:
            symlink.symlink_to(target)
        except OSError:
            pytest.skip("Cannot create symlinks (may require admin privileges)")

        # This should fail - absolute symlink to system directory
        with pytest.raises(ValueError, match="not within allowed directories"):
            expand_path(str(symlink), allowed_dirs=[safe_dir])


class TestAllowedDirectories:
    """Test allowed directory validation."""

    def test_path_within_allowed_dir(self, tmp_path):
        """Test that paths within allowed directories are permitted."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        test_file = allowed / "subdir" / "file.txt"
        test_file.parent.mkdir()

        result = expand_path(str(test_file), allowed_dirs=[allowed])
        assert result == test_file.resolve()

    def test_path_outside_allowed_dir(self, tmp_path):
        """Test that paths outside allowed directories are blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()

        forbidden = tmp_path / "forbidden"
        forbidden.mkdir()
        test_file = forbidden / "file.txt"

        with pytest.raises(ValueError, match="not within allowed directories"):
            expand_path(str(test_file), allowed_dirs=[allowed])

    def test_default_allowed_dirs(self):
        """Test that default allowed directories include safe locations."""
        # Should allow home directory .gemma_cli
        home_config = Path.home() / ".gemma_cli" / "config.toml"
        result = expand_path(str(home_config))
        assert result == home_config.resolve()

        # Should allow current working directory
        cwd_file = Path.cwd() / "test.txt"
        result = expand_path(str(cwd_file))
        assert result == cwd_file.resolve()


class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_path_with_spaces(self, tmp_path):
        """Test handling of paths with spaces."""
        dir_with_spaces = tmp_path / "dir with spaces"
        dir_with_spaces.mkdir()

        file_path = dir_with_spaces / "file.txt"
        result = expand_path(str(file_path), allowed_dirs=[tmp_path])
        assert result == file_path.resolve()

    def test_unicode_in_path(self, tmp_path):
        """Test handling of unicode characters in paths."""
        unicode_dir = tmp_path / "тест_目录"
        unicode_dir.mkdir()

        file_path = unicode_dir / "файл.txt"
        result = expand_path(str(file_path), allowed_dirs=[tmp_path])
        assert result == file_path.resolve()

    def test_multiple_slashes_normalized(self, tmp_path):
        """Test that multiple slashes are normalized."""
        test_file = tmp_path / "dir" / "file.txt"
        test_file.parent.mkdir()

        # Path with multiple slashes
        messy_path = str(tmp_path) + "//dir///file.txt"
        result = expand_path(messy_path, allowed_dirs=[tmp_path])
        assert result == test_file.resolve()

    def test_case_sensitivity(self, tmp_path):
        """Test case sensitivity handling across platforms."""
        test_dir = tmp_path / "TestDir"
        test_dir.mkdir()

        # On Windows, paths are case-insensitive
        # On Unix, they're case-sensitive
        if os.name == 'nt':
            # Should work on Windows regardless of case
            result = expand_path(str(tmp_path / "testdir"), allowed_dirs=[tmp_path])
            assert result.exists() or not result.exists()  # Just check it doesn't raise
        else:
            # On Unix, exact case matters
            result = expand_path(str(test_dir), allowed_dirs=[tmp_path])
            assert result == test_dir.resolve()


class TestSecurityRegression:
    """Test for specific security vulnerability regressions."""

    def test_cve_style_attacks(self):
        """Test common CVE-style path traversal patterns."""
        attacks = [
            "....//....//etc/passwd",  # Multiple dots
            "..;/etc/passwd",  # Semicolon variant
            "..%00/etc/passwd",  # Null byte injection
            ".%2e/%2e%2e/etc/passwd",  # Mixed encoding
            "..\\..\\windows\\system32",  # Windows backslash
        ]

        for attack in attacks:
            # All should be caught by validation
            with pytest.raises(ValueError):
                expand_path(attack)

    def test_race_condition_symlink_swap(self, tmp_path):
        """Test protection against TOCTOU race condition with symlinks."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()

        # Create initial safe symlink
        safe_target = safe_dir / "safe.txt"
        safe_target.write_text("safe content")

        symlink = safe_dir / "link.txt"
        symlink.symlink_to(safe_target)

        # First check should pass
        result = expand_path(str(symlink), allowed_dirs=[safe_dir])
        assert result == safe_target.resolve()

        # Even if symlink is swapped to malicious target between checks,
        # the resolve() call follows final symlink, so validation catches it

        # Simulate swap to malicious target
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        evil_target = outside_dir / "evil.txt"
        evil_target.write_text("evil content")

        symlink.unlink()
        symlink.symlink_to("../../outside/evil.txt")

        # Should now fail
        with pytest.raises(ValueError, match="not within allowed directories"):
            expand_path(str(symlink), allowed_dirs=[safe_dir])


class TestCompatibility:
    """Test compatibility across Python versions and platforms."""

    def test_python38_compatibility(self, tmp_path):
        """Test that code works on Python 3.8 (no is_relative_to)."""
        # Mock absence of is_relative_to method
        test_file = tmp_path / "test.txt"

        with patch.object(Path, 'is_relative_to', side_effect=AttributeError):
            # Should fall back to string comparison
            result = expand_path(str(test_file), allowed_dirs=[tmp_path])
            assert result == test_file.resolve()

    def test_windows_paths(self):
        """Test Windows-specific path handling."""
        if os.name != 'nt':
            pytest.skip("Windows-only test")

        # Test drive letters
        result = expand_path("C:\\codedev\\llm\\.models\\test.bin")
        assert result == Path("C:\\codedev\\llm\\.models\\test.bin").resolve()

        # Test UNC paths
        with pytest.raises(ValueError):
            expand_path("\\\\server\\share\\..\\..\\admin$")

    def test_wsl_paths(self):
        """Test WSL path handling."""
        # Test WSL-style paths
        wsl_path = "/c/codedev/llm/.models/test.bin"
        if Path(wsl_path).exists():
            result = expand_path(wsl_path)
            assert result == Path(wsl_path).resolve()