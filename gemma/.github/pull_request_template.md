<<<<<<< HEAD
# Pull Request

## Summary

<!-- Brief description of what this PR does -->

## Type of Change

<!-- Please delete options that are not relevant -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test addition or improvement
- [ ] 🔒 Security improvement
- [ ] 🏗️ Infrastructure/CI/CD changes

## Changes Made

<!-- Detailed description of the changes -->

## Testing

<!-- Describe the tests you ran to verify your changes -->

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Performance benchmarks (if applicable)

## Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is properly commented
- [ ] No debug prints or temporary code left
- [ ] Error handling is appropriate

### Testing & Documentation
- [ ] Tests have been added/updated for new functionality
- [ ] Documentation has been updated (if applicable)
- [ ] Breaking changes are documented
- [ ] Performance impact has been considered

### Security & Dependencies
- [ ] No sensitive information (tokens, passwords) committed
- [ ] Dependencies updated appropriately
- [ ] Security implications considered
- [ ] No new security vulnerabilities introduced

### Build & Integration
- [ ] All CI/CD checks pass
- [ ] No merge conflicts
- [ ] Build succeeds on all target platforms
- [ ] MCP server functionality verified (if applicable)
- [ ] Backend compatibility maintained (CUDA/SYCL/Vulkan)

## Performance Impact

<!-- If applicable, describe performance implications -->

- **Memory usage**: No change / Improved / Regression (specify impact)
- **CPU performance**: No change / Improved / Regression (specify impact)
- **GPU performance**: No change / Improved / Regression (specify impact)
- **Build time**: No change / Improved / Regression (specify impact)

## Related Issues

<!-- Link to related issues -->

Closes #
Fixes #
Related to #

## Screenshots/Logs

<!-- If applicable, add screenshots or relevant log outputs -->

## Additional Notes

<!-- Any additional information reviewers should know -->

---

**For Reviewers:**
- [ ] Code review completed
- [ ] Architecture review (for significant changes)
- [ ] Security review (for security-related changes)
- [ ] Performance review (for performance-critical changes)
||||||| empty tree
=======
# Pull Request

## Summary

Brief description of the changes in this PR.

## Type of Change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test coverage improvement

## Component Areas

Please check all that apply:

- [ ] Core Gemma engine (`gemma.cpp/`)
- [ ] MCP server (`mcp/`)
- [ ] Hardware backends (`backends/`)
- [ ] Testing framework (`tests/`)
- [ ] Build system (`CMakeLists.txt`, etc.)
- [ ] Documentation
- [ ] CI/CD workflows

## Changes Made

### Core Changes
-
-
-

### New Features
-
-
-

### Bug Fixes
-
-
-

### Breaking Changes
-
-
-

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance benchmarks added/updated
- [ ] All existing tests pass

### Platform Testing
- [ ] Linux (GCC)
- [ ] Linux (Clang)
- [ ] Windows (MSVC)
- [ ] macOS
- [ ] WSL

### Hardware Backend Testing
- [ ] CPU-only build tested
- [ ] CUDA backend tested (if applicable)
- [ ] SYCL backend tested (if applicable)
- [ ] Vulkan backend tested (if applicable)

## Performance Impact

### Benchmarks
- [ ] No performance regression
- [ ] Performance improvement measured
- [ ] Performance impact is acceptable for the feature

### Memory Usage
- [ ] No memory leak introduced
- [ ] Memory usage is reasonable
- [ ] Memory usage has been profiled

## Security Considerations

- [ ] No hardcoded secrets or credentials
- [ ] Input validation added where needed
- [ ] No known security vulnerabilities introduced
- [ ] Security scan passed

## Documentation

- [ ] Code comments added/updated
- [ ] README.md updated (if needed)
- [ ] API documentation updated (if needed)
- [ ] Build instructions updated (if needed)
- [ ] CHANGELOG.md updated

## Deployment Notes

### Configuration Changes
- [ ] No configuration changes required
- [ ] Configuration changes documented
- [ ] Backward compatibility maintained

### Database/Storage Changes
- [ ] No database changes
- [ ] Database migration included
- [ ] Storage format changes documented

## Checklist

### Before Submitting
- [ ] Self-review completed
- [ ] Code follows project style guidelines
- [ ] Comments added for complex logic
- [ ] No debugging code left in
- [ ] All CI checks pass

### Dependencies
- [ ] No new dependencies added
- [ ] New dependencies justified and documented
- [ ] License compatibility checked

### Compatibility
- [ ] Backward compatibility maintained
- [ ] API changes documented
- [ ] Version bumped if needed

## Additional Context

### Related Issues
- Fixes #
- Closes #
- Related to #

### Screenshots/Recordings
(If applicable, add screenshots or recordings to help explain your changes)

### Additional Notes
(Add any additional notes, concerns, or context for reviewers)

---

## For Reviewers

### Focus Areas
Please pay special attention to:
-
-
-

### Review Checklist for Maintainers
- [ ] Code quality and style
- [ ] Test coverage adequate
- [ ] Documentation complete
- [ ] Performance acceptable
- [ ] Security implications reviewed
- [ ] Breaking changes justified
- [ ] Release notes impact considered
>>>>>>> clean-refactor-branch
