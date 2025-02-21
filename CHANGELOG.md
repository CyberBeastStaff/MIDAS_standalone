# Changelog
All notable changes to the MIDAS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Improvements
- Enhanced image generation performance in ComfyUI integration
  - Implemented adaptive polling intervals
  - Optimized timeout and error handling mechanisms
  - Reduced overhead in image retrieval process

### Bug Fixes
- Fixed conversation title generation for both chat and image generation modes
- Resolved issues with duplicate message submissions
- Improved streaming response handling in chat interface
- Added more robust error handling for conversation and image generation workflows

### UI/UX Enhancements
- Improved chat interface responsiveness
- Better progress indicators for image generation
- More informative error messages

### Code Optimization
- Streamlined message handling in submit_message function
- Simplified timeout and error recovery in image generation
- Enhanced logging and error tracking

## [0.2.0] - 2025-02-21
### Fixed
- Resolved model dropdown population issue in new chat initialization
- Fixed image markdown rendering for generated images
- Corrected model name handling in dropdown and backend

### UI/UX Changes
- Updated file upload button with label
- Improved image generation and file serving mechanisms
- Refined default model selection process

### Minor Improvements
- Simplified model version handling
- Enhanced error tracking and logging
- Improved cross-component compatibility

## [0.1.0] - 2025-02-17
### Added
- Initial project setup
- README.md with comprehensive installation instructions
- `start_services.sh` and `start_services.ps1` for service management
- Dependency management via `requirements.txt`
- Comprehensive dependency list for MIDAS and ComfyUI

### Changed
- Updated Python version recommendation to 3.10-3.11
- Pinned specific versions of critical dependencies
- Improved package organization in `requirements.txt`

### Fixed
- Compatibility issues with Pydantic and FastAPI
- Resolved dependency conflicts
- Improved cross-platform support

## Initial Commit
- Project initialization
- Basic application structure
- Core functionality implementation
