# Contributing Guide

Thank you for your interest in this project! This is an academic project for UMBC DATA 621, but contributions, suggestions, and feedback are welcome.

## How to Contribute

### Reporting Issues

If you encounter a bug or have a question, please open a [GitHub Issue](https://github.com/lulumehippo/DeepFakeDetection/issues) and use the appropriate template:

- **Bug Report** — something in the notebooks is broken or produces incorrect results
- **Feature Request** — an idea for improvement (e.g. a new model architecture, better augmentation)

Please include:
- Which notebook and section the issue is in
- Your Python / PyTorch / CUDA version
- The full error message and traceback

### Submitting Changes

1. **Fork** this repository
2. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b fix/mtcnn-batch-detection
   git checkout -b feature/add-resnet-baseline
   ```
3. **Make your changes**, keeping each commit focused on one thing
4. **Test** that the notebooks run end-to-end without errors
5. **Open a Pull Request** against `main` with a clear description of what changed and why

### Code Style

- Follow the existing code style in the notebooks (comments, spacing, docstrings)
- Keep functions small and focused — one function, one job
- Add a docstring to every new function explaining Args and Returns

### Notebook Guidelines

- Clear all outputs before committing (`Kernel → Restart & Clear Output`)
- Keep markdown cells updated when you change code behavior
- Do not hard-code local file paths — use `Path` and relative references

## Project Structure

See [README.md](README.md) for the full folder layout and how each part fits together.

## Questions

Feel free to open an Issue with the label `question` if anything is unclear.
