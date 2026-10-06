# Third-party notices

This repository contains the FARR implementation and does not redistribute
datasets, model weights, forecast caches, or third-party checkpoints.

- **PyTorch** is used for tensor operations and model training. Follow its
  license and the license of the selected installation.
- **NumPy** is used for local array I/O and metric preparation.
- **PyYAML** is used to read experiment configuration files.
- The frozen SE-LLM/backbone adapter is intentionally supplied by the user and
  remains subject to the license of that backbone and its checkpoint.
- Each dataset must be downloaded from its official source and used under its
  own terms. Dataset files are not part of this release.
