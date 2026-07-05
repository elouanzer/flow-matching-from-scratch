# Flow Matching 

A from-scratch PyTorch R&D project exploring **Optimal Transport Flow Matching (OT-CFM)**, the current state-of-the-art framework for generative modeling.

## Project Architecture

* `configs/`: Configuration files for hyperparameters and model settings.
* `scripts/`: Executable entry points for training, sampling, and evaluation.
* `notebooks/`: Notebooks for studying.
* `src/`: Main source code package.
  * `src/data/`: Handles all data-related modules, dataset loading (MNIST), and preprocessing.
  * `src/models/`: Contains the neural network architectures, forward passes, and custom layers (e.g., Time Embeddings).
  * `src/flow/`: Manages the core algorithmic logic for Flow Matching, including interpolation paths and target vector computation.
* `tests/`: Unit tests verifying mathematical correctness, tensor broadcasting, and module integrity.

## Tests

Basic unit testing can be launched with pytest:

```bash
pytest tests\
```