# iCE-NGM: Improved cross-entropy importance sampling with non-parametric adaptive Gaussian mixtures and budget-informed stopping criterion

**Authors:** Tianyu Zhang, Jize Zhang

**Affiliation:** Department of Civil and Environmental Engineering, The Hong Kong University of Science and Technology, Hong Kong, China

## Abstract

Estimating the failure probability is an essential task in engineering reliability analysis, which can be challenging for applications featuring small failure probabilities and complex numerical models. Cross entropy (CE) importance sampling is a promising strategy to enhance the estimation efficiency, by searching for the proper proposal density that resembles the theoretically optimal choice. This paper introduces iCE-NGM, an approach that enriches the recently proposed improved cross entropy (iCE) method by a non-parametric adaptive Gaussian mixture model and a budget-informed stopping criterion. An over-parameterized Gaussian mixture model will be identified with a kernel density estimation inspired initialization and a constrained Expectation-Maximization fitting procedure. A novel budget-informed stopping criterion quantitatively balances between further refining proposal and reserving computational budget for final evaluation. A set of numerical examples demonstrate that the proposed approach performs consistently better than the classical distribution families and the existing stopping criteria.

## Project Structure

```
iCE-NGM/
├── src/                    # Source code
│   ├── iceais.py          # Main iCE-NGM implementation
│   ├── modules.py         # Supporting modules and utilities
│   ├── limit_state_funcs.py  # Limit state function definitions
│   └── __init__.py        # Package initialization
├── demos/                 # Jupyter notebooks with examples and demonstrations
│   ├── demo_1.ipynb       # First demonstration example
│   ├── demo_2.ipynb       # Second demonstration example
│   └── demo_3.ipynb       # Third demonstration example
└── .gitignore            # Git ignore file
```

## Description

This repository contains the implementation of the **iCE-NGM** (Improved cross-entropy importance sampling with non-parametric adaptive Gaussian mixtures and budget-informed stopping criterion) method for efficient failure probability estimation in engineering reliability analysis.

### Key features

- **Non-parametric adaptive Gaussian mixture models**: Enhanced proposal density estimation using adaptive Gaussian mixtures
- **Budget-informed stopping criterion**: Optimal balance between proposal refinement and computational budget allocation
- **Kernel density estimation initialization**: Improved initialization strategy for over-parameterized Gaussian mixture models
- **Constrained expectation-maximization**: Robust fitting procedure for mixture model parameters

### Main components

- **`src/iceais.py`**: Core implementation of the iCE-NGM algorithm
- **`src/modules.py`**: Supporting modules including Gaussian mixture models and utility functions
- **`src/limit_state_funcs.py`**: Definitions of limit state functions for reliability analysis

### Examples and demonstrations

The `demos/` directory contains comprehensive examples demonstrating the application of iCE-NGM to various reliability problems:

1. **demo_1.ipynb**: First demonstration example.
2. **demo_2.ipynb**: Second demonstration example.
3. **demo_3.ipynb**: Third demonstration example.

## Usage

To get started with iCE-NGM, explore the example notebooks in the `demos/` directory. These provide detailed demonstrations of how to apply the method to different types of reliability analysis problems.

## Requirements

This project is implemented in Python. Please ensure you have the necessary dependencies installed before running the code.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{ZHANG2025111322,
title = {iCE-NGM: Improved cross-entropy importance sampling with non-parametric adaptive Gaussian mixtures and budget-informed stopping criterion},
journal = {Reliability Engineering & System Safety},
pages = {111322},
year = {2025},
issn = {0951-8320},
doi = {https://doi.org/10.1016/j.ress.2025.111322},
url = {https://www.sciencedirect.com/science/article/pii/S095183202500523X},
author = {Tianyu Zhang and Jize Zhang},
keywords = {Reliability analysis, Importance sampling, Cross entropy method, Mixture model, Stopping criterion},
abstract = {Estimating the failure probability is an essential task in engineering reliability analysis, which can be challenging for applications featuring small failure probabilities and complex numerical models. Cross entropy (CE) importance sampling is a promising strategy to enhance the estimation efficiency, by searching for the proper proposal density that resembles the theoretically optimal choice. This paper introduces iCE-NGM, an approach that enriches the recently proposed improved cross entropy (iCE) method by a non-parametric adaptive Gaussian mixture model and a budget-informed stopping criterion. An over-parameterized Gaussian mixture model will be identified with a kernel density estimation-inspired initialization and a constrained Expectation-Maximization fitting procedure. A novel budget-informed stopping criterion quantitatively balances between further refining proposal and reserving computational budget for final evaluation. A set of numerical examples demonstrate that the proposed approach performs consistently better than the classical distribution families and the existing stopping criteria.}
}
``` 