# iCE-NGM: Improved Cross-Entropy Importance Sampling with Non-Parametric Adaptive Gaussian Mixtures and Budget-Informed Stopping Criterion

**Authors:** Tianyu Zhang, Jize Zhang

## Abstract

Estimating the failure probability is an essential task in engineering reliability analysis, which can be challenging for applications featuring small failure probabilities and complex numerical models. Cross entropy (CE) importance sampling is a promising strategy to enhance the estimation efficiency, by searching for the proper proposal density that resembles the theoretically optimal choice. This paper introduces iCE-NGM, an approach that enriches the recently proposed improved cross entropy (iCE) method by a non-parametric adaptive Gaussian mixture model and a budget-informed stopping criterion. An over-parameterized Gaussian mixture model will be identified with a kernel density estimation inspired initialization and a constrained Expectation-Maximization fitting procedure. A novel budget-informed stopping criterion quantitatively balances between further refining proposal and reserving computational budget for final evaluation. A set of numerical examples demonstrate that the proposed approach performs consistently better than the classical distribution families and the existing stopping criteria.

## Project Structure

```
iCE-NGM/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── iceais.py          # Main iCE-NGM implementation (18KB, 441 lines)
│   ├── modules.py         # Supporting modules and utilities (7.4KB, 182 lines)
│   └── utils.py           # Utility functions (1.3KB, 48 lines)
├── notebooks/             # Jupyter notebooks with examples and demonstrations
│   ├── __init__.py        # Package initialization
│   ├── demo_example_series_system.ipynb     # Series system reliability example (2.1MB)
│   └── demo_example_parabolic_1.ipynb       # Parabolic reliability problem example (874KB)
└── .gitignore            # Git ignore file
```

## Description

This repository contains the implementation of the **iCE-NGM** (improved Cross-Entropy with Non-parametric adaptive Gaussian Mixtures) method for efficient failure probability estimation in engineering reliability analysis.

### Key Features

- **Non-parametric Adaptive Gaussian Mixture Models**: Enhanced proposal density estimation using adaptive Gaussian mixtures
- **Budget-Informed Stopping Criterion**: Optimal balance between proposal refinement and computational budget allocation
- **Kernel Density Estimation Initialization**: Improved initialization strategy for over-parameterized Gaussian mixture models
- **Constrained Expectation-Maximization**: Robust fitting procedure for mixture model parameters

### Main Components

- **`src/iceais.py`**: Core implementation of the iCE-NGM algorithm
- **`src/modules.py`**: Supporting modules including Gaussian mixture models and utility functions
- **`src/utils.py`**: General utility functions for the framework

### Examples and Demonstrations

The `notebooks/` directory contains comprehensive examples demonstrating the application of iCE-NGM to various reliability problems:

1. **Series System Example**: Demonstrates the method on series system reliability analysis
2. **Parabolic Problem Example**: Shows application to parabolic reliability problems

## Usage

To get started with iCE-NGM, explore the example notebooks in the `notebooks/` directory. These provide detailed demonstrations of how to apply the method to different types of reliability analysis problems.

## Requirements

This project is implemented in Python. Please ensure you have the necessary dependencies installed before running the code.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{zhang2024ice,
  title={iCE-NGM: Improved Cross-Entropy Importance Sampling with Non-Parametric Adaptive Gaussian Mixtures and Budget-Informed Stopping Criterion},
  author={Zhang, Tianyu and Zhang, Jize},
  year={2025}
}
``` 