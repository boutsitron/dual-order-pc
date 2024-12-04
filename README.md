# Dual-Order-PC

This repository contains the implementation of a preconditioner capable of handling both Full Order Models (FOM) and Reduced Order Models (ROM) using the `appctx` of the Firedrake solver infrastructure.

## Contents

- **`dual_order_pc.py`**: Implements the dual-order preconditioner logic to handle FOM and ROM. This file contains the main functionality of the preconditioner.
- **`simple_mfe_FOM.py`**: Provides a Minimum Failing Example (MFE) to demonstrate the application of the FOM path of the dual-order preconditioner in a minimal finite element setup. Currently, the FOM path of the preconditioner is less performant compared to classic direct solvers with LU factorization and/or Firedrake's AssembledPC preconditioner.

## Requirements

This project requires Firedrake. To use or modify the files, ensure you have:

- **Firedrake**: [Installation guide](https://www.firedrakeproject.org/)
- Python 3.x

## Usage

1. Clone the repository:
   ```bash
   git clone git@github.com:boutsitron/dual-order-pc.git
   cd dual-order-pc
   ```

2. Set up Firedrake following their installation guide and source it.

3. Run the MFE:
   ```bash
   # Sequential mode
   python simple_mfe_FOM.py

   # Parallel mode (8 processes)
   mpiexec -np 8 python simple_mfe_FOM.py
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the Firedrake project developers for their robust FEM framework and to Julian Andrej, whose [work](https://github.com/JuLuSi/mor) inspired this preconditioner.
