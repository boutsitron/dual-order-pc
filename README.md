# Dual-Order-PC

This repository contains the implementation of a preconditioner capable of handling both Full Order Models (FOM) and Reduced Order Models (ROM) using the `appctx` of the Firedrake solver infrastructure.

## Contents

- **`dual_order_pc.py`**: Implements the dual-order preconditioner logic to handle FOM and ROM. This file contains the main functionality of the preconditioner.
- **`simple_mfe_FOM.py`**: Provides a Mininimum Failing Example (MFE)to demonstrate the application of the FOM path of the dual-order preconditioner in a minimal finite element setup. At the moment the FOM path of the preconditioner makes the call of the solver to take singificantly more time than expected.

## Requirements

This project is developed using the Firedrake library. To use or modify the files, ensure the following dependencies are installed:

- **Firedrake**: [Installation guide](https://www.firedrakeproject.org/)
- Python 3.x
- Other dependencies may be required; see the code files.

## Usage

1. Clone the repository:
   ```bash
   git@github.com:boutsitron/dual-order-pc.git
   cd dual-order-pc
   ```

2. Set up Firedrake and ensure it's properly installed on your system.

3. Run the MFE in sequential mode:
   ```bash
   python simple_mfe_FOM.py
   ```

   or the MFE in parallel mode:
   ```bash
   mpiexec -np 8 python simple_mfe_FOM.py
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the developers of the Firedrake project for their robust FEM framework and to the contributors who made this repository possible.
