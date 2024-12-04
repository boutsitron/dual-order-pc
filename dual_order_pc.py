"""This module contains the DualOrderPC class for preconditioning using the reduced basis"""

from __future__ import absolute_import, print_function

import gc
import logging

from firedrake import PCBase
from firedrake.assemble import get_assembler
from firedrake.petsc import PETSc
from pyadjoint import no_annotations


def log_msg(message: str) -> None:
    """Simple logging function for DualOrderPC.

    Args:
        message: Message to log
    """
    PETSc.Sys.Print(f"[DualOrderPC] {message}")


class DualOrderPC(PCBase):
    """DualOrderPC class for preconditioning using the reduced basis"""

    def __init__(self):
        """Initialize the DualOrderPC preconditioner"""
        log_msg("Running `__init__` for DualOrderPC preconditioner")
        super().__init__()

        # Create KSP solvers once
        self.KpInv = PETSc.KSP().create()  # Reduced space solver
        self.KInv = PETSc.KSP().create()  # Full space solver

        # Set MUMPS options once at initialization
        opts = PETSc.Options()

        # Set common MUMPS parameters
        for prefix in ["dual_order_pc_rom_", "dual_order_pc_fom_"]:
            opts.setValue(f"{prefix}mat_mumps_icntl_24", 1)

        # ROM solver options (small dense system)
        opts.setValue(
            "dual_order_pc_rom_mat_mumps_icntl_14", 50
        )  # Less working space needed
        # FOM solver options (large sparse system)
        opts.setValue("dual_order_pc_fom_mat_mumps_icntl_14", 200)  # More working space

        # Configure KSP solvers with distinct prefixes
        self.KpInv.setOptionsPrefix("dual_order_pc_rom_")  # Reduced space solver prefix
        self._configure_ksp(self.KpInv)

        self.KInv.setOptionsPrefix("dual_order_pc_fom_")  # Full space solver prefix
        self._configure_ksp(self.KInv)

    @no_annotations
    def _configure_ksp(self, ksp: PETSc.KSP) -> None:
        """Configure KSP solvers with settings optimized for their respective system sizes.

        ROM solver (small dense system):
            - Uses LU with sequential solver (good for small, dense matrices)
            - Less memory-intensive options as system is small

        FOM solver (large sparse system):
            - Uses MUMPS with parallel capabilities
            - Memory optimization settings for large systems

        Args:
            ksp: KSP solver to configure
        """
        prefix = ksp.getOptionsPrefix()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)

        if prefix == "dual_order_pc_rom_":
            # ROM: Small dense system - use sequential direct solver
            pc.setFactorSolverType("petsc")  # Sequential solver, good for small systems

        elif prefix == "dual_order_pc_fom_":
            # FOM: Large sparse system - use MUMPS with optimizations
            pc.setFactorSolverType("mumps")

        pc.setFromOptions()

        PETSc.garbage_cleanup()
        gc.collect()

    def __del__(self):
        """Destructor to ensure PETSc cleanup"""
        self._destroy()

    @no_annotations
    def initialize(self, pc: PETSc.PC) -> None:
        """Initialize the preconditioner

        Args:
            pc (PETSc.PC): PETSc preconditioner object
        """
        log_msg("Initializing DualOrderPC preconditioner")
        # Get context and set up assembler
        K_operator = pc.getOperators()[1]  # type: ignore[attr-defined]
        self.ctx = K_operator.getPythonContext()
        K_operator.destroy()

        # Set up the assembler
        self.prefix = pc.getOptionsPrefix()  # type: ignore[attr-defined]
        mat_type = PETSc.Options().getString(f"{self.prefix}assembled_mat_type", "aij")  # type: ignore[attr-defined]

        # Store the assembler configuration
        assembler = get_assembler(
            self.ctx.a,
            bcs=self.ctx.row_bcs,
            form_compiler_parameters=self.ctx.fc_params,
            mat_type=mat_type,
        )

        self.K = assembler.allocate()
        self._assemble_K = assembler.assemble

        Z = self._get_projection_matrix(pc)

        if self._is_rom(Z):
            self.ndofs = Z.getSize()[1]
            Kp = self._assemble_projected_jacobian(Z)
            self._update_ksp_solver(Kp, self.KpInv)
            Z.destroy()
        else:
            K = self._assemble_jacobian()
            self.ndofs = K.getSize()[0]
            self._update_ksp_solver(K, self.KInv)
            # Don't destroy K - it's owned by self.K and KInv

        PETSc.garbage_cleanup()  # type: ignore[attr-defined]
        gc.collect()

    @no_annotations
    def update(self, pc: PETSc.PC) -> None:
        """Update the preconditioner

        Args:
            pc (PETSc.PC): PETSc preconditioner object
        """
        log_msg("Updating DualOrderPC preconditioner")

        Z = self._get_projection_matrix(pc)

        if self._is_rom(Z):
            self._update_rom(Z)
        else:
            self._update_fom()

    def _update_rom(self, Z: PETSc.Mat) -> None:
        """Handle ROM-specific update logic"""
        self.ndofs = Z.getSize()[1]
        new_Kp = self._assemble_projected_jacobian(Z)

        # Set/Update matrix and solver
        self._update_ksp_solver(new_Kp, self.KpInv)
        new_Kp.destroy()

    def _update_fom(self) -> None:
        """Handle FOM-specific update logic"""
        K = self._assemble_jacobian()
        self.ndofs = K.getSize()[0]
        self._update_ksp_solver(K, self.KInv)

    @no_annotations
    def apply(self, pc: PETSc.PC, X: PETSc.Vec, Y: PETSc.Vec) -> None:
        """Apply the preconditioner

        Args:
            pc (PETSc.PC): PETSc preconditioner object
            X (PETSc.Vec): residual vector (input) in the full space
            Y (PETSc.Vec): solution increment vector (output) in the full space
        """
        log_msg("Applying DualOrderPC preconditioner")

        Z = self._get_projection_matrix(pc)

        if self._is_rom(Z):
            self._apply_rom(Z, X, Y)
        else:
            self._apply_fom(X, Y)

    @no_annotations
    # @track_memory_usage
    def _apply_rom(self, Z: PETSc.Mat, X: PETSc.Vec, Y: PETSc.Vec) -> None:
        """Apply ROM-specific preconditioner logic"""
        self.ndofs = Z.getSize()[1]

        # Create temporary vectors
        tmp_Xp = Z.createVecs()[0]
        tmp_Yp = Z.createVecs()[0]

        # Project X into reduced space Xp [kx1] = Z [mxk] * X [mx1]
        Z.multTranspose(X, tmp_Xp)
        # Solve in reduced space Yp [kx1] = KpInv [kxk] * Xp [kx1]
        self.KpInv.solve(tmp_Xp, tmp_Yp)
        # Project back to original space Y [mx1] = Z [mxk] * Yp [kx1]
        Z.mult(tmp_Yp, Y)
        # Clean up temporary vectors
        tmp_Xp.destroy()
        tmp_Yp.destroy()
        # X.destroy()

    @no_annotations
    def _apply_fom(self, X: PETSc.Vec, Y: PETSc.Vec) -> None:
        """Apply ROM-specific preconditioner logic"""
        self.KInv.solve(X, Y)  # type: ignore[attr-defined]

        # PETSc.garbage_cleanup()

    def applyTranspose(self, pc: PETSc.PC, X: PETSc.Vec, Y: PETSc.Vec) -> None:
        """Apply the transpose of the preconditioner

        Args:
            pc (PETSc.PC): PETSc preconditioner object
            X (PETSc.Vec): residual vector (input) in the full space
            Y (PETSc.Vec): solution increment vector (output) in the full space
        """
        raise NotImplementedError(
            "applyTranspose not implemented for custom POD-ROM PC"
        )

    @no_annotations
    def _update_ksp_solver(self, K: PETSc.Mat, ksp: PETSc.KSP):
        """Update KSP solver with new matrix"""
        ksp.reset()  # type: ignore[attr-defined]
        ksp.setOperators(K, K)  # type: ignore[attr-defined]
        ksp.setConvergenceHistory()  # type: ignore[attr-defined]
        ksp.setFromOptions()  # type: ignore[attr-defined]

        PETSc.garbage_cleanup(comm=K.comm)

    @no_annotations
    def _assemble_jacobian(self) -> PETSc.Mat:
        """Assemble the jacobian K using the current projection"""
        # Reuse existing tensor but return the PETSc handle
        self._assemble_K(tensor=self.K)
        return self.K.petscmat

    @no_annotations
    def _assemble_projected_jacobian(self, Z: PETSc.Mat) -> PETSc.Mat:
        """Assemble the projected matrix Kp from the full matrix K using the current projection matrix Z."""
        # Assume self.assembleK() exists and updates self.K
        K = self._assemble_jacobian()
        Kp = K.ptap(Z)

        m, k = Kp.getSize()
        assert m == k, "Projected matrix is not square!"

        return Kp

    @no_annotations
    def _get_projection_matrix(self, pc: PETSc.PC) -> PETSc.Mat:
        """Get the projection matrix

        Args:
            pc (PETSc.PC): PETSc preconditioner object

        Returns:
            PETSc.Mat: projection matrix
        """
        # Get the POD modes which is the projection matrix
        appctx = self.get_appctx(pc)  # type: ignore[attr-defined]
        Z = appctx.get("projection_mat", None)

        return Z

    def _is_rom(self, Z: PETSc.Mat) -> bool:
        """Check if the preconditioner is for a ROM"""
        return Z is not None

    def _destroy_common_objects(self):
        """Clean up common PETSc objects used in both ROM and FOM"""
        if hasattr(self, "ctx"):
            if hasattr(self.ctx, "a"):
                self.ctx.a = None  # type: ignore[union-attr]
            if hasattr(self.ctx, "row_bcs"):
                self.ctx.row_bcs = None  # type: ignore[union-attr]
            self.ctx = None
            self.K.petscmat.destroy()

    def _destroy(self):
        """Clean up all PETSc objects"""
        # Clean up common objects
        self._destroy_common_objects()

        # Destroy KSP solvers
        if hasattr(self, "KpInv"):
            self.KpInv.destroy()
        if hasattr(self, "KInv"):
            self.KInv.destroy()

        PETSc.garbage_cleanup()
        gc.collect()
