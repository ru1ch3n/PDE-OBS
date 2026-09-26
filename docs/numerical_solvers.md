# Numerical systems and generation

This document describes the implemented generator routes, not an accuracy claim
for every combination of factors. The release validation record distinguishes
small-grid executions from independent reference checks. Parameter values below
are numerical values on the code's unit domain; no SI units are assigned.

## Shared representation and coordinates

The primary scalar grid spans the unit square using cell centers
`x_j=(j+0.5)/W`, `y_i=(i+0.5)/H`, with `dx=1/W`, `dy=1/H`.
The bounded vorticity route instead uses node-centered spacings
`dx=1/(W-1)`, `dy=1/(H-1)`. The condition generator uses its own normalized
sample coordinates, and some network adapters construct endpoint-inclusive
coordinates. These are explicit historical conventions, not silently changed
by this release.

A `Sample` contains `condition[H,W,Cc]`, `trajectory[T,H,W,Cs]`, and
`geometry[H,W,Cg]`; the built-in geometry has one channel. Static systems save
one solution frame. Current scalar dynamic generators and vorticity NS save one
state channel. Numerical kernels generally use float64 internally and the
selected storage dtype is applied at the sample boundary.

## Seven generator routes

| Family | Equation and condition | Main numerical route | Regimes and default final time |
|---|---|---|---|
| Darcy | `-div(a grad u)=f`; condition is positive coefficient `a`; source is the specified combination of sines | Sparse variable-coefficient flux operator with arithmetic face coefficients; diagonally preconditioned CG | Coefficient contrast parameter 2, 8, 32 |
| Poisson | `-Delta u=f`; condition is source `f` | Sparse elliptic solve; compatibility/zero-mean handling for periodic and Neumann cases | Source scale 0.5, 1, 2 |
| Helmholtz | `-Delta u-k^2 u=f`; condition is source `f` | Real indefinite sparse solve using MINRES when the reaction term is negative | `k=0.5,1,2`; damping is zero and nonzero damping is rejected |
| Heat | `u_t=alpha Delta u`; condition is `u(t0)` | Fourier exponential diffusion for periodic data; Crank-Nicolson for bounded data | `alpha=0.005,0.02,0.08`; final time 0.25 |
| Reaction-diffusion | `u_t=0.008 Delta u+rho(u-u^3)`; condition is the scaled initial scalar field | Strang splitting with an analytic reaction step and diffusion | `rho=0.5,1.5,4`; final time 0.8 |
| Burgers | `u_t+u(u_x+u_y)=nu Delta u`; condition is the scalar initial state | Dealiased spectral convection for smooth periodic fields; Rusanov fluxes for nonsmooth or bounded cases; SSPRK2 nonlinear stages with diffusion | `nu=0.04,0.015,0.004`; final time 0.35 |
| Navier-Stokes | `omega_t+v dot grad omega=nu Delta omega+f`, with `-Delta psi=omega` and velocity derived from `psi` | Periodic dealiased vorticity stepping; bounded or obstacle-aware streamfunction/vorticity routes | `nu=0.02,0.008,0.0025`; initial scale 2, 3, 4; final time 0.4 |

Burgers is a **two-dimensional scalar** equation, not a two-component velocity
system. Reaction-diffusion is an Allen-Cahn-type scalar equation, not Gray-Scott.
The primary NS records are **vorticity**, not a silently interchangeable velocity
or pressure tensor. A state-conversion option must be recorded as such.

Elliptic kernels expose tolerance and iteration information. Default generator
routes use relative tolerance `1e-9`; the iteration bounds depend on the family
and grid. Failure to converge raises an error. NS periodic stepping bounds the
internal time step by `1e-4`; nonlinear routes also enforce stability/substep
limits. A stored-frame interval is not necessarily one internal integration step.

## Boundary semantics

The four registry labels are not identical physical operators across families:

| Label | Scalar elliptic/diffusive routes | Active bounded NS route |
|---|---|---|
| `periodic` | Periodic sampling/operator; source compatibility is applied when required | Periodic vorticity and streamfunction inversion |
| `dirichlet` | Zero values imposed at the outer stored grid layer | Closed no-slip walls; Thom wall-vorticity condition |
| `neumann` | Zero-normal-difference boundary relation; null-space compatibility where necessary | Free-slip walls represented by zero wall vorticity |
| `robin_obstacle` | Mixed scalar relation: horizontal zero boundaries and vertical Robin relation | Closed-wall domain with a seeded circular obstacle, treated by a masked streamfunction/vorticity solver |

For the scalar mixed relation the default coefficients are `alpha=1`,
`beta=0.15`. On the cell-centered grid, the implementation constrains the outer
stored samples. It must not be described as an independently verified
continuous-boundary stencil at endpoints `0` and `1`.

For NS, the `robin_obstacle` label is **not** an assertion that a scalar Robin
condition is applied to vorticity, and the active route is not an inflow/outflow
channel. Historical metadata can retain an `inflow_speed` value that is not used
by this active solver. Zero-valued pressure diagnostic fields do not establish
that an independent pressure solve converged. Additional MAC, LBM, projected
channel, and velocity routines remain in the numerical module as separate
internal capabilities; they are not all invoked by the registered NS generator.

## Ten condition-field generators

The setting registry produces deterministic fields from an explicit random
generator. A PDE family then interprets that field as a coefficient, source, or
initial state and applies its regime-dependent scaling.

| Setting | Construction |
|---|---|
| `smooth_grf` | Spectrally filtered Gaussian noise, correlation length 0.22 |
| `medium_grf` | Same family, correlation length 0.09 |
| `rough_grf` | Same family, correlation length 0.025 |
| `low_frequency_fourier` | Four random low-frequency modes with decaying amplitudes |
| `multi_frequency_fourier` | Twelve random modes with a broader frequency range |
| `gaussian_blobs` | Two to six randomly located, signed Gaussian blobs |
| `piecewise_blocks` | Random values on a coarse block partition, then standardized |
| `threshold_level_set` | A thresholded smooth random field with values -1 and +1 |
| `dipole_vortex_pair` | A signed pair of offset Gaussian structures |
| `front_ring_shock` | A front, ring, or hard-step pattern with sampled location/width |

These are not ten synonyms for a Gaussian random field. Standardization, fixed
binary values, and family scaling differ; there is no universal guarantee that
all conditions lie in `[-1,1]`. The sample seed is deterministically derived
from the generation specification and sample index using `derive_seed`, and
independent random streams are used within generators.

## Custom arrays, without rewriting a solver

The numerical module exposes array-based interfaces. For example, a positive
coefficient and a known source define a Darcy-type elliptic problem:

```python
import numpy as np
from pdeobs.pdes.numerics import solve_elliptic

n = 16
yy, xx = np.mgrid[:n, :n]
a = 1.0 + 0.2 * np.cos(2 * np.pi * xx / n)
f = np.sin(2 * np.pi * xx / n) * np.sin(2 * np.pi * yy / n)
u, info = solve_elliptic(
    f, boundary="periodic", dx=1.0 / n, dy=1.0 / n, coefficient=a
)
assert np.isfinite(u).all()
```

`crank_nicolson_diffusion` and the `advance_*` routines similarly accept arrays.
The installed E4 example exercises a concrete valid selection. Supply a
complete, well-posed physical problem with consistent boundary conditions.
Replacing unknown coefficients or sources with zero and solving the resulting
different problem is not a universal fair reconstruction baseline.

## Generation and storage

Configuration resolution applies included mappings, then the current mapping,
then explicit CLI overrides; environment placeholders are resolved by the config
loader. Generation builds a case specification, derives identities/seeds, calls
the registered solver, checks the sample, and writes the storage protocol
described in [Data schema](data_schema.md). Observation masks do not enter the
numerical solver.

For temporal production data, dense quality-audit trajectories and stored
trajectories are separate. Stored frames are selected by an integer stride from
computed frames rather than synthesized by interpolation. Record physical
`time_values`, stored frame indices, and dense quality time-step count when
interpreting a trajectory. `t1` in the paper means the next selected stored
frame, not one universal physical time shared by all PDE families.

## What constitutes numerical evidence

1. **Implemented:** a numerical route and error handling exist in source.
2. **Smoke-tested:** a stated small-grid call returned valid arrays and passed
   the selected checks. This is the role of the release demos and most unit tests.
3. **Reference-checked:** a specific solution was compared to an independent
   analytic or numerical reference, with the grid and error reported. The
   release's Heat check is deliberately narrow.
4. **Paper-evaluated:** a stated dataset and trained-model protocol were used in
   a paper result, with their scoring provenance preserved.

Linear-system residuals, finite/shape checks, initial/boundary-condition checks,
and operator-matched PDE residuals are useful diagnostics. They do not on their
own prove continuous-solution accuracy or a grid-convergence rate. In particular,
forming `b=A*u` and solving it back only demonstrates algebraic consistency.
The historical whole-corpus quality record and a fresh small-grid test are
different evidence. No full-corpus regeneration or independent all-factor
numerical convergence study is implied by this candidate.
