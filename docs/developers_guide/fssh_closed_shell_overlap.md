# Restricted closed-shell singlet overlaps in FSSH

The current production path uses real Gamma-point orbitals, `nspin=1`, all
occupied spatial orbitals, and singlet TDA amplitudes (`Y=0`). Its spatial
amplitudes obey `sum_ia X_ia^2 = 1` for a normalized state. The LR solver uses
the identity metric, redistribution and FSSH frame collection do not rescale
these amplitudes, and the spectrum calculation supplies the singlet factor
`sqrt(2)` in the transition dipole.

The corresponding CIS/TDA pseudowavefunction is

```text
|I> = sum_ia X^I_ia (|Phi_i_alpha^a_alpha> + |Phi_i_beta^a_beta>) / sqrt(2).
```

This convention defines an overlap of approximate states; it does not turn
LR-TDDFT into an exact many-electron wavefunction theory. It must not be
replaced by a one-spin determinant formula or by renormalizing X afterwards.

## Contraction

With old orbitals as rows and new orbitals as columns, let A, B, C and V be
the occupied/occupied, occupied/virtual, virtual/occupied and virtual/virtual
blocks of the cross-geometry MO overlap. All occupied orbitals enter A,
including orbitals whose excitation amplitudes vanish. Define

```text
d = det(A),  Q = inverse(A) B,  R = C inverse(A),  W = V - C Q
u_I = sum_ia Xold^I_ia R_ai
v_J = sum_jb Q_jb Xnew^J_jb
T_IJ = sum_ijab Xold^I_ia inverse(A)_ji W_ab Xnew^J_jb

Sigma_00 = d^2
Sigma_0J = sqrt(2) d^2 v_J
Sigma_I0 = sqrt(2) d^2 u_I
Sigma_IJ = d^2 (T_IJ + 2 u_I v_J).
```

The two same-spin excitation contributions sum to `d^2 (T + uv)`; the
opposite-spin contributions supply the other `d^2 uv`. The implementation
retains u and v directly instead of recovering their product by division by
a potentially tiny ground-state overlap. The determinant diagnostic continues
to return the spatial d, not d squared. Contraction is bilinear and does not
renormalize the input amplitudes.

The previous implementation used a single-spin ground determinant and lacked
the singlet factors and complete cross term. For a two-electron orbital
rotation `[[c,-s],[s,c]]`, the ground-plus-singlet raw overlap is
`[[c^2,-sqrt(2)sc],[sqrt(2)sc,cos(2 theta)]]`. The omitted double excitation
means this projected matrix need not be unitary. Polar processing must not
be used to disguise an incorrect raw overlap.

## Verification and limitations

`MODULE_LCAO_fssh_closed_shell_overlap` expands full spin determinants and
evaluates them with a small independent Leibniz determinant reference.
Its eight tests cover the analytic two-electron rotation, multiple occupied
and virtual orbitals, unexcited occupied orbitals, joint MO/X gauge changes,
bilinearity, frame reversal, a specific underflow case and the existing
conditioning rejection. The identical suite has six failures on the baseline
and passes all eight tests after the correction in the archived validation.

Existing callable diagnostics also need correct spin determinants and the
actual Gram or cross-Gram matrix when their trial amplitudes are not
orthonormal. Gauge changes must transform both orbitals and amplitudes.

The occupied overlap must remain invertible and well-conditioned. Computing
d squared can still underflow at extreme determinant magnitudes; the fix
does not promise arbitrary near-orthogonal stability. It does not add
frozen-core, open-shell, triplet, complex-k or full-Casida X/Y support.
Correct overlap contraction alone does not validate forces, hopping events,
energy conservation or material lifetimes.

The determinant-overlap framework is discussed by Plasser et al.,
*J. Chem. Theory Comput.* **12**, 1207-1219 (2016),
<https://doi.org/10.1021/acs.jctc.5b01148>. The ABACUS amplitude convention
above is established by its solver and data path, not inferred from that paper.
