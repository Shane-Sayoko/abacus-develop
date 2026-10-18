# Local-diabatization kernel: verification status

Checked 2026-10-19. This is a **verified numerical kernel, not an enabled production
FSSH propagator**. The default FSSH driver still uses its existing RK4 route.
No INPUT switch, new restart format, or molecular LD trajectory is claimed here.

The kernel in `source/source_lcao/module_lr/utils/fssh_local_diabatization.cpp`
implements the endpoint approximation of Granucci, Persico and Toniolo,
J. Chem. Phys. 114, 10608 (2001), DOI
[10.1063/1.1376633](https://doi.org/10.1063/1.1376633), Eq. (16) and Appendix B.
For raw adiabatic overlap `S=A0† A1`, its right polar factor is `Q`,
`Hbar=(D0+Q D1 Q†)/2`, and `c1=Q† exp(-i Hbar dt)c0`.
Energies use Hartree, time uses atomic units, and state labels remain the raw
adiabatic endpoint labels. Internal rotations do not define new force surfaces.

The probability routine implements Eq. (19) and (17), with
`r_KL=Re(U_KL c0_L conj(c1_K))` and
`B_KL=-r_KL (p1_K-p0_K)/(p1_K-r_KK)`.
The denominator is evaluated as the algebraically equivalent sum over `L!=K`
to reduce cancellation. Ratios are formed before products to avoid underflow at
very low active population. No endpoint NAC probability is mixed into this route.

## Production blocker

For a pure initial active state `c0=e_K`, all off-diagonal `r_KL` and the
denominator vanish, even when the endpoint propagation transfers population.
For three states, approaching the same pure state with a small amplitude in
different target channels yields different finite channel allocations. This is
an algebraic property of the original formula, not an electronic-structure error.
The approved requirement permits a zero denominator only for a proven
zero-transfer limit; consequently the kernel rejects this common initial step.
It must not silently set all probabilities to zero, normalize them, or substitute
`|U_LK|^2` or another publication's formula.

The zero-transfer exception is deliberately sufficient and conservative: the
raw overlap must be exactly diagonal, so the selected endpoint Hamiltonian
model has no inter-state coupling, and the propagated active channel must also
be exactly decoupled with population change limited to roundoff. This proves
zero transfer in this endpoint model, not along an unknown continuous nuclear path.
A finite-step return to the same populations alone is insufficient.

Later alternative probabilities are discussed in Aguilera-Porta et al., Comput.
Theor. Chem. 1152, 20–27 (2019),
[10.1016/j.comptc.2019.02.009](https://doi.org/10.1016/j.comptc.2019.02.009),
Appendix A. Adopting them would be a separate method decision.

## Tests and conservative gates

`unittests/test_fssh_local_diabatization.cpp` verifies two- and three-state complex
matrices against independent 80-digit matrix exponentials, channel probabilities,
phase/permutation covariance, common energy shifts, time reversal, degenerate
subspace rotations and a finite-gap negative control. It separately verifies
second-order convergence to an independent continuous two-state TDSE solution.
Other tests cover overlap leakage/rank, unresolved force labels, pure-state
singular probabilities, low active population, excessive outgoing probability,
finite-step returns and floating-point overflow.

The overlap singular-value interval `[0.9,1+1e-8]`, pre-normalization unitarity
limit `1e-12`, and outgoing probability bound `1+1e-12` are implementation
admission criteria. They are not universal physical accuracy criteria.
No output amplitude normalization is used to satisfy these gates.

The fixture `unittests/ld_reference.txt` is generated independently at 80 decimal
digits by the archived research workflow. The C++ implementation does not call
that reference implementation. Molecular fixed-path convergence, a corrected
closed-shell many-electron overlap, raw-frame replay/restart integration and
production acceptance remain separate requirements.
