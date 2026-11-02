#ifndef ABACUS_LR_FSSH_LOCAL_DIABATIZATION_H
#define ABACUS_LR_FSSH_LOCAL_DIABATIZATION_H

#include "source_base/complexmatrix.h"

#include <complex>
#include <vector>

namespace FsshLD
{
/** @brief Owned endpoint propagation data in the original adiabatic state order. */
struct Propagation
{
    ModuleBase::ComplexMatrix polar_overlap;        ///< Q, old-frame rows and new-frame columns.
    ModuleBase::ComplexMatrix propagator;           ///< U, new-frame rows and old-frame columns.
    std::vector<std::complex<double>> coefficients; ///< U c0; never renormalized.
    std::vector<double> singular_values;            ///< Ascending singular values of raw S.
    bool uncoupled_endpoints = false;               ///< Raw overlap is diagonal: no inter-state Hamiltonian coupling.
    double unitarity_residual = 0.0;                ///< Maximum element of U dagger U minus identity.
};

/** @brief Owned SHARC endpoint probabilities and dimensionless diagnostics before decoherence. */
struct HoppingProbabilities
{
    std::vector<double> probabilities; ///< Target order matches both endpoints; active entry is zero.
    double conditional_loss = 0.0;     ///< Positive active population loss divided by its old population.
    double total_probability = 0.0;    ///< Sum after negative-channel clipping; never rescaled.
    int clipped_channels = 0;          ///< Number of negative channel weights clipped to zero.
    bool pure_initial_state = false;   ///< All nonactive old amplitudes are exactly zero.
};

/**
 * @brief Evaluate SHARC finite-step probabilities (Mai et al. 2018, Eq.12).
 * @details Uses the target-row/source-column propagator U and b=U a BEFORE decoherence.
 * The exact pure-active-state limit is |U(J,K)|^2, including arbitrarily weak representable
 * transfers. General coherent states retain interference and the published net-loss gate.
 * Unlike SHARC's optional final rescaling, a total above 1+1e-12 is rejected here.
 * This endpoint prescription is not the general infinitesimal channel-flux FSSH rule.
 * @param[in] propagation Unitary U and its unmodified output amplitudes in matching state order.
 * @param[in] coefficients_old Normalized old amplitudes in the same state order.
 * @param[in] active_state Active state index; its old population must be positive.
 * @return Nonnegative probabilities and diagnostics, without normalization or fallback allocation.
 * @throw std::runtime_error For inconsistent/nonfinite inputs, unresolved denominators or total > 1+1e-12.
 */
HoppingProbabilities probabilities_sharc(
    const Propagation &propagation, const std::vector<std::complex<double>> &coefficients_old, int active_state);

/**
 * @brief Propagate with the endpoint LD Hamiltonian (Granucci et al. 2001, Eq.16).
 * @param[in] overlap Raw S=A0 dagger A1 in row-major project matrix storage.
 * @param[in] energy_old Original adiabatic endpoint energies in Hartree.
 * @param[in] energy_new Original adiabatic endpoint energies in Hartree.
 * @param[in] coefficients Normalized amplitudes in the old raw adiabatic frame.
 * @param[in] timestep Time interval in atomic units; negative intervals enable reversal tests.
 * @return Owned Q, U=Q dagger exp[-i(D0+Q D1 Q dagger)dt/2], amplitudes and diagnostics.
 * @throw std::runtime_error On rank/leakage gates (singular values outside [0.9,1+1e-8]),
 * non-unitary propagation, invalid input or a changed state window. No normalization is applied.
 */
Propagation propagate(
    const ModuleBase::ComplexMatrix &overlap,
    const std::vector<double> &energy_old,
    const std::vector<double> &energy_new,
    const std::vector<std::complex<double>> &coefficients,
    double timestep);

/**
 * @brief Evaluate only the original 2001 Eq.19/17 endpoint hopping probabilities.
 * @param[in] propagation Output of propagate, in raw adiabatic indices.
 * @param[in] coefficients_old The same old amplitudes supplied to propagate.
 * @param[in] active_state Old active surface index.
 * @return Nonnegative per-target probabilities; the active entry is zero.
 * @throw std::runtime_error For an unresolved zero denominator, zero active population,
 * inconsistent propagation data, or total probability above 1+1e-12. A zero-denominator
 * step is accepted only when the active channel is exactly decoupled in U and has zero
 * population transfer within floating-point error. No fallback allocation or rescaling.
 */
std::vector<double> probabilities_2001(
    const Propagation &propagation, const std::vector<std::complex<double>> &coefficients_old, int active_state);

/**
 * @brief Reject an active surface whose force label is unresolved by LR residuals.
 * @param[in] energies Adiabatic energies in Hartree.
 * @param[in] residuals Nonnegative eigenpair residual bounds in Hartree.
 * @param[in] active_state Surface for which a single adiabatic force is requested.
 * @throw std::runtime_error If another energy interval intersects the active interval.
 * This conservative gate does not prohibit degenerate-subspace propagation itself.
 */
void require_resolved_active_surface(
    const std::vector<double> &energies, const std::vector<double> &residuals, int active_state);
} // namespace FsshLD
#endif
