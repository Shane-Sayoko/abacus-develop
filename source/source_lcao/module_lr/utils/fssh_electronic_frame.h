#ifndef ABACUS_LR_FSSH_ELECTRONIC_FRAME_H
#define ABACUS_LR_FSSH_ELECTRONIC_FRAME_H

#include "source_base/complexmatrix.h"

#include <utility>
#include <vector>

/** @brief Spin-adapted closed-shell excitation manifold used by FSSH. */
enum class FsshSpinChannel
{
    singlet = 0,
    triplet = 1
};

/// @brief Structure to hold LR-TDDFT Casida equation results for a single excitation.
struct CasidaWavefunction
{
    double omega = 0.0;           ///< Excitation energy in Hartree
    std::vector<double> X_coeffs; ///< FSSH: normalized singlet spatial amplitudes, index i * nvirt_lr + a.
    std::vector<double> Y_coeffs; ///< Casida Y coefficients (de-excitations)
    int nocc_lr = 0;              ///< LR module's actual nocc (may differ from KS nocc)
    int nvirt_lr = 0;             ///< LR module's actual nvirt (may differ from KS nvirt)

    CasidaWavefunction() = default;
    CasidaWavefunction(
        double w,
        std::vector<double> x,
        std::vector<double> y,
        int nolr = 0,
        int nvlr = 0)
        : omega(w), X_coeffs(std::move(x)), Y_coeffs(std::move(y)), nocc_lr(nolr), nvirt_lr(nvlr)
    {
    }
};

/**
 * @brief Electronic data needed to advance one FSSH nuclear geometry.
 * Energies are in Hartree; matrix rows are KS bands and columns are AO basis functions.
 */
struct FsshElectronicFrame
{
    ModuleBase::ComplexMatrix mo_coefficients;
    std::vector<double> ks_energies_hartree;
    std::vector<double> surface_energies_hartree;
    std::vector<CasidaWavefunction> casida_states;
    std::vector<double> eigen_residuals_hartree; ///< Surface-indexed LR residuals; ground entry is zero.
    int occupied_bands = 0;
};

#endif
