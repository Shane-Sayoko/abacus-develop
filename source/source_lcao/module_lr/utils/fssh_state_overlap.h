#ifndef ABACUS_LR_FSSH_STATE_OVERLAP_H
#define ABACUS_LR_FSSH_STATE_OVERLAP_H

#include "fssh_electronic_frame.h"

namespace LR
{
/**
 * @brief Contract real-Gamma MO overlaps without phase alignment or state rotation.
 * @param[in] coef_old Old MO coefficients, band rows and AO columns.
 * @param[in] coef_new New MO coefficients in the same convention as new Casida amplitudes.
 * @param[in] s_ao_dense Dimensionless AO overlap, old rows and new columns.
 * @param[in] orbital_indices KS indices of the retained orbital rows.
 * @param[in] nbasis Number of AO columns in both frames.
 * @return Dimensionless row-major real MO overlap on the requested orbital space.
 * @pre Real Gamma orbitals and a matching AO basis; complex-k states are not supported.
 */
std::vector<double> contract_fssh_mo_overlap(
    const ModuleBase::ComplexMatrix &coef_old,
    const ModuleBase::ComplexMatrix &coef_new,
    const ModuleBase::ComplexMatrix &s_ao_dense,
    const std::vector<int> &orbital_indices,
    int nbasis);

/**
 * @brief Evaluate raw closed-shell singlet determinant overlaps by Lowdin contraction.
 * @param[in] coef_old Old band-by-AO coefficients.
 * @param[in] coef_new New band-by-AO coefficients, unrotated relative to casida_new.
 * @param[in] s_ao_dense Dimensionless cross-geometry AO overlap, old rows/new columns.
 * @param[in] casida_old Old states, ground placeholder first; X is occupied-major.
 * @param[in] casida_new New states in the corresponding order; no rephasing is performed.
 * @param[in] occupied_total Total occupied KS bands, locating the first virtual band.
 * @param[in] state_count Number of states including the ground placeholder.
 * @param[in] nocc_lr Occupied orbitals in the contraction (all occupied bands for LD).
 * @param[in] nvirt_lr Virtual orbitals in each X vector.
 * @param[in] occ_offset First occupied KS index; zero for the LD raw-frame interface.
 * @param[out] determinant Spatial occupied determinant diagnostic, or nullptr.
 * @return Owned raw state overlap with old rows/new columns; no normalization or polar projection.
 * @pre Real Gamma singlet TDA frames with consistent dimensions and well-conditioned occupied overlap.
 */
ModuleBase::ComplexMatrix closed_shell_fssh_overlap(
    const ModuleBase::ComplexMatrix &coef_old,
    const ModuleBase::ComplexMatrix &coef_new,
    const ModuleBase::ComplexMatrix &s_ao_dense,
    const std::vector<CasidaWavefunction> &casida_old,
    const std::vector<CasidaWavefunction> &casida_new,
    int occupied_total,
    int state_count,
    int nocc_lr,
    int nvirt_lr,
    int occ_offset,
    double *determinant);
} // namespace LR
#endif
