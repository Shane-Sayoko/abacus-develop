#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_UTILS_FSSH_FRAME_H
#define ABACUS_SOURCE_LCAO_MODULE_LR_UTILS_FSSH_FRAME_H

#include "fssh_electronic_frame.h"
#include "source_base/matrix.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_psi/psi.h"

#include <ATen/core/tensor.h>

namespace LR
{
/**
 * @brief Gather a closed-shell Gamma-point TDA frame without global input state.
 * @param[in] state_count Number of surfaces including the ground state.
 * @param[in] basis_count Global number of atomic orbitals.
 * @param[in] occupied Full occupied KS space size.
 * @param[in] virtual_count Number of virtual orbitals in the LR window.
 * @param[in] local_state_size Distributed Casida storage per excited state.
 * @param[in] ground_energy_ry Ground-state total energy in Ry.
 * @param[in] ks_energies_ry KS eigenvalues, indexed by k point and band, in Ry.
 * @param[in] occupations Weighted KS occupations in the same layout.
 * @param[in] excitation_energies_ry LR excitation energies in Ry.
 * @param[in] orbitals Distributed KS orbitals, k-first storage.
 * @param[in] pv KS orbital distribution over the world communicator.
 * @param[in] px Casida distribution, virtual rows and occupied columns.
 * @param[in] casida_amplitudes State-major distributed TDA amplitudes.
 * @param[in] excitation_residuals_ry Global normalized excited-state residuals in Ry.
 * @return Replicated frame with Hartree energies and occupied-major amplitudes.
 * @note The caller must establish Gamma-only, nspin=1, kpar=1 and completed LR.
 */
template <typename T>
FsshElectronicFrame make_fssh_frame(
    int state_count,
    int basis_count,
    int occupied,
    int virtual_count,
    int local_state_size,
    double ground_energy_ry,
    const ModuleBase::matrix &ks_energies_ry,
    const ModuleBase::matrix &occupations,
    const ModuleBase::matrix &excitation_energies_ry,
    const psi::Psi<T> &orbitals,
    const Parallel_Orbitals &pv,
    const Parallel_2D &px,
    const ct::Tensor &casida_amplitudes,
    const std::vector<double> &excitation_residuals_ry);
} // namespace LR

#endif // ABACUS_SOURCE_LCAO_MODULE_LR_UTILS_FSSH_FRAME_H
