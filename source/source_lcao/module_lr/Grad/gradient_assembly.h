#ifndef ABACUS_LR_GRADIENT_ASSEMBLY_H
#define ABACUS_LR_GRADIENT_ASSEMBLY_H
#include "source_lcao/module_lr/Grad/force/lr_force.h"
#include "source_psi/psi.h"

#include <ATen/core/tensor.h>
#include <fstream>

namespace LR
{
/**
 * @brief Explicit, borrowed inputs for LR force assembly after the Z-vector solve.
 * All referenced objects must outlive the call. Energies and returned forces
 * use Ry and Ry/Bohr; X and Z store contiguous states with spin/k blocks and
 * column-major local virtual-by-occupied matrices described by px.
 */
template <typename T>
struct GradientContext
{
    int nk;
    int nspin;
    int nbasis;
    int nstates;
    int local_state_size;
    const std::vector<int>& nocc;
    const std::vector<int>& nvirt;
    const std::vector<Parallel_2D>& px;
    const Parallel_2D& pc;
    const Parallel_Orbitals& pmat;
    const UnitCell& ucell;
    const K_Vectors& kv;
    const Grid_Driver& grid;
    const std::vector<double>& orbital_cutoffs;
    const psi::Psi<T>& orbitals;
    const ModuleBase::matrix& ks_energies;
    const ModuleBase::matrix& excitation_energies;
    const std::vector<std::shared_ptr<PotHxcLR>>& potentials;
    const std::shared_ptr<PotHxcLR>& ground_potential;
    const std::string& xc_kernel;
    const std::vector<std::string>& spin_types;
    bool test_force;
    bool excited_relax;
    std::ofstream& log;
    double ground_channel_factor;
#ifdef __EXX
    const std::shared_ptr<Exx_LRI<T>>& exx_lri;
    double hybrid_alpha;
    bool ground_is_hybrid;
#endif
};
/**
 * @brief Assemble excited-state Lagrangian forces in the full Z-vector window.
 * @param[in] context Explicit geometry, orbitals, potentials and output options.
 * @param[in,out] lr_force Force-integral engine owned by the composition point.
 * @param[in] ground_density Ground-state density with its original occupations.
 * @param[in] Xz Casida amplitudes padded to the Z-vector window (dimensionless).
 * @param[in] Z Solved orbital-response amplitudes in the same layout as Xz.
 * @param[in] ist_begin First physical excitation index, inclusive.
 * @param[in] ist_end Last physical excitation index, exclusive.
 * @param[in] ispin Closed-shell singlet/triplet channel index.
 * @return One owned atom-by-Cartesian force matrix in Ry/Bohr per state.
 */
template <typename T>
std::vector<ModuleBase::matrix> assemble_closed_shell_forces(const GradientContext<T>& context,
                                                             LR_Force<T>& lr_force,
                                                             const elecstate::DensityMatrix<T, double>& ground_density,
                                                             const ct::Tensor& Xz,
                                                             const ct::Tensor& Z,
                                                             const int ist_begin,
                                                             const int ist_end,
                                                             const int ispin);
/**
 * @brief Assemble excited-state Lagrangian forces in the full Z-vector window.
 * @param[in] context Explicit geometry, orbitals, potentials and output options.
 * @param[in,out] lr_force Force-integral engine owned by the composition point.
 * @param[in] ground_density Ground-state density with its original occupations.
 * @param[in] Xz Casida amplitudes padded to the Z-vector window (dimensionless).
 * @param[in] Z Solved orbital-response amplitudes in the same layout as Xz.
 * @param[in] ist_begin First physical excitation index, inclusive.
 * @param[in] ist_end Last physical excitation index, exclusive.
 * @return One owned atom-by-Cartesian force matrix in Ry/Bohr per state.
 */
template <typename T>
std::vector<ModuleBase::matrix> assemble_open_shell_forces(const GradientContext<T>& context,
                                                           LR_Force<T>& lr_force,
                                                           const elecstate::DensityMatrix<T, double>& ground_density,
                                                           const ct::Tensor& Xz,
                                                           const ct::Tensor& Z,
                                                           const int ist_begin,
                                                           const int ist_end);
} // namespace LR
#endif
