#ifndef ABACUS_LR_FSSH_SURFACE_H
#define ABACUS_LR_FSSH_SURFACE_H

#include "source_base/matrix.h"

namespace LR
{
/** @brief One adiabatic surface in MD units; force rows follow global atom order. */
struct FsshSurface
{
    double energy_hartree = 0.0;
    ModuleBase::matrix force_hartree_per_bohr;
};

/**
 * @brief Combine KS and LR contributions and convert Rydberg units to Hartree.
 * @param[in] ground_energy_ry Ground-state total energy, in Ry.
 * @param[in] excitation_energy_ry Excitation energy, in Ry; zero for the ground surface.
 * @param[in] ground_force_ry Ground-state force, nat by 3, in Ry/Bohr.
 * @param[in] excitation_force_ry Borrowed LR force correction, in Ry/Bohr; nullptr for S0.
 * @return Total energy and force in Hartree and Hartree/Bohr, with independent storage.
 * @throws std::invalid_argument For invalid dimensions or nonfinite input.
 */
FsshSurface make_fssh_surface(
    double ground_energy_ry,
    double excitation_energy_ry,
    const ModuleBase::matrix &ground_force_ry,
    const ModuleBase::matrix *excitation_force_ry);
} // namespace LR

#endif
