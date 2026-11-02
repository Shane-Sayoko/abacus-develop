#include "fssh_surface.h"

#include <cmath>
#include <stdexcept>

namespace LR
{
FsshSurface make_fssh_surface(
    const double ground_energy_ry,
    const double excitation_energy_ry,
    const ModuleBase::matrix &ground_force_ry,
    const ModuleBase::matrix *excitation_force_ry)
{
    if (!std::isfinite(ground_energy_ry) || !std::isfinite(excitation_energy_ry) || ground_force_ry.nr <= 0 ||
        ground_force_ry.nc != 3)
    {
        throw std::invalid_argument("Invalid KS energy or force for FSSH");
    }
    if (excitation_force_ry != nullptr &&
        (excitation_force_ry->nr != ground_force_ry.nr || excitation_force_ry->nc != 3))
    {
        throw std::invalid_argument("KS and LR forces have incompatible atom dimensions");
    }
    FsshSurface surface;
    surface.energy_hartree = 0.5 * (ground_energy_ry + excitation_energy_ry);
    surface.force_hartree_per_bohr.create(ground_force_ry.nr, 3);
    for (int atom = 0; atom < ground_force_ry.nr; ++atom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const double ground_force = ground_force_ry(atom, axis);
            double correction = 0.0;
            if (excitation_force_ry != nullptr)
            {
                correction = (*excitation_force_ry)(atom, axis);
            }
            if (!std::isfinite(ground_force) || !std::isfinite(correction))
            {
                throw std::invalid_argument("Nonfinite KS or LR force for FSSH");
            }
            surface.force_hartree_per_bohr(atom, axis) = 0.5 * (ground_force + correction);
        }
    }
    return surface;
}
} // namespace LR
