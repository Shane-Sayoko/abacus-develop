#include "fssh_mdcell.h"

#include "source_base/matrix.h"
#include "source_base/parallel_reduce.h"
#include "source_cell/mdcell.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace FsshMDCell
{
double kinetic_energy(const MDCell& mdcell)
{
    double energy = 0.0;
    for (const LocalAtom& atom : mdcell.owned_atoms())
    {
        if (!std::isfinite(atom.mass) || atom.mass <= 0.0 || !std::isfinite(atom.vel.x)
            || !std::isfinite(atom.vel.y) || !std::isfinite(atom.vel.z))
        {
            throw std::invalid_argument("FSSH kinetic energy requires finite velocities and positive masses");
        }
        energy += 0.5 * atom.mass * atom.vel.norm2();
    }
    Parallel_Reduce::reduce_all(energy);
    return energy;
}

void set_owned_forces(MDCell& mdcell, const ModuleBase::matrix& forces)
{
    if (static_cast<std::int64_t>(forces.nr) != mdcell.nat() || forces.nc != 3)
    {
        throw std::invalid_argument("FSSH force matrix must contain one three-component row per atom");
    }

    const std::vector<std::int64_t>& counts = mdcell.type_atom_counts();
    std::vector<std::int64_t> offsets(counts.size() + 1, 0);
    for (std::size_t type = 0; type < counts.size(); ++type)
    {
        offsets[type + 1] = offsets[type] + counts[type];
    }
    if (offsets.back() != mdcell.nat())
    {
        throw std::invalid_argument("FSSH atom type counts do not match the MD cell atom count");
    }

    std::vector<ModuleBase::Vector3<double>> staged_forces;
    staged_forces.reserve(mdcell.owned_atoms().size());
    for (const LocalAtom& atom : mdcell.owned_atoms())
    {
        if (atom.type < 0 || static_cast<std::size_t>(atom.type) >= counts.size()
            || atom.type_index < 0 || atom.type_index >= counts[atom.type])
        {
            throw std::out_of_range("FSSH owned atom has an invalid global atom index");
        }
        const std::int64_t index = offsets[atom.type] + atom.type_index;
        const int row = static_cast<int>(index);
        ModuleBase::Vector3<double> force(forces(row, 0), forces(row, 1), forces(row, 2));
        if (!std::isfinite(force.x) || !std::isfinite(force.y) || !std::isfinite(force.z))
        {
            throw std::invalid_argument("FSSH force matrix contains a non-finite component");
        }
        staged_forces.push_back(force);
    }

    std::vector<LocalAtom>& atoms = mdcell.owned_atoms();
    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        atoms[i].force = staged_forces[i];
    }
}

void scale_owned_velocities(MDCell& mdcell, const double factor)
{
    if (!std::isfinite(factor) || factor < 0.0)
    {
        throw std::invalid_argument("FSSH velocity scale factor must be finite and nonnegative");
    }
    for (LocalAtom& atom : mdcell.owned_atoms())
    {
        atom.vel *= factor;
    }
}
} // namespace FsshMDCell
