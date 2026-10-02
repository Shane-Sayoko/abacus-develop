#ifndef FSSH_MDCELL_H
#define FSSH_MDCELL_H

class MDCell;

namespace ModuleBase
{
class matrix;
}

namespace FsshMDCell
{
/**
 * @brief Calculate the nuclear kinetic energy from atoms owned by each MD rank.
 * @param mdcell MD cell whose atom masses and velocities are in atomic units.
 * @return Total kinetic energy over the world communicator, in Hartree.
 */
double kinetic_energy(const MDCell& mdcell);

/**
 * @brief Set owned-atom forces from a globally indexed force matrix.
 * @param mdcell MD cell whose owned atoms receive the forces.
 * @param forces Force matrix in global atom order, in Hartree per Bohr.
 */
void set_owned_forces(MDCell& mdcell, const ModuleBase::matrix& forces);

/**
 * @brief Apply a caller-supplied uniform scale to owned-atom velocities.
 * @param mdcell MD cell whose owned atom velocities are updated.
 * @param factor Nonnegative dimensionless velocity scale factor.
 */
void scale_owned_velocities(MDCell& mdcell, double factor);
} // namespace FsshMDCell

#endif // FSSH_MDCELL_H
