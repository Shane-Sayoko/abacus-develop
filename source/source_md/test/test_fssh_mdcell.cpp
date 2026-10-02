#include "source_md/fssh_mdcell.h"

#include "source_base/matrix.h"
#include "source_base/matrix3.h"
#include "source_base/parallel_cell.h"
#include "source_cell/mdcell.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
MDCell make_mdcell()
{
    const ModuleBase::CommunicationDomain domain = ModuleBase::world_comm_domain();
    ModuleBase::Matrix3 lattice;
    lattice.e11 = 4.0;
    lattice.e22 = 4.0;
    lattice.e33 = 4.0;

    LocalAtom atom;
    atom.type_index = domain.rank();
    atom.owner_rank = domain.rank();
    atom.mass = 2.0;
    atom.vel.set(1.0, 0.0, 0.0);

    MDCell cell;
    cell.initialize_from_owned_atoms(lattice,
                                     lattice.Inverse(),
                                     1.0,
                                     64.0,
                                     domain.size(),
                                     std::vector<LocalAtom>(1, atom),
                                     std::vector<std::string>(1, "X"),
                                     std::vector<double>(1, 2.0),
                                     std::vector<std::int64_t>(1, domain.size()),
                                     0.4,
                                     domain);
    return cell;
}
} // namespace

TEST(FsshMDCell, OwnedForcesAndVelocityRescaling)
{
    MDCell cell = make_mdcell();
    const ModuleBase::CommunicationDomain domain = ModuleBase::world_comm_domain();
    ModuleBase::matrix forces(domain.size(), 3);
    for (int row = 0; row < domain.size(); ++row)
    {
        forces(row, 0) = row + 1.0;
        forces(row, 1) = -2.0;
        forces(row, 2) = 3.0;
    }

    FsshMDCell::set_owned_forces(cell, forces);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].force.x, domain.rank() + 1.0);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].force.y, -2.0);
    EXPECT_DOUBLE_EQ(FsshMDCell::kinetic_energy(cell), static_cast<double>(domain.size()));

    FsshMDCell::scale_owned_velocities(cell, 2.0);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].vel.x, 2.0);
    EXPECT_DOUBLE_EQ(FsshMDCell::kinetic_energy(cell), 4.0 * domain.size());
}

TEST(FsshMDCell, InvalidInputLeavesOwnedAtomsUnchanged)
{
    MDCell cell = make_mdcell();
    ModuleBase::matrix wrong_shape(1, 2);
    EXPECT_THROW(FsshMDCell::set_owned_forces(cell, wrong_shape), std::invalid_argument);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].force.x, 0.0);

    ModuleBase::matrix forces(static_cast<int>(cell.nat()), 3);
    forces.fill_out(1.0);
    forces(static_cast<int>(cell.owned_atoms()[0].type_index), 2) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(FsshMDCell::set_owned_forces(cell, forces), std::invalid_argument);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].force.x, 0.0);

    EXPECT_THROW(FsshMDCell::scale_owned_velocities(cell, -1.0), std::invalid_argument);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].vel.x, 1.0);
}

TEST(FsshMDCell, ForceRowsUseAtomTypeOffsets)
{
    const ModuleBase::CommunicationDomain domain = ModuleBase::world_comm_domain();
    ModuleBase::Matrix3 lattice;
    lattice.e11 = 4.0;
    lattice.e22 = 4.0;
    lattice.e33 = 4.0;
    LocalAtom atom;
    atom.type = 1;
    atom.type_index = 0;

    MDCell cell;
    cell.initialize_from_owned_atoms(lattice,
                                     lattice.Inverse(),
                                     1.0,
                                     64.0,
                                     2,
                                     std::vector<LocalAtom>(1, atom),
                                     std::vector<std::string>{"A", "B"},
                                     std::vector<double>{1.0, 1.0},
                                     std::vector<std::int64_t>{1, 1},
                                     0.4,
                                     domain);
    ModuleBase::matrix forces(2, 3);
    forces.fill_out(0.0);
    forces(1, 0) = 7.0;
    FsshMDCell::set_owned_forces(cell, forces);
    EXPECT_DOUBLE_EQ(cell.owned_atoms()[0].force.x, 7.0);
}
