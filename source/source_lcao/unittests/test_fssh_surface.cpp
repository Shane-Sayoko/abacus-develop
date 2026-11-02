#include "source_lcao/module_lr/utils/fssh_surface.h"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

TEST(FsshSurface, TotalForceMatchesDerivativeOfTotalSurfaceEnergy)
{
    // Independent quadratic KS and excitation energies, in Ry, at x in Bohr.
    const double x = 0.7;
    const double dx = 1.0e-4;
    ModuleBase::matrix ground_force(1, 3);
    ModuleBase::matrix excitation_force(1, 3);
    ground_force(0, 0) = -2.0 * x;
    excitation_force(0, 0) = 0.4 * x;
    const double ground_energy = -3.0 + x * x;
    const double excitation_energy = 0.5 - 0.2 * x * x;
    const LR::FsshSurface surface = LR::make_fssh_surface(
        ground_energy,
        excitation_energy,
        ground_force,
        &excitation_force);
    const double plus = 0.5 * (-2.5 + 0.8 * (x + dx) * (x + dx));
    const double minus = 0.5 * (-2.5 + 0.8 * (x - dx) * (x - dx));
    EXPECT_NEAR(surface.force_hartree_per_bohr(0, 0), -(plus - minus) / (2.0 * dx), 1.0e-11);
    EXPECT_DOUBLE_EQ(surface.energy_hartree, 0.5 * (ground_energy + excitation_energy));
    EXPECT_DOUBLE_EQ(surface.force_hartree_per_bohr(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(ground_force(0, 0), -1.4);
}

TEST(FsshSurface, GroundSurfaceHasIndependentForceStorage)
{
    ModuleBase::matrix force(2, 3);
    force(1, 2) = -0.8;
    const LR::FsshSurface surface = LR::make_fssh_surface(
        -4.0,
        0.0,
        force,
        nullptr);
    force(1, 2) = 7.0;
    EXPECT_DOUBLE_EQ(surface.energy_hartree, -2.0);
    EXPECT_DOUBLE_EQ(surface.force_hartree_per_bohr(1, 2), -0.4);
}

TEST(FsshSurface, RejectsInvalidDimensionsAndNonfiniteData)
{
    ModuleBase::matrix force(2, 3);
    ModuleBase::matrix wrong(1, 3);
    EXPECT_THROW(
        LR::make_fssh_surface(
            -4.0,
            0.2,
            force,
            &wrong),
        std::invalid_argument);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        LR::make_fssh_surface(
            nan,
            0.0,
            force,
            nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        LR::make_fssh_surface(
            -4.0,
            nan,
            force,
            nullptr),
        std::invalid_argument);
    force(0, 1) = nan;
    EXPECT_THROW(
        LR::make_fssh_surface(
            -4.0,
            0.0,
            force,
            nullptr),
        std::invalid_argument);
    force.zero_out();
    ModuleBase::matrix correction(2, 3);
    correction(1, 2) = nan;
    EXPECT_THROW(
        LR::make_fssh_surface(
            -4.0,
            0.2,
            force,
            &correction),
        std::invalid_argument);
}
