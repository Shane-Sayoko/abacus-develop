#include "source_esolver/lr_postprocess.h"

#include <gtest/gtest.h>

TEST(LrPostprocess, FsshUsesForcesAlreadyEvaluatedDuringMd)
{
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "md",
            "fssh",
            true,
            false));
}

TEST(LrPostprocess, SinglePointKeepsAllStateForces)
{
    EXPECT_TRUE(
        ModuleESolver::needs_lr_postprocess_forces(
            "scf",
            "nve",
            true,
            false));
    EXPECT_TRUE(
        ModuleESolver::needs_lr_postprocess_forces(
            "nscf",
            "nve",
            true,
            false));
}

TEST(LrPostprocess, OtherMdIntegratorsKeepExistingBehavior)
{
    EXPECT_TRUE(
        ModuleESolver::needs_lr_postprocess_forces(
            "md",
            "nve",
            true,
            false));
    EXPECT_TRUE(
        ModuleESolver::needs_lr_postprocess_forces(
            "md",
            "nvt",
            true,
            false));
}

TEST(LrPostprocess, FsshNameAloneDoesNotSuppressNonMdOutput)
{
    EXPECT_TRUE(
        ModuleESolver::needs_lr_postprocess_forces(
            "scf",
            "fssh",
            true,
            false));
}

TEST(LrPostprocess, RelaxationKeepsTargetForceBehavior)
{
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "relax",
            "nve",
            true,
            true));
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "relax",
            "fssh",
            true,
            true));
}

TEST(LrPostprocess, DisabledForceOutputStaysDisabled)
{
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "scf",
            "nve",
            false,
            false));
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "md",
            "fssh",
            false,
            false));
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "md",
            "nve",
            false,
            false));
    EXPECT_FALSE(
        ModuleESolver::needs_lr_postprocess_forces(
            "relax",
            "nve",
            false,
            true));
}
