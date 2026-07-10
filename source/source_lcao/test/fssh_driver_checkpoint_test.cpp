#include <gtest/gtest.h>

#include <sstream>

#include "source_lcao/module_operator_lcao/fssh_driver.h"

TEST(FsshDriverCheckpoint, RoundTripsDeterministicState)
{
    FsshDriver original;
    original.init(4, 3, 2.5, 1, 2, 5, true, 42, 1.0e-3);
    original.set_casida_cache({CasidaWavefunction(0.0, {}, {}, 2, 3),
                               CasidaWavefunction(0.1, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {}, 2, 3),
                               CasidaWavefunction(0.2, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, {}, 2, 3)});

    std::stringstream checkpoint;
    ASSERT_TRUE(original.save_checkpoint(checkpoint));

    FsshDriver restored;
    ASSERT_TRUE(restored.load_checkpoint(checkpoint));
    EXPECT_EQ(restored.get_nstates(), 3);
    EXPECT_EQ(restored.get_current_state(), 1);
}

TEST(FsshDriverCheckpoint, RejectsInvalidHeader)
{
    FsshDriver driver;
    std::stringstream invalid("not-a-checkpoint\n");
    EXPECT_FALSE(driver.load_checkpoint(invalid));
}
