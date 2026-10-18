#include "source_lcao/module_lr/utils/lr_util.h"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(LrOccupations, RoundsTheSpinTotalRatherThanEachBand)
{
    ModuleBase::matrix occupations(2, 3);
    occupations(0, 0) = 1.0;
    occupations(0, 1) = 1.0;
    occupations(0, 2) = 1.0;
    occupations(1, 0) = 1.0;
    occupations(1, 1) = 0.5000000052;
    occupations(1, 2) = 0.4999999947;
    EXPECT_EQ(LR_Util::spin_imbalance(occupations), 1);
}

TEST(LrOccupations, SumsWeightedKPointsInSpinMajorOrder)
{
    ModuleBase::matrix occupations(4, 2);
    occupations(0, 0) = 0.25;
    occupations(1, 0) = 0.75;
    occupations(0, 1) = 0.25;
    occupations(1, 1) = 0.75;
    occupations(2, 0) = 0.25;
    occupations(3, 0) = 0.75;
    EXPECT_EQ(LR_Util::spin_imbalance(occupations), 1);
}

TEST(LrOccupations, RejectsIncompleteSpinBlocks)
{
    ModuleBase::matrix empty;
    ModuleBase::matrix odd(3, 2);
    EXPECT_THROW(LR_Util::spin_imbalance(empty), std::invalid_argument);
    EXPECT_THROW(LR_Util::spin_imbalance(odd), std::invalid_argument);
}
