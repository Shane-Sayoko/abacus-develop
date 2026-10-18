#include "../lr_util_print.h"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

TEST(LRStateIO, PreservesSmallComplexAmplitudes)
{
    const std::string file = "lr_io_roundtrip.dat";
    const std::complex<double> input[] = {{1.0e-13, -2.0e-15}, {0.125, -0.75}};
    std::complex<double> output[2];
    ASSERT_EQ(LR_Util::write_value(file, 17, input, 1, 2), 2);
    ASSERT_EQ(LR_Util::read_value(file, output, 1, 2), 2);
    EXPECT_EQ(output[0], input[0]);
    EXPECT_EQ(output[1], input[1]);
    std::remove(file.c_str());
}

TEST(LRStateIO, WavefunctionRoundTripAlsoWorksWithNdebug)
{
    const std::string prefix = "lr_psi_roundtrip";
    const double input[] = {0.25, -0.5, 1.0e-14, 0.75};
    LR_Util::write_psi_bandfirst(input, 2, 1, 2, prefix, 0, 0.0, 17);
    const auto output = LR_Util::read_psi_bandfirst<double>(prefix, 0);
    ASSERT_EQ(output.size(), 4);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_DOUBLE_EQ(output.get_pointer()[i], input[i]);
    }
    std::remove((prefix + "_0.dat").c_str());
}

TEST(LRStateIO, RejectsMissingAndTruncatedWavefunctions)
{
    const std::string prefix = "lr_psi_truncated";
    const std::string file = prefix + "_0.dat";
    std::remove(file.c_str());
    EXPECT_THROW(LR_Util::read_psi_bandfirst<double>(prefix, 0), std::runtime_error);
    {
        std::ofstream stream(file);
        stream << "2 1 2\n0.25 -0.5\n";
    }
    EXPECT_THROW(LR_Util::read_psi_bandfirst<double>(prefix, 0), std::runtime_error);
    std::remove(file.c_str());
}
