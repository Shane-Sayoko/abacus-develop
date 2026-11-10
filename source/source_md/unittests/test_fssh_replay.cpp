#ifdef __MPI
#include <mpi.h>
#endif
#include "source_md/fssh_replay.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

/** @brief A three-state analytic replay with one constant real coupling. */
std::string replay_text(const std::string &version, const std::string &model_field, bool couple_ground)
{
    const double angle = 0.1;
    const double probability = 0.2 * std::tan(angle);
    std::ostringstream text;
    text << std::setprecision(17) << version << '\n'
         << "nstates 3\nnsteps 1\ndt_au 0.5\ninitial_state 1\ndecoherence 0\nalpha 0.1\n";
    if (version == "FSSH_REPLAY_V3" || version == "FSSH_REPLAY_V4")
    {
        text << model_field;
    }
    if (version != "FSSH_REPLAY_V1")
    {
        text << "initial_coefficients 0 0 1 0 0 0\n";
    }
    text << "data\n1 1 1 ";
    if (couple_ground)
    {
        text << "0 0.005 1\n0 0 0\n0 0.2 0 -0.2 0 0 0 0 0\n";
        text << -std::sin(angle) << " 0 " << std::cos(angle) << " 0 0 0\n";
        text << -std::sin(angle) << " 0 " << std::cos(angle) << " 0 0 0\n";
        text << probability << " 0 0\n";
    }
    else
    {
        // Deliberately supply a large S0 coupling to test the selected projection.
        text << "2 0.005 1\n0 0 0\n0 1e100 0 -1e100 0 0.2 0 -0.2 0\n";
        text << "0 0 " << std::cos(angle) << " 0 " << std::sin(angle) << " 0\n";
        text << "0 0 " << std::cos(angle) << " 0 " << std::sin(angle) << " 0\n";
        text << "0 0 " << probability << '\n';
    }
    text << "inf inf inf\n1 1 1\n";
    return text.str();
}

class FsshReplayModel : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        std::string pattern = ::testing::TempDir() + "abacus_fssh_replay_XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char *directory = mkdtemp(writable.data());
        ASSERT_NE(directory, nullptr);
        directory_ = directory;
    }

    void TearDown() override
    {
        for (const std::string &name : {"input", "comparison.csv", "summary.json"})
        {
            std::remove((directory_ + "/" + name).c_str());
        }
        rmdir(directory_.c_str());
    }

    FsshReplaySummary run(const std::string &text)
    {
        const std::string input_path = directory_ + "/input";
        std::ofstream input(input_path);
        input << text;
        input.close();
        const FsshReplay replay;
        return replay.run(input_path, directory_ + "/comparison.csv", directory_ + "/summary.json");
    }

    void expect_analytic_result(const FsshReplaySummary &result)
    {
        EXPECT_EQ(result.steps, 1U);
        EXPECT_EQ(result.proposed_state_mismatches, 0U);
        EXPECT_NEAR(result.max_hop_probability_error, 0.0, 2.0e-14);
        EXPECT_NEAR(result.max_pre_decoherence_error, 0.0, 2.0e-14);
        EXPECT_NEAR(result.max_post_decoherence_error, 0.0, 2.0e-14);
    }

    std::string directory_;
};

TEST_F(FsshReplayModel, VersionThreeUsesTheExcitedOnlyModel)
{
    expect_analytic_result(run(replay_text("FSSH_REPLAY_V3", "ground_state_coupling 0\n", false)));
}

TEST_F(FsshReplayModel, VersionFourSelectsTripletWithoutGroundPopulation)
{
    expect_analytic_result(run(replay_text(
        "FSSH_REPLAY_V4", "ground_state_coupling 0\nspin_channel 1\n", false)));
    EXPECT_THROW(run(replay_text(
        "FSSH_REPLAY_V4", "ground_state_coupling 1\nspin_channel 1\n", true)), std::runtime_error);
    EXPECT_THROW(run(replay_text(
        "FSSH_REPLAY_V4", "ground_state_coupling 0\nspin_channel 2\n", false)), std::runtime_error);
}

TEST_F(FsshReplayModel, LegacyVersionsAndExplicitTrueKeepGroundDynamics)
{
    expect_analytic_result(run(replay_text("FSSH_REPLAY_V1", "", true)));
    expect_analytic_result(run(replay_text("FSSH_REPLAY_V2", "", true)));
    expect_analytic_result(run(replay_text("FSSH_REPLAY_V3", "ground_state_coupling 1\n", true)));
}

TEST_F(FsshReplayModel, RejectsMissingOrInvalidVersionThreeModel)
{
    for (const std::string &field : {"",
             "ground_state_coupling 2\n",
             "ground_state_coupling -1\n",
             "ground_state_coupling true\n",
             "ground_state_coupling 0.5\n"})
    {
        EXPECT_THROW(run(replay_text("FSSH_REPLAY_V3", field, false)), std::runtime_error);
    }
}

TEST_F(FsshReplayModel, RejectsGroundPopulationAndGroundInitialSurface)
{
    std::string text = replay_text("FSSH_REPLAY_V3", "ground_state_coupling 0\n", false);
    const std::string initial_coefficients = "initial_coefficients 0 0 1 0 0 0";
    const size_t coefficients_position = text.find(initial_coefficients);
    ASSERT_NE(coefficients_position, std::string::npos);
    text.replace(coefficients_position, initial_coefficients.size(), "initial_coefficients 0.6 0 0.8 0 0 0");
    EXPECT_THROW(run(text), std::runtime_error);

    text = replay_text("FSSH_REPLAY_V3", "ground_state_coupling 0\n", false);
    const std::string initial_state = "initial_state 1";
    const size_t state_position = text.find(initial_state);
    ASSERT_NE(state_position, std::string::npos);
    text.replace(state_position, initial_state.size(), "initial_state 0");
    EXPECT_THROW(run(text), std::runtime_error);
}

} // namespace

int main(int argc, char **argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
