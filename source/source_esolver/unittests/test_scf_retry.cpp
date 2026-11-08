#include "../scf_retry.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace
{
struct RetryProbe
{
    std::vector<int> events;
    std::vector<bool> outcomes;
    int solve_count = 0;
    bool retry_config = false;
    bool published = false;
    bool final_convergence = false;

    bool run(bool enabled)
    {
        const auto solve = [this]() -> bool
        {
            events.push_back(retry_config ? 2 : 1);
            return outcomes.at(solve_count++);
        };
        const auto configure = [this](bool retry)
        {
            retry_config = retry;
            events.push_back(retry ? 3 : 4);
        };
        const auto finish = [this](bool converged)
        {
            EXPECT_FALSE(published);
            published = true;
            final_convergence = converged;
            events.push_back(5);
        };
        return ModuleESolver::run_scf_retry(
            enabled,
            solve,
            configure,
            finish);
    }
};
} // namespace

TEST(ScfRetry, SuccessfulBaselineDoesNotChangeConfiguration)
{
    RetryProbe probe;
    probe.outcomes = {true};
    EXPECT_TRUE(probe.run(true));
    EXPECT_EQ(probe.events, std::vector<int>({1, 5}));
}

TEST(ScfRetry, DisabledRetainsOriginalFailure)
{
    RetryProbe probe;
    probe.outcomes = {false};
    EXPECT_FALSE(probe.run(false));
    EXPECT_EQ(probe.events, std::vector<int>({1, 5}));
}

TEST(ScfRetry, RetryPublishesOnlyFinalEndpointAndRestoresBaseline)
{
    RetryProbe probe;
    probe.outcomes = {false, true};
    EXPECT_TRUE(probe.run(true));
    EXPECT_TRUE(probe.final_convergence);
    EXPECT_FALSE(probe.retry_config);
    EXPECT_EQ(
        probe.events, std::vector<int>({1,
                          3,
                          2,
                          5,
                          4}));
    probe.published = false;
    probe.outcomes.push_back(true);
    EXPECT_TRUE(probe.run(true));
    EXPECT_EQ(probe.events.back(), 5);
    EXPECT_EQ(probe.events.at(5), 1);
}

TEST(ScfRetry, ExhaustionStillRejectsEndpoint)
{
    RetryProbe probe;
    probe.outcomes = {false, false};
    EXPECT_FALSE(probe.run(true));
    EXPECT_FALSE(probe.final_convergence);
    EXPECT_FALSE(probe.retry_config);
}

TEST(ScfRetry, ExceptionRestoresBaselineWithoutPublishing)
{
    int solves = 0;
    bool retry_config = false;
    bool published = false;
    const auto solve = [&]() -> bool
    {
        if (++solves == 2)
        {
            throw std::runtime_error("electronic solver failure");
        }
        return false;
    };
    const auto configure = [&](bool retry) { retry_config = retry; };
    const auto finish = [&](bool) { published = true; };
    EXPECT_THROW(
        ModuleESolver::run_scf_retry(
            true,
            solve,
            configure,
            finish),
        std::runtime_error);
    EXPECT_FALSE(retry_config);
    EXPECT_FALSE(published);
}
