#include "source_lcao/module_operator_lcao/fssh_driver.h"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace
{
FsshModelConfig triplet_model()
{
    FsshModelConfig model;
    model.ground_state_coupling = false;
    model.spin_channel = FsshSpinChannel::triplet;
    return model;
}

template <typename T> void read_value(std::istream &stream, T &value)
{
    if (!(stream >> value))
    {
        throw std::runtime_error("Malformed Newton-X microstep reference input");
    }
}
} // namespace

TEST(FsshDriverTriplet, AcceptsAndRejectsHopsWithinMicrosteps)
{
    for (double kinetic : {0.05, 0.005})
    {
        FsshDriver driver;
        const auto model = triplet_model();
        driver.init(model, 0, 3, 0.1, 1, 0, 0, false, 17U, 0.0);
        const std::vector<double> energies = {0.0, 0.0, 0.02};
        ModuleBase::ComplexMatrix nac(3, 3);
        nac(1, 2) = 0.1;
        nac(2, 1) = -0.1;
        std::vector<double> variates(20, 0.999);
        variates[0] = 0.0;
        const auto records = driver.run_triplet_step(nac, energies, energies, kinetic, variates);
        EXPECT_EQ(records[0].proposed, 2);
        const bool accepted = kinetic > 0.02;
        EXPECT_EQ(records[0].accepted, accepted);
        const int expected_state = accepted ? 2 : 1;
        EXPECT_EQ(records[0].active_after, expected_state);
        EXPECT_EQ(records[1].active_before, expected_state);
    }
}

TEST(FsshDriverTriplet, InvalidReferenceLeavesStateUnchanged)
{
    FsshDriver driver;
    const auto model = triplet_model();
    driver.init(model, 0, 3, 0.1, 1, 0, 0, true, 17U, 0.0);
    const std::vector<double> energies(3, 0.0);
    ModuleBase::ComplexMatrix nac(3, 3);
    std::vector<double> variates(20, 0.9);
    variates.back() = 1.0;
    EXPECT_THROW(driver.run_triplet_step(nac, energies, energies, 0.1, variates), std::invalid_argument);
    EXPECT_EQ(driver.get_current_state(), 1);
    variates.back() = 0.9;
    nac(1, 2) = 0.1;
    EXPECT_THROW(driver.run_triplet_step(nac, energies, energies, 0.1, variates), std::invalid_argument);
}

TEST(FsshDriverTriplet, RestartsAfterHopWithButcherAndDecoherenceHistory)
{
    for (bool decoherence : {false, true})
    {
        const auto model = triplet_model();
        FsshDriver driver;
        driver.init(model, 0, 3, 4.1341373335, 1, 0, 0, decoherence, 17U, 0.0);
        const std::vector<double> energies = {0.0, 0.0, 0.02};
        ModuleBase::ComplexMatrix nac(3, 3);
        nac(1, 2) = 0.15;
        nac(2, 1) = -0.15;
        std::vector<double> variates(20, 0.999);
        variates[13] = 0.01;
        const auto first = driver.run_triplet_step(nac, energies, energies, 0.05, variates);
        ASSERT_TRUE(first[13].accepted);
        ASSERT_EQ(driver.get_current_state(), 2);
        std::stringstream checkpoint;
        ASSERT_TRUE(driver.save_checkpoint(checkpoint));
        FsshDriver restored;
        ASSERT_TRUE(restored.load_checkpoint(checkpoint, model))
            << "decoherence=" << decoherence << " header=" << checkpoint.str().substr(0, 240)
            << " tail=" << checkpoint.str().substr(checkpoint.str().size() - 1000);
        const std::vector<double> no_hops(20, 0.999);
        const auto continuous = driver.run_triplet_step(nac, energies, energies, 0.03, no_hops);
        const auto restarted = restored.run_triplet_step(nac, energies, energies, 0.03, no_hops);
        for (std::size_t step = 0; step < continuous.size(); ++step)
        {
            EXPECT_EQ(continuous[step].after_decoherence, restarted[step].after_decoherence);
            EXPECT_EQ(continuous[step].probabilities, restarted[step].probabilities);
            EXPECT_EQ(continuous[step].active_after, restarted[step].active_after);
        }
        std::string old_checkpoint = checkpoint.str();
        old_checkpoint[old_checkpoint.find('5')] = '4';
        std::stringstream old_input(old_checkpoint);
        EXPECT_FALSE(restored.load_checkpoint(old_input, model));
        const std::string truncated = checkpoint.str().substr(0, checkpoint.str().size() - 20);
        std::stringstream corrupt_input(truncated);
        EXPECT_FALSE(restored.load_checkpoint(corrupt_input, model));
    }
}

// This fixture is supplied by a Slurm workflow executing the unmodified NX binary.
TEST(FsshDriverNewtonX, ExternalMicrostepReference)
{
    const char *input_path = std::getenv("TT_NX_INPUT");
    const char *output_path = std::getenv("TT_NX_OUTPUT");
    if (input_path == nullptr || output_path == nullptr)
    {
        GTEST_SKIP() << "Newton-X reference is supplied by the para Slurm workflow";
    }
    std::ifstream input(input_path);
    std::ofstream output(output_path);
    ASSERT_TRUE(input.good());
    ASSERT_TRUE(output.good());
    output << std::setprecision(17);
    int macros = 0;
    int nstates = 0;
    int initial = 0;
    int decoherence = 0;
    double dt = 0.0;
    read_value(input, macros);
    read_value(input, nstates);
    read_value(input, dt);
    read_value(input, initial);
    read_value(input, decoherence);
    ASSERT_GT(macros, 0);
    ASSERT_GE(nstates, 3);
    FsshDriver driver;
    const bool enable_decoherence = decoherence != 0;
    const auto model = triplet_model();
    driver.init(model, 0, nstates, dt, initial, 0, 0, enable_decoherence, 17U, 0.0);
    for (int macro = 0; macro < macros; ++macro)
    {
        double kinetic = 0.0;
        read_value(input, kinetic);
        std::vector<double> old_energies(nstates);
        std::vector<double> energies(nstates);
        for (double &value : old_energies)
        {
            read_value(input, value);
        }
        for (double &value : energies)
        {
            read_value(input, value);
        }
        ModuleBase::ComplexMatrix nac(nstates, nstates);
        for (int row = 0; row < nstates; ++row)
        {
            for (int column = 0; column < nstates; ++column)
            {
                double value = 0.0;
                read_value(input, value);
                nac(row, column) = value;
            }
        }
        std::vector<double> variates(20);
        for (double &value : variates)
        {
            read_value(input, value);
        }
        const auto records = driver.run_triplet_step(nac, old_energies, energies, kinetic, variates);
        for (const auto &record : records)
        {
            output << record.active_before << ' ' << record.proposed << ' ' << record.active_after << ' '
                   << record.accepted << ' ' << record.kinetic_after << ' ' << record.random_number;
            for (int state = 1; state < nstates; ++state)
            {
                output << ' ' << record.probabilities[state] << ' '
                       << record.before_decoherence[state].real() << ' ' << record.before_decoherence[state].imag()
                       << ' ' << record.after_decoherence[state].real()
                       << ' ' << record.after_decoherence[state].imag();
            }
            output << '\n';
        }
    }
    input >> std::ws;
    EXPECT_TRUE(input.eof());
    EXPECT_TRUE(output.good());
}
