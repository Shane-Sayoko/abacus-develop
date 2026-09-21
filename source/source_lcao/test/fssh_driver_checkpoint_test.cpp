#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

#include <mpi.h>

#include "source_lcao/module_operator_lcao/fssh_driver.h"

namespace {

class MpiScope {
public:
    MpiScope()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) {
            MPI_Init(nullptr, nullptr);
            finalize_ = true;
        }
    }

    ~MpiScope()
    {
        if (finalize_) MPI_Finalize();
    }

private:
    bool finalize_ = false;
};

} // namespace

/// @brief Test-only access to the private Löwdin overlap contraction.
struct FsshDriverTestAccess {
    static void compute_lowdin_sigma(FsshDriver& driver,
                                     const ModuleBase::ComplexMatrix& coef_old,
                                     const ModuleBase::ComplexMatrix& coef_new,
                                     const ModuleBase::ComplexMatrix& s_ao_dense,
                                     const std::vector<CasidaWavefunction>& casida_old,
                                     const std::vector<CasidaWavefunction>& casida_new,
                                     int nocc_lr,
                                     int nvirt_lr,
                                     int occ_offset,
                                     const std::vector<double>& occ_phase,
                                     ModuleBase::ComplexMatrix& sigma)
    {
        driver.compute_lowdin_sigma(coef_old,
                                    coef_new,
                                    s_ao_dense,
                                    casida_old,
                                    casida_new,
                                    nocc_lr,
                                    nvirt_lr,
                                    occ_offset,
                                    occ_phase,
                                    sigma);
    }

    static void calculate_nac_from_dense(FsshDriver& driver,
                                         const ModuleBase::ComplexMatrix& coef_old,
                                         const ModuleBase::ComplexMatrix& coef_new,
                                         const ModuleBase::ComplexMatrix& s_ao_dense,
                                         ModuleBase::ComplexMatrix& sigma,
                                         const std::vector<CasidaWavefunction>& casida_old,
                                         const std::vector<CasidaWavefunction>& casida_new,
                                         std::vector<CasidaWavefunction>* aligned)
    {
        driver.calculate_nac_from_dense(
            coef_old, coef_new, s_ao_dense, sigma, true, casida_old, casida_new, aligned);
    }
};

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

TEST(FsshDriverNacGauge, OccupiedOrbitalAndCasidaSignFlipKeepsStationaryNacZero)
{
    constexpr int nocc = 2;
    constexpr int nvirt = 1;
    constexpr int nbasis = nocc + nvirt;
    constexpr int nstates = 3;
    constexpr double dt = 1.0;

    FsshDriver driver;
    driver.init(nbasis, nstates, dt, 0, nocc, nbasis);

    ModuleBase::ComplexMatrix coef_old(nbasis, nbasis);
    ModuleBase::ComplexMatrix coef_new(nbasis, nbasis);
    ModuleBase::ComplexMatrix s_ao(nbasis, nbasis);
    for (int row = 0; row < nbasis; ++row) {
        for (int col = 0; col < nbasis; ++col) {
            const double value = row == col ? 1.0 : 0.0;
            coef_old(row, col) = {value, 0.0};
            coef_new(row, col) = {value, 0.0};
            s_ao(row, col) = {value, 0.0};
        }
    }

    // The first new occupied MO has the opposite sign.  The matching first
    // occupied index of every Casida vector is transformed at the same time,
    // so old and new inputs represent identical stationary many-electron states.
    coef_new(0, 0) = {-1.0, 0.0};
    const double theta = std::acos(-1.0) * 25.0 / 180.0;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    std::vector<CasidaWavefunction> casida_old{
        CasidaWavefunction(0.0, {}, {}, nocc, nvirt),
        CasidaWavefunction(0.1, {cosine, sine}, {}, nocc, nvirt),
        CasidaWavefunction(0.2, {-sine, cosine}, {}, nocc, nvirt)};
    std::vector<CasidaWavefunction> casida_new{
        CasidaWavefunction(0.0, {}, {}, nocc, nvirt),
        CasidaWavefunction(0.1, {-cosine, sine}, {}, nocc, nvirt),
        CasidaWavefunction(0.2, {sine, cosine}, {}, nocc, nvirt)};

    ModuleBase::ComplexMatrix sigma(nstates, nstates);
    FsshDriverTestAccess::compute_lowdin_sigma(driver,
                                               coef_old,
                                               coef_new,
                                               s_ao,
                                               casida_old,
                                               casida_new,
                                               nocc,
                                               nvirt,
                                               0,
                                               {-1.0, 1.0},
                                               sigma);

    for (int i = 0; i < nstates; ++i) {
        for (int j = 0; j < nstates; ++j) {
            const std::complex<double> nac = (sigma(i, j) - std::conj(sigma(j, i))) / (2.0 * dt);
            EXPECT_NEAR(std::abs(nac), 0.0, 1.0e-12) << "NAC(" << i << ", " << j << ")";
        }
    }
}

TEST(FsshDriverNacGauge, AlignsSmallRotationOfExactlyDegenerateStateBlock)
{
    MpiScope mpi;
    constexpr int nocc = 2;
    constexpr int nvirt = 2;
    constexpr int nbasis = nocc + nvirt;
    constexpr int nstates = 5;
    constexpr double dt = 20.670686667591056;

    FsshDriver driver;
    driver.init(nbasis, nstates, dt, 0, nocc, nbasis);

    ModuleBase::ComplexMatrix coef_old(nbasis, nbasis);
    ModuleBase::ComplexMatrix coef_new(nbasis, nbasis);
    ModuleBase::ComplexMatrix s_ao(nbasis, nbasis);
    for (int row = 0; row < nbasis; ++row) {
        for (int col = 0; col < nbasis; ++col) {
            const double value = row == col ? 1.0 : 0.0;
            coef_old(row, col) = {value, 0.0};
            coef_new(row, col) = {value, 0.0};
            s_ao(row, col) = {value, 0.0};
        }
    }

    const double theta = std::acos(-1.0) * 20.0 / 180.0;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    std::vector<CasidaWavefunction> casida_old{
        CasidaWavefunction(0.0, {}, {}, nocc, nvirt),
        CasidaWavefunction(0.1, {1.0, 0.0, 0.0, 0.0}, {}, nocc, nvirt),
        CasidaWavefunction(0.1, {0.0, 1.0, 0.0, 0.0}, {}, nocc, nvirt),
        CasidaWavefunction(0.3, {0.0, 0.0, 1.0, 0.0}, {}, nocc, nvirt),
        CasidaWavefunction(0.4, {0.0, 0.0, 0.0, 1.0}, {}, nocc, nvirt)};
    std::vector<CasidaWavefunction> casida_new{
        CasidaWavefunction(0.0, {}, {}, nocc, nvirt),
        CasidaWavefunction(0.1, {cosine, sine, 0.0, 0.0}, {}, nocc, nvirt),
        CasidaWavefunction(0.1, {-sine, cosine, 0.0, 0.0}, {}, nocc, nvirt),
        CasidaWavefunction(0.3, {0.0, 0.0, 1.0, 0.0}, {}, nocc, nvirt),
        CasidaWavefunction(0.4, {0.0, 0.0, 0.0, 1.0}, {}, nocc, nvirt)};

    ModuleBase::ComplexMatrix sigma(nstates, nstates);
    std::vector<CasidaWavefunction> aligned;
    FsshDriverTestAccess::calculate_nac_from_dense(
        driver, coef_old, coef_new, s_ao, sigma, casida_old, casida_new, &aligned);

    for (int i = 0; i < nstates; ++i) {
        for (int j = 0; j < nstates; ++j) {
            EXPECT_NEAR(std::abs(sigma(i, j)), 0.0, 1.0e-12) << "NAC(" << i << ", " << j << ")";
        }
    }
    ASSERT_EQ(aligned.size(), casida_old.size());
    for (int state = 1; state < nstates; ++state) {
        ASSERT_EQ(aligned[state].X_coeffs.size(), casida_old[state].X_coeffs.size());
        for (size_t index = 0; index < aligned[state].X_coeffs.size(); ++index) {
            EXPECT_NEAR(aligned[state].X_coeffs[index], casida_old[state].X_coeffs[index], 1.0e-12);
        }
    }
}

TEST(FsshDriverReplay, ReportsProductionDecoherenceDiagnostics)
{
    FsshDriver driver;
    driver.init(0, 2, 0.1, 0, 0, 0, true, 7U);

    FsshReplayStepInput input;
    input.energies = {0.0, 0.2};
    input.time_nac.create(2, 2);
    input.time_nac(0, 0) = {0.0, 0.0};
    input.time_nac(0, 1) = {0.1, 0.0};
    input.time_nac(1, 0) = {-0.1, 0.0};
    input.time_nac(1, 1) = {0.0, 0.0};
    input.kinetic_energy = 0.5;
    input.random_number = 0.999;
    input.active_state_before = 0;
    input.active_state_after = 0;

    const FsshReplayStepResult result = driver.run_step_replay(input);
    const double expected_tau = 1.0 / 0.2 * (1.0 + 0.1 / 0.5);
    const double expected_damping = std::exp(-0.1 / expected_tau);

    ASSERT_EQ(result.hopping_probabilities.size(), 2U);
    ASSERT_EQ(result.coefficients_before_decoherence.size(), 2U);
    ASSERT_EQ(result.coefficients_after_decoherence.size(), 2U);
    EXPECT_EQ(result.active_state_before, 0);
    EXPECT_EQ(result.active_state_after, 0);
    EXPECT_NEAR(result.decoherence_tau[1], expected_tau, 1.0e-14);
    EXPECT_NEAR(result.decoherence_damping[1], expected_damping, 1.0e-14);
    EXPECT_NEAR(std::abs(result.coefficients_after_decoherence[1]),
                std::abs(result.coefficients_before_decoherence[1]) * expected_damping,
                1.0e-14);
}

TEST(FsshDriverReplay, RejectsActiveStateDiscontinuity)
{
    FsshDriver driver;
    driver.init(0, 2, 0.1, 0, 0, 0, false, 7U);

    FsshReplayStepInput input;
    input.energies = {0.0, 0.2};
    input.time_nac.create(2, 2);
    input.active_state_before = 1;
    input.active_state_after = 1;

    EXPECT_THROW(driver.run_step_replay(input), std::runtime_error);
}

TEST(FsshDriverReplay, CapsIndividualHoppingProbabilitiesAtOne)
{
    FsshDriver driver;
    driver.init(0, 2, 0.1, 0, 0, 0, false, 7U);

    FsshReplayStepInput input;
    input.energies = {0.0, 0.0};
    input.time_nac.create(2, 2);
    input.time_nac(0, 0) = {0.0, 0.0};
    input.time_nac(0, 1) = {100.0, 0.0};
    input.time_nac(1, 0) = {-100.0, 0.0};
    input.time_nac(1, 1) = {0.0, 0.0};
    input.kinetic_energy = 1.0;
    input.random_number = 0.999;
    input.active_state_before = 0;
    input.active_state_after = 0;

    const FsshReplayStepResult result = driver.run_step_replay(input);
    ASSERT_EQ(result.hopping_probabilities.size(), 2U);
    EXPECT_DOUBLE_EQ(result.hopping_probabilities[1], 1.0);
    EXPECT_EQ(result.proposed_state, 1);
}

TEST(FsshDriverReplay, AcceptsSavedSegmentBoundaryState)
{
    FsshDriver driver;
    driver.init(0, 2, 0.1, 0, 0, 0, false, 7U);
    driver.set_replay_state(1, {{0.6, 0.0}, {0.8, 0.0}});

    FsshReplayStepInput input;
    input.energies = {0.0, 0.0};
    input.time_nac.create(2, 2);
    input.time_nac(0, 0) = {0.0, 0.0};
    input.time_nac(0, 1) = {0.0, 0.0};
    input.time_nac(1, 0) = {0.0, 0.0};
    input.time_nac(1, 1) = {0.0, 0.0};
    input.kinetic_energy = 1.0;
    input.random_number = 0.5;
    input.active_state_before = 1;
    input.active_state_after = 1;

    const FsshReplayStepResult result = driver.run_step_replay(input);
    EXPECT_EQ(result.active_state_before, 1);
    EXPECT_EQ(result.active_state_after, 1);
    EXPECT_NEAR(std::abs(result.coefficients_before_decoherence[0]), 0.6, 1.0e-14);
    EXPECT_NEAR(std::abs(result.coefficients_before_decoherence[1]), 0.8, 1.0e-14);
}

TEST(FsshDriverVelocityRescale, ConservesEnergyForAcceptedHops)
{
    FsshDriver driver;

    for (const auto& test_case : std::vector<std::pair<double, double>>{{1.0e-12, 0.5e-12},
                                                                         {1.0e-8, -0.4e-8},
                                                                         {1.0, 0.0},
                                                                         {1.0, 0.25},
                                                                         {1.0, -0.75},
                                                                         {1.0, 1.0}}) {
        const FsshVelocityRescaleResult result =
            driver.evaluate_velocity_rescale(test_case.first, 0.0, test_case.second);
        ASSERT_TRUE(result.accepted);
        EXPECT_TRUE(std::isfinite(result.scale_factor));
        EXPECT_NEAR(result.kinetic_energy_after + test_case.second, test_case.first, 1.0e-14);
        EXPECT_NEAR(result.total_energy_residual, 0.0, 1.0e-14);
    }
}

TEST(FsshDriverVelocityRescale, RejectsInfeasibleAndZeroKineticEnergyCases)
{
    FsshDriver driver;

    EXPECT_FALSE(driver.evaluate_velocity_rescale(0.1, 0.0, 0.2).accepted);
    EXPECT_TRUE(driver.evaluate_velocity_rescale(0.0, 0.0, 0.0).accepted);
    EXPECT_FALSE(driver.evaluate_velocity_rescale(0.0, 0.0, -0.1).accepted);
    EXPECT_THROW(driver.evaluate_velocity_rescale(-1.0, 0.0, 0.0), std::invalid_argument);
}
