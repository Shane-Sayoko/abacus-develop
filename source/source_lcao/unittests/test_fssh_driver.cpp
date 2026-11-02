#include "source_base/timer.h"
#include "source_lcao/module_operator_lcao/fssh_driver.h"

#include <cmath>
#include <gtest/gtest.h>
#ifdef __MPI
#include <mpi.h>
#endif
#include <sstream>
#include <utility>
#include <vector>

namespace
{

class MpiScope
{
  public:
    MpiScope()
    {
#ifdef __MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0)
        {
            MPI_Init(nullptr, nullptr);
        }
#endif
    }
};

} // namespace

/// @brief Test-only access to the private Löwdin overlap contraction.
struct FsshDriverTestAccess
{
    static std::vector<double> contract_mo_overlap(
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const ModuleBase::ComplexMatrix &s_ao,
        const std::vector<int> &orbitals,
        int nbasis)
    {
        return FsshDriver::contract_mo_overlap(
            coef_old,
            coef_new,
            s_ao,
            orbitals,
            nbasis);
    }

    static void transform_casida_amplitudes(
        std::vector<double> &amplitudes,
        const std::vector<double> &occupied_transform,
        const std::vector<double> &virtual_transform,
        int nocc,
        int nvirt)
    {
        FsshDriver::transform_casida_amplitudes(
            amplitudes,
            occupied_transform,
            virtual_transform,
            nocc,
            nvirt);
    }

    static void parse_latest_csr(FsshDriver &driver, const std::string &filename) { driver.parse_latest_csr(filename); }

    static void compute_lowdin_sigma(
        FsshDriver &driver,
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const ModuleBase::ComplexMatrix &s_ao_dense,
        const std::vector<CasidaWavefunction> &casida_old,
        const std::vector<CasidaWavefunction> &casida_new,
        int nocc_lr,
        int nvirt_lr,
        int occ_offset,
        const std::vector<double> &occ_phase,
        ModuleBase::ComplexMatrix &sigma)
    {
        driver.compute_lowdin_sigma(
            coef_old,
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

    static void calculate_nac_from_dense(
        FsshDriver &driver,
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const ModuleBase::ComplexMatrix &s_ao_dense,
        ModuleBase::ComplexMatrix &sigma,
        const std::vector<CasidaWavefunction> &casida_old,
        const std::vector<CasidaWavefunction> &casida_new,
        std::vector<CasidaWavefunction> *aligned,
        const std::vector<double> &ks_energies_old = {},
        const std::vector<double> &ks_energies_new = {},
        ModuleBase::ComplexMatrix *coef_aligned = nullptr)
    {
        driver.calculate_nac_from_dense(
            coef_old,
            coef_new,
            s_ao_dense,
            sigma,
            true,
            casida_old,
            casida_new,
            aligned,
            ks_energies_old,
            ks_energies_new,
            coef_aligned);
    }
};

TEST(FsshDriverContraction, ComplexMoOverlapMatchesFourIndexReference)
{
    const int nbasis = 7;
    ModuleBase::ComplexMatrix coef_old(6, nbasis);
    ModuleBase::ComplexMatrix coef_new(5, nbasis);
    ModuleBase::ComplexMatrix s_ao(nbasis, nbasis);
    for (int p = 0; p < coef_old.nr; ++p)
    {
        for (int mu = 0; mu < nbasis; ++mu)
        {
            coef_old(p, mu) = {std::sin(0.3 * (p + 1) * (mu + 1)), 0.2 * std::cos(p + 2 * mu)};
            if (p < coef_new.nr)
            {
                coef_new(p, mu) = {std::cos(0.4 * (p + 2) * (mu + 1)), 0.3 * std::sin(2 * p - mu)};
            }
        }
    }
    for (int mu = 0; mu < nbasis; ++mu)
    {
        for (int nu = 0; nu < nbasis; ++nu)
        {
            // Inter-time overlap need not be symmetric or Hermitian.
            s_ao(mu, nu) = {0.1 * (mu + 1) - 0.03 * nu, 0.02 * (mu - 2 * nu + 1)};
        }
    }
    // Noncontiguous, reordered orbitals, including a row missing only from the new matrix.
    const std::vector<int> orbitals{3,
        1,
        5,
        0};
    const int nmo = static_cast<int>(orbitals.size());
    const auto overlap = FsshDriverTestAccess::contract_mo_overlap(
        coef_old,
        coef_new,
        s_ao,
        orbitals,
        nbasis);
    ASSERT_EQ(overlap.size(), static_cast<size_t>(nmo * nmo));
    for (int p = 0; p < nmo; ++p)
    {
        for (int q = 0; q < nmo; ++q)
        {
            double reference = 0.0;
            if (orbitals[p] < coef_old.nr && orbitals[q] < coef_new.nr)
            {
                for (int mu = 0; mu < nbasis; ++mu)
                {
                    for (int nu = 0; nu < nbasis; ++nu)
                    {
                        reference +=
                            (std::conj(coef_old(orbitals[p], mu)) * s_ao(mu, nu) * coef_new(orbitals[q], nu)).real();
                    }
                }
            }
            EXPECT_NEAR(overlap[p * nmo + q], reference, 2.0e-13) << "MO(" << p << ", " << q << ")";
        }
    }
    EXPECT_TRUE(
        FsshDriverTestAccess::contract_mo_overlap(
            coef_old,
            coef_new,
            s_ao, {},
            nbasis)
            .empty());
}

TEST(FsshDriverContraction, RectangularCasidaAmplitudesMatchFourIndexReference)
{
    const int nocc = 3;
    const int nvirt = 5;
    std::vector<double> occupied_transform(nocc * nocc, 0.0);
    std::vector<double> virtual_transform(nvirt * nvirt, 0.0);
    const double cosine = std::cos(0.37);
    const double sine = std::sin(0.37);
    occupied_transform[0] = cosine;
    occupied_transform[1] = -sine;
    occupied_transform[nocc] = sine;
    occupied_transform[nocc + 1] = cosine;
    occupied_transform[2 * nocc + 2] = -1.0;
    // A signed cyclic permutation makes the virtual rotation nonsymmetric.
    for (int b = 0; b < nvirt; ++b)
    {
        virtual_transform[b * nvirt + (b + 1) % nvirt] = b % 2 == 0 ? 1.0 : -1.0;
    }
    for (int component = 0; component < 2; ++component)
    {
        std::vector<double> amplitudes(nocc * nvirt);
        for (int i = 0; i < nocc * nvirt; ++i)
        {
            amplitudes[i] = std::sin(0.21 * (i + 1)) + 0.07 * component * (i - 3);
        }
        const std::vector<double> original = amplitudes;
        FsshDriverTestAccess::transform_casida_amplitudes(
            amplitudes,
            occupied_transform,
            virtual_transform,
            nocc,
            nvirt);
        for (int i = 0; i < nocc; ++i)
        {
            for (int a = 0; a < nvirt; ++a)
            {
                double reference = 0.0;
                for (int j = 0; j < nocc; ++j)
                {
                    for (int b = 0; b < nvirt; ++b)
                    {
                        reference += occupied_transform[j * nocc + i] * original[j * nvirt + b] *
                                     virtual_transform[b * nvirt + a];
                    }
                }
                EXPECT_NEAR(amplitudes[i * nvirt + a], reference, 2.0e-14);
            }
        }
    }
}

TEST(FsshDriverContraction, LeavesAbsentOrMismatchedCasidaAmplitudesUnchanged)
{
    const std::vector<double> occupied_transform{1.0};
    const std::vector<double> virtual_transform{1.0,
        0.0,
        0.0,
        1.0};
    std::vector<double> absent;
    FsshDriverTestAccess::transform_casida_amplitudes(
        absent,
        occupied_transform,
        virtual_transform,
        1,
        2);
    EXPECT_TRUE(absent.empty());
    std::vector<double> mismatched{0.4};
    FsshDriverTestAccess::transform_casida_amplitudes(
        mismatched,
        occupied_transform,
        virtual_transform,
        1,
        2);
    EXPECT_EQ(mismatched, std::vector<double>({0.4}));
}

TEST(FsshDriverCheckpoint, RoundTripsDeterministicState)
{
    FsshDriver original;
    original.init(
        4,
        3,
        2.5,
        1,
        2,
        5,
        true,
        42,
        1.0e-3);
    original.set_casida_cache(
        {CasidaWavefunction(
             0.0, {}, {},
             2,
             3),
            CasidaWavefunction(
                0.1,
                {1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0},
                {},
                2,
                3),
            CasidaWavefunction(
                0.2,
                {0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0},
                {},
                2,
                3)});

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
    driver.init(
        nbasis,
        nstates,
        dt,
        0,
        nocc,
        nbasis);

    ModuleBase::ComplexMatrix coef_old(nbasis, nbasis);
    ModuleBase::ComplexMatrix coef_new(nbasis, nbasis);
    ModuleBase::ComplexMatrix s_ao(nbasis, nbasis);
    for (int row = 0; row < nbasis; ++row)
    {
        for (int col = 0; col < nbasis; ++col)
        {
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
    std::vector<CasidaWavefunction> casida_old{CasidaWavefunction(
                                                   0.0, {}, {},
                                                   nocc,
                                                   nvirt),
        CasidaWavefunction(
            0.1, {cosine, sine}, {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.2, {-sine, cosine}, {},
            nocc,
            nvirt)};
    std::vector<CasidaWavefunction> casida_new{CasidaWavefunction(
                                                   0.0, {}, {},
                                                   nocc,
                                                   nvirt),
        CasidaWavefunction(
            0.1, {-cosine, sine}, {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.2, {sine, cosine}, {},
            nocc,
            nvirt)};

    ModuleBase::ComplexMatrix sigma(nstates, nstates);
    FsshDriverTestAccess::compute_lowdin_sigma(
        driver,
        coef_old,
        coef_new,
        s_ao,
        casida_old,
        casida_new,
        nocc,
        nvirt,
        0, {-1.0, 1.0},
        sigma);

    for (int i = 0; i < nstates; ++i)
    {
        for (int j = 0; j < nstates; ++j)
        {
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
    driver.init(
        nbasis,
        nstates,
        dt,
        0,
        nocc,
        nbasis);

    ModuleBase::ComplexMatrix coef_old(nbasis, nbasis);
    ModuleBase::ComplexMatrix coef_new(nbasis, nbasis);
    ModuleBase::ComplexMatrix s_ao(nbasis, nbasis);
    for (int row = 0; row < nbasis; ++row)
    {
        for (int col = 0; col < nbasis; ++col)
        {
            const double value = row == col ? 1.0 : 0.0;
            coef_old(row, col) = {value, 0.0};
            coef_new(row, col) = {value, 0.0};
            s_ao(row, col) = {value, 0.0};
        }
    }

    const double theta = std::acos(-1.0) * 20.0 / 180.0;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    std::vector<CasidaWavefunction> casida_old{CasidaWavefunction(
                                                   0.0, {}, {},
                                                   nocc,
                                                   nvirt),
        CasidaWavefunction(
            0.1,
            {1.0,
                0.0,
                0.0,
                0.0},
            {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.1,
            {0.0,
                1.0,
                0.0,
                0.0},
            {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.3,
            {0.0,
                0.0,
                1.0,
                0.0},
            {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.4,
            {0.0,
                0.0,
                0.0,
                1.0},
            {},
            nocc,
            nvirt)};
    std::vector<CasidaWavefunction> casida_new{CasidaWavefunction(
                                                   0.0, {}, {},
                                                   nocc,
                                                   nvirt),
        CasidaWavefunction(
            0.1,
            {cosine,
                sine,
                0.0,
                0.0},
            {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.1,
            {-sine,
                cosine,
                0.0,
                0.0},
            {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.3,
            {0.0,
                0.0,
                1.0,
                0.0},
            {},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.4,
            {0.0,
                0.0,
                0.0,
                1.0},
            {},
            nocc,
            nvirt)};

    ModuleBase::ComplexMatrix sigma(nstates, nstates);
    std::vector<CasidaWavefunction> aligned;
    const auto &timer = ModuleBase::timer::timer_pool["FsshDriver"]["calculate_nac_from_dense"];
    const std::size_t calls_before = timer.calls;
    FsshDriverTestAccess::calculate_nac_from_dense(
        driver,
        coef_old,
        coef_new,
        s_ao,
        sigma,
        casida_old,
        casida_new,
        &aligned);

    EXPECT_EQ(timer.calls, calls_before + 1);
    EXPECT_TRUE(timer.start_flag); // The TDA early return must close the function timer.
    EXPECT_GE(timer.cpu_second, 0.0);

    for (int i = 0; i < nstates; ++i)
    {
        for (int j = 0; j < nstates; ++j)
        {
            EXPECT_NEAR(std::abs(sigma(i, j)), 0.0, 1.0e-12) << "NAC(" << i << ", " << j << ")";
        }
    }
    ASSERT_EQ(aligned.size(), casida_old.size());
    for (int state = 1; state < nstates; ++state)
    {
        ASSERT_EQ(aligned[state].X_coeffs.size(), casida_old[state].X_coeffs.size());
        for (size_t index = 0; index < aligned[state].X_coeffs.size(); ++index)
        {
            EXPECT_NEAR(aligned[state].X_coeffs[index], casida_old[state].X_coeffs[index], 1.0e-12);
        }
    }
}

TEST(FsshDriverNacGauge, AlignsDegenerateKsOrbitalsAndCovariantCasidaAmplitudes)
{
    MpiScope mpi;
    constexpr int nocc = 2;
    constexpr int nvirt = 2;
    constexpr int nstates = 5;
    FsshDriver driver;
    driver.init(
        4,
        nstates,
        1.0,
        0,
        nocc,
        nocc + nvirt);

    ModuleBase::ComplexMatrix coef_old(4, 4);
    ModuleBase::ComplexMatrix coef_new(4, 4);
    ModuleBase::ComplexMatrix s_ao(4, 4);
    for (int i = 0; i < 4; ++i)
    {
        coef_old(i, i) = {1.0, 0.0};
        s_ao(i, i) = {1.0, 0.0};
    }
    const double occ_angle = 20.0 * std::acos(-1.0) / 180.0;
    const double virt_angle = 35.0 * std::acos(-1.0) / 180.0;
    const double co = std::cos(occ_angle), so = std::sin(occ_angle);
    const double cv = std::cos(virt_angle), sv = std::sin(virt_angle);
    // Solver orbitals are C_new = U^T C_old in both exactly degenerate blocks.
    coef_new(0, 0) = {co, 0.0};
    coef_new(0, 1) = {so, 0.0};
    coef_new(1, 0) = {-so, 0.0};
    coef_new(1, 1) = {co, 0.0};
    coef_new(2, 2) = {cv, 0.0};
    coef_new(2, 3) = {sv, 0.0};
    coef_new(3, 2) = {-sv, 0.0};
    coef_new(3, 3) = {cv, 0.0};

    std::vector<CasidaWavefunction> casida_old{CasidaWavefunction(
                                                   0.0, {}, {},
                                                   nocc,
                                                   nvirt),
        CasidaWavefunction(
            0.10,
            {1.0,
                0.0,
                0.0,
                0.0},
            {0.25,
                0.0,
                0.0,
                0.0},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.20,
            {0.0,
                1.0,
                0.0,
                0.0},
            {0.0,
                0.25,
                0.0,
                0.0},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.30,
            {0.0,
                0.0,
                1.0,
                0.0},
            {0.0,
                0.0,
                0.25,
                0.0},
            nocc,
            nvirt),
        CasidaWavefunction(
            0.40,
            {0.0,
                0.0,
                0.0,
                1.0},
            {0.0,
                0.0,
                0.0,
                0.25},
            nocc,
            nvirt)};
    std::vector<CasidaWavefunction> casida_new = casida_old;
    for (int state = 1; state < nstates; ++state)
    {
        const std::vector<double> x = casida_old[state].X_coeffs;
        const std::vector<double> y = casida_old[state].Y_coeffs;
        for (int i = 0; i < nocc; ++i)
        {
            for (int a = 0; a < nvirt; ++a)
            {
                double x_value = 0.0, y_value = 0.0;
                for (int j = 0; j < nocc; ++j)
                {
                    const double u_occ_transpose = (i == 0 && j == 0)   ? co
                                                   : (i == 0 && j == 1) ? so
                                                   : (i == 1 && j == 0) ? -so
                                                                        : co;
                    for (int b = 0; b < nvirt; ++b)
                    {
                        const double u_virt = (b == 0 && a == 0)   ? cv
                                              : (b == 0 && a == 1) ? -sv
                                              : (b == 1 && a == 0) ? sv
                                                                   : cv;
                        x_value += u_occ_transpose * x[j * nvirt + b] * u_virt;
                        y_value += u_occ_transpose * y[j * nvirt + b] * u_virt;
                    }
                }
                casida_new[state].X_coeffs[i * nvirt + a] = x_value;
                casida_new[state].Y_coeffs[i * nvirt + a] = y_value;
            }
        }
    }

    ModuleBase::ComplexMatrix sigma(nstates, nstates);
    ModuleBase::ComplexMatrix coef_aligned;
    std::vector<CasidaWavefunction> casida_aligned;
    const std::vector<double> ks_energies{-0.50,
        -0.50,
        0.20,
        0.20};
    FsshDriverTestAccess::calculate_nac_from_dense(
        driver,
        coef_old,
        coef_new,
        s_ao,
        sigma,
        casida_old,
        casida_new,
        &casida_aligned,
        ks_energies,
        ks_energies,
        &coef_aligned);

    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            EXPECT_NEAR(coef_aligned(row, column).real(), coef_old(row, column).real(), 1.0e-12);
        }
    }
    for (int state = 1; state < nstates; ++state)
    {
        for (size_t index = 0; index < casida_old[state].X_coeffs.size(); ++index)
        {
            EXPECT_NEAR(casida_aligned[state].X_coeffs[index], casida_old[state].X_coeffs[index], 1.0e-12);
            EXPECT_NEAR(casida_aligned[state].Y_coeffs[index], casida_old[state].Y_coeffs[index], 1.0e-12);
        }
    }
    for (int i = 0; i < nstates; ++i)
    {
        for (int j = 0; j < nstates; ++j)
        {
            EXPECT_NEAR(sigma(i, j).real(), 0.0, 1.0e-12);
        }
    }
}

TEST(FsshDriverReplay, ReportsProductionDecoherenceDiagnostics)
{
    MpiScope mpi;
    FsshDriver driver;
    driver.init(
        0,
        2,
        0.1,
        0,
        0,
        0,
        true,
        7U);

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

    const auto &timer = ModuleBase::timer::timer_pool["FsshDriver"]["propagate_rk4"];
    const std::size_t calls_before = timer.calls;
    const FsshReplayStepResult result = driver.run_step_replay(input);
    EXPECT_EQ(timer.calls, calls_before + 1);
    EXPECT_TRUE(timer.start_flag);
    EXPECT_GE(timer.cpu_second, 0.0);
    const double expected_tau = 1.0 / 0.2 * (1.0 + 0.1 / 0.5);
    const double expected_damping = std::exp(-0.1 / expected_tau);

    ASSERT_EQ(result.hopping_probabilities.size(), 2U);
    ASSERT_EQ(result.coefficients_before_decoherence.size(), 2U);
    ASSERT_EQ(result.coefficients_after_decoherence.size(), 2U);
    EXPECT_EQ(result.active_state_before, 0);
    EXPECT_EQ(result.active_state_after, 0);
    EXPECT_NEAR(result.decoherence_tau[1], expected_tau, 1.0e-14);
    EXPECT_NEAR(result.decoherence_damping[1], expected_damping, 1.0e-14);
    EXPECT_NEAR(
        std::abs(result.coefficients_after_decoherence[1]),
        std::abs(result.coefficients_before_decoherence[1]) * expected_damping, 1.0e-14);
}

TEST(FsshDriverTiming, ClosesCsrTimerAfterReadFailure)
{
    MpiScope mpi;
    FsshDriver driver;
    driver.init(
        2,
        2,
        0.1,
        0,
        1,
        2);
    const auto &timer = ModuleBase::timer::timer_pool["FsshDriver"]["parse_latest_csr"];
    const std::size_t calls_before = timer.calls;

    // Two failures must preserve the original error and leave the timer reusable.
    EXPECT_THROW(FsshDriverTestAccess::parse_latest_csr(driver, ""), std::runtime_error);
    EXPECT_TRUE(timer.start_flag);
    EXPECT_THROW(FsshDriverTestAccess::parse_latest_csr(driver, ""), std::runtime_error);
    EXPECT_TRUE(timer.start_flag);
    EXPECT_EQ(timer.calls, calls_before + 2);
}

TEST(FsshDriverReplay, RejectsActiveStateDiscontinuity)
{
    FsshDriver driver;
    driver.init(
        0,
        2,
        0.1,
        0,
        0,
        0,
        false,
        7U);

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
    driver.init(
        0,
        2,
        0.1,
        0,
        0,
        0,
        false,
        7U);

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
    driver.init(
        0,
        2,
        0.1,
        0,
        0,
        0,
        false,
        7U);
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

    for (const auto &test_case : std::vector<std::pair<double, double>>{
             {1.0e-12, 0.5e-12}, {1.0e-8, -0.4e-8}, {1.0, 0.0}, {1.0, 0.25}, {1.0, -0.75}, {1.0, 1.0}})
    {
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

TEST(FsshDriverStateSpace, ProjectsGroundCouplingsWithoutRemovingExcitedTransfer)
{
    FsshModelConfig model;
    model.ground_state_coupling = false;
    FsshDriver driver;
    const double dt = 0.5;
    const double coupling = 0.2;
    driver.init(
        model,
        0,
        3,
        dt,
        1,
        0,
        0,
        false,
        7U,
        0.0);

    FsshReplayStepInput input;
    input.energies = {0.0, 0.0, 0.0};
    input.time_nac.create(3, 3);
    input.time_nac(0, 1) = {1.0e200, 0.0};
    input.time_nac(1, 0) = {-1.0e200, 0.0};
    input.time_nac(1, 2) = {coupling, 0.0};
    input.time_nac(2, 1) = {-coupling, 0.0};
    input.kinetic_energy = 1.0;
    input.random_number = 0.005;
    input.active_state_before = 1;
    input.active_state_after = 1;
    const FsshReplayStepResult result = driver.run_step_replay(input);

    EXPECT_EQ(result.coefficients_before_decoherence[0], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(result.coefficients_after_decoherence[0], std::complex<double>(0.0, 0.0));
    EXPECT_DOUBLE_EQ(result.hopping_probabilities[0], 0.0);
    EXPECT_NEAR(result.coefficients_before_decoherence[1].real(), std::cos(coupling * dt), 2.0e-14);
    EXPECT_NEAR(result.coefficients_before_decoherence[2].real(), std::sin(coupling * dt), 2.0e-14);
    const double probability = 2.0 * dt * coupling * std::tan(coupling * dt);
    EXPECT_NEAR(result.hopping_probabilities[2], probability, 2.0e-14);
    EXPECT_EQ(result.proposed_state, 2);
}

TEST(FsshDriverStateSpace, RejectsGroundInitialAndReplayStates)
{
    FsshModelConfig model;
    model.ground_state_coupling = false;
    FsshDriver driver;
    EXPECT_THROW(
        driver.init(
            model,
            0,
            2,
            0.1,
            0,
            0,
            0,
            false,
            7U,
            0.0),
        std::invalid_argument);
    driver.init(
        model,
        0,
        2,
        0.1,
        1,
        0,
        0,
        false,
        7U,
        0.0);
    EXPECT_THROW(driver.set_replay_state(0, {{0.0, 0.0}, {1.0, 0.0}}), std::runtime_error);
    EXPECT_THROW(driver.set_replay_state(1, {{0.6, 0.0}, {0.8, 0.0}}), std::runtime_error);
    FsshReplayStepInput input;
    input.energies = {0.0, 0.1};
    input.time_nac.create(2, 2);
    input.active_state_before = 1;
    input.active_state_after = 0;
    EXPECT_THROW(driver.run_step_replay(input), std::runtime_error);
}

TEST(FsshDriverStateSpace, DefaultPreservesGroundStateDynamics)
{
    const FsshModelConfig model;
    ASSERT_TRUE(model.ground_state_coupling);
    FsshDriver legacy;
    FsshDriver explicit_model;
    legacy.init(
        0,
        2,
        0.1,
        1,
        0,
        0,
        false,
        7U,
        0.0);
    explicit_model.init(
        model,
        0,
        2,
        0.1,
        1,
        0,
        0,
        false,
        7U,
        0.0);
    FsshReplayStepInput input;
    input.energies = {0.0, 0.0};
    input.time_nac.create(2, 2);
    input.time_nac(0, 1) = {0.4, 0.0};
    input.time_nac(1, 0) = {-0.4, 0.0};
    input.kinetic_energy = 1.0;
    input.random_number = 0.0;
    input.active_state_before = 1;
    input.active_state_after = 1;
    const FsshReplayStepResult old_result = legacy.run_step_replay(input);
    const FsshReplayStepResult new_result = explicit_model.run_step_replay(input);
    EXPECT_EQ(old_result.coefficients_after_decoherence, new_result.coefficients_after_decoherence);
    EXPECT_EQ(old_result.hopping_probabilities, new_result.hopping_probabilities);
    EXPECT_GT(new_result.hopping_probabilities[0], 0.0);
    EXPECT_EQ(new_result.proposed_state, 0);
}

TEST(FsshDriverStateSpace, ProjectsAfterGaugeAlignmentAndPreservesExcitedNac)
{
    MpiScope mpi;
    FsshModelConfig model;
    model.ground_state_coupling = false;
    FsshDriver projected;
    FsshDriver full;
    projected.init(
        model,
        3,
        3,
        0.5,
        1,
        1,
        3,
        false,
        7U,
        0.0);
    full.init(
        3,
        3,
        0.5,
        1,
        1,
        3,
        false,
        7U,
        0.0);
    ModuleBase::ComplexMatrix orbitals(3, 3);
    ModuleBase::ComplexMatrix overlap(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        orbitals(i, i) = {1.0, 0.0};
    }
    const double c = std::cos(0.1);
    const double s = std::sin(0.1);
    const double cv = std::cos(0.2);
    const double sv = std::sin(0.2);
    overlap(0, 0) = {c, 0.0};
    overlap(0, 1) = {-s * cv, 0.0};
    overlap(0, 2) = {s * sv, 0.0};
    overlap(1, 0) = {s, 0.0};
    overlap(1, 1) = {c * cv, 0.0};
    overlap(1, 2) = {-c * sv, 0.0};
    overlap(2, 1) = {sv, 0.0};
    overlap(2, 2) = {cv, 0.0};
    std::vector<CasidaWavefunction> states{CasidaWavefunction(
                                               0.0, {}, {},
                                               1,
                                               2),
        CasidaWavefunction(
            0.1, {1.0, 0.0}, {},
            1,
            2),
        CasidaWavefunction(
            0.2, {0.0, 1.0}, {},
            1,
            2)};
    std::vector<CasidaWavefunction> new_states = states;
    new_states[1].X_coeffs[0] = -1.0;
    ModuleBase::ComplexMatrix full_nac(3, 3);
    ModuleBase::ComplexMatrix projected_nac(3, 3);
    std::vector<CasidaWavefunction> full_aligned;
    std::vector<CasidaWavefunction> projected_aligned;
    FsshDriverTestAccess::calculate_nac_from_dense(
        full,
        orbitals,
        orbitals,
        overlap,
        full_nac,
        states,
        new_states,
        &full_aligned);
    FsshDriverTestAccess::calculate_nac_from_dense(
        projected,
        orbitals,
        orbitals,
        overlap,
        projected_nac,
        states,
        new_states,
        &projected_aligned);
    EXPECT_GT(std::abs(full_nac(0, 1)), 1.0e-4);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(projected_nac(0, i), std::complex<double>(0.0, 0.0));
        EXPECT_EQ(projected_nac(i, 0), std::complex<double>(0.0, 0.0));
    }
    for (int i = 1; i < 3; ++i)
    {
        EXPECT_EQ(projected_aligned[i].X_coeffs, full_aligned[i].X_coeffs);
        for (int j = 1; j < 3; ++j)
        {
            EXPECT_EQ(projected_nac(i, j), full_nac(i, j));
        }
    }
    EXPECT_GT(std::abs(projected_nac(1, 2)), 1.0e-4);
}

TEST(FsshDriverStateSpace, CheckpointVersionAndConfigurationAreExplicit)
{
    FsshModelConfig excited;
    excited.ground_state_coupling = false;
    const FsshModelConfig full;
    FsshDriver source;
    source.init(
        excited,
        0,
        3,
        0.1,
        1,
        0,
        0,
        true,
        7U,
        0.0);
    source.set_replay_state(1, {{0.0, 0.0}, {0.6, 0.0}, {0.0, 0.8}});
    std::stringstream checkpoint;
    ASSERT_TRUE(source.save_checkpoint(checkpoint));
    FsshDriver restored;
    ASSERT_TRUE(restored.load_checkpoint(checkpoint, excited));
    std::stringstream roundtrip;
    ASSERT_TRUE(restored.save_checkpoint(roundtrip));
    EXPECT_EQ(roundtrip.str(), checkpoint.str());
    std::stringstream mismatch(checkpoint.str());
    EXPECT_FALSE(restored.load_checkpoint(mismatch, full));
    std::stringstream after_failure;
    ASSERT_TRUE(restored.save_checkpoint(after_failure));
    EXPECT_EQ(after_failure.str(), roundtrip.str());

    FsshReplayStepInput input;
    input.energies = {0.0, 0.1, 0.2};
    input.time_nac.create(3, 3);
    input.time_nac(0, 1) = {0.5, 0.0};
    input.time_nac(1, 0) = {-0.5, 0.0};
    input.time_nac(1, 2) = {0.2, 0.0};
    input.time_nac(2, 1) = {-0.2, 0.0};
    input.kinetic_energy = 1.0;
    input.random_number = 0.01;
    input.active_state_before = 1;
    input.active_state_after = 1;
    const FsshReplayStepResult uninterrupted = source.run_step_replay(input);
    const FsshReplayStepResult restarted = restored.run_step_replay(input);
    EXPECT_EQ(restarted.coefficients_after_decoherence, uninterrupted.coefficients_after_decoherence);
    EXPECT_EQ(restarted.hopping_probabilities, uninterrupted.hopping_probabilities);
    EXPECT_EQ(restarted.proposed_state, uninterrupted.proposed_state);
    EXPECT_EQ(restarted.coefficients_after_decoherence[0], std::complex<double>(0.0, 0.0));
    EXPECT_DOUBLE_EQ(restarted.hopping_probabilities[0], 0.0);

    std::stringstream corrupted_source(checkpoint.str());
    std::string corrupted_header;
    std::string corrupted_metadata;
    std::string coefficient_count;
    std::string ground_coefficient;
    std::getline(corrupted_source, corrupted_header);
    std::getline(corrupted_source, corrupted_metadata);
    std::getline(corrupted_source, coefficient_count);
    std::getline(corrupted_source, ground_coefficient);
    const std::string remainder((std::istreambuf_iterator<char>(corrupted_source)), std::istreambuf_iterator<char>());
    std::stringstream corrupted(
        corrupted_header + "\n" + corrupted_metadata + "\n" + coefficient_count + "\n1e-20 0\n" + remainder);
    EXPECT_FALSE(restored.load_checkpoint(corrupted, excited));

    FsshDriver legacy_source;
    legacy_source.init(
        0,
        2,
        0.1,
        0,
        0,
        0,
        false,
        7U,
        0.0);
    std::stringstream version2;
    ASSERT_TRUE(legacy_source.save_checkpoint(version2));
    std::string header;
    std::string metadata;
    std::getline(version2, header);
    std::getline(version2, metadata);
    const std::string body((std::istreambuf_iterator<char>(version2)), std::istreambuf_iterator<char>());
    const std::string legacy_text =
        "ABACUS_FSSH_CHECKPOINT 1\n" + metadata.substr(0, metadata.find_last_of(' ')) + "\n" + body;
    std::stringstream legacy(legacy_text);
    FsshDriver legacy_restored;
    EXPECT_TRUE(legacy_restored.load_checkpoint(legacy));
    std::stringstream excluded_legacy(legacy_text);
    EXPECT_FALSE(restored.load_checkpoint(excluded_legacy, excited));
}
