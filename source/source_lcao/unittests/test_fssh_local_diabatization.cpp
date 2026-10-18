#include "source_lcao/module_lr/utils/fssh_local_diabatization.h"

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>

namespace
{
using Matrix = ModuleBase::ComplexMatrix;
using Complex = std::complex<double>;
Matrix rotation(double angle)
{
    Matrix result(2, 2);
    result(0, 0) = std::cos(angle);
    result(0, 1) = -std::sin(angle);
    result(1, 0) = std::sin(angle);
    result(1, 1) = std::cos(angle);
    return result;
}
std::vector<Complex> apply_matrix(const Matrix& matrix, const std::vector<Complex>& values)
{
    std::vector<Complex> result(matrix.nr, Complex(0.0));
    for (int row = 0; row < matrix.nr; ++row)
    {
        for (int column = 0; column < matrix.nc; ++column)
        {
            result[row] += matrix(row, column) * values[column];
        }
    }
    return result;
}
void expect_same_density(const std::vector<Complex>& left, const std::vector<Complex>& right, double tolerance)
{
    ASSERT_EQ(left.size(), right.size());
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        for (std::size_t j = 0; j < left.size(); ++j)
        {
            EXPECT_LT(std::abs(left[i] * std::conj(left[j]) - right[i] * std::conj(right[j])), tolerance);
        }
    }
}
Complex read_complex(std::istream& input)
{
    double real = 0.0;
    double imaginary = 0.0;
    input >> real >> imaginary;
    return Complex(real, imaginary);
}
} // namespace

TEST(FsshLocalDiabatization, Independent80DigitTwoAndThreeStateReferences)
{
    std::ifstream input(LD_REFERENCE_PATH);
    ASSERT_TRUE(input.good());
    int cases = 0;
    input >> cases;
    ASSERT_EQ(cases, 6);
    double maximum_error = 0.0;
    double maximum_unitarity = 0.0;
    for (int sample = 0; sample < cases; ++sample)
    {
        int n = 0;
        double dt = 0.0;
        input >> n >> dt;
        Matrix overlap(n, n);
        for (int i = 0; i < overlap.size; ++i)
        {
            overlap.c[i] = read_complex(input);
        }
        std::vector<double> old_energies(n);
        std::vector<double> new_energies(n);
        for (auto& value: old_energies)
        {
            input >> value;
        }
        for (auto& value: new_energies)
        {
            input >> value;
        }
        std::vector<Complex> old_coefficients(n);
        for (auto& value: old_coefficients)
        {
            value = read_complex(input);
        }
        Matrix expected_u(n, n);
        for (int i = 0; i < expected_u.size; ++i)
        {
            expected_u.c[i] = read_complex(input);
        }
        std::vector<Complex> expected_coefficients(n);
        for (auto& value: expected_coefficients)
        {
            value = read_complex(input);
        }
        ASSERT_TRUE(input.good());
        const auto result = FsshLD::propagate(overlap, old_energies, new_energies, old_coefficients, dt);
        for (int i = 0; i < expected_u.size; ++i)
        {
            maximum_error = std::max(maximum_error, std::abs(result.propagator.c[i] - expected_u.c[i]));
        }
        maximum_unitarity = std::max(maximum_unitarity, result.unitarity_residual);
        expect_same_density(result.coefficients, expected_coefficients, 1.0e-10);
        for (int active = 0; active < n; ++active)
        {
            std::vector<double> expected_probabilities(n);
            double total = 0.0;
            for (auto& value: expected_probabilities)
            {
                input >> value;
                total += value;
            }
            ASSERT_TRUE(input.good());
            if (total > 1.0 + 1.0e-12)
            {
                EXPECT_THROW(FsshLD::probabilities_2001(result, old_coefficients, active), std::runtime_error);
            }
            else
            {
                const auto probabilities = FsshLD::probabilities_2001(result, old_coefficients, active);
                for (int target = 0; target < n; ++target)
                {
                    EXPECT_NEAR(probabilities[target], expected_probabilities[target], 1.0e-12);
                }
            }
        }
    }
    EXPECT_LE(maximum_error, 1.0e-10);
    EXPECT_LE(maximum_unitarity, 1.0e-12);
    std::cout << std::setprecision(17) << "LD_REFERENCE max_U_error=" << maximum_error
              << " max_unitarity=" << maximum_unitarity << '\n';
}

TEST(FsshLocalDiabatization, AnalyticConstantDiagonalHamiltonian)
{
    Matrix overlap(2, 2);
    overlap.set_as_identity_matrix();
    const std::vector<double> energies = {0.2, 0.7};
    const std::vector<Complex> initial = {std::sqrt(0.3), Complex(0.0, std::sqrt(0.7))};
    const double dt = 0.4;
    const auto result = FsshLD::propagate(overlap, energies, energies, initial, dt);
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_LT(std::abs(result.coefficients[i] - initial[i] * std::exp(Complex(0.0, -energies[i] * dt))), 1.0e-14);
    }
    const auto probabilities = FsshLD::probabilities_2001(result, initial, 0);
    EXPECT_DOUBLE_EQ(probabilities[0], 0.0);
    EXPECT_DOUBLE_EQ(probabilities[1], 0.0);
}

TEST(FsshLocalDiabatization, PhasePermutationAndEnergyOriginCovariance)
{
    const Matrix overlap = rotation(0.19);
    const std::vector<double> e0 = {0.1, 0.6};
    const std::vector<double> e1 = {0.12, 0.58};
    const std::vector<Complex> c0 = {std::sqrt(0.8), Complex(0.0, std::sqrt(0.2))};
    const auto reference = FsshLD::propagate(overlap, e0, e1, c0, 0.7);
    Matrix g0(2, 2);
    Matrix g1(2, 2);
    g0(0, 1) = std::exp(Complex(0.0, 0.73));
    g0(1, 0) = std::exp(Complex(0.0, -0.21));
    g1(0, 0) = std::exp(Complex(0.0, -0.32));
    g1(1, 1) = std::exp(Complex(0.0, 0.48));
    const Matrix transformed_s = ModuleBase::transpose(g0, true) * overlap * g1;
    const auto transformed_c = apply_matrix(ModuleBase::transpose(g0, true), c0);
    const std::vector<double> transformed_e0 = {e0[1], e0[0]};
    const auto transformed = FsshLD::propagate(transformed_s, transformed_e0, e1, transformed_c, 0.7);
    expect_same_density(apply_matrix(g1, transformed.coefficients), reference.coefficients, 1.0e-12);
    const std::vector<double> shifted0 = {e0[0] + 19.0, e0[1] + 19.0};
    const std::vector<double> shifted1 = {e1[0] + 19.0, e1[1] + 19.0};
    const auto shifted = FsshLD::propagate(overlap, shifted0, shifted1, c0, 0.7);
    expect_same_density(shifted.coefficients, reference.coefficients, 1.0e-12);
}

TEST(FsshLocalDiabatization, TimeReversalWithoutRenormalization)
{
    const Matrix overlap = 0.96 * rotation(0.23);
    const std::vector<double> e0 = {0.1, 0.6};
    const std::vector<double> e1 = {0.12, 0.58};
    const std::vector<Complex> initial = {std::sqrt(0.8), Complex(0.0, std::sqrt(0.2))};
    const auto forward = FsshLD::propagate(overlap, e0, e1, initial, 0.7);
    const auto backward = FsshLD::propagate(ModuleBase::transpose(overlap, true), e1, e0, forward.coefficients, -0.7);
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_LT(std::abs(backward.coefficients[i] - initial[i]), 1.0e-12);
    }
}

TEST(FsshLocalDiabatization, DegenerateRotationRequiresHamiltonianCovariance)
{
    Matrix identity(2, 2);
    identity.set_as_identity_matrix();
    const Matrix gauge = rotation(0.6);
    const std::vector<Complex> c0 = {1.0, 0.0};
    const std::vector<double> degenerate = {0.3, 0.3};
    const auto reference = FsshLD::propagate(identity, degenerate, degenerate, c0, 0.7);
    const auto rotated = FsshLD::propagate(gauge, degenerate, degenerate, c0, 0.7);
    expect_same_density(apply_matrix(gauge, rotated.coefficients), reference.coefficients, 1.0e-12);
    // A finite-gap rotation is not an adiabatic gauge: keeping D diagonal changes H.
    const std::vector<double> separated = {0.1, 0.8};
    const auto invalid_rotation = FsshLD::propagate(gauge, separated, separated, c0, 0.7);
    const auto original = FsshLD::propagate(identity, separated, separated, c0, 0.7);
    const auto recovered = apply_matrix(gauge, invalid_rotation.coefficients);
    EXPECT_GT(std::abs(recovered[0] * std::conj(recovered[1])
                       - original.coefficients[0] * std::conj(original.coefficients[1])),
              1.0e-3);
    EXPECT_THROW(FsshLD::require_resolved_active_surface(degenerate, {0.0, 0.0}, 0), std::runtime_error);
    EXPECT_NO_THROW(FsshLD::require_resolved_active_surface(separated, {1.0e-10, 1.0e-10}, 0));
    EXPECT_THROW(FsshLD::require_resolved_active_surface({0.1, 0.10000001}, {1.0e-8, 1.0e-8}, 0), std::runtime_error);
}

TEST(FsshLocalDiabatization, RejectsOverlapLeakageRankAndChangedWindow)
{
    const std::vector<double> energies = {0.1, 0.5};
    const std::vector<Complex> c0 = {1.0, 0.0};
    Matrix overlap = rotation(0.2);
    EXPECT_THROW(FsshLD::propagate(0.89 * overlap, energies, energies, c0, 0.7), std::runtime_error);
    EXPECT_THROW(FsshLD::propagate(1.000001 * overlap, energies, energies, c0, 0.7), std::runtime_error);
    overlap.zero_out();
    EXPECT_THROW(FsshLD::propagate(overlap, energies, energies, c0, 0.7), std::runtime_error);
    EXPECT_THROW(FsshLD::propagate(rotation(0.2), energies, {0.1}, c0, 0.7), std::runtime_error);
}

TEST(FsshLocalDiabatization, OriginalProbabilityMatchesNonSingularTwoStatePopulationLoss)
{
    const std::vector<Complex> initial = {std::sqrt(0.8), std::sqrt(0.2)};
    const std::vector<double> energies = {0.0, 0.2};
    const auto result = FsshLD::propagate(rotation(-0.1), energies, energies, initial, 0.3);
    const auto probabilities = FsshLD::probabilities_2001(result, initial, 0);
    const double expected
        = std::max(0.0, (std::norm(initial[0]) - std::norm(result.coefficients[0])) / std::norm(initial[0]));
    EXPECT_NEAR(probabilities[1], expected, 1.0e-14);
}

TEST(FsshLocalDiabatization, PureStateNonzeroTransferHasNoPermittedZeroDenominatorLimit)
{
    const std::vector<double> energies = {0.0, 0.2};
    const std::vector<Complex> initial = {1.0, 0.0};
    const auto result = FsshLD::propagate(rotation(0.2), energies, energies, initial, 0.3);
    EXPECT_GT(std::norm(result.coefficients[1]), 0.01);
    EXPECT_THROW(FsshLD::probabilities_2001(result, initial, 0), std::runtime_error);
    EXPECT_THROW(FsshLD::probabilities_2001(result, initial, 1), std::runtime_error);
}

TEST(FsshLocalDiabatization, MultiChannelPureStateExtensionIsPathDependent)
{
    Matrix q(3, 3);
    const double ca = 99.0 / 101.0;
    const double sa = 20.0 / 101.0;
    const double cb = 399.0 / 401.0;
    const double sb = 40.0 / 401.0;
    q(0, 0) = ca * cb;
    q(0, 1) = -sa;
    q(0, 2) = -ca * sb;
    q(1, 0) = sa * cb;
    q(1, 1) = ca;
    q(1, 2) = -sa * sb;
    q(2, 0) = sb;
    q(2, 2) = cb;
    const std::vector<double> e0 = {0.0, 0.3, 0.8};
    const std::vector<double> e1 = {0.02, 0.31, 0.79};
    const std::vector<Complex> pure = {1.0, 0.0, 0.0};
    const auto pure_step = FsshLD::propagate(q, e0, e1, pure, 0.7);
    EXPECT_THROW(FsshLD::probabilities_2001(pure_step, pure, 0), std::runtime_error);
    std::vector<std::vector<double>> probabilities;
    for (int target = 1; target < 3; ++target)
    {
        std::vector<Complex> perturbed = {std::sqrt(1.0 - 1.0e-12), 0.0, 0.0};
        perturbed[target] = 1.0e-6;
        const auto step = FsshLD::propagate(q, e0, e1, perturbed, 0.7);
        probabilities.push_back(FsshLD::probabilities_2001(step, perturbed, 0));
    }
    EXPECT_GT(probabilities[0][1], 0.04);
    EXPECT_DOUBLE_EQ(probabilities[0][2], 0.0);
    EXPECT_DOUBLE_EQ(probabilities[1][1], 0.0);
    EXPECT_GT(probabilities[1][2], 0.04);
}

TEST(FsshLocalDiabatization, LowActivePopulationDoesNotUnderflowConditionalProbability)
{
    const std::vector<Complex> initial = {1.0e-90, 1.0};
    const std::vector<double> energies = {0.0, 0.0};
    const auto result = FsshLD::propagate(rotation(-0.5e-90), energies, energies, initial, 0.3);
    const auto probabilities = FsshLD::probabilities_2001(result, initial, 0);
    EXPECT_NEAR(probabilities[1], 0.75, 1.0e-14);
}

TEST(FsshLocalDiabatization, ExcessOutgoingProbabilityIsRejectedWithoutNormalization)
{
    Matrix u(3, 3);
    const double values[9] = {0.581115702110278,
                              0.7091128806410308,
                              0.3993287658932893,
                              0.7403552431500067,
                              -0.6643735359402692,
                              0.10238124184886774,
                              0.3379033215302456,
                              0.23614979832979185,
                              -0.9110733329692098};
    for (int i = 0; i < 9; ++i)
    {
        u.c[i] = values[i];
    }
    const std::vector<Complex> initial = {0.15709790290460018, -0.8435222662334182, 0.5136053302599334};
    const std::vector<double> energies = {0.0, 0.0, 0.0};
    const auto result = FsshLD::propagate(ModuleBase::transpose(u, true), energies, energies, initial, 0.3);
    EXPECT_THROW(FsshLD::probabilities_2001(result, initial, 0), std::runtime_error);
}

TEST(FsshLocalDiabatization, FiniteInputsWhosePhaseOverflowsAreRejected)
{
    Matrix identity(2, 2);
    identity.set_as_identity_matrix();
    const std::vector<Complex> initial = {1.0, 0.0};
    const std::vector<double> energies = {1.0e308, 1.0e308};
    EXPECT_THROW(FsshLD::propagate(identity, energies, energies, initial, 10.0), std::runtime_error);
}

TEST(FsshLocalDiabatization, SecondOrderConvergenceToIndependentContinuousTdse)
{
    // Independent 80-digit Taylor reference: reports/.../data/continuous_tdse_reference.json.
    const std::vector<Complex> reference
        = {0.75907772254419214101405055235171,
           Complex(0.25245877881076940144019280878491, -0.60005464429374885468177290033239)};
    const auto eigenvectors = [](double time) {
        const double angle = 0.5 * std::atan2(0.3, 0.4 * time);
        Matrix result(2, 2);
        result(0, 0) = -std::sin(angle);
        result(0, 1) = std::cos(angle);
        result(1, 0) = std::cos(angle);
        result(1, 1) = std::sin(angle);
        return result;
    };
    double previous_error = 0.0;
    for (int steps: {6, 12, 24, 48})
    {
        const double dt = 2.4 / steps;
        Matrix frame = eigenvectors(-1.2);
        std::vector<Complex> coefficients = {frame(0, 0), frame(0, 1)};
        for (int step = 0; step < steps; ++step)
        {
            const double old_time = -1.2 + step * dt;
            const double new_time = -1.2 + (step + 1) * dt;
            const double old_energy = std::hypot(0.4 * old_time, 0.3);
            const double new_energy = std::hypot(0.4 * new_time, 0.3);
            Matrix next_frame = eigenvectors(new_time);
            const Matrix overlap = ModuleBase::transpose(frame, true) * next_frame;
            const auto result
                = FsshLD::propagate(overlap, {-old_energy, old_energy}, {-new_energy, new_energy}, coefficients, dt);
            coefficients = result.coefficients;
            frame = std::move(next_frame);
        }
        const auto diabatic = apply_matrix(frame, coefficients);
        double error = 0.0;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                error
                    = std::max(error,
                               std::abs(diabatic[i] * std::conj(diabatic[j]) - reference[i] * std::conj(reference[j])));
            }
        }
        std::cout << std::setprecision(17) << "LD_CONVERGENCE dt=" << dt << " density_error=" << error << '\n';
        if (previous_error > 0.0)
        {
            EXPECT_GT(previous_error / error, 3.8);
            EXPECT_LT(previous_error / error, 4.2);
        }
        previous_error = error;
    }
}

TEST(FsshLocalDiabatization, FiniteStepReturnDoesNotProveUncoupledEvolution)
{
    const double pi = std::acos(-1.0);
    const double angle = pi / 4.0;
    const double dt = pi / (2.0 * std::cos(angle));
    const std::vector<double> energies = {1.0, -1.0};
    const std::vector<Complex> initial = {1.0, 0.0};
    const auto result = FsshLD::propagate(rotation(angle), energies, energies, initial, dt);
    EXPECT_FALSE(result.uncoupled_endpoints);
    EXPECT_LT(std::abs(result.propagator(0, 1)), 1.0e-14);
    EXPECT_LT(std::abs(result.propagator(1, 0)), 1.0e-14);
    EXPECT_THROW(FsshLD::probabilities_2001(result, initial, 0), std::runtime_error);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
