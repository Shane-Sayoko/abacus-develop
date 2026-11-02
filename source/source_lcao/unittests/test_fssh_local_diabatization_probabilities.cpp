#include "source_lcao/module_lr/utils/fssh_local_diabatization.h"

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace
{
using Complex = std::complex<double>;
using Matrix = ModuleBase::ComplexMatrix;

FsshLD::Propagation apply_propagator(const Matrix &matrix, const std::vector<Complex> &initial)
{
    FsshLD::Propagation result;
    result.propagator = matrix;
    result.coefficients.assign(initial.size(), Complex(0.0));
    for (int i = 0; i < matrix.nr; ++i)
    {
        for (int j = 0; j < matrix.nc; ++j)
        {
            result.coefficients[i] += matrix(i, j) * initial[j];
        }
    }
    return result;
}

Complex read_complex(std::istream &stream)
{
    double real = 0.0;
    double imaginary = 0.0;
    stream >> real >> imaginary;
    return Complex(real, imaginary);
}
} // namespace

TEST(FsshLdProbabilities, PureRabiIncludingSubtractionInvisibleTransfer)
{
    for (double angle : {0.0,
             1.0e-10,
             0.2,
             1.1})
    {
        Matrix u(2, 2);
        u(0, 0) = std::cos(angle);
        u(1, 1) = std::cos(angle);
        u(0, 1) = Complex(0.0, -std::sin(angle));
        u(1, 0) = u(0, 1);
        const std::vector<Complex> initial = {Complex(0.0, 1.0), 0.0};
        const auto propagation = apply_propagator(u, initial);
        const auto result = FsshLD::probabilities_sharc(propagation, initial, 0);
        const double expected = std::pow(std::sin(angle), 2);
        EXPECT_TRUE(result.pure_initial_state);
        EXPECT_DOUBLE_EQ(result.probabilities[0], 0.0);
        EXPECT_NEAR(result.probabilities[1], expected, 1.0e-30 + 1.0e-12 * expected);
    }
}

TEST(FsshLdProbabilities, PureThreeStateStarHasAnalyticBranching)
{
    const double g1 = 0.02;
    const double g2 = 0.03;
    const double g = std::hypot(g1, g2);
    for (double time : {0.0,
             0.1,
             1.0,
             10.0})
    {
        const double cosine = std::cos(g * time);
        const double sine = std::sin(g * time);
        const std::vector<double> coupling = {g1, g2};
        Matrix u(3, 3);
        u(0, 0) = cosine;
        for (int j = 0; j < 2; ++j)
        {
            u(0, j + 1) = Complex(0.0, -sine * coupling[j] / g);
            u(j + 1, 0) = u(0, j + 1);
            for (int k = 0; k < 2; ++k)
            {
                double value = (cosine - 1.0) * coupling[j] * coupling[k] / (g * g);
                if (j == k)
                {
                    value += 1.0;
                }
                u(j + 1, k + 1) = value;
            }
        }
        const std::vector<Complex> initial = {1.0, 0.0, 0.0};
        const auto propagation = apply_propagator(u, initial);
        const auto result = FsshLD::probabilities_sharc(propagation, initial, 0);
        EXPECT_NEAR(result.probabilities[1], g1 * g1 / (g * g) * sine * sine, 1.0e-14);
        EXPECT_NEAR(result.probabilities[2], g2 * g2 / (g * g) * sine * sine, 1.0e-14);
        EXPECT_NEAR(result.total_probability, sine * sine, 1.0e-14);
    }
}

TEST(FsshLdProbabilities, IndependentDecimalCoherentReferences)
{
    std::ifstream input(LD_PROBABILITY_REFERENCE_PATH);
    ASSERT_TRUE(input.good());
    int samples = 0;
    input >> samples;
    ASSERT_EQ(samples, 8);
    for (int sample = 0; sample < samples; ++sample)
    {
        SCOPED_TRACE(sample);
        int dimension = 0;
        int active = 0;
        int reject = 0;
        input >> dimension >> active >> reject;
        FsshLD::Propagation propagation;
        propagation.propagator.create(dimension, dimension);
        for (int index = 0; index < propagation.propagator.size; ++index)
        {
            propagation.propagator.c[index] = read_complex(input);
        }
        std::vector<Complex> initial(dimension);
        for (auto &value : initial)
        {
            value = read_complex(input);
        }
        propagation.coefficients.resize(dimension);
        for (auto &value : propagation.coefficients)
        {
            value = read_complex(input);
        }
        std::vector<double> expected(dimension);
        for (double &value : expected)
        {
            input >> value;
        }
        ASSERT_TRUE(input.good());
        if (reject)
        {
            EXPECT_THROW(FsshLD::probabilities_sharc(propagation, initial, active), std::runtime_error);
        }
        else
        {
            const auto result = FsshLD::probabilities_sharc(propagation, initial, active);
            for (int state = 0; state < dimension; ++state)
            {
                EXPECT_NEAR(result.probabilities[state], expected[state], 1.0e-12);
            }
        }
    }
}

TEST(FsshLdProbabilities, RejectsDecoheredNonunitaryAndInvalidData)
{
    Matrix u(2, 2);
    u(0, 0) = 0.8;
    u(1, 1) = 0.8;
    u(0, 1) = -0.6;
    u(1, 0) = 0.6;
    const std::vector<Complex> initial = {1.0, 0.0};
    auto result = apply_propagator(u, initial);
    result.coefficients = {1.0, 0.0};
    EXPECT_THROW(FsshLD::probabilities_sharc(result, initial, 0), std::runtime_error);
    result = apply_propagator(u, initial);
    result.propagator(1, 1) = 0.7;
    EXPECT_THROW(FsshLD::probabilities_sharc(result, initial, 0), std::runtime_error);
    result = apply_propagator(u, initial);
    EXPECT_THROW(FsshLD::probabilities_sharc(result, initial, 1), std::runtime_error);
    EXPECT_THROW(FsshLD::probabilities_sharc(result, initial, -1), std::runtime_error);
    result.propagator(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(FsshLD::probabilities_sharc(result, initial, 0), std::runtime_error);
}

TEST(FsshLdProbabilities, TinyCoherentActivePopulationIsNotClassifiedAsPure)
{
    Matrix swap(2, 2);
    swap(0, 1) = 1.0;
    swap(1, 0) = 1.0;
    const std::vector<Complex> initial = {1.0e-150, 1.0};
    const auto propagation = apply_propagator(swap, initial);
    const auto result = FsshLD::probabilities_sharc(propagation, initial, 0);
    EXPECT_FALSE(result.pure_initial_state);
    EXPECT_DOUBLE_EQ(result.total_probability, 0.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
