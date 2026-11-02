#include "source_basis/module_nao/numerical_radial.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
std::vector<double> uniform_grid(const int count, const double spacing)
{
    std::vector<double> grid(count);
    for (int i = 0; i < count; ++i)
    {
        grid[i] = i * spacing;
    }
    return grid;
}

std::vector<double> gaussian(const std::vector<double> &grid, const double exponent)
{
    const double pi = std::acos(-1.0);
    const double norm = 2.0 * std::pow(2.0 * exponent, 0.75) / std::pow(pi, 0.25);
    std::vector<double> values(grid.size());
    for (int i = 0; i < grid.size(); ++i)
    {
        values[i] = norm * std::exp(-exponent * grid[i] * grid[i]);
    }
    return values;
}

double translated_integral(
    const Numerical_Orbital_Lm &left,
    const Numerical_Orbital_Lm &right,
    const double distance,
    const bool kinetic)
{
    double result = 0.0;
    for (int i = 0; i < left.getNk(); ++i)
    {
        const double k = left.getKpoint(i);
        const double kr = k * distance;
        double bessel = 1.0;
        if (kr != 0.0)
        {
            bessel = std::sin(kr) / kr;
        }
        double integrand = left.getPsif(i) * right.getPsif(i) * k * k * bessel;
        if (kinetic)
        {
            integrand *= 0.5 * k * k;
        }
        double weight = 2.0;
        if (i == 0 || i == left.getNk() - 1)
        {
            weight = 1.0;
        }
        else if (i % 2 == 1)
        {
            weight = 4.0;
        }
        result += weight * integrand;
    }
    return result * left.getDk() / 3.0;
}
} // namespace

TEST(NumericalRadialLegacy, CommonSupportMatchesOriginalConversionExactly)
{
    const std::vector<double> grid = uniform_grid(401, 0.01);
    const std::vector<double> values = gaussian(grid, 0.2);
    NumericalRadial radial;
    radial.build(
        0,
        true,
        grid.size(),
        grid.data(),
        values.data(),
        0,
        0,
        "H",
        0,
        false);
    std::vector<double> rab(grid.size(), 0.01);
    Numerical_Orbital_Lm reference;
    reference.set_orbital_info(
        "H",
        0,
        0,
        0,
        grid.size(),
        rab.data(),
        grid.data(),
        Numerical_Orbital_Lm::Psi_Type::Psi,
        values.data(),
        101,
        0.1,
        0.001,
        false,
        true,
        true);
    Numerical_Orbital_Lm exported;
    radial.to_numerical_orbital_lm(
        exported,
        101,
        0.1,
        false,
        true,
        grid.size(),
        grid.data());
    EXPECT_EQ(reference.get_psi(), exported.get_psi());
    EXPECT_EQ(reference.get_psif(), exported.get_psif());
    EXPECT_EQ(reference.psi_uniform, exported.psi_uniform);
    EXPECT_EQ(reference.dpsi_uniform, exported.dpsi_uniform);
    EXPECT_EQ(reference.ddpsi_uniform, exported.ddpsi_uniform);
}

TEST(NumericalRadialLegacy, PreservesTheExistingAccumulatedGridTolerance)
{
    std::vector<double> grid(80, 0.0);
    for (int i = 1; i < grid.size(); ++i)
    {
        grid[i] = grid[i - 1] + 0.1;
    }
    // Endpoint-based uniformity accepts this grid; using i * grid[1] would reject it.
    EXPECT_GT(std::abs(grid.back() - 79 * grid[1]), 1e-14);
    const std::vector<double> values = gaussian(grid, 0.2);
    NumericalRadial radial;
    radial.build(
        0,
        true,
        grid.size(),
        grid.data(),
        values.data(),
        0,
        0,
        "H",
        0,
        false);
    Numerical_Orbital_Lm output;
    EXPECT_NO_THROW(radial.to_numerical_orbital_lm(
        output,
        101,
        0.1,
        false,
        true));
    EXPECT_EQ(output.getNr(), grid.size());
    EXPECT_EQ(output.get_psi(), values);
}

TEST(NumericalRadialLegacy, UnequalGaussianSupportPreservesAnalyticTwoCenterIntegrals)
{
    const std::vector<double> grid = uniform_grid(801, 0.02);
    const std::vector<double> tight_values = gaussian(grid, 0.8);
    const std::vector<double> diffuse_values = gaussian(grid, 0.2);
    NumericalRadial tight;
    NumericalRadial diffuse;
    tight.build(
        0,
        true,
        grid.size(),
        grid.data(),
        tight_values.data(),
        0,
        0,
        "H",
        0,
        false);
    diffuse.build(
        0,
        true,
        grid.size(),
        grid.data(),
        diffuse_values.data(),
        0,
        1,
        "H",
        0,
        false);
    ASSERT_LT(tight.rcut(), diffuse.rcut());
    Numerical_Orbital_Lm left;
    Numerical_Orbital_Lm right;
    tight.to_numerical_orbital_lm(
        left,
        401,
        0.05,
        false,
        true,
        grid.size(),
        grid.data());
    diffuse.to_numerical_orbital_lm(
        right,
        401,
        0.05,
        false,
        true,
        grid.size(),
        grid.data());
    const double exponents_product_over_sum = 0.16;
    for (const double distance : {0.0,
             0.7,
             2.0,
             6.0})
    {
        const double overlap = std::pow(0.8, 1.5) * std::exp(-exponents_product_over_sum * distance * distance);
        const double kinetic =
            exponents_product_over_sum * (3.0 - 2.0 * exponents_product_over_sum * distance * distance) * overlap;
        EXPECT_NEAR(
            translated_integral(
                left,
                right,
                distance,
                false),
            overlap, 1e-9);
        EXPECT_NEAR(
            translated_integral(
                left,
                right,
                distance,
                true),
            kinetic, 1e-9);
    }
    // The consistent upstream spline can have tiny interior tails. Exact zero
    // padding is guaranteed beyond the exported common cutoff.
    const int last = static_cast<int>(std::ceil(left.getRcut() / left.dr_uniform));
    EXPECT_DOUBLE_EQ(left.psi_uniform.at(last + 3), 0.0);
    EXPECT_TRUE(std::isfinite(left.dpsi_uniform.at(last + 3)));
}

TEST(NumericalRadialLegacy, RejectsNonfiniteAndIncompatibleDataInRelease)
{
    std::vector<double> grid = uniform_grid(101, 0.01);
    std::vector<double> values = gaussian(grid, 0.8);
    NumericalRadial radial;
    radial.build(
        0,
        true,
        grid.size(),
        grid.data(),
        values.data(),
        0,
        0,
        "H",
        0,
        false);
    Numerical_Orbital_Lm output;
    EXPECT_THROW(
        radial.to_numerical_orbital_lm(
            output,
            101,
            0.1,
            false,
            true,
            2,
            grid.data()),
        std::invalid_argument);
    EXPECT_THROW(
        radial.to_numerical_orbital_lm(
            output,
            101,
            0.1,
            false,
            true,
            50,
            grid.data()),
        std::invalid_argument);
    std::vector<double> bad_grid = grid;
    bad_grid[50] += 0.001;
    EXPECT_THROW(
        radial.to_numerical_orbital_lm(
            output,
            101,
            0.1,
            false,
            true,
            grid.size(),
            bad_grid.data()),
        std::invalid_argument);
    bad_grid[50] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        radial.to_numerical_orbital_lm(
            output,
            101,
            0.1,
            false,
            true,
            grid.size(),
            bad_grid.data()),
        std::invalid_argument);
    values[30] = std::numeric_limits<double>::quiet_NaN();
    radial.build(
        0,
        true,
        grid.size(),
        grid.data(),
        values.data(),
        0,
        0,
        "H",
        0,
        false);
    EXPECT_THROW(
        radial.to_numerical_orbital_lm(
            output,
            101,
            0.1,
            false,
            true),
        std::invalid_argument);
    const std::vector<double> short_grid{0.0, 0.01};
    const std::vector<double> short_values{0.0, 0.0};
    radial.build(
        0,
        true,
        2,
        short_grid.data(),
        short_values.data(),
        0,
        0,
        "H",
        0,
        false);
    EXPECT_THROW(
        radial.to_numerical_orbital_lm(
            output,
            101,
            0.1,
            false,
            true),
        std::invalid_argument);
}
