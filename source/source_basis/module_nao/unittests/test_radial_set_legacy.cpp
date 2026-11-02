#include "source_basis/module_nao/radial_set.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
/** @brief A synthetic element with analytically compact s and p radial functions. */
class CompactRadialSet : public RadialSet
{
  public:
    /**
     * @param[in] zero_first Whether the shorter s radial is identically zero.
     * @param[in] common_support Whether both radials end at three Bohr.
     * @param[in] no_tail Whether both radials remain nonzero at the last grid point.
     */
    CompactRadialSet(const bool zero_first, const bool common_support, const bool no_tail)
    {
        symbol_ = "H";
        lmax_ = 1;
        nchi_ = 2;
        nzeta_max_ = 1;
        nzeta_ = new int[2]{1, 1};
        chi_ = new NumericalRadial[2];
        std::vector<double> grid(401);
        for (int i = 0; i < grid.size(); ++i)
        {
            grid[i] = i * 0.01;
        }
        for (int l = 0; l < 2; ++l)
        {
            double cutoff = 2.0 + l;
            if (common_support)
            {
                cutoff = 3.0;
            }
            if (no_tail)
            {
                cutoff = 5.0;
            }
            std::vector<double> values(grid.size(), 0.0);
            for (int i = 0; i < grid.size(); ++i)
            {
                if (grid[i] < cutoff && !(zero_first && l == 0))
                {
                    const double polynomial = 1.0 - grid[i] * grid[i] / (cutoff * cutoff);
                    values[i] = std::pow(grid[i], l) * polynomial * polynomial * polynomial;
                }
            }
            chi_[l].build(
                l,
                true,
                grid.size(),
                grid.data(),
                values.data(),
                0,
                0,
                symbol_,
                0,
                false);
        }
        indexing();
        set_rcut_max();
    }

    /** @param[in] grid Replacement grid for invalid-input tests, without resampling. */
    void replace_second_grid(const std::vector<double> &grid)
    {
        std::vector<double> values(grid.size(), 1.0);
        chi_[1].build(
            1,
            true,
            grid.size(),
            grid.data(),
            values.data(),
            0,
            0,
            symbol_,
            0,
            false);
    }

    /** @brief Replace every radial by zero on its original grid. */
    void zero_all_radials()
    {
        for (int l = 0; l < 2; ++l)
        {
            const std::vector<double> grid(chi_[l].rgrid(), chi_[l].rgrid() + chi_[l].nr());
            const std::vector<double> values(grid.size(), 0.0);
            chi_[l].build(
                l,
                true,
                grid.size(),
                grid.data(),
                values.data(),
                0,
                0,
                symbol_,
                0,
                false);
        }
        set_rcut_max();
    }

    /** @return An owning copy, following the RadialSet clone interface. */
    CompactRadialSet *clone() const override { return new CompactRadialSet(*this); }
};
} // namespace

TEST(RadialSetLegacy, PadsEveryRadialToTheElementMaximum)
{
    const CompactRadialSet source(false, false, false);
    Numerical_Orbital orbital;
    source.to_numerical_orbital(
        orbital,
        101,
        0.1,
        false,
        true);
    EXPECT_DOUBLE_EQ(orbital.getRcut(), 3.0);
    const Numerical_Orbital_Lm &s = orbital.PhiLN(0, 0);
    const Numerical_Orbital_Lm &p = orbital.PhiLN(1, 0);
    EXPECT_EQ(s.getNr(), p.getNr());
    EXPECT_EQ(s.getNruniform(), p.getNruniform());
    EXPECT_DOUBLE_EQ(s.getRcut(), p.getRcut());
    for (int i = 0; i < s.getNr(); ++i)
    {
        if (s.getRadial(i) > source.chi(0, 0).rcut())
        {
            EXPECT_DOUBLE_EQ(s.getPsi(i), 0.0);
        }
        else
        {
            EXPECT_DOUBLE_EQ(s.getPsi(i), source.chi(0, 0).rvalue(i));
        }
    }
    // Gint evaluates all radial arrays using a common atom-level index and ip+3.
    // Check the exported cutoff guard, not an interior spline sample.
    const int ip = static_cast<int>(std::ceil(s.getRcut() / s.dr_uniform));
    EXPECT_DOUBLE_EQ(s.psi_uniform.at(ip + 3), 0.0);
    EXPECT_TRUE(std::isfinite(s.dpsi_uniform.at(ip + 3)));
    EXPECT_TRUE(std::isfinite(s.ddpsi_uniform.at(ip + 3)));
}

TEST(RadialSetLegacy, ZeroRadialUsesTheSameSafeStorageExtent)
{
    const CompactRadialSet source(true, false, false);
    Numerical_Orbital orbital;
    source.to_numerical_orbital(
        orbital,
        101,
        0.1,
        false,
        true);
    const Numerical_Orbital_Lm &zero = orbital.PhiLN(0, 0);
    EXPECT_DOUBLE_EQ(zero.getRcut(), orbital.getRcut());
    for (const double value : zero.psi_uniform)
    {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
    for (const double value : zero.dpsi_uniform)
    {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
}

TEST(RadialSetLegacy, AllZeroRadialsHaveThreePointsForLegacyInterpolation)
{
    CompactRadialSet source(false, false, false);
    source.zero_all_radials();
    Numerical_Orbital orbital;
    source.to_numerical_orbital(
        orbital,
        101,
        0.1,
        false,
        true);
    for (int l = 0; l < 2; ++l)
    {
        const Numerical_Orbital_Lm &zero = orbital.PhiLN(l, 0);
        EXPECT_EQ(zero.getNr(), 3);
        EXPECT_DOUBLE_EQ(zero.getRcut(), 0.02);
        for (const double value : zero.psi_uniform)
        {
            EXPECT_DOUBLE_EQ(value, 0.0);
        }
        for (const double value : zero.dpsi_uniform)
        {
            EXPECT_DOUBLE_EQ(value, 0.0);
        }
        Numerical_Orbital_Lm individual;
        source.chi(l, 0).to_numerical_orbital_lm(
            individual,
            101,
            0.1,
            false,
            true);
        EXPECT_EQ(individual.psi_uniform, zero.psi_uniform);
    }
    source.replace_second_grid({0.0, 0.01});
    EXPECT_THROW(
        source.to_numerical_orbital(
            orbital,
            101,
            0.1,
            false,
            true),
        std::invalid_argument);
}

TEST(RadialSetLegacy, CommonCutoffMatchesIndividualExportsExactlyWithAndWithoutTail)
{
    for (const bool no_tail : {false, true})
    {
        const CompactRadialSet source(false, true, no_tail);
        Numerical_Orbital orbital;
        source.to_numerical_orbital(
            orbital,
            101,
            0.1,
            false,
            true);
        for (int l = 0; l < 2; ++l)
        {
            Numerical_Orbital_Lm reference;
            source.chi(l, 0).to_numerical_orbital_lm(
                reference,
                101,
                0.1,
                false,
                true);
            const Numerical_Orbital_Lm &converted = orbital.PhiLN(l, 0);
            EXPECT_EQ(reference.getNr(), converted.getNr());
            EXPECT_EQ(reference.get_psi(), converted.get_psi());
            EXPECT_EQ(reference.get_psif(), converted.get_psif());
            EXPECT_EQ(reference.psi_uniform, converted.psi_uniform);
            EXPECT_EQ(reference.dpsi_uniform, converted.dpsi_uniform);
            EXPECT_EQ(reference.ddpsi_uniform, converted.ddpsi_uniform);
        }
    }
}

#ifndef __DEBUG
TEST(RadialSetLegacy, RejectsInvalidCommonGridBeforeAnyExport)
{
    CompactRadialSet source(false, false, false);
    std::vector<double> grid(401);
    for (int i = 0; i < grid.size(); ++i)
    {
        grid[i] = 0.01 * i;
    }
    grid[399] = 5.0;
    source.replace_second_grid(grid);
    Numerical_Orbital orbital;
    EXPECT_THROW(
        source.to_numerical_orbital(
            orbital,
            101,
            0.1,
            false,
            true),
        std::invalid_argument);
    grid[399] = 3.99;
    grid[10] = std::numeric_limits<double>::quiet_NaN();
    source.replace_second_grid(grid);
    EXPECT_THROW(
        source.to_numerical_orbital(
            orbital,
            101,
            0.1,
            false,
            true),
        std::invalid_argument);
}
#endif
