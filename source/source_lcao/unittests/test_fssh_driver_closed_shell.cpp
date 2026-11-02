#include "source_lcao/module_operator_lcao/fssh_driver.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <vector>

/// @brief Narrow access to the production raw overlap, before gauge processing.
struct FsshDriverTestAccess
{
    static void overlap(
        FsshDriver &driver,
        const ModuleBase::ComplexMatrix &orbitals,
        const ModuleBase::ComplexMatrix &cross_overlap,
        const std::vector<CasidaWavefunction> &old_states,
        const std::vector<CasidaWavefunction> &new_states,
        int occupied,
        int virtual_count,
        ModuleBase::ComplexMatrix &result)
    {
        const std::vector<double> unused_phases(occupied, 1.0);
        driver.compute_lowdin_sigma(
            orbitals,
            orbitals,
            cross_overlap,
            old_states,
            new_states,
            occupied,
            virtual_count,
            0,
            unused_phases,
            result);
    }
};

namespace
{

/// @brief One normalized spin determinant with a real expansion coefficient.
struct DeterminantTerm
{
    std::vector<int> orbitals; ///< Alpha occupied slots followed by beta slots.
    double coefficient;
};

/**
 * @brief Evaluates a small determinant by the independent Leibniz formula.
 * @param[in] matrix Row-major square matrix.
 * @param[in] size Dimension, at most six in these tests.
 * @return Determinant without an occupied-block inverse or Schur complement.
 */
double determinant(const std::vector<double> &matrix, int size)
{
    std::vector<int> permutation(size);
    std::iota(permutation.begin(), permutation.end(), 0);
    double value = 0.0;
    do
    {
        double product = 1.0;
        int inversions = 0;
        for (int row = 0; row < size; ++row)
        {
            product *= matrix[row * size + permutation[row]];
            for (int next = row + 1; next < size; ++next)
            {
                if (permutation[row] > permutation[next])
                {
                    ++inversions;
                }
            }
        }
        if (inversions % 2 != 0)
        {
            product = -product;
        }
        value += product;
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return value;
}

/**
 * @brief Expands a real singlet CIS state in normalized spin determinants.
 * @param[in] state Ground placeholder or occupied-major, virtual-minor amplitudes.
 * @param[in] ground Whether to construct the closed-shell ground determinant.
 * @return Determinants in a fixed slot ordering, each excitation weighted X/sqrt(2).
 */
std::vector<DeterminantTerm> expand(const CasidaWavefunction &state, bool ground)
{
    const int occupied = state.nocc_lr;
    const int virtual_count = state.nvirt_lr;
    const int spatial = occupied + virtual_count;
    DeterminantTerm reference;
    reference.coefficient = 1.0;
    for (int spin = 0; spin < 2; ++spin)
    {
        for (int index = 0; index < occupied; ++index)
        {
            reference.orbitals.push_back(spin * spatial + index);
        }
    }
    if (ground)
    {
        return {reference};
    }
    std::vector<DeterminantTerm> result;
    for (int index = 0; index < occupied; ++index)
    {
        for (int virt = 0; virt < virtual_count; ++virt)
        {
            const double coefficient = state.X_coeffs[index * virtual_count + virt];
            if (coefficient == 0.0)
            {
                continue;
            }
            for (int spin = 0; spin < 2; ++spin)
            {
                DeterminantTerm term = reference;
                term.coefficient = coefficient / std::sqrt(2.0);
                term.orbitals[spin * occupied + index] = spin * spatial + occupied + virt;
                result.push_back(term);
            }
        }
    }
    return result;
}

/**
 * @brief Computes <old|new> directly from full spin-orbital determinants.
 * @param[in] overlap Spatial old/new orbital overlaps.
 * @param[in] old_terms Old normalized determinants and expansion coefficients.
 * @param[in] new_terms New normalized determinants and expansion coefficients.
 * @return Raw many-electron overlap; no normalization or orthogonalization.
 */
double explicit_overlap(
    const ModuleBase::ComplexMatrix &overlap, const std::vector<DeterminantTerm> &old_terms,
    const std::vector<DeterminantTerm> &new_terms)
{
    const int spatial = overlap.nr;
    double result = 0.0;
    for (const auto &old_term : old_terms)
    {
        for (const auto &new_term : new_terms)
        {
            const int electrons = static_cast<int>(old_term.orbitals.size());
            std::vector<double> spin_overlap(electrons * electrons, 0.0);
            for (int row = 0; row < electrons; ++row)
            {
                const int old_orbital = old_term.orbitals[row];
                for (int col = 0; col < electrons; ++col)
                {
                    const int new_orbital = new_term.orbitals[col];
                    if (old_orbital / spatial == new_orbital / spatial)
                    {
                        spin_overlap[row * electrons + col] =
                            overlap(old_orbital % spatial, new_orbital % spatial).real();
                    }
                }
            }
            result += old_term.coefficient * new_term.coefficient * determinant(spin_overlap, electrons);
        }
    }
    return result;
}

ModuleBase::ComplexMatrix identity(int size)
{
    ModuleBase::ComplexMatrix result(size, size);
    for (int index = 0; index < size; ++index)
    {
        result(index, index) = 1.0;
    }
    return result;
}

void rotate_columns(
    ModuleBase::ComplexMatrix &matrix,
    int first,
    int second,
    double angle)
{
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (int row = 0; row < matrix.nr; ++row)
    {
        const auto old_first = matrix(row, first);
        const auto old_second = matrix(row, second);
        matrix(row, first) = cosine * old_first + sine * old_second;
        matrix(row, second) = -sine * old_first + cosine * old_second;
    }
}

ModuleBase::ComplexMatrix production_overlap(
    const ModuleBase::ComplexMatrix &overlap, const std::vector<CasidaWavefunction> &old_states,
    const std::vector<CasidaWavefunction> &new_states)
{
    const int occupied = old_states[0].nocc_lr;
    const int virtual_count = old_states[0].nvirt_lr;
    const int count = static_cast<int>(old_states.size());
    FsshDriver driver;
    driver.init(
        overlap.nr,
        count,
        1.0,
        0,
        occupied,
        overlap.nr);
    const auto orbitals = identity(overlap.nr);
    ModuleBase::ComplexMatrix result(count, count);
    FsshDriverTestAccess::overlap(
        driver,
        orbitals,
        overlap,
        old_states,
        new_states,
        occupied,
        virtual_count,
        result);
    return result;
}

void compare_expansion(
    const ModuleBase::ComplexMatrix &overlap, const std::vector<CasidaWavefunction> &old_states,
    const std::vector<CasidaWavefunction> &new_states)
{
    const auto actual = production_overlap(overlap, old_states, new_states);
    for (int row = 0; row < actual.nr; ++row)
    {
        const auto old_terms = expand(old_states[row], row == 0);
        for (int col = 0; col < actual.nc; ++col)
        {
            const auto new_terms = expand(new_states[col], col == 0);
            const double reference = explicit_overlap(overlap, old_terms, new_terms);
            EXPECT_NEAR(actual(row, col).real(), reference, 1.0e-12) << row << ", " << col;
            EXPECT_NEAR(actual(row, col).imag(), 0.0, 1.0e-14);
        }
    }
}

} // namespace

TEST(FsshDriverClosedShell, TwoElectronAnalyticRotation)
{
    const std::vector<CasidaWavefunction> states{CasidaWavefunction(
                                                     0.0, {}, {},
                                                     1,
                                                     1),
        CasidaWavefunction(
            0.1, {1.0}, {},
            1,
            1)};
    for (const double angle : {0.0,
             0.01,
             0.05,
             0.2})
    {
        auto overlap = identity(2);
        rotate_columns(
            overlap,
            0,
            1,
            angle);
        const auto result = production_overlap(overlap, states, states);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        EXPECT_NEAR(result(0, 0).real(), cosine * cosine, 1.0e-12);
        EXPECT_NEAR(result(0, 1).real(), -std::sqrt(2.0) * sine * cosine, 1.0e-12);
        EXPECT_NEAR(result(1, 0).real(), std::sqrt(2.0) * sine * cosine, 1.0e-12);
        EXPECT_NEAR(result(1, 1).real(), std::cos(2.0 * angle), 1.0e-12);
        compare_expansion(overlap, states, states);
    }
}

TEST(FsshDriverClosedShell, MultipleOccupiedAndVirtualDeterminants)
{
    for (const int occupied : {2, 3})
    {
        const int virtual_count = 2;
        const int size = occupied + virtual_count;
        auto overlap = identity(size);
        for (int first = 0; first < size; ++first)
        {
            for (int second = first + 1; second < size; ++second)
            {
                const double angle = 0.07 * (first + 1) - 0.03 * (second + 1);
                rotate_columns(
                    overlap,
                    first,
                    second,
                    angle);
            }
        }
        std::vector<CasidaWavefunction> old_states{CasidaWavefunction(
            0.0, {}, {},
            occupied,
            virtual_count)};
        std::vector<CasidaWavefunction> new_states = old_states;
        for (int state = 1; state <= 3; ++state)
        {
            std::vector<double> old_x(occupied * virtual_count);
            std::vector<double> new_x(occupied * virtual_count);
            double old_norm = 0.0;
            double new_norm = 0.0;
            for (int pair = 0; pair < occupied * virtual_count; ++pair)
            {
                old_x[pair] = std::sin(0.7 * state + pair);
                new_x[pair] = std::cos(0.3 * state + pair);
                old_norm += old_x[pair] * old_x[pair];
                new_norm += new_x[pair] * new_x[pair];
            }
            for (int pair = 0; pair < occupied * virtual_count; ++pair)
            {
                old_x[pair] /= std::sqrt(old_norm);
                new_x[pair] /= std::sqrt(new_norm);
            }
            old_states.emplace_back(
                0.1 * state,
                old_x,
                std::vector<double>{},
                occupied,
                virtual_count);
            new_states.emplace_back(
                0.1 * state,
                new_x,
                std::vector<double>{},
                occupied,
                virtual_count);
        }
        compare_expansion(overlap, old_states, new_states);
        // A finite orbital window can leak: the projected cross-overlap need not be unitary.
        for (int col = 0; col < size; ++col)
        {
            overlap(0, col) *= 0.93;
        }
        compare_expansion(overlap, old_states, new_states);
    }
}

TEST(FsshDriverClosedShell, UnexcitedOccupiedOrbitalRemainsInDeterminants)
{
    const std::vector<CasidaWavefunction> states{CasidaWavefunction(
                                                     0.0, {}, {},
                                                     2,
                                                     1),
        CasidaWavefunction(
            0.1, {0.0, 1.0}, {},
            2,
            1)};
    auto overlap = identity(3);
    overlap(0, 0) = 0.8;
    rotate_columns(
        overlap,
        1,
        2,
        0.2);
    compare_expansion(overlap, states, states);
    const auto result = production_overlap(overlap, states, states);
    EXPECT_NEAR(result(1, 1).real(), 0.64 * std::cos(0.4), 1.0e-12);
}

TEST(FsshDriverClosedShell, OccupiedGaugeSignCancelsForBothSpins)
{
    const std::vector<CasidaWavefunction> old_states{CasidaWavefunction(
                                                         0.0, {}, {},
                                                         2,
                                                         1),
        CasidaWavefunction(
            0.1, {1.0, 0.0}, {},
            2,
            1),
        CasidaWavefunction(
            0.2, {0.0, 1.0}, {},
            2,
            1)};
    auto new_states = old_states;
    new_states[1].X_coeffs[0] = -1.0;
    auto overlap = identity(3);
    overlap(0, 0) = -1.0;
    const auto result = production_overlap(overlap, old_states, new_states);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            const double expected = row == col ? 1.0 : 0.0;
            EXPECT_NEAR(result(row, col).real(), expected, 1.0e-12);
        }
    }
    compare_expansion(overlap, old_states, new_states);
}

TEST(FsshDriverClosedShell, AmplitudesRemainBilinearWithoutRenormalization)
{
    const std::vector<CasidaWavefunction> old_states{CasidaWavefunction(
                                                         0.0, {}, {},
                                                         1,
                                                         1),
        CasidaWavefunction(
            0.1, {0.6}, {},
            1,
            1)};
    auto new_states = old_states;
    new_states[1].X_coeffs[0] = 0.8;
    auto overlap = identity(2);
    rotate_columns(
        overlap,
        0,
        1,
        0.15);
    compare_expansion(overlap, old_states, new_states);
}

TEST(FsshDriverClosedShell, ReversingFramesTransposesRawOverlap)
{
    const std::vector<CasidaWavefunction> old_states{CasidaWavefunction(
                                                         0.0, {}, {},
                                                         2,
                                                         1),
        CasidaWavefunction(
            0.1, {0.6, 0.8}, {},
            2,
            1)};
    auto new_states = old_states;
    new_states[1].X_coeffs = {0.8, -0.6};
    auto overlap = identity(3);
    rotate_columns(
        overlap,
        0,
        2,
        0.17);
    rotate_columns(
        overlap,
        1,
        2,
        -0.09);
    ModuleBase::ComplexMatrix reverse(3, 3);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            reverse(row, col) = overlap(col, row);
        }
    }
    const auto forward_result = production_overlap(overlap, old_states, new_states);
    const auto reverse_result = production_overlap(reverse, new_states, old_states);
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 2; ++col)
        {
            EXPECT_NEAR(forward_result(row, col).real(), reverse_result(col, row).real(), 1.0e-12);
        }
    }
}

TEST(FsshDriverClosedShell, SmallDeterminantDoesNotDivideByGroundOverlap)
{
    const std::vector<CasidaWavefunction> states{CasidaWavefunction(
                                                     0.0, {}, {},
                                                     2,
                                                     1),
        CasidaWavefunction(
            0.1, {1.0, 0.0}, {},
            2,
            1)};
    auto overlap = identity(3);
    for (int index = 0; index < 3; ++index)
    {
        overlap(index, index) = 1.0e-100;
    }
    const auto result = production_overlap(overlap, states, states);
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 2; ++col)
        {
            EXPECT_TRUE(std::isfinite(result(row, col).real()));
            EXPECT_EQ(result(row, col).real(), 0.0);
        }
    }
}

TEST(FsshDriverClosedShell, IllConditionedOccupiedBlockStillRejected)
{
    const std::vector<CasidaWavefunction> states{CasidaWavefunction(
                                                     0.0, {}, {},
                                                     2,
                                                     1),
        CasidaWavefunction(
            0.1, {1.0, 0.0}, {},
            2,
            1)};
    auto overlap = identity(3);
    overlap(0, 0) = 1.0e-12;
    EXPECT_THROW(production_overlap(overlap, states, states), std::runtime_error);
}
