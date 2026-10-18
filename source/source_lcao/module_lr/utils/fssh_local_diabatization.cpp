#include "fssh_local_diabatization.h"

#include "source_base/module_external/lapack_connector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace FsshLD
{
namespace
{
using Matrix = ModuleBase::ComplexMatrix;
using Complex = std::complex<double>;

bool finite(const Complex value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

std::vector<double> diagonalize(Matrix& matrix)
{
    const int dimension = matrix.nr;
    for (int index = 0; index < matrix.size; ++index)
    {
        if (!finite(matrix.c[index]))
        {
            throw std::runtime_error("LD matrix arithmetic overflowed before diagonalization");
        }
    }
    std::vector<double> eigenvalues(dimension);
    const int workspace_size = std::max(1, 4 * dimension);
    std::vector<Complex> workspace(workspace_size);
    std::vector<double> real_workspace(std::max(1, 3 * dimension - 2));
    int status = 0;
    LapackConnector::zheev('V',
                           'U',
                           dimension,
                           matrix,
                           dimension,
                           eigenvalues.data(),
                           workspace.data(),
                           workspace_size,
                           real_workspace.data(),
                           &status);
    if (status != 0)
    {
        throw std::runtime_error("LD Hermitian eigendecomposition failed");
    }
    return eigenvalues;
}

Matrix adjoint(const Matrix& matrix)
{
    return ModuleBase::transpose(matrix, true);
}

double norm(const std::vector<Complex>& coefficients)
{
    double value = 0.0;
    for (const auto coefficient: coefficients)
    {
        if (!finite(coefficient))
        {
            throw std::runtime_error("LD amplitudes must be finite");
        }
        value += std::norm(coefficient);
    }
    return value;
}

Matrix spectral_matrix(const Matrix& vectors, const std::vector<Complex>& values)
{
    Matrix scaled(vectors);
    for (int row = 0; row < scaled.nr; ++row)
    {
        for (int column = 0; column < scaled.nc; ++column)
        {
            scaled(row, column) *= values[column];
        }
    }
    return scaled * adjoint(vectors);
}
} // namespace

Propagation propagate(const Matrix& overlap,
                      const std::vector<double>& energy_old,
                      const std::vector<double>& energy_new,
                      const std::vector<Complex>& coefficients,
                      double timestep)
{
    const int dimension = overlap.nr;
    if (dimension <= 0 || overlap.nc != dimension || energy_old.size() != static_cast<std::size_t>(dimension)
        || energy_new.size() != energy_old.size() || coefficients.size() != energy_old.size())
    {
        throw std::runtime_error("LD state window or endpoint dimensions changed");
    }
    if (!std::isfinite(timestep) || std::abs(norm(coefficients) - 1.0) > 1.0e-12)
    {
        throw std::runtime_error("LD requires finite time and normalized input amplitudes");
    }
    for (int row = 0; row < dimension; ++row)
    {
        if (!std::isfinite(energy_old[row]) || !std::isfinite(energy_new[row]))
        {
            throw std::runtime_error("LD energies must be finite");
        }
        for (int column = 0; column < dimension; ++column)
        {
            if (!finite(overlap(row, column)))
            {
                throw std::runtime_error("LD overlap must be finite");
            }
        }
    }

    Propagation result;
    result.uncoupled_endpoints = true;
    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            if (row != column && overlap(row, column) != Complex(0.0))
            {
                result.uncoupled_endpoints = false;
            }
        }
    }
    Matrix gram = adjoint(overlap) * overlap;
    const std::vector<double> gram_eigenvalues = diagonalize(gram);
    std::vector<Complex> inverse_roots(dimension);
    result.singular_values.resize(dimension);
    for (int state = 0; state < dimension; ++state)
    {
        if (gram_eigenvalues[state] <= 0.0 || !std::isfinite(gram_eigenvalues[state]))
        {
            throw std::runtime_error("LD overlap has insufficient rank");
        }
        const double singular_value = std::sqrt(gram_eigenvalues[state]);
        result.singular_values[state] = singular_value;
        if (singular_value < 0.9 || singular_value > 1.0 + 1.0e-8)
        {
            std::ostringstream message;
            message.precision(17);
            message << "LD overlap singular value " << singular_value << " outside [0.9, 1+1e-8]";
            throw std::runtime_error(message.str());
        }
        inverse_roots[state] = 1.0 / singular_value;
    }
    result.polar_overlap = overlap * spectral_matrix(gram, inverse_roots);
    Matrix rotated(result.polar_overlap);
    // A common energy origin only changes the final global phase.
    const double reference = energy_old.front();
    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            rotated(row, column) *= energy_new[column] - reference;
        }
    }
    Matrix hamiltonian = 0.5 * (rotated * adjoint(result.polar_overlap));
    for (int state = 0; state < dimension; ++state)
    {
        hamiltonian(state, state) += 0.5 * (energy_old[state] - reference);
    }
    const std::vector<double> eigenvalues = diagonalize(hamiltonian);
    std::vector<Complex> phases(dimension);
    for (int state = 0; state < dimension; ++state)
    {
        const double phase = -(eigenvalues[state] + reference) * timestep;
        if (!std::isfinite(phase))
        {
            throw std::runtime_error("LD propagation phase overflowed");
        }
        phases[state] = std::exp(Complex(0.0, phase));
    }
    result.propagator = adjoint(result.polar_overlap) * spectral_matrix(hamiltonian, phases);
    const Matrix unitary_check = adjoint(result.propagator) * result.propagator;
    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            if (!finite(result.propagator(row, column)) || !finite(unitary_check(row, column)))
            {
                throw std::runtime_error("LD propagator or unitarity diagnostic is non-finite");
            }
            Complex residual = unitary_check(row, column);
            if (row == column)
            {
                residual -= 1.0;
            }
            result.unitarity_residual = std::max(result.unitarity_residual, std::abs(residual));
        }
    }
    if (result.unitarity_residual > 1.0e-12)
    {
        throw std::runtime_error("LD propagator violates the 1e-12 unitarity gate before normalization");
    }
    result.coefficients.assign(dimension, Complex(0.0));
    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            result.coefficients[row] += result.propagator(row, column) * coefficients[column];
        }
    }
    if (!std::isfinite(norm(result.coefficients)))
    {
        throw std::runtime_error("LD propagated amplitudes are non-finite");
    }
    return result;
}

std::vector<double> probabilities_2001(const Propagation& propagation,
                                       const std::vector<Complex>& coefficients_old,
                                       int active_state)
{
    const int dimension = static_cast<int>(coefficients_old.size());
    if (dimension <= 0 || active_state < 0 || active_state >= dimension
        || propagation.coefficients.size() != coefficients_old.size() || propagation.propagator.nr != dimension
        || propagation.propagator.nc != dimension || std::abs(norm(coefficients_old) - 1.0) > 1.0e-12
        || std::abs(norm(propagation.coefficients) - 1.0) > 1.0e-12)
    {
        throw std::runtime_error("LD probability input is inconsistent");
    }
    const double population_old = std::norm(coefficients_old[active_state]);
    const double population_new = std::norm(propagation.coefficients[active_state]);
    if (population_old == 0.0)
    {
        throw std::runtime_error("LD active surface has zero old population");
    }
    std::vector<double> weights(dimension, 0.0);
    double denominator = 0.0;
    double weight_scale = 0.0;
    bool exactly_decoupled = true;
    for (int state = 0; state < dimension; ++state)
    {
        if (state == active_state)
        {
            continue;
        }
        const Complex coupling = propagation.propagator(active_state, state);
        if (!finite(coupling))
        {
            throw std::runtime_error("LD transition matrix contains a non-finite value");
        }
        exactly_decoupled = exactly_decoupled && coupling == Complex(0.0);
        weights[state]
            = std::real(coupling * coefficients_old[state] * std::conj(propagation.coefficients[active_state]));
        // Algebraically p1_K-Re[U_KK c0_K conj(c1_K)]. Summation of off-diagonal
        // terms avoids catastrophic cancellation without assigning missing channels.
        denominator += weights[state];
        weight_scale += std::abs(weights[state]);
    }
    const double population_change = population_new - population_old;
    const double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    const double denominator_floor = roundoff * std::max(population_new, weight_scale);
    std::vector<double> probabilities(dimension, 0.0);
    if (std::abs(denominator) <= denominator_floor)
    {
        if (propagation.uncoupled_endpoints && exactly_decoupled
            && std::abs(population_change) <= roundoff * population_old)
        {
            return probabilities;
        }
        std::ostringstream message;
        message.precision(17);
        message << "LD 2001 Eq.19 has an unresolved denominator: " << denominator
                << ", active population change=" << population_change << "; no unique channel allocation is assumed";
        throw std::runtime_error(message.str());
    }
    double total_probability = 0.0;
    for (int state = 0; state < dimension; ++state)
    {
        // Ratios first: multiplying two O(population) values can underflow even
        // when their final conditional probability is O(1).
        const double probability = -(weights[state] / denominator) * (population_change / population_old);
        if (!std::isfinite(probability))
        {
            throw std::runtime_error("LD conditional probability is non-finite");
        }
        probabilities[state] = std::max(0.0, probability);
        if (!std::isfinite(probabilities[state]))
        {
            throw std::runtime_error("LD probability is non-finite");
        }
        total_probability += probabilities[state];
    }
    if (total_probability > 1.0 + 1.0e-12)
    {
        throw std::runtime_error("LD total outgoing probability exceeds one; reduce the nuclear timestep");
    }
    return probabilities;
}

void require_resolved_active_surface(const std::vector<double>& energies,
                                     const std::vector<double>& residuals,
                                     int active_state)
{
    if (energies.empty() || residuals.size() != energies.size() || active_state < 0
        || active_state >= static_cast<int>(energies.size()))
    {
        throw std::runtime_error("LD active-surface diagnostic dimensions are inconsistent");
    }
    for (std::size_t state = 0; state < energies.size(); ++state)
    {
        if (!std::isfinite(energies[state]) || !std::isfinite(residuals[state]) || residuals[state] < 0.0)
        {
            throw std::runtime_error("LD active-surface diagnostics must be finite with nonnegative residuals");
        }
    }
    for (std::size_t state = 0; state < energies.size(); ++state)
    {
        if (state == static_cast<std::size_t>(active_state))
        {
            continue;
        }
        const double scale = std::max(1.0, std::max(std::abs(energies[state]), std::abs(energies[active_state])));
        const double resolution
            = residuals[state] + residuals[active_state] + 64.0 * std::numeric_limits<double>::epsilon() * scale;
        if (std::abs(energies[state] - energies[active_state]) <= resolution)
        {
            throw std::runtime_error("LD active force surface is not resolved by the LR residuals");
        }
    }
}
} // namespace FsshLD
