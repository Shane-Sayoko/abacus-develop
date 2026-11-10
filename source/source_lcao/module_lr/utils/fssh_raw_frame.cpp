#include "fssh_raw_frame.h"
#include "fssh_state_overlap.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace LR
{
namespace
{
/** @brief Check finite Cartesian geometry and explicit basis/frame identities. */
void validate_geometry(const FsshFrameGeometry &geometry)
{
    if (geometry.frame_id < 0 || !std::isfinite(geometry.time_au) || geometry.basis_id.empty() ||
        geometry.positions_bohr.empty() || geometry.positions_bohr.size() % 3 != 0)
    {
        throw std::runtime_error("LD raw frame requires explicit geometry, time and basis identity");
    }
    for (double value : geometry.cell_bohr)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("LD raw cell must be finite");
        }
    }
    const auto &a = geometry.cell_bohr;
    const double volume =
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(volume) || volume <= 0.0)
    {
        throw std::runtime_error("LD raw cell must have positive finite volume");
    }
    for (double value : geometry.positions_bohr)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("LD raw positions must be finite");
        }
    }
}

/** @brief Require that AO-integral provenance names the actual electronic endpoint. */
void require_same_geometry(const FsshFrameGeometry &actual, const FsshFrameGeometry &expected)
{
    validate_geometry(actual);
    if (actual.frame_id != expected.frame_id || actual.basis_id != expected.basis_id ||
        actual.time_au != expected.time_au || actual.positions_bohr.size() != expected.positions_bohr.size())
    {
        throw std::runtime_error("LD AO overlap endpoint identity does not match electronic frame");
    }
    for (std::size_t i = 0; i < actual.cell_bohr.size(); ++i)
    {
        if (std::abs(actual.cell_bohr[i] - expected.cell_bohr[i]) > 1.0e-12)
        {
            throw std::runtime_error("LD AO overlap cell does not match electronic frame");
        }
    }
    for (std::size_t i = 0; i < actual.positions_bohr.size(); ++i)
    {
        if (std::abs(actual.positions_bohr[i] - expected.positions_bohr[i]) > 1.0e-12)
        {
            throw std::runtime_error("LD AO overlap coordinates do not match electronic frame");
        }
    }
}

/** @brief Invert a full permutation of solver roots 0..N-1, rejecting duplicates. */
std::vector<int> root_indices(const std::vector<int> &roots)
{
    std::vector<int> indices(roots.size(), -1);
    for (std::size_t i = 0; i < roots.size(); ++i)
    {
        const int root = roots[i];
        if (root < 0 || root >= static_cast<int>(roots.size()) || indices[root] != -1)
        {
            throw std::runtime_error("LD solver-root labels must be a permutation of 0..N-1");
        }
        indices[root] = static_cast<int>(i);
    }
    return indices;
}

/** @brief Reject unsupported complex-Gamma data rather than discarding its imaginary part. */
void require_real_finite(const ModuleBase::ComplexMatrix &matrix)
{
    for (int i = 0; i < matrix.size; ++i)
    {
        if (!std::isfinite(matrix.c[i].real()) || matrix.c[i].imag() != 0.0)
        {
            throw std::runtime_error("LD raw orbital/AO interface supports finite real Gamma data only");
        }
    }
}

/** @brief Validate TDA state normalization and consistency of energies, roots and orbital dimensions. */
void validate_electronic(const FsshElectronicFrame &frame, const std::vector<int> &roots)
{
    const int states = static_cast<int>(roots.size());
    const int bands = frame.mo_coefficients.nr;
    if (states < 2 || frame.mo_coefficients.nc <= 0 || frame.occupied_bands <= 0 || bands <= frame.occupied_bands ||
        frame.ks_energies_hartree.size() != static_cast<std::size_t>(bands) ||
        frame.surface_energies_hartree.size() != roots.size() || frame.casida_states.size() != roots.size() ||
        frame.eigen_residuals_hartree.size() != roots.size())
    {
        throw std::runtime_error("LD raw electronic dimensions are inconsistent");
    }
    require_real_finite(frame.mo_coefficients);
    const auto indices = root_indices(roots);
    const double ground = frame.surface_energies_hartree[indices[0]];
    const int virtual_count = frame.casida_states[indices[1]].nvirt_lr;
    if (virtual_count <= 0 || frame.occupied_bands + virtual_count > bands)
    {
        throw std::runtime_error("LD raw virtual orbital window is invalid");
    }
    for (double value : frame.ks_energies_hartree)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("LD raw KS energies must be finite");
        }
    }
    for (int i = 0; i < states; ++i)
    {
        const double energy = frame.surface_energies_hartree[i];
        const double residual = frame.eigen_residuals_hartree[i];
        const auto &state = frame.casida_states[i];
        const double energy_tolerance = 1.0e-10 + 16.0 * std::numeric_limits<double>::epsilon() * std::abs(ground);
        if (!std::isfinite(energy) || !std::isfinite(residual) || residual < 0.0 || !std::isfinite(state.omega) ||
            std::abs(energy - ground - state.omega) > energy_tolerance)
        {
            throw std::runtime_error("LD raw surface energies/residuals disagree with their Casida labels");
        }
        for (double value : state.Y_coeffs)
        {
            if (value != 0.0)
            {
                throw std::runtime_error("LD raw frame requires TDA (Y=0)");
            }
        }
        if (roots[i] == 0)
        {
            if (!state.X_coeffs.empty() || state.omega != 0.0)
            {
                throw std::runtime_error("LD S0 must be the empty ground-state placeholder");
            }
            continue;
        }
        if (state.nocc_lr != frame.occupied_bands || state.nvirt_lr != virtual_count ||
            state.X_coeffs.size() != static_cast<std::size_t>(frame.occupied_bands) * virtual_count)
        {
            throw std::runtime_error("LD raw frame requires the complete occupied space and matching X dimensions");
        }
        for (double value : state.X_coeffs)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("LD raw Casida amplitudes must be finite");
            }
        }
    }
    for (int r = 1; r < states; ++r)
    {
        const auto &x = frame.casida_states[indices[r]].X_coeffs;
        for (int s = 1; s <= r; ++s)
        {
            const auto &y = frame.casida_states[indices[s]].X_coeffs;
            double product = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                product += x[i] * y[i];
            }
            if (r == s)
            {
                product -= 1.0;
            }
            if (std::abs(product) > 1.0e-10)
            {
                throw std::runtime_error("LD raw Casida states are not orthonormal; no normalization is applied");
            }
        }
    }
}
} // namespace

FsshRawFrame::FsshRawFrame(
    const FsshElectronicFrame &electronic, const std::vector<int> &solver_roots, const FsshFrameGeometry &geometry)
{
    validate_geometry(geometry);
    validate_electronic(electronic, solver_roots);
    electronic_ = electronic;
    solver_roots_ = solver_roots;
    geometry_ = geometry;
}

const FsshElectronicFrame &FsshRawFrame::electronic() const { return electronic_; }

const std::vector<int> &FsshRawFrame::solver_roots() const { return solver_roots_; }

const FsshFrameGeometry &FsshRawFrame::geometry() const { return geometry_; }

FsshLDEndpoints prepare_fssh_ld_endpoints(
    const FsshRawFrame &old_frame,
    const FsshRawFrame &new_frame,
    const FsshRawAoOverlap &ao_overlap,
    bool ground_state_coupling)
{
    require_same_geometry(ao_overlap.old_geometry, old_frame.geometry());
    require_same_geometry(ao_overlap.new_geometry, new_frame.geometry());
    const auto &old = old_frame.electronic();
    const auto &next = new_frame.electronic();
    if (old_frame.geometry().frame_id == new_frame.geometry().frame_id ||
        new_frame.geometry().time_au <= old_frame.geometry().time_au ||
        old_frame.geometry().basis_id != new_frame.geometry().basis_id ||
        old_frame.geometry().cell_bohr != new_frame.geometry().cell_bohr ||
        old_frame.geometry().positions_bohr.size() != new_frame.geometry().positions_bohr.size() ||
        old.mo_coefficients.nr != next.mo_coefficients.nr || old.mo_coefficients.nc != next.mo_coefficients.nc ||
        old.occupied_bands != next.occupied_bands || old.casida_states.size() != next.casida_states.size() ||
        ao_overlap.values.nr != old.mo_coefficients.nc || ao_overlap.values.nc != next.mo_coefficients.nc)
    {
        throw std::runtime_error("LD endpoints have changed basis/window, invalid time or AO dimensions");
    }
    require_real_finite(ao_overlap.values);
    const auto old_indices = root_indices(old_frame.solver_roots());
    const auto new_indices = root_indices(new_frame.solver_roots());
    const int virtual_count = old.casida_states[old_indices[1]].nvirt_lr;
    if (virtual_count != next.casida_states[new_indices[1]].nvirt_lr)
    {
        throw std::runtime_error("LD virtual orbital window changed between endpoints");
    }
    const int states = static_cast<int>(old_indices.size());
    std::vector<CasidaWavefunction> old_states(states);
    std::vector<CasidaWavefunction> new_states(states);
    for (int root = 0; root < states; ++root)
    {
        old_states[root] = old.casida_states[old_indices[root]];
        new_states[root] = next.casida_states[new_indices[root]];
    }
    const auto full_overlap = closed_shell_fssh_overlap(
        FsshSpinChannel::singlet,
        old.mo_coefficients,
        next.mo_coefficients,
        ao_overlap.values,
        old_states,
        new_states,
        old.occupied_bands,
        states,
        old.occupied_bands,
        virtual_count,
        0,
        nullptr);
    int first_root = 0;
    if (!ground_state_coupling)
    {
        first_root = 1;
    }
    FsshLDEndpoints result;
    result.full_state_count = states;
    result.timestep_au = new_frame.geometry().time_au - old_frame.geometry().time_au;
    result.overlap.create(states - first_root, states - first_root);
    for (int root = first_root; root < states; ++root)
    {
        result.solver_roots.push_back(root);
        result.old_indices.push_back(old_indices[root]);
        result.new_indices.push_back(new_indices[root]);
        result.energy_old.push_back(old.surface_energies_hartree[old_indices[root]]);
        result.energy_new.push_back(next.surface_energies_hartree[new_indices[root]]);
        for (int other = first_root; other < states; ++other)
        {
            result.overlap(root - first_root, other - first_root) = full_overlap(root, other);
        }
    }
    return result;
}

FsshLDFrameStep advance_fssh_ld_endpoints(
    const FsshLDEndpoints &endpoints, const std::vector<std::complex<double>> &coefficients_old, int active_old_index)
{
    const std::size_t count = endpoints.solver_roots.size();
    if (endpoints.full_state_count <= 0 ||
        coefficients_old.size() != static_cast<std::size_t>(endpoints.full_state_count) ||
        endpoints.old_indices.size() != count || endpoints.new_indices.size() != count || count == 0 ||
        !std::isfinite(endpoints.timestep_au) || endpoints.timestep_au <= 0.0)
    {
        throw std::runtime_error("LD endpoint map or full coefficient dimensions are invalid");
    }
    std::vector<bool> seen_old(endpoints.full_state_count, false);
    std::vector<bool> seen_new(endpoints.full_state_count, false);
    std::vector<std::complex<double>> retained(count);
    int active = -1;
    for (std::size_t i = 0; i < count; ++i)
    {
        const int old_index = endpoints.old_indices[i];
        const int new_index = endpoints.new_indices[i];
        if (old_index < 0 || new_index < 0 || old_index >= endpoints.full_state_count ||
            new_index >= endpoints.full_state_count || seen_old[old_index] || seen_new[new_index])
        {
            throw std::runtime_error("LD endpoint storage map is not one-to-one");
        }
        seen_old[old_index] = true;
        seen_new[new_index] = true;
        retained[i] = coefficients_old[old_index];
        if (old_index == active_old_index)
        {
            active = static_cast<int>(i);
        }
    }
    if (active < 0)
    {
        throw std::runtime_error("LD active state is outside the retained model");
    }
    for (int i = 0; i < endpoints.full_state_count; ++i)
    {
        if (!seen_old[i] && coefficients_old[i] != std::complex<double>(0.0))
        {
            throw std::runtime_error("LD excluded state has nonzero population");
        }
    }
    FsshLDFrameStep result;
    result.propagation = FsshLD::propagate(
        endpoints.overlap,
        endpoints.energy_old,
        endpoints.energy_new,
        retained,
        endpoints.timestep_au);
    result.hopping = FsshLD::probabilities_sharc(result.propagation, retained, active);
    result.coefficients_new.assign(endpoints.full_state_count, std::complex<double>(0.0));
    result.probabilities_new.assign(endpoints.full_state_count, 0.0);
    result.active_new_index = endpoints.new_indices[active];
    for (std::size_t i = 0; i < count; ++i)
    {
        result.coefficients_new[endpoints.new_indices[i]] = result.propagation.coefficients[i];
        result.probabilities_new[endpoints.new_indices[i]] = result.hopping.probabilities[i];
    }
    return result;
}

int resolved_fssh_force_root(const FsshRawFrame &frame, int raw_state_index, bool ground_state_coupling)
{
    if (raw_state_index < 0 || raw_state_index >= static_cast<int>(frame.solver_roots().size()))
    {
        throw std::runtime_error("LD force storage index is out of range");
    }
    std::vector<double> energies;
    std::vector<double> residuals;
    int active = -1;
    for (std::size_t i = 0; i < frame.solver_roots().size(); ++i)
    {
        if (!ground_state_coupling && frame.solver_roots()[i] == 0)
        {
            continue;
        }
        if (static_cast<int>(i) == raw_state_index)
        {
            active = static_cast<int>(energies.size());
        }
        energies.push_back(frame.electronic().surface_energies_hartree[i]);
        residuals.push_back(frame.electronic().eigen_residuals_hartree[i]);
    }
    FsshLD::require_resolved_active_surface(energies, residuals, active);
    return frame.solver_roots()[raw_state_index];
}
} // namespace LR
