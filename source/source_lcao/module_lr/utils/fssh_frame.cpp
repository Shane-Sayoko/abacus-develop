#include "fssh_frame.h"

#include "source_base/parallel_reduce.h"

#include <cmath>
#include <stdexcept>

namespace LR
{
template <typename T>
FsshElectronicFrame make_fssh_frame(int state_count,
                                    int basis_count,
                                    int occupied,
                                    int virtual_count,
                                    int local_state_size,
                                    double ground_energy_ry,
                                    const ModuleBase::matrix& ks_energies_ry,
                                    const ModuleBase::matrix& occupations,
                                    const ModuleBase::matrix& excitation_energies_ry,
                                    const psi::Psi<T>& orbitals,
                                    const Parallel_Orbitals& pv,
                                    const Parallel_2D& px,
                                    const ct::Tensor& casida_amplitudes,
                                    const std::vector<double>& excitation_residuals_ry)
{
    const int ks_band_count = ks_energies_ry.nc;
    if (ks_band_count <= 0 || basis_count <= 0 || occupations.nc != ks_band_count)
    {
        throw std::logic_error("KS orbitals or occupations are incomplete for FSSH");
    }

    FsshElectronicFrame frame;
    if (excitation_residuals_ry.size() < static_cast<std::size_t>(state_count - 1))
    {
        throw std::logic_error("LR residuals are incomplete for FSSH");
    }
    frame.eigen_residuals_hartree.assign(state_count, 0.0);
    for (int state = 1; state < state_count; ++state)
    {
        const double residual = excitation_residuals_ry[state - 1];
        if (!std::isfinite(residual) || residual < 0.0)
        {
            throw std::logic_error("Invalid LR residual for FSSH");
        }
        frame.eigen_residuals_hartree[state] = 0.5 * residual;
    }
    frame.mo_coefficients.create(ks_band_count, basis_count);
    std::vector<double> coefficients(static_cast<std::size_t>(ks_band_count) * basis_count, 0.0);
    for (int local_band = 0; local_band < orbitals.get_nbands(); ++local_band)
    {
        const int band = pv.local2global_col(local_band);
        if (band < 0 || band >= ks_band_count)
        {
            continue;
        }
        for (int local_ao = 0; local_ao < orbitals.get_nbasis(); ++local_ao)
        {
            const int ao = pv.local2global_row(local_ao);
            if (ao < 0 || ao >= basis_count)
            {
                continue;
            }
            coefficients[static_cast<std::size_t>(band) * basis_count + ao]
                = std::real(orbitals(0, local_band, local_ao));
        }
    }
    Parallel_Reduce::reduce_all(coefficients.data(), static_cast<int>(coefficients.size()));
    frame.ks_energies_hartree.reserve(ks_band_count);
    for (int band = 0; band < ks_band_count; ++band)
    {
        frame.ks_energies_hartree.push_back(0.5 * ks_energies_ry(0, band));
        if (occupations(0, band) > 0.5)
        {
            ++frame.occupied_bands;
        }
        for (int ao = 0; ao < basis_count; ++ao)
        {
            frame.mo_coefficients(band, ao)
                = std::complex<double>(coefficients[static_cast<std::size_t>(band) * basis_count + ao], 0.0);
        }
    }
    if (occupied != frame.occupied_bands || virtual_count <= 0 || occupied + virtual_count > ks_band_count)
    {
        throw std::logic_error("FSSH-Lowdin requires the full occupied KS space and available virtual bands");
    }

    frame.surface_energies_hartree.reserve(state_count);
    frame.casida_states.reserve(state_count);
    frame.surface_energies_hartree.push_back(0.5 * ground_energy_ry);
    frame.casida_states.emplace_back(0.0, std::vector<double>{}, std::vector<double>{}, occupied, virtual_count);

    const T* amplitudes = casida_amplitudes.template data<T>();
    for (int state = 1; state < state_count; ++state)
    {
        CasidaWavefunction casida;
        casida.omega = 0.5 * excitation_energies_ry(0, state - 1);
        casida.nocc_lr = occupied;
        casida.nvirt_lr = virtual_count;
        casida.X_coeffs.assign(static_cast<std::size_t>(occupied) * virtual_count, 0.0);
        for (int local_occ = 0; local_occ < px.get_col_size(); ++local_occ)
        {
            const int occ = px.local2global_col(local_occ);
            if (occ < 0 || occ >= occupied)
            {
                continue;
            }
            for (int local_virtual = 0; local_virtual < px.get_row_size(); ++local_virtual)
            {
                const int virt = px.local2global_row(local_virtual);
                if (virt < 0 || virt >= virtual_count)
                {
                    continue;
                }
                casida.X_coeffs[static_cast<std::size_t>(occ) * virtual_count + virt]
                    = std::real(amplitudes[static_cast<std::size_t>(state - 1) * local_state_size
                                           + local_occ * px.get_row_size() + local_virtual]);
            }
        }
        Parallel_Reduce::reduce_all(casida.X_coeffs.data(), static_cast<int>(casida.X_coeffs.size()));
        double norm_squared = 0.0;
        for (const double coefficient: casida.X_coeffs)
        {
            norm_squared += coefficient * coefficient;
        }
        if (!std::isfinite(norm_squared) || norm_squared <= 1.0e-15)
        {
            throw std::logic_error("FSSH received an empty or invalid Casida excitation vector");
        }
        frame.surface_energies_hartree.push_back(0.5 * ground_energy_ry + casida.omega);
        frame.casida_states.push_back(std::move(casida));
    }
    return frame;
}

template FsshElectronicFrame make_fssh_frame<double>(int,
                                                     int,
                                                     int,
                                                     int,
                                                     int,
                                                     double,
                                                     const ModuleBase::matrix&,
                                                     const ModuleBase::matrix&,
                                                     const ModuleBase::matrix&,
                                                     const psi::Psi<double>&,
                                                     const Parallel_Orbitals&,
                                                     const Parallel_2D&,
                                                     const ct::Tensor&,
                                                     const std::vector<double>&);

template FsshElectronicFrame make_fssh_frame<std::complex<double>>(int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   double,
                                                                   const ModuleBase::matrix&,
                                                                   const ModuleBase::matrix&,
                                                                   const ModuleBase::matrix&,
                                                                   const psi::Psi<std::complex<double>>&,
                                                                   const Parallel_Orbitals&,
                                                                   const Parallel_2D&,
                                                                   const ct::Tensor&,
                                                                   const std::vector<double>&);
} // namespace LR
