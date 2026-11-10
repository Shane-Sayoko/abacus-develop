#include "esolver_lr_lcao_tddft.h"

#include "source_lcao/module_lr/utils/fssh_frame.h"
#include "source_lcao/module_lr/utils/fssh_surface.h"

#include <stdexcept>
#include <type_traits>

namespace ModuleESolver
{
template <typename T, typename TR>
LR::FsshSurface ESolver_LR<T, TR>::evaluate_fssh_surface(BaseCell &basecell, const int state)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell &ucell = static_cast<UnitCell &>(basecell);
    if (!this->inp_ || this->inp_->calculation != "md" || this->inp_->mdp.md_type != "fssh" ||
        this->inp_->esolver_type != "ks-lr" || !this->inp_->cal_force)
    {
        throw std::logic_error("FSSH surface evaluation requires ks-lr MD with cal_force=1");
    }
    if (!this->lr_runner_completed_ || !this->ks_ || &ucell != this->ucell_ || !this->pelec)
    {
        throw std::logic_error("Complete the LR solver for this geometry before evaluating a FSSH surface");
    }
    const bool triplet = this->inp_->mdp.fssh_spin_channel == "triplet";
    const int expected_nspin = triplet ? 2 : 1;
    if (!std::is_same<T, double>::value || this->nk != 1 || this->nspin != expected_nspin || this->openshell)
    {
        throw std::logic_error("Analytic FSSH forces require the selected closed-shell Gamma LR channel");
    }
    if (state < 0 || state > this->nstates)
    {
        throw std::out_of_range("FSSH surface index is outside the solved LR states");
    }
    if (triplet && state == 0)
    {
        throw std::logic_error("Triplet FSSH cannot select the S0 placeholder");
    }

    ModuleBase::matrix ground_force;
    this->ks_->cal_force(ucell, ground_force);
    if (state == 0)
    {
        return LR::make_fssh_surface(
            this->etot_gs_,
            0.0,
            ground_force,
            nullptr);
    }
    const int excited_state = state - 1;
    const int excitation_channel = triplet ? 1 : 0;
    const std::vector<ModuleBase::matrix> correction = this->cal_force(excitation_channel, excited_state);
    const double excitation_energy = this->pelec->ekb(excitation_channel, excited_state);
    const ModuleBase::matrix *excitation_force = &correction.at(0);
    return LR::make_fssh_surface(
        this->etot_gs_,
        excitation_energy,
        ground_force,
        excitation_force);
}

template <typename T, typename TR>
FsshElectronicFrame ESolver_LR<T, TR>::collect_fssh_frame(const int state_count) const
{
    if (!this->inp_ || this->inp_->calculation != "md" || this->inp_->mdp.md_type != "fssh" ||
        !this->lr_runner_completed_ || !this->ks_ || !this->psi_ks_all_ || !this->pelec)
    {
        throw std::logic_error("FSSH electronic data requires a completed ks-lr MD step");
    }
    const bool triplet = this->inp_->mdp.fssh_spin_channel == "triplet";
    const int expected_nspin = triplet ? 2 : 1;
    if (!std::is_same<T, double>::value || this->nk != 1 || this->nspin != expected_nspin || this->openshell ||
        this->inp_->kpar != 1)
    {
        throw std::logic_error("FSSH electronic data requires the selected closed-shell Gamma LR channel and kpar=1");
    }
    const int excitation_channel = triplet ? 1 : 0;
    if (state_count < 2 || state_count > this->nstates + 1 ||
        this->X.size() <= static_cast<std::size_t>(excitation_channel) ||
        this->paraX_.size() <= static_cast<std::size_t>(excitation_channel) ||
        this->lr_residuals_.size() <= static_cast<std::size_t>(excitation_channel))
    {
        throw std::out_of_range("FSSH requests more electronic surfaces than the LR solver provides");
    }
    return LR::make_fssh_frame(
        state_count,
        excitation_channel,
        this->nbasis,
        this->nocc[0],
        this->nvirt[0],
        this->nloc_per_state,
        this->etot_gs_,
        this->eig_ks_all,
        this->wg_ks_all,
        this->pelec->ekb,
        *this->psi_ks_all_,
        this->ks_->pv,
        this->paraX_[excitation_channel],
        this->X[excitation_channel],
        this->lr_residuals_.at(excitation_channel));
}

template LR::FsshSurface ESolver_LR<double, double>::evaluate_fssh_surface(BaseCell &, int);
template LR::FsshSurface ESolver_LR<std::complex<double>, double>::evaluate_fssh_surface(BaseCell &, int);
template FsshElectronicFrame ESolver_LR<double, double>::collect_fssh_frame(int) const;
template FsshElectronicFrame ESolver_LR<std::complex<double>, double>::collect_fssh_frame(int) const;
} // namespace ModuleESolver
