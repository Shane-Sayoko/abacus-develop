#include "fssh.h"

#include "fssh_mdcell.h"
#include "md_func.h"
#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/tool_quit.h"
#include "source_cell/unitcell.h"
#include "source_esolver/esolver_lr_lcao_tddft.h"
#include "source_io/module_output/print_info.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

FsshMD::FsshMD(const Parameter& param_in, MDCell& mdcell_in)
    : MD_base(param_in, mdcell_in), param_(param_in)
{
}

void FsshMD::setup(ModuleESolver::ESolver* p_esolver,
                   const std::string& global_readin_dir,
                   DomainDecomposition& decomp)
{
    (void)decomp;
    if (!mdcell.has_backing_unitcell() || param_.inp.cal_syns.empty() || !param_.inp.cal_syns[0]
        || param_.inp.cal_stress)
    {
        ModuleBase::WARNING_QUIT("FsshMD", "FSSH requires a UnitCell, cal_syns=1 and cal_stress=0");
    }
    lr_solver_ = dynamic_cast<ModuleESolver::ESolver_LR<double, double>*>(p_esolver);
    if (lr_solver_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("FsshMD", "FSSH requires the real gamma-point ks-lr solver");
    }
    if (mdp.md_restart) { restart(global_readin_dir); }

    ModuleIO::print_screen(0, 0, step_ + step_rst_ + 1);
    mdcell.sync_backing_unitcell();
    UnitCell& unitcell = mdcell.backing_unitcell();
    unitcell.ionic_position_updated = true;
    lr_solver_->runner(unitcell, step_rst_);
    FsshElectronicFrame frame = lr_solver_->collect_fssh_frame(mdp.fssh_nstate);
    if (restored_)
    {
        if (previous_frame_.mo_coefficients.nr != frame.mo_coefficients.nr
            || previous_frame_.mo_coefficients.nc != frame.mo_coefficients.nc
            || previous_frame_.ks_energies_hartree.size() != frame.ks_energies_hartree.size()
            || driver_.get_nstates() != mdp.fssh_nstate)
        {
            ModuleBase::WARNING_QUIT("FsshMD", "FSSH restart does not match current electronic dimensions");
        }
    }
    else
    {
        driver_.init(frame.mo_coefficients.nc,
                     mdp.fssh_nstate,
                     md_dt,
                     mdp.fssh_init_state,
                     frame.occupied_bands,
                     frame.mo_coefficients.nr,
                     mdp.decoherence != 0,
                     static_cast<unsigned int>(mdp.fssh_random_seed),
                     mdp.fssh_degen_energy_threshold);
        driver_.set_casida_cache(frame.casida_states);
        previous_frame_ = std::move(frame);
    }
    apply_surface(driver_.get_current_state());
    MD_func::compute_stress(mdcell, false, virial, stress);
    t_current = MD_func::current_temp(kinetic, mdcell, frozen_freedom_);
}

void FsshMD::advance_surface(ModuleESolver::ESolver* p_esolver, const int ionic_step)
{
    if (p_esolver != lr_solver_)
    {
        throw std::logic_error("FSSH solver changed during the MD trajectory");
    }
    mdcell.sync_backing_unitcell();
    UnitCell& unitcell = mdcell.backing_unitcell();
    unitcell.ionic_position_updated = true;
    lr_solver_->runner(unitcell, ionic_step);
    FsshElectronicFrame frame = lr_solver_->collect_fssh_frame(mdp.fssh_nstate);
    ModuleBase::ComplexMatrix aligned;
    const int state = driver_.run_step_advanced(previous_frame_.mo_coefficients,
                                                 frame.mo_coefficients,
                                                 param_.globalv.global_out_dir + "syns_nao.csr",
                                                 frame.ks_energies_hartree,
                                                 mdcell,
                                                 true,
                                                 frame.surface_energies_hartree,
                                                 frame.casida_states,
                                                 previous_frame_.ks_energies_hartree,
                                                 &aligned);
    frame.mo_coefficients = std::move(aligned);
    previous_frame_ = std::move(frame);
    mdcell.sync_backing_unitcell();
    apply_surface(state);
}

void FsshMD::apply_surface(const int state)
{
    const ModuleESolver::FsshSurface surface
        = lr_solver_->evaluate_fssh_surface(mdcell.backing_unitcell(), state);
    FsshMDCell::set_owned_forces(mdcell, surface.force_hartree_per_bohr);
    potential = surface.energy_hartree;
    virial.zero_out();
}

void FsshMD::write_restart(const std::string& global_out_dir)
{
    MD_base::write_restart(global_out_dir);
    if (my_rank != 0) { return; }
    std::ofstream checkpoint(global_out_dir + "Restart_fssh.chk");
    checkpoint << std::setprecision(17);
    if (!checkpoint || !driver_.save_checkpoint(checkpoint))
    {
        ModuleBase::WARNING_QUIT("FsshMD", "could not write FSSH electronic checkpoint");
    }
    const auto& mo = previous_frame_.mo_coefficients;
    checkpoint << mo.nr << ' ' << mo.nc << '\n';
    for (int i = 0; i < mo.nr; ++i)
    {
        for (int j = 0; j < mo.nc; ++j)
        {
            checkpoint << mo(i, j).real() << ' ' << mo(i, j).imag() << '\n';
        }
    }
    checkpoint << previous_frame_.ks_energies_hartree.size() << '\n';
    for (double energy : previous_frame_.ks_energies_hartree) { checkpoint << energy << '\n'; }
    if (!checkpoint) { ModuleBase::WARNING_QUIT("FsshMD", "incomplete FSSH electronic checkpoint"); }
}

void FsshMD::restart(const std::string& global_readin_dir)
{
    MD_base::restart(global_readin_dir);
    std::ifstream checkpoint(global_readin_dir + "Restart_fssh.chk");
    if (!checkpoint || !driver_.load_checkpoint(checkpoint))
    {
        ModuleBase::WARNING_QUIT("FsshMD", "missing or invalid FSSH electronic checkpoint");
    }
    int rows = 0;
    int cols = 0;
    if (!(checkpoint >> rows >> cols) || rows <= 0 || cols <= 0)
    {
        ModuleBase::WARNING_QUIT("FsshMD", "invalid previous MO dimensions in FSSH checkpoint");
    }
    previous_frame_.mo_coefficients.create(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double real = 0.0;
            double imag = 0.0;
            if (!(checkpoint >> real >> imag))
            {
                ModuleBase::WARNING_QUIT("FsshMD", "truncated FSSH MO checkpoint");
            }
            previous_frame_.mo_coefficients(i, j) = {real, imag};
        }
    }
    std::size_t count = 0;
    if (!(checkpoint >> count) || count != static_cast<std::size_t>(rows))
    {
        ModuleBase::WARNING_QUIT("FsshMD", "invalid FSSH KS energies in checkpoint");
    }
    previous_frame_.ks_energies_hartree.resize(count);
    for (double& energy : previous_frame_.ks_energies_hartree)
    {
        if (!(checkpoint >> energy))
        {
            ModuleBase::WARNING_QUIT("FsshMD", "truncated FSSH KS energies in checkpoint");
        }
    }
    restored_ = true;
}
