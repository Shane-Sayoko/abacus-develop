#include "fssh.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_lr/utils/fssh_surface.h"

#include "fssh_mdcell.h"
#include "fssh_restart.h"
#include "md_func.h"
#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_cell.h"
#include "source_base/tool_quit.h"
#include "source_cell/unitcell.h"
#include "source_esolver/esolver_lr_lcao_tddft.h"
#include "source_io/module_output/print_info.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

FsshMD::FsshMD(const Parameter &param_in, MDCell &mdcell_in, std::ostream &log)
    : MD_base(
          param_in.mdp,
          param_in.inp.cal_stress,
          param_in.inp.init_vel,
          param_in.globalv.myrank,
          mdcell_in),
      log_(log), param_(param_in)
{
}

void FsshMD::setup(ModuleESolver::ESolver *p_esolver, const std::string &global_readin_dir, DomainDecomposition &decomp)
{
    (void)decomp;
    if (!mdcell.has_backing_unitcell() || param_.inp.cal_syns.empty() || !param_.inp.cal_syns[0] ||
        param_.inp.cal_stress)
    {
        ModuleBase::WARNING_QUIT("FsshMD", "FSSH requires a UnitCell, cal_syns=1 and cal_stress=0");
    }
    lr_solver_ = dynamic_cast<ModuleESolver::ESolver_LR<double, double> *>(p_esolver);
    if (lr_solver_ == nullptr)
    {
        ModuleBase::WARNING_QUIT("FsshMD", "FSSH requires the real gamma-point ks-lr solver");
    }
    if (mdp.md_restart)
    {
        restart(global_readin_dir);
    }

    ModuleIO::print_screen(0, 0, step_ + step_rst_ + 1);
    const bool positions_updated = true;
    mdcell.sync_backing_unitcell(positions_updated);
    UnitCell &unitcell = mdcell.backing_unitcell();
    lr_solver_->runner(unitcell, step_rst_);
    FsshElectronicFrame frame = lr_solver_->collect_fssh_frame(mdp.fssh_nstate);
    if (restored_)
    {
        if (previous_frame_.mo_coefficients.nr != frame.mo_coefficients.nr ||
            previous_frame_.mo_coefficients.nc != frame.mo_coefficients.nc ||
            previous_frame_.ks_energies_hartree.size() != frame.ks_energies_hartree.size() ||
            driver_.get_nstates() != mdp.fssh_nstate)
        {
            ModuleBase::WARNING_QUIT("FsshMD", "FSSH restart does not match current electronic dimensions");
        }
        const double restart_energy_tolerance_hartree = 1.0e-6;
        if (previous_frame_.occupied_bands != frame.occupied_bands ||
            previous_frame_.surface_energies_hartree.size() != frame.surface_energies_hartree.size())
        {
            ModuleBase::WARNING_QUIT("FsshMD", "FSSH restart occupied subspace or surface count changed");
        }
        for (std::size_t i = 0; i < frame.ks_energies_hartree.size(); ++i)
        {
            if (std::abs(frame.ks_energies_hartree[i] - previous_frame_.ks_energies_hartree[i]) >
                restart_energy_tolerance_hartree)
            {
                ModuleBase::WARNING_QUIT(
                    "FsshMD", "Recomputed FSSH restart KS spectrum differs by more than 1e-6 Hartree");
            }
        }
        for (std::size_t i = 0; i < frame.surface_energies_hartree.size(); ++i)
        {
            if (std::abs(frame.surface_energies_hartree[i] - previous_frame_.surface_energies_hartree[i]) >
                restart_energy_tolerance_hartree)
            {
                ModuleBase::WARNING_QUIT("FsshMD", "Recomputed FSSH restart surfaces differ by more than 1e-6 Hartree");
            }
        }
    }
    else
    {
        FsshModelConfig model;
        model.ground_state_coupling = mdp.fssh_ground_state_coupling;
        model.spin_channel = mdp.fssh_spin_channel == "triplet" ? FsshSpinChannel::triplet : FsshSpinChannel::singlet;
        driver_.init(
            model,
            frame.mo_coefficients.nc,
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
    MD_func::compute_stress(
        mdcell,
        false,
        virial,
        stress);
    t_current = MD_func::current_temp(kinetic, mdcell, frozen_freedom_);
}

void FsshMD::advance_surface(ModuleESolver::ESolver *p_esolver, const int ionic_step)
{
    if (p_esolver != lr_solver_)
    {
        throw std::logic_error("FSSH solver changed during the MD trajectory");
    }
    const bool positions_updated = true;
    mdcell.sync_backing_unitcell(positions_updated);
    UnitCell &unitcell = mdcell.backing_unitcell();
    lr_solver_->runner(unitcell, ionic_step);
    FsshElectronicFrame frame = lr_solver_->collect_fssh_frame(mdp.fssh_nstate);
    ModuleBase::ComplexMatrix aligned;
    log_ << "FSSH_STEP " << ionic_step << '\n';
    const int state = driver_.run_step_advanced(
        previous_frame_.mo_coefficients,
        frame.mo_coefficients,
        param_.globalv.global_out_dir + "syns_nao.csr",
        frame.ks_energies_hartree,
        mdcell,
        true,
        frame.surface_energies_hartree,
        previous_frame_.surface_energies_hartree,
        frame.casida_states,
        previous_frame_.ks_energies_hartree,
        &aligned,
        log_);
    frame.mo_coefficients = std::move(aligned);
    previous_frame_ = std::move(frame);
    const bool positions_updated_by_hop = false;
    mdcell.sync_backing_unitcell(positions_updated_by_hop);
    apply_surface(state);
}

void FsshMD::apply_surface(const int state)
{
    const LR::FsshSurface surface = lr_solver_->evaluate_fssh_surface(mdcell.backing_unitcell(), state);
    FsshMDCell::set_owned_forces(mdcell, surface.force_hartree_per_bohr);
    potential = surface.energy_hartree;
    virial.zero_out();
}

std::string FsshMD::restart_signature() const
{
    const UnitCell &cell = mdcell.backing_unitcell();
    const Input_para &input = param_.inp;
    std::ostringstream identity;
    identity << std::setprecision(17) << mdp.md_dt << ' ' << mdp.fssh_nstate << ' ' << mdp.decoherence << ' '
             << mdp.fssh_ground_state_coupling << ' ' << mdp.fssh_degen_energy_threshold << ' ' << input.nelec << ' '
             << input.nspin << ' ' << input.dft_functional << ' ' << input.xc_kernel << ' ' << input.ecutwfc << ' '
             << input.nbands << ' ' << input.lr_nstates << ' ' << input.nocc << ' ' << input.nvirt << ' '
             << input.nelec_delta;
    if (mdp.fssh_spin_channel == "triplet")
    {
        identity << " triplet";
    }
    identity << '\n';
    for (int type = 0; type < cell.ntype; ++type)
    {
        identity << cell.atoms[type].label << ' ' << cell.atoms[type].na << ' ' << cell.atoms[type].mass << '\n';
        identity << FsshRestart::checksum(FsshRestart::read_file(input.pseudo_dir + cell.pseudo_fn[type])) << '\n';
        identity << FsshRestart::checksum(FsshRestart::read_file(input.orbital_dir + cell.orbital_fn[type])) << '\n';
    }
    return FsshRestart::checksum(identity.str());
}

void FsshMD::write_restart(const std::string &global_out_dir)
{
    int failed = 0;
    std::string error;
    if (my_rank == 0)
    {
        try
        {
            std::ostringstream checkpoint;
            checkpoint << std::setprecision(17);
            if (!driver_.save_checkpoint(checkpoint))
            {
                throw std::runtime_error("Could not serialize FSSH driver");
            }
            const auto &mo = previous_frame_.mo_coefficients;
            checkpoint << mo.nr << ' ' << mo.nc << '\n';
            for (int i = 0; i < mo.nr; ++i)
            {
                for (int j = 0; j < mo.nc; ++j)
                {
                    checkpoint << mo(i, j).real() << ' ' << mo(i, j).imag() << '\n';
                }
            }
            checkpoint << previous_frame_.ks_energies_hartree.size() << '\n';
            for (double energy : previous_frame_.ks_energies_hartree)
            {
                checkpoint << energy << '\n';
            }
            checkpoint << previous_frame_.occupied_bands << ' ' << previous_frame_.surface_energies_hartree.size()
                       << '\n';
            for (double energy : previous_frame_.surface_energies_hartree)
            {
                checkpoint << energy << '\n';
            }
            const int step = step_ + step_rst_;
            const std::string structure_path = param_.globalv.global_stru_dir + "STRU_MD_" + std::to_string(step);
            const std::string structure = FsshRestart::read_file(structure_path);
            const std::string signature = restart_signature();
            const std::string path = global_out_dir + "Restart_md.txt";
            FsshRestart::write(
                path,
                step,
                md_tfirst,
                signature,
                structure,
                checkpoint.str());
            log_ << "FSSH restart written: " << FsshRestart::absolute_path(path) << " step " << step << '\n';
        }
        catch (const std::exception &exception)
        {
            failed = 1;
            error = exception.what();
        }
    }
    const ModuleBase::CommunicationDomain domain = ModuleBase::world_comm_domain();
    if (domain.max(failed) != 0)
    {
        ModuleBase::WARNING_QUIT("FsshMD::write_restart", "FSSH checkpoint not committed: " + error);
    }
}

void FsshMD::restart(const std::string &global_readin_dir)
{
    MD_base::restart(global_readin_dir);
    const std::string path = global_readin_dir + "Restart_md.txt";
    try
    {
        const std::string data = FsshRestart::read_file(path);
        const std::string structure = FsshRestart::read_file(param_.globalv.global_in_stru);
        const std::string signature = restart_signature();
        const std::string payload = FsshRestart::read(
            data,
            step_rst_,
            signature,
            structure);
        std::istringstream checkpoint(payload);
        FsshModelConfig model;
        model.ground_state_coupling = mdp.fssh_ground_state_coupling;
        model.spin_channel = mdp.fssh_spin_channel == "triplet" ? FsshSpinChannel::triplet : FsshSpinChannel::singlet;
        if (!driver_.load_checkpoint(checkpoint, model))
        {
            throw std::runtime_error("Invalid FSSH electronic state");
        }
        if (!driver_.restart_matches(
                md_dt,
                mdp.fssh_nstate,
                mdp.decoherence != 0,
                mdp.fssh_degen_energy_threshold))
        {
            throw std::runtime_error("FSSH time step or model differs from checkpoint");
        }
        int rows = 0;
        int cols = 0;
        if (!(checkpoint >> rows >> cols) || rows <= 0 || cols <= 0 ||
            static_cast<std::size_t>(rows) > payload.size() / static_cast<std::size_t>(cols) / 4)
        {
            throw std::runtime_error("Invalid FSSH MO dimensions");
        }
        previous_frame_.mo_coefficients.create(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                double real = 0.0;
                double imag = 0.0;
                if (!(checkpoint >> real >> imag) || !std::isfinite(real) || !std::isfinite(imag))
                {
                    throw std::runtime_error("Invalid FSSH MO coefficient");
                }
                previous_frame_.mo_coefficients(i, j) = {real, imag};
            }
        }
        std::size_t count = 0;
        if (!(checkpoint >> count) || count != static_cast<std::size_t>(rows))
        {
            throw std::runtime_error("Invalid FSSH KS energy count");
        }
        previous_frame_.ks_energies_hartree.resize(count);
        for (double &energy : previous_frame_.ks_energies_hartree)
        {
            if (!(checkpoint >> energy) || !std::isfinite(energy))
            {
                throw std::runtime_error("Invalid FSSH KS energy");
            }
        }
        std::size_t surfaces = 0;
        if (!(checkpoint >> previous_frame_.occupied_bands >> surfaces) || previous_frame_.occupied_bands <= 0 ||
            previous_frame_.occupied_bands > rows || surfaces != static_cast<std::size_t>(mdp.fssh_nstate))
        {
            throw std::runtime_error("Invalid FSSH occupied subspace or surface energy count");
        }
        previous_frame_.surface_energies_hartree.resize(surfaces);
        for (double &energy : previous_frame_.surface_energies_hartree)
        {
            if (!(checkpoint >> energy) || !std::isfinite(energy))
            {
                throw std::runtime_error("Invalid FSSH surface energy");
            }
        }
        checkpoint >> std::ws;
        if (!checkpoint.eof())
        {
            throw std::runtime_error("Unexpected FSSH payload data");
        }
        restored_ = true;
        log_ << "FSSH restart read: " << FsshRestart::absolute_path(path) << " step " << step_rst_ << '\n';
        if (my_rank == 0)
        {
            std::cout << "FSSH restart read: " << FsshRestart::absolute_path(path) << " step " << step_rst_
                      << std::endl;
        }
    }
    catch (const std::exception &exception)
    {
        ModuleBase::WARNING_QUIT("FsshMD::restart", exception.what());
    }
}
