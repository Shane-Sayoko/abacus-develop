#include "fssh.h"

#include "source_esolver/esolver_ks.h"
#include "source_lcao/module_lr/esolver_lrtd_lcao.h"
#include "source_lcao/module_lr/utils/lr_util.hpp"
#include "source_lcao/module_lr/utils/lr_util_print.h"
#include "source_base/parallel_2d.h"
#include "source_base/parallel_global.h"
#include "source_psi/psi.h"
#include "source_cell/print_cell.h"
#include "source_io/module_wf/read_wfc_nao.h"
#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include <mpi.h>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unistd.h> // for readlink

#ifndef DEBUG_NAC
#define DEBUG_NAC 0
#endif

// Helper to get current executable path
std::string get_self_path() {
    char buffer[1024];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        return std::string(buffer);
    }
    return "abacus"; // fallback
}

// ====================================================================
// Accessor: 用于访问ESolver的protected成员
// ====================================================================
namespace FsshIntegration {

    template <typename T, typename TR>
    class EsolverKsLcaoAccessor : public ModuleESolver::ESolver_KS_LCAO<T, TR> {
    public:
        static auto get_psi(ModuleESolver::ESolver_KS_LCAO<T, TR>* ks) -> decltype(static_cast<EsolverKsLcaoAccessor*>(ks)->psi)& { return static_cast<EsolverKsLcaoAccessor*>(ks)->psi; }
        static auto get_pelec(ModuleESolver::ESolver_KS_LCAO<T, TR>* ks) -> decltype(static_cast<EsolverKsLcaoAccessor*>(ks)->pelec)& { return static_cast<EsolverKsLcaoAccessor*>(ks)->pelec; }
        static const Parallel_Orbitals& get_pv(ModuleESolver::ESolver_KS_LCAO<T, TR>* ks) { return static_cast<EsolverKsLcaoAccessor*>(ks)->pv; }
    };

    template <typename T, typename TR>
    class EsolverLrAccessor : public LR::ESolver_LR<T, TR> {
    public:
        static const std::vector<ct::Tensor>* get_X(LR::ESolver_LR<T, TR>* esolver) { return &(static_cast<const EsolverLrAccessor*>(esolver)->X); }
        static int get_nstates(LR::ESolver_LR<T, TR>* esolver) { return static_cast<const EsolverLrAccessor*>(esolver)->nstates; }
        static auto get_pelec(LR::ESolver_LR<T, TR>* lr) -> decltype(static_cast<EsolverLrAccessor*>(lr)->pelec)& { return static_cast<EsolverLrAccessor*>(lr)->pelec; }
        static int get_nocc(LR::ESolver_LR<T, TR>* esolver) { return static_cast<const EsolverLrAccessor*>(esolver)->nocc[0]; }
        static int get_nvirt(LR::ESolver_LR<T, TR>* esolver) { return static_cast<const EsolverLrAccessor*>(esolver)->nvirt[0]; }
        static const std::vector<Parallel_2D>& get_paraX(LR::ESolver_LR<T, TR>* esolver) { return static_cast<const EsolverLrAccessor*>(esolver)->paraX_; }
        
        static void setup_paraX(LR::ESolver_LR<T, TR>* esolver, int blacs_ctxt) {
            auto lr = static_cast<EsolverLrAccessor*>(esolver);
            lr->paraX_.clear();
            lr->paraX_.resize(1); // spin 0
            int nocc_lr = lr->nocc[0];
            int nvirt_lr = lr->nvirt[0];
#ifdef __MPI
            LR_Util::setup_2d_division(lr->paraX_[0], 1, nvirt_lr, nocc_lr, blacs_ctxt);
#else
            lr->paraX_[0].set_serial(nvirt_lr, nocc_lr);
#endif
        }
    };
} // namespace FsshIntegration

FsshMD::FsshMD(const Parameter& param_in, UnitCell& unit_in) 
    : MD_base(param_in, unit_in), fssh_initialized_(false) {
}

FsshMD::~FsshMD() {}

void FsshMD::setup(ModuleESolver::ESolver* p_esolver, const std::string& global_readin_dir) {
    MD_base::setup(p_esolver, global_readin_dir);
    if (has_restored_force_) {
        for (int atom = 0; atom < ucell.nat; ++atom) force[atom] = restored_force_[atom];
        potential = restored_potential_;
        has_restored_force_ = false;
    }
}

void FsshMD::write_restart(const std::string& global_out_dir) {
    MD_base::write_restart(global_out_dir);
    if (my_rank != 0 || !fssh_initialized_) return;

    std::ofstream checkpoint(global_out_dir + "Restart_fssh.chk");
    if (!checkpoint.is_open()) {
        ModuleBase::WARNING_QUIT("FsshMD", "cannot write Restart_fssh.chk");
    }
    if (!fssh_engine_.save_checkpoint(checkpoint)) {
        ModuleBase::WARNING_QUIT("FsshMD", "failed to write Restart_fssh.chk");
    }
    checkpoint << coef_old_.nr << ' ' << coef_old_.nc << '\n';
    checkpoint << std::setprecision(17);
    for (int i = 0; i < coef_old_.nr; ++i) {
        for (int j = 0; j < coef_old_.nc; ++j) {
            checkpoint << coef_old_(i, j).real() << ' ' << coef_old_(i, j).imag() << '\n';
        }
    }
    checkpoint << tddft_energies_cache_.size() << '\n';
    for (double energy : tddft_energies_cache_) checkpoint << energy << '\n';
    checkpoint << potential << ' ' << ucell.nat << '\n';
    for (int atom = 0; atom < ucell.nat; ++atom) {
        checkpoint << force[atom].x << ' ' << force[atom].y << ' ' << force[atom].z << '\n';
    }
    if (!checkpoint) ModuleBase::WARNING_QUIT("FsshMD", "incomplete Restart_fssh.chk write");
}

void FsshMD::restart(const std::string& global_readin_dir) {
    MD_base::restart(global_readin_dir);
    std::ifstream checkpoint(global_readin_dir + "Restart_fssh.chk");
    if (!checkpoint.is_open() || !fssh_engine_.load_checkpoint(checkpoint)) {
        ModuleBase::WARNING_QUIT("FsshMD", "missing or invalid Restart_fssh.chk for FSSH restart");
    }
    int nr = 0, nc = 0;
    if (!(checkpoint >> nr >> nc) || nr <= 0 || nc <= 0) {
        ModuleBase::WARNING_QUIT("FsshMD", "missing previous MO coefficients in Restart_fssh.chk");
    }
    coef_old_.create(nr, nc);
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nc; ++j) {
            double re = 0.0, im = 0.0;
            if (!(checkpoint >> re >> im)) ModuleBase::WARNING_QUIT("FsshMD", "truncated MO coefficients in Restart_fssh.chk");
            coef_old_(i, j) = {re, im};
        }
    }
    size_t nenergy = 0;
    if (!(checkpoint >> nenergy)) ModuleBase::WARNING_QUIT("FsshMD", "missing TDDFT energies in Restart_fssh.chk");
    tddft_energies_cache_.resize(nenergy);
    for (double& energy : tddft_energies_cache_) {
        if (!(checkpoint >> energy)) ModuleBase::WARNING_QUIT("FsshMD", "truncated TDDFT energies in Restart_fssh.chk");
    }
    int natom = 0;
    if (!(checkpoint >> restored_potential_ >> natom) || natom != ucell.nat) {
        ModuleBase::WARNING_QUIT("FsshMD", "missing or incompatible force state in Restart_fssh.chk");
    }
    restored_force_.resize(natom);
    for (auto& value : restored_force_) {
        if (!(checkpoint >> value.x >> value.y >> value.z)) {
            ModuleBase::WARNING_QUIT("FsshMD", "truncated force state in Restart_fssh.chk");
        }
    }
    has_restored_force_ = true;
    fssh_initialized_ = true;
    skip_hopping_once_ = true;
}

[[noreturn]] void FsshMD::checkpoint_and_fail(const std::string& message, const Parameter& param_in) {
    if (fssh_initialized_) write_restart(param_in.globalv.global_out_dir);
    ModuleBase::WARNING_QUIT("FsshMD", message);
    std::abort();
}

void FsshMD::first_half(std::ofstream& ofs) {
    update_vel(this->force);
    update_pos();
}

void FsshMD::second_half() {
    update_vel(this->force);
}

void FsshMD::sync_vel_to_ucell() {
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it) {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia) {
            ucell.atoms[it].vel[ia] = this->vel[iat];
            iat++;
        }
    }
}

void FsshMD::sync_vel_from_ucell() {
    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it) {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia) {
            this->vel[iat] = ucell.atoms[it].vel[ia];
            iat++;
        }
    }
}

void FsshMD::execute_hopping(ModuleESolver::ESolver* p_esolver, const Parameter& param_in) {
    if (!param_in.inp.cal_syns) {
        checkpoint_and_fail("cal_syns = 1 is required for md_type = fssh", param_in);
    }
    if (skip_hopping_once_) {
        skip_hopping_once_ = false;
        return;
    }

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    bool use_tddft = (param_in.inp.esolver_type == "ks-lr");

    if (param_in.inp.esolver_type != "ksdft" && param_in.inp.esolver_type != "ks-lr") {
        if (my_rank == 0) {
            GlobalV::ofs_warning << "[FSSH ERROR] Only esolver_type = ksdft or ks-lr is supported for FSSH!" << std::endl;
        }
        exit(1);
    }

    const int fssh_nstate = param_in.mdp.fssh_nstate;
    const int fssh_init_state = param_in.mdp.fssh_init_state;

    int nbasis = 0;
    int nstates_ks = 0;
    int nocc = 0;
    ModuleBase::ComplexMatrix coef_new;
    std::vector<double> ks_bands;
    std::vector<double> tddft_energies;
    std::vector<CasidaWavefunction> casida_wfcs;

    try {
    if (auto ks_gamma = dynamic_cast<ModuleESolver::ESolver_KS_LCAO<double, double>*>(p_esolver)) {

#if FSSH_USE_FD_STATES
        this->update_states_kslr_fd(param_in, ks_gamma, tddft_energies, casida_wfcs);

        nbasis = param_in.globalv.nlocal;
        nstates_ks = param_in.inp.nbands;

        auto pelec_ptr = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_pelec(ks_gamma);
        ks_bands.resize(nstates_ks);
        // ABACUS Kohn-Sham eigenvalues are in Ry; FSSH uses Hartree.
        for (int is = 0; is < nstates_ks; ++is) { ks_bands[is] = pelec_ptr->ekb(0, is) / 2.0; }

        nocc = 0;
        for (int ib = 0; ib < nstates_ks; ++ib) {
            if (pelec_ptr->wg(0, ib) > 0.5) nocc++;
        }

        auto psi_ptr = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_psi(ks_gamma);
        const auto& pv = ks_gamma->get_pv();
        const int nrow_local = psi_ptr[0].get_nbasis();
        const int ncol_local = psi_ptr[0].get_nbands();
        coef_new.create(nstates_ks, nbasis);
        {
            std::vector<double> full_psi(static_cast<size_t>(nstates_ks) * nbasis, 0.0);
            for (int j = 0; j < ncol_local; ++j) {
                int g_band = pv.local2global_col(j);
                if (g_band >= nstates_ks) continue;
                for (int i = 0; i < nrow_local; ++i) {
                    int g_basis = pv.local2global_row(i);
                    full_psi[g_band * nbasis + g_basis] = psi_ptr[0].operator()(0, j, i);
                }
            }
#ifdef __MPI
            MPI_Allreduce(MPI_IN_PLACE, full_psi.data(), nstates_ks * nbasis, MPI_DOUBLE, MPI_SUM, pv.comm());
#endif
            for (int ib = 0; ib < nstates_ks; ++ib) {
                for (int mu = 0; mu < nbasis; ++mu) {
                    coef_new(ib, mu) = std::complex<double>(full_psi[ib * nbasis + mu], 0.0);
                }
            }
        }
#else
        auto psi_ptr = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_psi(ks_gamma);
        auto pelec_ptr = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_pelec(ks_gamma);

        nbasis = param_in.globalv.nlocal;
        nstates_ks = param_in.inp.nbands;

        const auto& pv = ks_gamma->get_pv();
        const int nrow_local = psi_ptr[0].get_nbasis();
        const int ncol_local = psi_ptr[0].get_nbands();

        coef_new.create(nstates_ks, nbasis);
        {
            std::vector<double> full_psi(static_cast<size_t>(nstates_ks) * nbasis, 0.0);
            for (int j = 0; j < ncol_local; ++j) {
                int g_band = pv.local2global_col(j);
                if (g_band >= nstates_ks) continue;
                for (int i = 0; i < nrow_local; ++i) {
                    int g_basis = pv.local2global_row(i);
                    full_psi[g_band * nbasis + g_basis] = psi_ptr[0].operator()(0, j, i);
                }
            }
#ifdef __MPI
            MPI_Allreduce(MPI_IN_PLACE, full_psi.data(),
                          nstates_ks * nbasis, MPI_DOUBLE, MPI_SUM, pv.comm());
#endif
            for (int ib = 0; ib < nstates_ks; ++ib) {
                for (int mu = 0; mu < nbasis; ++mu) {
                    coef_new(ib, mu) = std::complex<double>(full_psi[ib * nbasis + mu], 0.0);
                }
            }
        }

        ks_bands.resize(nstates_ks);
        // ABACUS Kohn-Sham eigenvalues are in Ry; FSSH uses Hartree.
        for (int is = 0; is < nstates_ks; ++is) { ks_bands[is] = pelec_ptr->ekb(0, is) / 2.0; }

        nocc = 0;
        for (int ib = 0; ib < nstates_ks; ++ib) {
            if (pelec_ptr->wg(0, ib) > 0.5) nocc++;
        }

        if (use_tddft) {
            LR::ESolver_LR<double, double> lr_solver(*ks_gamma, param_in.inp, this->ucell);
            lr_solver.runner(this->ucell, param_in.mdp.md_nstep);
            const std::vector<ct::Tensor>* X_tensor = FsshIntegration::EsolverLrAccessor<double, double>::get_X(&lr_solver);
            int num_excitations = FsshIntegration::EsolverLrAccessor<double, double>::get_nstates(&lr_solver);
            auto lr_pelec = FsshIntegration::EsolverLrAccessor<double, double>::get_pelec(&lr_solver);
            int nocc_lr = FsshIntegration::EsolverLrAccessor<double, double>::get_nocc(&lr_solver);
            int nvirt_lr = FsshIntegration::EsolverLrAccessor<double, double>::get_nvirt(&lr_solver);
            const auto& paraX = FsshIntegration::EsolverLrAccessor<double, double>::get_paraX(&lr_solver);

            if (nocc_lr != nocc) {
                throw std::runtime_error("FSSH-Löwdin requires nocc_lr == nocc_ks; frozen occupied orbitals are unsupported");
            }
            if (nvirt_lr <= 0 || nvirt_lr > nstates_ks - nocc) {
                throw std::runtime_error("invalid LR virtual subspace for FSSH-Löwdin");
            }
            if (fssh_nstate > num_excitations + 1) {
                throw std::runtime_error("fssh_nstate exceeds ground state plus available LR excited states");
            }
            if (!X_tensor || X_tensor->empty() || !lr_pelec || paraX.empty()) {
                throw std::runtime_error("incomplete Casida amplitudes or LR excitation energies");
            }
            tddft_energies.push_back(0.0);
            casida_wfcs.push_back(CasidaWavefunction(0.0, {}, {}, nocc_lr, nvirt_lr));

            {
                const Parallel_2D& px = paraX[0];
                int full_size = nocc_lr * nvirt_lr;

                for (int i = 0; i < num_excitations; ++i) {
                    CasidaWavefunction wfc;
                    wfc.omega = lr_pelec->ekb(0, i) / 2.0;
                    wfc.nocc_lr = nocc_lr;
                    wfc.nvirt_lr = nvirt_lr;
                    tddft_energies.push_back(wfc.omega);
                    wfc.X_coeffs.resize(full_size, 0.0);
                    const double* band_data = (*X_tensor)[0].template data<double>() + i * px.get_local_size();
#ifdef __MPI
                    LR_Util::gather_2d_to_full(px, band_data, wfc.X_coeffs.data(),
                                                false, nvirt_lr, nocc_lr);
#else
                    for (int j = 0; j < full_size; ++j) {
                        wfc.X_coeffs[j] = band_data[j];
                    }
#endif
                    casida_wfcs.push_back(wfc);
                }
            }
        }
#endif
    } else {
        throw std::runtime_error("FSSH requires Gamma-point LCAO ESolver_KS_LCAO");
    }

    if (use_tddft) {
        if (fssh_nstate > static_cast<int>(casida_wfcs.size())
            || fssh_nstate > static_cast<int>(tddft_energies.size())) {
            throw std::runtime_error("fssh_nstate exceeds complete ground-state plus LR excitation data");
        }
        if (casida_wfcs.size() < 2 || casida_wfcs[1].nocc_lr != nocc) {
            throw std::runtime_error("FSSH-Löwdin requires nocc_lr == nocc_ks; frozen occupied orbitals are unsupported");
        }
        const int nvirt_ks = nstates_ks - nocc;
        const int nvirt_lr = casida_wfcs[1].nvirt_lr;
        if (nvirt_lr <= 0 || nvirt_lr > nvirt_ks) {
            throw std::runtime_error("FSSH-Löwdin requires 0 < nvirt_lr <= nvirt_ks");
        }
        const size_t x_size = static_cast<size_t>(nocc) * nvirt_lr;
        for (int state = 1; state < fssh_nstate; ++state) {
            const auto& wfc = casida_wfcs[state];
            if (wfc.nocc_lr != nocc || wfc.nvirt_lr != nvirt_lr || wfc.X_coeffs.size() != x_size) {
                throw std::runtime_error("incomplete or inconsistent Casida X coefficients");
            }
            double norm2 = 0.0;
            for (double coefficient : wfc.X_coeffs) norm2 += coefficient * coefficient;
            if (!std::isfinite(norm2) || norm2 <= 1.0e-15) {
                throw std::runtime_error("zero or invalid Casida X coefficient norm");
            }
        }
    }

    // TDDFT energies and CasidaWavefunction::omega are Hartree throughout FSSH.
    this->tddft_energies_cache_ = tddft_energies;

    if (!fssh_initialized_) {
        // The input time step is in fs; FSSH and MD internals use atomic units.
        double md_dt_au = param_in.mdp.md_dt / ModuleBase::AU_to_FS;
        const bool enable_decoherence = (param_in.mdp.decoherence == 1);
        fssh_engine_.init(nbasis, fssh_nstate, md_dt_au, fssh_init_state,
                          nocc, nstates_ks, enable_decoherence,
                          static_cast<unsigned int>(param_in.mdp.fssh_random_seed),
                          param_in.mdp.fssh_degen_energy_threshold);
        if (my_rank == 0) {
            std::cout << "[FSSH INFO] Initialized: nstate=" << fssh_nstate
                      << ", init_state=" << fssh_init_state
                      << ", nocc=" << nocc << ", nks_total=" << nstates_ks
                      << ", use_tddft=" << (use_tddft ? "True" : "False")
                      << ", decoherence=" << (enable_decoherence ? "True" : "False")
                      << std::endl;
        }
        coef_old_ = coef_new;
        if (use_tddft && !casida_wfcs.empty()) {
            fssh_engine_.set_casida_cache(casida_wfcs);
        }
        fssh_initialized_ = true;

        // [Crucial Fix] Calculate Step 0 forces for the upcoming first_half in istep=0
#if FSSH_USE_FD_FORCE
        this->update_force_active_state_fd(param_in);
#endif
    } else {
        std::string csr_file = param_in.globalv.global_out_dir + "syns_nao.csr";
#if FSSH_USE_FD_STATES
        const std::string step_dir = "Step" + std::to_string(this->step_) + "/";
        const std::string work_dir = step_dir + "sp_force/";
        const std::string suffix = param_in.inp.suffix;
        csr_file = work_dir + "OUT." + suffix + "/syns_nao.csr";
#endif
        if (my_rank == 0) {
            std::cout << "[FSSH DEBUG] Reading asynchronous overlap from: " << csr_file << std::endl <<std::flush;
        }
        this->sync_vel_to_ucell();
        fssh_engine_.run_step_advanced(
            coef_old_, coef_new, csr_file, ks_bands, this->ucell, use_tddft, tddft_energies, casida_wfcs
        );
        this->sync_vel_from_ucell();
        coef_old_ = coef_new;
#if FSSH_USE_FD_FORCE
        this->update_force_active_state_fd(param_in);
#endif
    }
    } catch (const std::exception& error) {
        checkpoint_and_fail(error.what(), param_in);
    }
}

void FsshMD::update_states_kslr_fd(const Parameter& param_in, ModuleESolver::ESolver* p_esolver,
                                   std::vector<double>& tddft_energies,
                                   std::vector<CasidaWavefunction>& casida_wfcs) {
    int my_rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // [Step-wise directory isolation]
    const std::string step_dir = "Step" + std::to_string(this->step_) + "/";
    const std::string work_dir = step_dir + "sp_force/";
    const std::string suffix = param_in.inp.suffix;

    if (my_rank == 0) {
        std::cout << "[FSSH FD-HIJACK] Step " << this->step_ << ": Calling abacus-fd for electronic states..." << std::endl;
        std::system(("mkdir -p " + work_dir).c_str());
        unitcell::print_stru_file(this->ucell, this->ucell.atoms, this->ucell.latvec, work_dir + "STRU", 
                                  param_in.inp.nspin, false, false, false, true, false, 0);
        std::system(("cp INPUT " + work_dir).c_str());
        if (std::ifstream("KPT")) std::system(("cp KPT " + work_dir).c_str());

        std::string abacus_bin = get_self_path();
        std::string cmd = "abacus-fd kslr-states -a " + abacus_bin + " -d " + work_dir 
                        + " -n " + std::to_string(param_in.mdp.fssh_fd_nproc) + " > " + work_dir + "fd_states.log 2>&1";

        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "[FSSH ERROR] abacus-fd states calculation failed! Check " << work_dir << "fd_states.log" << std::endl;
        }
        int fd_success = ret == 0 ? 1 : 0;
        MPI_Bcast(&fd_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (fd_success == 0) throw std::runtime_error("abacus-fd kslr-states failed");
    } else {
        int fd_success = 0;
        MPI_Bcast(&fd_success, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (fd_success == 0) throw std::runtime_error("abacus-fd kslr-states failed");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (auto ks_gamma = dynamic_cast<ModuleESolver::ESolver_KS_LCAO<double, double>*>(p_esolver)) {
        auto psi_ptr = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_psi(ks_gamma);
        auto pelec_ptr = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_pelec(ks_gamma);
        std::vector<int> ik2iktot = {0}; // Gamma only
        ModuleIO::read_wfc_nao<double>(work_dir + "OUT." + suffix + "/", ks_gamma->get_pv(), psi_ptr[0], 
                                        pelec_ptr->ekb, pelec_ptr->wg, ik2iktot, 1, param_in.inp.nspin, 0, -1);
    }

    if (param_in.inp.esolver_type == "ks-lr") {
        if (auto ks_gamma = dynamic_cast<ModuleESolver::ESolver_KS_LCAO<double, double>*>(p_esolver)) {
            LR::ESolver_LR<double, double> lr_meta(*ks_gamma, param_in.inp, this->ucell);
#ifdef __MPI
            const auto& pv = FsshIntegration::EsolverKsLcaoAccessor<double, double>::get_pv(ks_gamma);
            FsshIntegration::EsolverLrAccessor<double, double>::setup_paraX(&lr_meta, pv.blacs_ctxt);
#else
            FsshIntegration::EsolverLrAccessor<double, double>::setup_paraX(&lr_meta, -1);
#endif
            int nocc_lr = FsshIntegration::EsolverLrAccessor<double, double>::get_nocc(&lr_meta);
            int nvirt_lr = FsshIntegration::EsolverLrAccessor<double, double>::get_nvirt(&lr_meta);
            int num_excitations = FsshIntegration::EsolverLrAccessor<double, double>::get_nstates(&lr_meta);
            int full_size = nocc_lr * nvirt_lr;

            tddft_energies.clear();
            casida_wfcs.clear();
            double etot_gs_ry = 0.0;
            int energy_file_ok = 1;
            if (my_rank == 0) {
                std::string log_file = work_dir + "OUT." + suffix + "/running_scf.log";
                std::ifstream ifs_log(log_file);
                std::string line;
                while (std::getline(ifs_log, line)) {
                    if (line.find("E_KohnSham") != std::string::npos) {
                        size_t pos = line.find("E_KohnSham");
                        std::string sub = line.substr(pos + 10);
                        std::stringstream ss(sub);
                        ss >> etot_gs_ry;
                        if (!ss) energy_file_ok = 0;
                        break;
                    }
                }
                ifs_log.close();
                if (etot_gs_ry == 0.0) energy_file_ok = 0;
                // std::cout << "[FSSH FD-HIJACK] Parsed Ground State Energy: " << etot_gs_ry << " Ry" << std::endl;
            }
            MPI_Bcast(&energy_file_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (energy_file_ok == 0) throw std::runtime_error("missing or invalid E_KohnSham in FD output");
            MPI_Bcast(&etot_gs_ry, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            tddft_energies.push_back(etot_gs_ry / 2.0);
            casida_wfcs.push_back(CasidaWavefunction(0.0, {}, {}, nocc_lr, nvirt_lr));

            std::string out_dir = work_dir + "OUT." + suffix + "/";
            // ABACUS writes Excitation_Energy_singlet.dat in Rydberg units.
            // Keep the file values in Ry until converting to Hartree below.
            std::vector<double> all_energies_ry(num_excitations);
            int excitation_energy_ok = 1;
            if (my_rank == 0) {
                std::ifstream energies_file(out_dir + "Excitation_Energy_singlet.dat");
                for (double& energy : all_energies_ry) {
                    if (!(energies_file >> energy)) {
                        excitation_energy_ok = 0;
                        break;
                    }
                }
            }
            MPI_Bcast(&excitation_energy_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (excitation_energy_ok == 0) throw std::runtime_error("missing or incomplete TDDFT excitation-energy file");
            MPI_Bcast(all_energies_ry.data(), num_excitations, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> merged_amplitudes(static_cast<size_t>(num_excitations) * full_size, 0.0);
            int amplitude_file_ok = 1;
            if (my_rank == 0) {
                std::ifstream amplitude_file(out_dir + "Excitation_Amplitude_singlet.dat");
                double value = 0.0;
                size_t index = 0;
                while (amplitude_file >> value && index < merged_amplitudes.size()) {
                    merged_amplitudes[index++] = value;
                }
                if (!amplitude_file.eof() || index != merged_amplitudes.size()) amplitude_file_ok = 0;
            }
            MPI_Bcast(&amplitude_file_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (amplitude_file_ok == 0) {
                throw std::runtime_error("missing or malformed merged external FD Casida-amplitude file");
            }
            MPI_Bcast(merged_amplitudes.data(), static_cast<int>(merged_amplitudes.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (int i = 0; i < num_excitations; ++i) {
                CasidaWavefunction wfc;
                // 1 Ry = 0.5 Hartree.  The previous code treated these native
                // Ry values as eV, underestimating every excitation gap.
                wfc.omega = all_energies_ry[i] / 2.0;
                wfc.nocc_lr = nocc_lr;
                wfc.nvirt_lr = nvirt_lr;
                tddft_energies.push_back(etot_gs_ry / 2.0 + wfc.omega);
                const auto begin = merged_amplitudes.begin() + static_cast<size_t>(i) * full_size;
                wfc.X_coeffs.assign(begin, begin + full_size);
                casida_wfcs.push_back(wfc);
            }
        }
    }
}

void FsshMD::update_force_active_state_fd(const Parameter& param_in) {
    const int active_state = fssh_engine_.get_current_state();
    const int natom = ucell.nat;
    std::vector<double> fd_forces(natom * 3, 0.0);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // [Step-wise directory isolation]
    const std::string step_dir = "Step" + std::to_string(this->step_) + "/";
    const std::string sp_dir = step_dir + "sp_force/";
    const std::string fd_dir = step_dir + "fd_forces/";

    if (active_state < this->tddft_energies_cache_.size()) {
        this->potential = this->tddft_energies_cache_[active_state];
        if (my_rank == 0) {
            //  std::cout << "[FSSH FD-HIJACK] Active potential updated to: " << this->potential * 2.0 << " Ry" << std::endl;
        }
    }

    if (my_rank == 0) {
        std::string force_file;
        if (active_state == 0) {
            // [基态优化] 直接从Step*/sp_force目录读取单点计算生成的解析力
            force_file = sp_dir + "ground_forces.txt";
            std::cout << "[FSSH FD-HIJACK] Active state: 0 (Ground). Reading analytical forces from " << force_file << std::endl;
        } else {
            // [激发态] 调用abacus-fd进行有限差分
            std::cout << "[FSSH FD-HIJACK] Active state: " << active_state << ". Calling abacus-fd for FD forces (Finite Difference)..." << std::endl;
            std::system(("mkdir -p " + fd_dir).c_str());
            unitcell::print_stru_file(this->ucell, this->ucell.atoms, this->ucell.latvec, fd_dir + "STRU", 
                                        param_in.inp.nspin, false, false, false, true, false, 0);
            std::system(("cp INPUT " + fd_dir).c_str());
            if (std::ifstream("KPT")) std::system(("cp KPT " + fd_dir).c_str());
            std::string abacus_bin = get_self_path();
            std::string cmd = "abacus-fd " + std::string(param_in.inp.esolver_type == "ks-lr" ? "kslr-all" : "gs-all")
                + " -a " + abacus_bin + " -d " + fd_dir + " -n " + std::to_string(param_in.mdp.fssh_fd_nproc)
                + " -j " + std::to_string(param_in.mdp.fssh_fd_nparallel) + " > " + fd_dir + "fd_force.log 2>&1";
            std::system(cmd.c_str());
            force_file = fd_dir + "excited_forces.txt";
        }

        std::ifstream ifs(force_file);
        if (ifs.is_open()) {
            std::string line;
            const int force_state_idx = active_state - 1;
            std::vector<bool> force_seen(static_cast<size_t>(natom), false);
            while (std::getline(ifs, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::stringstream ss(line);
                if (active_state == 0) {
                    int a_idx; double fx, fy, fz;
                    if (ss >> a_idx >> fx >> fy >> fz && a_idx >= 0 && a_idx < natom) {
                        fd_forces[a_idx * 3 + 0] = fx; fd_forces[a_idx * 3 + 1] = fy; fd_forces[a_idx * 3 + 2] = fz;
                    }
                } else {
                    std::string type; int s_idx, a_idx; double fx, fy, fz;
                    // FSSH currently propagates singlet states.  The finite-
                    // difference file stores excited states as S 0..N-1,
                    // while FSSH uses S0..SN, so subtract the ground state
                    // offset and never let triplet rows overwrite singlet data.
                    if (ss >> type >> s_idx >> a_idx >> fx >> fy >> fz
                        && type == "S" && s_idx == force_state_idx && a_idx >= 0 && a_idx < natom) {
                        fd_forces[a_idx * 3 + 0] = fx; fd_forces[a_idx * 3 + 1] = fy; fd_forces[a_idx * 3 + 2] = fz;
                        force_seen[static_cast<size_t>(a_idx)] = true;
                    }
                }
            }
            ifs.close();
            if (active_state > 0
                && std::find(force_seen.begin(), force_seen.end(), false) != force_seen.end()) {
                throw std::runtime_error("incomplete singlet finite-difference force data for active FSSH state");
            }
        } else {
            throw std::runtime_error("cannot open FSSH force file: " + force_file);
        }
    }
#ifdef __MPI
    MPI_Bcast(fd_forces.data(), natom * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    // [Unit Conversion] Convert eV/Angstrom to Hartree/Bohr (Atomic Units)
    // Force(a.u.) = Force(eV/A) / (Hartree_to_eV * ANGSTROM_AU)
    const double eV_A_to_au = 1.0 / (ModuleBase::Hartree_to_eV * ModuleBase::ANGSTROM_AU);
    for (int i = 0; i < natom * 3; ++i) {
        fd_forces[i] *= eV_A_to_au;
    }

    if (my_rank == 0) {
        std::cout << " --- FSSH Nuclear Forces (Hartree/Bohr) ---" << std::endl;
        for (int i = 0; i < natom; ++i) {
            std::cout << " Atom " << i << ": " << std::fixed << std::setprecision(8) 
                      << fd_forces[i*3+0] << " " << fd_forces[i*3+1] << " " << fd_forces[i*3+2] << std::endl;
        }
        std::cout << " ------------------------------------------" << std::endl;
    }

    if (this->force != nullptr) {
        for (int i = 0; i < natom; ++i) {
            this->force[i].x = fd_forces[i * 3 + 0];
            this->force[i].y = fd_forces[i * 3 + 1];
            this->force[i].z = fd_forces[i * 3 + 2];
        }
    }

    int iat = 0;
    for (int it = 0; it < ucell.ntype; ++it) {
        if (ucell.atoms[it].force.size() != ucell.atoms[it].na) ucell.atoms[it].force.resize(ucell.atoms[it].na);
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia) {
            ucell.atoms[it].force[ia].x = fd_forces[iat * 3 + 0];
            ucell.atoms[it].force[ia].y = fd_forces[iat * 3 + 1];
            ucell.atoms[it].force[ia].z = fd_forces[iat * 3 + 2];
            iat++;
        }
    }
}
