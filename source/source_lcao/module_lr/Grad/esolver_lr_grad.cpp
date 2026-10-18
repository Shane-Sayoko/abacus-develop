#include "source_esolver/esolver_lr_lcao_tddft.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_io/module_output/output_log.h"
#include "source_lcao/module_lr/Grad/force/lr_force.h"
#include "source_lcao/module_lr/Grad/gradient_assembly.h"
#include "source_lcao/module_lr/Grad/multipliers/zeq_solver.h"

#include <stdexcept>
#include <type_traits>

using namespace LR;

///========================= excited-state geometry relaxation =========================

template <typename T, typename TR>
ModuleESolver::FsshSurface ModuleESolver::ESolver_LR<T, TR>::evaluate_fssh_surface(BaseCell& basecell, const int state)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell& ucell = static_cast<UnitCell&>(basecell);
    if (!this->inp_ || this->inp_->calculation != "md" || this->inp_->mdp.md_type != "fssh"
        || this->inp_->esolver_type != "ks-lr")
    {
        throw std::logic_error("FSSH surface evaluation requires ks-lr molecular dynamics");
    }
    if (!this->lr_runner_completed_ || !this->ks_ || &ucell != this->ucell_ || !this->pelec)
    {
        throw std::logic_error("run the LR solver on the current UnitCell before evaluating a FSSH surface");
    }
    if (!std::is_same<T, double>::value || this->nk != 1 || this->nspin != 1 || this->openshell)
    {
        throw std::logic_error("analytic FSSH force currently requires gamma-only, closed-shell singlet LR");
    }
    if (state < 0 || state > this->nstates)
    {
        throw std::out_of_range("FSSH surface index is outside the solved LR states");
    }

    ModuleBase::matrix force_ry;
    this->ks_->cal_force(ucell, force_ry);
    double energy_ry = this->etot_gs_;
    if (state > 0)
    {
        energy_ry += this->pelec->ekb(0, state - 1);
        force_ry = force_ry + this->cal_force(0, state - 1).at(0);
    }

    FsshSurface surface;
    surface.energy_hartree = 0.5 * energy_ry;
    surface.force_hartree_per_bohr = force_ry;
    surface.force_hartree_per_bohr *= 0.5;
    return surface;
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::setup_relax_target_()
{
    this->excited_relax_ = (this->inp_->calculation == "relax");
    if (!this->excited_relax_)
    {
        return;
    }

    // The Z-vector equation has no complex solver (see Grad/multipliers/zeq_solver.hpp), so an
    // excited-state gradient only exists at gamma. Fail here rather than after the SCF.
    if (!std::is_same<T, double>::value)
    {
        ModuleBase::WARNING_QUIT("ESolver_LR",
                                 "excited-state relaxation currently requires gamma-only sampling: the complex "
                                 "Z-vector solver is not implemented.");
    }
    if (this->inp_->lr_target_state >= this->nstates)
    {
        ModuleBase::WARNING_QUIT("ESolver_LR",
                                 "lr_target_state is beyond the states actually solved (lr_nstates <= 0 expands to "
                                 "all particle-hole pairs, which may be fewer than requested).");
    }

    // `openshell` is only settled after the ground-state occupations have been read, which is why
    // this cannot live in the input-file checks
    const std::string& spin = this->inp_->lr_target_spin;
    if (this->openshell)
    {
        // An open-shell calculation solves one spin-conserving channel, so there is nothing to
        // choose: whatever lr_target_spin says, this is the state that gets relaxed. Only an
        // explicit `triplet` is worth mentioning -- `singlet` is the default and expresses no
        // intent, and `updown` is already the right name for this channel.
        if (spin == "triplet")
        {
            GlobalV::ofs_running << " WARNING: lr_target_spin=triplet is ignored. This is an"
                                    " open-shell calculation with a single spin-conserving channel (updown), which is"
                                    " what the relaxation will follow."
                                 << std::endl;
        }
        this->target_is_ = 0;
    }
    else
    {
        // Closed shell is the opposite case: singlet and triplet are genuinely different states
        // with different gradients, so `updown` here is ambiguous rather than redundant -- it
        // usually means lr_unrestricted was meant to be set.
        if (spin == "updown")
        {
            ModuleBase::WARNING_QUIT("ESolver_LR",
                                     "lr_target_spin=updown, but this is a closed-shell calculation, where singlet and "
                                     "triplet are separate states with separate gradients. Pick one of them, or set "
                                     "lr_unrestricted to run spin-unrestricted.");
        }
        this->target_is_ = (spin == "triplet") ? 1 : 0;
        if (this->target_is_ >= this->nspin)
        {
            ModuleBase::WARNING_QUIT("ESolver_LR",
                                     "lr_target_spin=triplet requires nspin=2: the triplet channel is not built here.");
        }
    }
    this->force_gs_.create(this->ucell_->nat, 3);
    this->lr_force_.create(this->ucell_->nat, 3);
    GlobalV::ofs_running << " Excited-state relaxation follows state " << this->inp_->lr_target_state << " of the "
                         << (this->openshell ? "updown" : (this->target_is_ == 1 ? "triplet" : "singlet"))
                         << " channel." << std::endl;
}

template <typename T, typename TR>
double ModuleESolver::ESolver_LR<T, TR>::cal_energy()
{
    // Outside a relaxation nothing consumes this, and returning a non-zero value would change
    // what the existing single-point outputs report.
    if (!this->excited_relax_)
    {
        return 0.0;
    }
    return this->etot_gs_ + this->pelec->ekb.c[this->target_ekb_offset_()];
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::cal_force(BaseCell& basecell, ModuleBase::matrix& force)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    const UnitCell& ucell = static_cast<const UnitCell&>(basecell);
    if (!this->excited_relax_)
    { // single-point runs print the gradients of every state from `after_all_runners` instead
        return;
    }
    if (this->lr_force_.nr != ucell.nat)
    {
        ModuleBase::WARNING_QUIT("ESolver_LR::cal_force",
                                 "the excited-state gradient has not been computed for this geometry.");
    }
    // Both halves already follow the ABACUS force convention F = -dE/dR (Ry/Bohr), so they add.
    // The "Gradients of each excited state" heading that `cal_force(int)` prints under is a
    // misnomer: the finite-difference reference it was validated against (`abacus-fd lr-custom`)
    // computes (E(-h) - E(+h))/h, which is -d(Omega)/dR, and the two agree in sign.
    force.create(ucell.nat, 3);
    force = this->force_gs_ + this->lr_force_;
    ModuleIO::print_force(GlobalV::ofs_running, ucell, "EXCITED-STATE TOTAL-FORCE (eV/Angstrom)", force, false);
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::cal_stress(BaseCell& basecell, ModuleBase::matrix& stress)
{
    static_cast<void>(stress);
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    ModuleBase::WARNING_QUIT("ESolver_LR::cal_stress",
                             "the excited-state stress is not implemented (every stress term under module_lr/Grad is a "
                             "dummy passed with isstress=false).");
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::init_pot_groundstate(const Charge& chg_gs)
{
    ModuleBase::TITLE("ESolver_LR", "init_pot_gs");
    std::vector<std::string> pot_register;
    if (this->inp_->vl_in_h)
    {
        if (!this->ks_)
        { // on the `ks-lr` path `sfac()`/`vloc()` alias the ground-state solver's, which
            // `ESolver_FP::before_scf` already refreshed for the current geometry
            //! 11) calculate the structure factor
            this->sfac().setup(&(*this->ucell_), pgrid(), this->pw_rhod);
            this->vloc().init_vloc((*this->ucell_), this->pw_rho);
        }
        pot_register.push_back("local");
    }
    if (this->inp_->vh_in_h)
    {
        pot_register.push_back("hartree");
    }
    pot_register.push_back("xc");

    // initialize the ground state potential
    this->pot_gs = LR_Util::make_unique<elecstate::Potential>(this->pw_rhod,
                                                              this->pw_rho,
                                                              &(*this->ucell_),
                                                              &this->vloc().vloc,
                                                              &this->sfac(),
                                                              &this->solvent,
                                                              &this->etxc_gs,
                                                              &this->vtxc_gs);
    this->pot_gs.get()->pot_register(pot_register);
    XC_Functional::set_xc_type((*this->ucell_).atoms[0].ncpp.xc_func); // set XC type of the ground state
    this->pot_gs->init_pot(&chg_gs);                                   // call update_from_charge inside
    if (LR_Util::has_local_xc(this->xc_kernel))
    {
        XC_Functional::set_xc_type(this->xc_kernel); // recover the excited state xc kernel type
    }
    if (this->inp_->test_force)
    {
        this->pot_gs_hartree = LR_Util::make_unique<elecstate::Potential>(this->pw_rhod,
                                                                          this->pw_rho,
                                                                          &(*this->ucell_),
                                                                          &this->vloc().vloc,
                                                                          &this->sfac(),
                                                                          &this->solvent,
                                                                          &this->etxc_gs,
                                                                          &this->vtxc_gs);
        this->pot_gs_hartree->pot_register({"hartree"});
        this->pot_gs_hartree->init_pot(&chg_gs); // call update_from_charge inside
    }
}

template <typename T, typename TR>
ct::Tensor ModuleESolver::ESolver_LR<T, TR>::pad_X_to_z_(const int ispin, const int istate_begin, const int nst) const
{
    ct::Tensor Xz = LR_Util::newTensor<T>({nst, this->nloc_per_state_z_});
    Xz.zero();
    // Closed shell: one channel, and `ispin` selects the spin COMBINATION (singlet/triplet)
    // whose X is being widened -- `nocc`/`nvirt`/`paraX_` are the same for both.
    // Open shell: one eigenvector holding both channels back to back, so both sub-blocks are
    // widened and the second one is re-based, because widening the first moves where it starts.
    const std::vector<int> chan = this->openshell ? std::vector<int>{0, 1} : std::vector<int>{ispin};
    const int ix = this->openshell ? 0 : ispin; // which entry of `X`
    // channel offsets inside one state's block
    int soff_ch = 0;
    int zoff_ch = 0;
    for (const int is: chan)
    {
        const Parallel_2D& pxs = this->paraX_[is];
        const Parallel_2D& pxz = this->paraX_z_[is];
        for (int ist = 0; ist < nst; ++ist)
        {
            const T* const src = this->X[ix].template data<T>() + (istate_begin + ist) * this->nloc_per_state + soff_ch;
            T* const dst = Xz.data<T>() + ist * this->nloc_per_state_z_ + zoff_ch;
            for (int ik = 0; ik < this->nk; ++ik)
            {
                const int soff = ik * pxs.get_local_size();
                const int zoff = ik * pxz.get_local_size();
                // Both windows are block-cyclic with nb2d = 1 on the same process grid and share
                // the occupied dimension, so a global (virt, occ) element lives on the same
                // process in both -- only its local index differs. The widening is therefore a
                // purely local copy.
                for (int o = 0; o < this->nocc[is]; ++o)
                {
                    for (int v = 0; v < this->nvirt[is]; ++v)
                    {
                        if (!pxs.in_this_processor(v, o))
                        {
                            continue;
                        }
                        dst[zoff + pxz.global2local_col(o) * pxz.get_row_size() + pxz.global2local_row(v)]
                            = src[soff + pxs.global2local_col(o) * pxs.get_row_size() + pxs.global2local_row(v)];
                    }
                }
            }
        }
        soff_ch += this->nk * pxs.get_local_size();
        zoff_ch += this->nk * pxz.get_local_size();
    }
    return Xz;
}

template <typename T, typename TR>
ct::Tensor ModuleESolver::ESolver_LR<T, TR>::solve_zvector_eqation(const int ispin,
                                                                   const int istate_only,
                                                                   const ct::Tensor& Xz)
{
    ModuleBase::TITLE("ESolver_LR", "cal_force");
    ModuleBase::timer::start("ESolver_LR", "solve_zvector_eqation");
    // `Z_vector_equation` treats X and Z as `nstates` independent blocks of `nloc_per_state`,
    // so a single state is just the corresponding block with nstates = 1
    const int nst = (istate_only < 0) ? this->nstates : 1;
    // X arrives already widened into the Z window (`Xz`), which spans every virtual band the
    // ground state produced rather than the `nvirt` window X was solved in -- the Brillouin
    // condition the Z-vector enforces holds in EVERY occupied-virtual rotation. The padded
    // entries are zero, so every X-derived quantity (D^X, T) is unchanged; only the space Z is
    // solved in grows. Both the closed- and the open-shell path go through here.
    ct::Tensor Z = LR_Util::newTensor<T>({nst, this->nloc_per_state_z_});
    // construct and solve the Z-vector equation
    Z_vector_equation(Xz.template data<T>(),
                      Z.template data<T>(),
                      this->xc_kernel,
                      nst,
                      this->nspin,
                      this->nbasis,
                      this->nocc,
                      this->nvirt_z_,
                      (*this->ucell_),
                      orb_cutoff_,
                      this->gd(),
                      *this->psi_ks_z_,
                      this->eig_ks_z_,
#ifdef __EXX
                      std::weak_ptr<Exx_LRI<T>>(this->exx_lri),
                      this->exx_info.info_global.hybrid_alpha,
#endif
                      std::weak_ptr<PotHxcLR>(this->pot[ispin]),
                      std::weak_ptr<PotHxcLR>(this->pot_hxc_gs),
                      this->kv,
                      this->paraX_z_,
                      this->paraC_z_,
                      this->paraMat_,
                      this->spin_types[ispin],
                      this->in_dir,
                      this->out_dir,
                      this->openshell);
    ModuleBase::timer::end("ESolver_LR", "solve_zvector_eqation");
    return Z;
}

template <typename T, typename TR>
std::vector<ModuleBase::matrix> ModuleESolver::ESolver_LR<T, TR>::cal_force(const int ispin, const int istate_only)
{
    if (this->inp_->test_force && ispin == 0)
    {
        this->test_force();
    }
    if (this->openshell)
    {
        return this->cal_force_openshell(istate_only);
    }

    // for each state, calculate dm_trans, dm_relaxed_diff, edm and force
    const int ist_begin = (istate_only < 0) ? 0 : istate_only;
    const int ist_end = (istate_only < 0) ? this->nstates : istate_only + 1;

    // The whole closed-shell gradient runs in the Z window (every virtual band the ground state
    // produced), not the `nvirt` window X was solved in: only there does the Z-vector enforce
    // the Brillouin condition in every occupied-virtual rotation. X is zero-padded into it, so
    // D^X and T come out bit-identical -- and Omega is untouched, so an existing finite-difference
    // reference stays valid. See `fill_z_window_`.
    const ct::Tensor Xz = this->pad_X_to_z_(ispin, ist_begin, ist_end - ist_begin);

    const ct::Tensor& Z = this->solve_zvector_eqation(ispin, istate_only, Xz);
    const auto ground_density = this->cal_dm_gs();
    LR_Force<T> lr_force((*this->ucell_),
                         this->kv.kvec_d,
                         this->paraMat_,
                         *this->pw_rhod,
                         *this->pw_rho,
                         this->vloc(),
                         this->sfac(),
                         this->gd(),
                         this->tcb()
#ifdef __EXX
                             ,
                         std::weak_ptr<Exx_LRI<T>>(this->exx_lri),
                         this->exx_info.info_global.hybrid_alpha
#endif
    );
    const double ground_channel_factor = this->nspin == 1 ? 0.5 : 1.0;
#ifdef __EXX
    const std::string ground_functional = LR_Util::tolower(this->inp_->dft_functional);
    const bool ground_is_hybrid = LR::exx_kernel_list().count(ground_functional) > 0;
#endif
    const LR::GradientContext<T> context{
        this->nk,
        this->nspin,
        this->nbasis,
        this->nstates,
        this->nloc_per_state_z_,
        this->nocc,
        this->nvirt_z_,
        this->paraX_z_,
        this->paraC_z_,
        this->paraMat_,
        *this->ucell_,
        this->kv,
        this->gd(),
        this->orb_cutoff_,
        *this->psi_ks_z_,
        this->eig_ks_z_,
        this->pelec->ekb,
        this->pot,
        this->pot_hxc_gs,
        this->xc_kernel,
        this->spin_types,
        this->inp_->test_force,
        this->excited_relax_,
        GlobalV::ofs_running,
        ground_channel_factor,
#ifdef __EXX
        this->exx_lri,
        this->exx_info.info_global.hybrid_alpha,
        ground_is_hybrid,
#endif
    };
    return LR::assemble_closed_shell_forces(context, lr_force, ground_density, Xz, Z, ist_begin, ist_end, ispin);
}

template <typename T, typename TR>
std::vector<ModuleBase::matrix> ModuleESolver::ESolver_LR<T, TR>::cal_force_openshell(const int istate_only)
{

    // Open shell: there is a single eigenproblem whose vector is the concatenation
    // [up-block | down-block], and every density matrix has two independent channels.
    // The spin-orbital formulas apply verbatim -- unlike the closed-shell singlet/triplet
    // algorithm, X here is normalized over BOTH channels, so it carries no implicit sqrt(2)
    // and none of the collapsed 2/4 factors are needed.
    const int ist_begin_ = (istate_only < 0) ? 0 : istate_only;
    const int ist_end_ = (istate_only < 0) ? this->nstates : istate_only + 1;
    // Like the closed-shell path, the whole gradient runs in the Z window: X is zero-padded
    // into it (both channels, each re-based -- see `pad_X_to_z_`), so D^X and T are unchanged
    // while the Z-vector gets every occupied-virtual rotation the AO basis supports.
    const ct::Tensor Xz = this->pad_X_to_z_(0, ist_begin_, ist_end_ - ist_begin_);

    const ct::Tensor& Z = this->solve_zvector_eqation(0, istate_only, Xz);
    const auto ground_density = this->cal_dm_gs();
    LR_Force<T> lr_force((*this->ucell_),
                         this->kv.kvec_d,
                         this->paraMat_,
                         *this->pw_rhod,
                         *this->pw_rho,
                         this->vloc(),
                         this->sfac(),
                         this->gd(),
                         this->tcb()
#ifdef __EXX
                             ,
                         std::weak_ptr<Exx_LRI<T>>(this->exx_lri),
                         this->exx_info.info_global.hybrid_alpha
#endif
    );
    const double ground_channel_factor = this->nspin == 1 ? 0.5 : 1.0;
#ifdef __EXX
    const std::string ground_functional = LR_Util::tolower(this->inp_->dft_functional);
    const bool ground_is_hybrid = LR::exx_kernel_list().count(ground_functional) > 0;
#endif
    const LR::GradientContext<T> context{
        this->nk,
        this->nspin,
        this->nbasis,
        this->nstates,
        this->nloc_per_state_z_,
        this->nocc,
        this->nvirt_z_,
        this->paraX_z_,
        this->paraC_z_,
        this->paraMat_,
        *this->ucell_,
        this->kv,
        this->gd(),
        this->orb_cutoff_,
        *this->psi_ks_z_,
        this->eig_ks_z_,
        this->pelec->ekb,
        this->pot,
        this->pot_hxc_gs,
        this->xc_kernel,
        this->spin_types,
        this->inp_->test_force,
        this->excited_relax_,
        GlobalV::ofs_running,
        ground_channel_factor,
#ifdef __EXX
        this->exx_lri,
        this->exx_info.info_global.hybrid_alpha,
        ground_is_hybrid,
#endif
    };
    return LR::assemble_open_shell_forces(context, lr_force, ground_density, Xz, Z, ist_begin_, ist_end_);
}

template <typename T, typename TR>
elecstate::DensityMatrix<T, double> ModuleESolver::ESolver_LR<T, TR>::cal_dm_gs()
{
    elecstate::DensityMatrix<T, double> dm_gs(&this->paraMat_, this->nspin, this->kv.kvec_d, this->nk);
    // The ks-lr path aliases the KS solver's wave functions, distributed by its own descriptor.
    const Parallel_Orbitals* const pv_all = this->ks_ ? &this->ks_->pv : &this->paraMat_all_;
    elecstate::cal_dm_psi(pv_all, this->wg_ks_all, *this->psi_ks_all_, dm_gs); // nbands is important here
    LR_Util::initialize_DMR(dm_gs,
                            this->paraMat_,
                            (*this->ucell_),
                            this->gd(),
                            this->orb_cutoff_); // nbands is not important here
    dm_gs.cal_DMR();
    return dm_gs;
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::test_force()
{
    LR_Force<T> lr_force((*this->ucell_),
                         this->kv.kvec_d,
                         this->paraMat_,
                         *this->pw_rhod,
                         *this->pw_rho,
                         this->vloc(),
                         this->sfac(),
                         this->gd(),
                         this->tcb()
#ifdef __EXX
                             ,
                         std::weak_ptr<Exx_LRI<T>>(this->exx_lri),
                         this->exx_info.info_global.hybrid_alpha
#endif
    );

    const elecstate::DensityMatrix<T, double>& dm_gs = this->cal_dm_gs();
    // LR_Util::print_DMR(dm_gs, "DM(R) of ground state");
    ///========================== test 1: reproduce the force of ground state =========================
    // energy density matrix of the ground state
    elecstate::DensityMatrix<T, double> edm_gs(&this->paraMat_, this->nspin, this->kv.kvec_d, this->nk); // DX
    ModuleBase::matrix wg_ekb_ks_all(nspin, this->inp_->nbands);
    std::transform(this->wg_ks_all.c,
                   this->wg_ks_all.c + nspin * this->inp_->nbands,
                   this->eig_ks_all.c,
                   wg_ekb_ks_all.c,
                   std::multiplies<double>());
    const Parallel_Orbitals* const pv_all = this->ks_ ? &this->ks_->pv : &this->paraMat_all_;
    elecstate::cal_dm_psi(pv_all, wg_ekb_ks_all, *this->psi_ks_all_, edm_gs);
    LR_Util::initialize_DMR(edm_gs, this->paraMat_, (*this->ucell_), this->gd(), this->orb_cutoff_);
    edm_gs.cal_DMR();
    // ground-state force
    ModuleBase::matrix force_gs = lr_force.reproduce_force_gs(kv, dm_gs, edm_gs);
    ModuleIO::print_force(GlobalV::ofs_running, (*this->ucell_), "Ground State FORCE (eV/Angstrom)", force_gs, false);
    /// ======================================= END test 1 =========================================
    ///========================== test 2: reproduce the DX Hartree term =========================
    ModuleBase::matrix f_hxc_potgs = lr_force.reproduce_force_gs_loc(dm_gs, *this->pot_gs_hartree);
    ModuleIO::print_force(GlobalV::ofs_running,
                          (*this->ucell_),
                          "GS Hartree force calculated by 'cal_pulay_fs' from potential (eV/Angstrom)",
                          f_hxc_potgs,
                          false);
    ModuleBase::matrix f_hxc_potlr = lr_force.cal_force_hxc_dmtrans(dm_gs, *this->pot[0]);
    ModuleIO::print_force(GlobalV::ofs_running,
                          (*this->ucell_),
                          "GS Hxc force calculated by 'LR_Force' from kernel (eV/Angstrom)",
                          f_hxc_potlr,
                          false);
    // `cal_force_hxc_dmtrans` now includes the Pulay -> Pulay+Hellmann-Feynman factor 2 itself,
    // so this must match the ground-state Hartree force directly (dm_gs is already symmetric).
    /// ======================================= END test 2 =========================================
    ///========================== test 3: H2 SZ 4-center gradients =========================
    if (this->nbasis == 2 && (*this->ucell_).nat == 2)
    {
        // lr_force.cal_H2_sz_center2_deriv(orb_cutoff_, kv);  // for gradient
        // lr_force.cal_H2_sz_center4(orb_cutoff_, kv, /*is_grad=*/false);  // for 4-center integrals
        // lr_force.cal_H2_sz_center4(orb_cutoff_, kv, /*is_grad=*/true);   // for gradient
        // exit(0);
    }
}

template class ModuleESolver::ESolver_LR<double, double>;
template class ModuleESolver::ESolver_LR<std::complex<double>, double>;
