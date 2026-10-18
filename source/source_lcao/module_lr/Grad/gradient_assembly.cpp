#include "gradient_assembly.h"

#include "source_io/module_output/output_log.h"
#include "source_lcao/module_lr/Grad/multipliers/cal_edm_from_multipliers.h"
namespace LR
{
namespace
{
template <typename Tstream>
inline void print_force(const std::vector<ModuleBase::matrix>& force, Tstream& ofs, const int istate_begin = 0)
{
    const int nstate = force.size();
    ofs << "Forces (-gradients) of each excited state: (eV/Angstrom)" << std::endl;
    ofs << std::setprecision(6) << std::setw(6) << "state" << std::setw(6) << "atom" << std::setw(15) << "x"
        << std::setw(15) << "y" << std::setw(15) << "z" << std::endl;
    const double fac = ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A;
    for (int i = 0; i < nstate; ++i)
    {
        for (int iat = 0; iat < force[i].nr; ++iat)
        {
            std::string istate = iat == 0 ? std::to_string(istate_begin + i) : " ";
            ofs << std::setw(6) << istate << std::setw(6) << iat << std::setw(6) << "force";
            for (int ixyz = 0; ixyz < 3; ++ixyz)
            {
                ofs << std::setw(15) << force[i](iat, ixyz) * fac;
            }
            ofs << std::endl;
        }
    }
}

// check C_uaC_va-C_uiC_vi of lumo-homo, nocc=1,  nk=1
template <typename T>
inline void test_dm_diff_H2(const T* dm, const psi::Psi<T>& c, const int nbasis)
{
    std::cout << "difference dm cal: " << std::endl;
    LR_Util::print_value(dm, nbasis, nbasis);
    std::cout << "difference dm ref: " << std::endl;
    for (int i = 0; i < nbasis; ++i)
    {
        for (int j = 0; j < nbasis; ++j)
        {
            std::cout << c(0, 1, i) * c(0, 1, j) - c(0, 0, i) * c(0, 0, j) << " ";
        }
        std::cout << std::endl;
    }
}

// check e_aC_uaC_va-e_iC_uiC_vi of lumo-homo, nocc=1,  nk=1
template <typename T>
inline void test_edm_H2(const T* const edm, const double* const eig_ks, const psi::Psi<T>& c, const int nbasis)
{
    std::cout << "edm cal: " << std::endl;
    LR_Util::print_value(edm, nbasis, nbasis);
    std::cout << "edm ref: " << std::endl;
    for (int i = 0; i < nbasis; ++i)
    {
        for (int j = 0; j < nbasis; ++j)
        {
            std::cout << eig_ks[1] * c(0, 1, i) * c(0, 1, j) - eig_ks[0] * c(0, 0, i) * c(0, 0, j) << " ";
        }
        std::cout << std::endl;
    }
}

} // namespace
template <typename T>
std::vector<ModuleBase::matrix> assemble_closed_shell_forces(const GradientContext<T>& context,
                                                             LR_Force<T>& lr_force,
                                                             const elecstate::DensityMatrix<T, double>& ground_density,
                                                             const ct::Tensor& Xz,
                                                             const ct::Tensor& Z,
                                                             const int ist_begin,
                                                             const int ist_end,
                                                             const int ispin)
{
    ModuleBase::TITLE("LR_Gradient", "assemble_closed_shell_forces");
    ModuleBase::timer::start("LR_Gradient", "assemble_closed_shell_forces");
    const auto& nvirt_g = context.nvirt;
    const auto& paraX_g = context.px;
    const int nloc_g = context.local_state_size;
    const auto c = LR_Util::get_psi_spin(context.orbitals, 0, context.nk);
    context.log << "Start to calculate excited-state force of " << context.spin_types[ispin] << std::endl;
    // ground state dm for currrent spin (only for test the correctness of the force)
    // elecstate::DensityMatrix<T, T> dm_gs(context.pmat, 1, context.kv.kvec_d, context.nk);

    std::vector<ModuleBase::matrix> forces(ist_end - ist_begin);
    for (int istate = ist_begin; istate < ist_end; ++istate)
    {
        const int offset = (istate - ist_begin) * nloc_g; // block of X (widened into the Z window)
        const int zoffset = offset;                       // block of Z
        // The imag part will be cancelled in the force calculation, so we use double DM(R) to calculate force.
        // But complex transition DM(R) is still used in energy density matrix calculation.
        const auto& dm_trans_k = cal_dm_trans_pblas(Xz.data<T>() + offset,
                                                    paraX_g[ispin],
                                                    c,
                                                    context.pc,
                                                    context.nbasis,
                                                    context.nocc[ispin],
                                                    nvirt_g[ispin],
                                                    context.pmat);
        // D(X) complex, for the EXX (LibRI) force. Built FIRST and left UN-symmetrized:
        // the exchange kernel (mu kappa | nu lambda) puts the two indices of one D^X into
        // different electron coordinates, so Tr[D^X D^X K_exx] = (aa|ii) requires the full
        // non-symmetric D^X. Symmetrizing would give 1/2[(aa|ii)+(ai|ia)], which is wrong.
        // (`cal_force_exx_dm_trans` feeds the same tensor to both slots, so it is consistent.)
        auto dm_trans = // D(X) complex
            LR_Util::build_dm_from_dmk<T, T>(dm_trans_k,
                                             context.pmat,
                                             context.nk,
                                             context.kv.kvec_d,
                                             context.ucell,
                                             context.grid,
                                             context.orbital_cutoffs);
        LR_Util::transpose_DMR(dm_trans, context.ucell.nat);
        // D(X) real, for the grid Hxc force. The Coulomb kernel (mu nu | kappa lambda) is
        // symmetric within each index pair, so it only ever sees the symmetric part of D^X.
        // In `PulayForceStress::cal_pulay_fs`, `cal_gint_rho` (which builds v) symmetrizes
        // implicitly, while `cal_gint_fvl`'s internal factor 2 assumes D_{mu nu} = D_{nu mu}.
        // Passing an un-symmetrized D^X makes the two slots of the bilinear form disagree.
        // NOTE: `build_dm_from_dmk` symmetrizes `dm_trans_k` IN PLACE, hence the ordering.
        auto dm_trans_real = // D(X), double (FIXME: not enough for periodic system!)
            LR_Util::build_dm_from_dmk<T, double>(dm_trans_k,
                                                  context.pmat,
                                                  context.nk,
                                                  context.kv.kvec_d,
                                                  context.ucell,
                                                  context.grid,
                                                  context.orbital_cutoffs,
                                                  /*symmetrize=*/true);
        LR_Util::transpose_DMR(dm_trans_real, context.ucell.nat);
        // LR_Util::print_DMR(dm_trans, "dm_trans of istate " + std::to_string(istate));
        // difference density matrix
        std::vector<ct::Tensor> dm_diff_k = cal_dm_diff_pblas(Xz.data<T>() + offset,
                                                              paraX_g[ispin],
                                                              c,
                                                              context.pc,
                                                              context.nbasis,
                                                              context.nocc[ispin],
                                                              nvirt_g[ispin],
                                                              context.pmat);
        std::cout << "dm_diff_k T(k) before symmetrization, istate " + std::to_string(istate) << std::endl;
        LR_Util::print_value(dm_diff_k[0].data<T>(), context.pmat.get_col_size(), context.pmat.get_row_size());
        // for (auto& d : dm_diff_k) { LR_Util::matsym(d.data<T>(), context.nbasis, context.pmat); }   // symmetrize
        // std::cout << "dm_diff_k T(k) after symmetrization, istate " + std::to_string(istate) << std::endl;
        // LR_Util::print_value(dm_diff_k[0].data<T>(), context.pmat.get_col_size(), context.pmat.get_row_size());

        const std::vector<ct::Tensor>& dm_relaxed_k = cal_dm_trans_pblas(Z.template data<T>() + zoffset,
                                                                         paraX_g[ispin],
                                                                         c,
                                                                         context.pc,
                                                                         context.nbasis,
                                                                         context.nocc[ispin],
                                                                         nvirt_g[ispin],
                                                                         context.pmat);
        std::cout << "dm_relaxed_k Z(k) before symmetrization, istate " + std::to_string(istate) << std::endl;
        LR_Util::print_value(dm_relaxed_k[0].data<T>(), context.pmat.get_col_size(), context.pmat.get_row_size());
        for (auto& d: dm_relaxed_k)
        {
            LR_Util::matsym(d.data<T>(), context.nbasis, context.pmat);
        } // symmetrize
        std::cout << "dm_relaxed_k Z(k) after symmetrization, istate " + std::to_string(istate) << std::endl;
        LR_Util::print_value(dm_relaxed_k[0].data<T>(), context.pmat.get_col_size(), context.pmat.get_row_size());
        // relaxed difference density matrix
        const std::vector<ct::Tensor>& relaxed_diff_dm_k = dm_diff_k + dm_relaxed_k;
        const elecstate::DensityMatrix<T, T>& diff_dm = LR_Util::build_dm_from_dmk<T, T>(dm_diff_k,
                                                                                         context.pmat,
                                                                                         context.nk,
                                                                                         context.kv.kvec_d,
                                                                                         context.ucell,
                                                                                         context.grid,
                                                                                         context.orbital_cutoffs);
        const elecstate::DensityMatrix<T, T>& relaxed_diff_dm
            = LR_Util::build_dm_from_dmk<T, T>(relaxed_diff_dm_k,
                                               context.pmat,
                                               context.nk,
                                               context.kv.kvec_d,
                                               context.ucell,
                                               context.grid,
                                               context.orbital_cutoffs);
        // LR_Util::print_DMR(relaxed_diff_dm, "relaxed_diff_dm T+Z (Z symmetrized) of istate " +
        // std::to_string(istate));

        // elecstate::DensityMatrix<T, T> relaxed_diff_dm =    // T+D(Z), (R) can be complex
        //     LR_Util::build_dm_from_dmk<T, T>(
        //         // LR_Util::operator+(
        //             cal_dm_diff_pb las(Xz.data<T>() + offset, paraX_g[ispin], c, context.pc, context.nbasis,
        //             context.nocc[ispin], nvirt_g[ispin], context.pmat)
        //             + cal_dm_trans_pblas(Z.template data<T>() + offset, paraX_g[ispin], c, context.pc,
        //             context.nbasis, context.nocc[ispin], nvirt_g[ispin], context.pmat)
        //             ,// ),
        //         context.pmat, context.nk, context.kv.kvec_d, context.ucell, context.grid, context.orbital_cutoffs);
        // LR_Util::print_DMR(relaxed_diff_dm, "relaxed_diff_dm of istate " + std::to_string(istate));
        elecstate::DensityMatrix<T, double> relaxed_diff_dm_real(&context.pmat, 1, context.kv.kvec_d, context.nk);
        LR_Util::initialize_DMR(relaxed_diff_dm_real,
                                context.pmat,
                                context.ucell,
                                context.grid,
                                context.orbital_cutoffs);
        LR_Util::get_DMR_real_imag_part(relaxed_diff_dm, relaxed_diff_dm_real, 'R');

        // get edm of type DensityMatrix
        // weak_ptr here is to avoid "could not match 'weak_ptr' against 'shared_ptr'"
        // but why there're no bug in the previous code (HamiltLR and HamiltULF)?
        // seems because those two are classes having constructors
        // but `cal_edm_from_XZ_istate` here is a functions
        std::weak_ptr<PotHxcLR> pot_weak = context.potentials[ispin];
        std::weak_ptr<PotHxcLR> pot_hxc_gs_weak = context.ground_potential;
#ifdef __EXX
        std::weak_ptr<Exx_LRI<T>> exx_lri_weak = context.exx_lri;
#endif
        const std::vector<ct::Tensor>& edm_k
            = cal_edm_from_XZ_istate(Xz.data<T>() + offset,
                                     Z.template data<T>() + zoffset,
                                     context.excitation_energies.c[ispin * context.nstates + istate],
                                     // pack the following as a struct or use parameter package
                                     context.ks_energies.c,
                                     dm_trans,
                                     c,
                                     context.nspin,
                                     context.nbasis,
                                     context.nocc,
                                     nvirt_g,
                                     context.ucell,
                                     context.orbital_cutoffs,
#ifdef __EXX
                                     exx_lri_weak,
                                     context.hybrid_alpha,
#endif
                                     pot_weak,
                                     pot_hxc_gs_weak,
                                     context.kv,
                                     context.grid,
                                     paraX_g,
                                     context.pc,
                                     context.pmat,
                                     context.xc_kernel,
                                     context.spin_types[ispin]);
        if (context.test_force && context.nocc[0] == 1 && nvirt_g[0] == 1)
        {
            const std::vector<ct::Tensor>& dm_diff = cal_dm_diff_pblas(Xz.data<T>() + offset,
                                                                       paraX_g[0],
                                                                       c,
                                                                       context.pc,
                                                                       context.nbasis,
                                                                       context.nocc[0],
                                                                       nvirt_g[0],
                                                                       context.pmat);
            // test_dm_diff_H2<T>(relaxed_diff_dm.get_DMK_pointer(0), c, context.nbasis);
            test_dm_diff_H2<T>(dm_diff[0].data<T>(), c, context.nbasis);
            test_edm_H2<T>(edm_k[0].data<T>(), context.ks_energies.c, c, context.nbasis);
        }
        elecstate::DensityMatrix<T, double> edm_real = LR_Util::build_dm_from_dmk<T, double>(edm_k,
                                                                                             context.pmat,
                                                                                             context.nk,
                                                                                             context.kv.kvec_d,
                                                                                             context.ucell,
                                                                                             context.grid,
                                                                                             context.orbital_cutoffs,
                                                                                             /*symmetrize=*/true);
        // print edm_real (R)
        if (context.test_force)
        {
            LR_Util::save_DMR(edm_real,
                              "data-EDMR-sparse"
                                  + std::string(context.excited_relax ? "_state" + std::to_string(istate) : ""),
                              context.pmat);
            // LR_Util::print_DMR(edm_real, "edm_real (R) of istate " + std::to_string(istate));
        }

        ModuleBase::matrix force_hxc_dmtrans
            = lr_force.cal_force_hxc_dmtrans(dm_trans_real, *context.potentials[ispin]);
        if (context.test_force)
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "HXC DMTRANS FORCE (eV/Angstrom)",
                                  force_hxc_dmtrans,
                                  false);

        const elecstate::DensityMatrix<T, double>& dm_gs = ground_density;

        // the $g^{xc}$ half of $\partial_x K[D^X]D^X$, i.e. the derivative of the xc kernel through
        // the ground-state density (see `cal_force_gxc_dmtrans`). Only for local kernels.
        if (LR_Util::has_local_xc(context.xc_kernel))
        {
            PotGradXCLR pot_grad(context.ground_potential->xc_kernel_components(),
                                 context.ground_potential->get_rho_basis(),
                                 context.ucell,
                                 context.ground_potential->nrxx,
                                 context.ground_potential->nspin,
                                 context.spin_types[ispin] == "triplet");
            ModuleBase::matrix force_gxc_dmtrans = lr_force.cal_force_gxc_dmtrans(dm_trans_real, dm_gs, pot_grad);
            if (context.test_force)
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "GXC DMTRANS FORCE (eV/Angstrom)",
                                      force_gxc_dmtrans,
                                      false);
            force_hxc_dmtrans += force_gxc_dmtrans;
        }
        ModuleBase::matrix force_hamiltgs_relaxed_diff
            = lr_force.cal_force_hamilt_gs_dm_relaxed_diff(relaxed_diff_dm_real,
                                                           dm_gs,
                                                           false,
                                                           context.ground_potential.get());
        if (context.test_force)
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "H_GS-(T+Z) FORCE (without EXX) (eV/Angstrom)",
                                  force_hamiltgs_relaxed_diff,
                                  false);

        ModuleBase::matrix force_overlap_edm
            = lr_force.cal_force_overlap_edm(edm_real); // "-" sign has been included in the force factor
        if (context.test_force)
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "OVERLAP-EDM FORCE (eV/Angstrom)",
                                  force_overlap_edm,
                                  false);

        if (context.test_force)
        {
            // test H[T] force (Z=0), non-EXX part
            elecstate::DensityMatrix<T, double> diff_dm_real(&context.pmat, 1, context.kv.kvec_d, context.nk);
            LR_Util::initialize_DMR(diff_dm_real, context.pmat, context.ucell, context.grid, context.orbital_cutoffs);
            LR_Util::get_DMR_real_imag_part(diff_dm, diff_dm_real, 'R');

            context.log << "========== [TEST H_GS-(T) force (Z=0), non-EXX part] ===========" << std::endl;
            ModuleBase::matrix force_hamiltgs_diff = lr_force.cal_force_hamilt_gs_dm_relaxed_diff(diff_dm_real, dm_gs);
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "H_GS-T FORCE (without EXX) (eV/Angstrom)",
                                  force_hamiltgs_diff,
                                  false);
            context.log << "========== [\\TEST H_GS-(T) force (Z=0), non-EXX part] ===========" << std::endl;
        }

#ifdef __EXX
        const double& alpha = context.hybrid_alpha;

        if (LR::exx_kernel_list().count(context.xc_kernel))
        {
            const auto& Ds_trans = LR_Util::get_exx_Ds_spin1(dm_trans, context.ucell, context.kv, context.pmat);
            ModuleBase::matrix force_exx_dmtrans
                = lr_force.cal_force_exx_dm_trans(Ds_trans, alpha * 4.0); // cancel the two 0.5s in Ds
            if (context.test_force)
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "EXX DMTRANS FORCE (eV/Angstrom)",
                                      force_exx_dmtrans,
                                      false);
            force_hxc_dmtrans += force_exx_dmtrans;
        }

        if (context.ground_is_hybrid)
        {
            const auto& Ds_gs
                = LR_Util::get_exx_Ds_spin1(dm_gs, context.ucell, context.kv, context.pmat); // returns 0.5*D[0]
            const auto& Ds_relaxed_diff = LR_Util::get_exx_Ds_spin1(relaxed_diff_dm,
                                                                    context.ucell,
                                                                    context.kv,
                                                                    context.pmat); // returns 0.5*D[0]
            // LR_Util::print_CV(Ds_relaxed_diff, "Ds_relaxed_diff for EXX force");
            // `get_exx_Ds_spin1` feeds `split_m2D_ktoR(..., nspin=1)`, which reads only channel 0
            // with a 0.5 prefactor. For `dm_gs` that channel is $D^\text{gs}_\uparrow$ at nspin=2
            // but the spin-summed $D^\text{gs}$ at nspin=1, i.e. twice as large.
            ModuleBase::matrix force_exx_gs_relaxed_diff
                = lr_force.cal_force_exx_gs_dm_relaxed_diff(Ds_gs, Ds_relaxed_diff, alpha * 4.0)
                  * context.ground_channel_factor; // cancel the two 0.5s in Ds
            if (context.test_force)
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "EXX GS-(T+Z) FORCE (eV/Angstrom)",
                                      force_exx_gs_relaxed_diff,
                                      false);
            force_hamiltgs_relaxed_diff += force_exx_gs_relaxed_diff;

            if (context.test_force)
            {
                // test H[T] force (Z=0), EXX part
                const auto& Ds_diff
                    = LR_Util::get_exx_Ds_spin1(diff_dm, context.ucell, context.kv, context.pmat); // returns 0.5*D[0]
                context.log << "========== [TEST H_GS-(T) force (Z=0), EXX part] ===========" << std::endl;
                ModuleBase::matrix force_exx_gs_diff
                    = lr_force.cal_force_exx_gs_dm_relaxed_diff(Ds_gs, Ds_diff, alpha * 4.0)
                      * context.ground_channel_factor; // cancel the two 0.5s in Ds
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "H_GS-T EXX FORCE (Z=0) (eV/Angstrom)",
                                      force_exx_gs_diff,
                                      false);
                context.log << "========== [\\TEST H_GS-(T) force (Z=0), EXX part] ===========" << std::endl;
            }
        }
#endif
        forces[istate - ist_begin] = force_hxc_dmtrans + force_hamiltgs_relaxed_diff + force_overlap_edm;
    }
    ModuleBase::timer::end("LR_Gradient", "assemble_closed_shell_forces");
    // total force
    print_force(forces, std::cout, ist_begin);
    print_force(forces, context.log, ist_begin);
    return forces;
}
template <typename T>
std::vector<ModuleBase::matrix> assemble_open_shell_forces(const GradientContext<T>& context,
                                                           LR_Force<T>& lr_force,
                                                           const elecstate::DensityMatrix<T, double>& ground_density,
                                                           const ct::Tensor& Xz,
                                                           const ct::Tensor& Z,
                                                           const int ist_begin,
                                                           const int ist_end)
{
    ModuleBase::TITLE("LR_Gradient", "assemble_open_shell_forces");
    ModuleBase::timer::start("LR_Gradient", "assemble_open_shell_forces");
    const auto& nvirt_g = context.nvirt;
    const auto& paraX_g = context.px;
    const int nloc_g = context.local_state_size;
    const std::vector<int> ld_x = {static_cast<int>(context.nk * paraX_g[0].get_local_size()),
                                   static_cast<int>(context.nk * paraX_g[1].get_local_size())};
    const std::vector<int> off_x = {0, ld_x[0]};
    std::vector<psi::Psi<T>> c_spin;
    for (int is: {0, 1})
    {
        c_spin.push_back(LR_Util::get_psi_spin(context.orbitals, is, context.nk));
    }

    context.log << "Start to calculate excited-state force of updown (open shell)" << std::endl;

    std::vector<ModuleBase::matrix> forces(ist_end - ist_begin);
    for (int istate = ist_begin; istate < ist_end; ++istate)
    {
        const int offset = (istate - ist_begin) * nloc_g; // X widened into the Z window
        const T* const X_istate = Xz.data<T>() + offset;
        const T* const Z_istate = Z.template data<T>() + offset;

        // 1. the k-space blocks of each spin channel
        std::vector<std::vector<ct::Tensor>> dmx_k(2), dmdiff_k(2), relaxed_k(2);
        for (int is: {0, 1})
        {
            dmx_k[is] = cal_dm_trans_pblas(X_istate + off_x[is],
                                           paraX_g[is],
                                           c_spin[is],
                                           context.pc,
                                           context.nbasis,
                                           context.nocc[is],
                                           nvirt_g[is],
                                           context.pmat);
            dmdiff_k[is] = cal_dm_diff_pblas(X_istate + off_x[is],
                                             paraX_g[is],
                                             c_spin[is],
                                             context.pc,
                                             context.nbasis,
                                             context.nocc[is],
                                             nvirt_g[is],
                                             context.pmat);
            std::vector<ct::Tensor> dmz_k = cal_dm_trans_pblas(Z_istate + off_x[is],
                                                               paraX_g[is],
                                                               c_spin[is],
                                                               context.pc,
                                                               context.nbasis,
                                                               context.nocc[is],
                                                               nvirt_g[is],
                                                               context.pmat);
            for (auto& d: dmz_k)
            {
                LR_Util::matsym(d.template data<T>(), context.nbasis, context.pmat);
            }
            relaxed_k[is] = dmdiff_k[is] + dmz_k;
        }

        // 2. $D^X$. Complex and UN-symmetrized first (the EXX kernel needs the full
        //    non-symmetric $D^X$), then the real symmetrized copy for the grid Hxc force --
        //    `build_dm_from_dmk_spin` symmetrizes IN PLACE, hence the ordering.
        auto dm_trans = LR_Util::build_dm_from_dmk_spin<T, T>(dmx_k,
                                                              context.pmat,
                                                              context.nk,
                                                              context.kv.kvec_d,
                                                              context.ucell,
                                                              context.grid,
                                                              context.orbital_cutoffs);
        LR_Util::transpose_DMR(dm_trans, context.ucell.nat);
        auto dm_trans_real = LR_Util::build_dm_from_dmk_spin<T, double>(dmx_k,
                                                                        context.pmat,
                                                                        context.nk,
                                                                        context.kv.kvec_d,
                                                                        context.ucell,
                                                                        context.grid,
                                                                        context.orbital_cutoffs,
                                                                        /*symmetrize=*/true);
        LR_Util::transpose_DMR(dm_trans_real, context.ucell.nat);

        // 3. the relaxed difference density matrix $T+D^Z$
        const elecstate::DensityMatrix<T, T>& relaxed_diff_dm
            = LR_Util::build_dm_from_dmk_spin<T, T>(relaxed_k,
                                                    context.pmat,
                                                    context.nk,
                                                    context.kv.kvec_d,
                                                    context.ucell,
                                                    context.grid,
                                                    context.orbital_cutoffs);
        elecstate::DensityMatrix<T, double> relaxed_diff_dm_real(&context.pmat, 2, context.kv.kvec_d, context.nk);
        LR_Util::initialize_DMR(relaxed_diff_dm_real,
                                context.pmat,
                                context.ucell,
                                context.grid,
                                context.orbital_cutoffs);
        LR_Util::get_DMR_real_imag_part(relaxed_diff_dm, relaxed_diff_dm_real, 'R');

        // 4. the energy-weighted density matrix
        std::weak_ptr<PotHxcLR> pot_weak = context.potentials[0];
        std::weak_ptr<PotHxcLR> pot_hxc_gs_weak = context.ground_potential;
#ifdef __EXX
        std::weak_ptr<Exx_LRI<T>> exx_lri_weak = context.exx_lri;
#endif
        const std::vector<std::vector<ct::Tensor>>& edm_k
            = cal_edm_from_XZ_istate_openshell(X_istate,
                                               Z_istate,
                                               context.excitation_energies.c[istate],
                                               context.ks_energies.c,
                                               dm_trans,
                                               context.orbitals,
                                               context.nspin,
                                               context.nbasis,
                                               context.nocc,
                                               nvirt_g,
                                               context.ucell,
                                               context.orbital_cutoffs,
#ifdef __EXX
                                               exx_lri_weak,
                                               context.hybrid_alpha,
#endif
                                               pot_weak,
                                               pot_hxc_gs_weak,
                                               context.kv,
                                               context.grid,
                                               paraX_g,
                                               context.pc,
                                               context.pmat,
                                               context.xc_kernel);
        elecstate::DensityMatrix<T, double> edm_real
            = LR_Util::build_dm_from_dmk_spin<T, double>(edm_k,
                                                         context.pmat,
                                                         context.nk,
                                                         context.kv.kvec_d,
                                                         context.ucell,
                                                         context.grid,
                                                         context.orbital_cutoffs,
                                                         /*symmetrize=*/true);

        // 5. the force terms
        ModuleBase::matrix force_hxc_dmtrans = lr_force.cal_force_hxc_dmtrans(dm_trans_real, *context.potentials[0]);
        if (context.test_force)
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "HXC DMTRANS FORCE (eV/Angstrom)",
                                  force_hxc_dmtrans,
                                  false);

        const elecstate::DensityMatrix<T, double>& dm_gs = ground_density;

        // the $g^{xc}$ half of $\partial_x K[D^X]D^X$, i.e. the derivative of the xc kernel
        // through the ground-state density. Only for local kernels.
        if (LR_Util::has_local_xc(context.xc_kernel))
        {
            PotGradXCLR pot_grad(context.ground_potential->xc_kernel_components(),
                                 context.ground_potential->get_rho_basis(),
                                 context.ucell,
                                 context.ground_potential->nrxx,
                                 context.ground_potential->nspin,
                                 /*triplet=*/false);
            ModuleBase::matrix force_gxc_dmtrans
                = lr_force.cal_force_gxc_dmtrans_openshell(dm_trans_real, dm_gs, pot_grad);
            if (context.test_force)
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "GXC DMTRANS FORCE (eV/Angstrom)",
                                      force_gxc_dmtrans,
                                      false);
            force_hxc_dmtrans += force_gxc_dmtrans;
        }

        ModuleBase::matrix force_hamiltgs_relaxed_diff
            = lr_force.cal_force_hamilt_gs_dm_relaxed_diff(relaxed_diff_dm_real,
                                                           dm_gs,
                                                           false,
                                                           context.ground_potential.get());
        if (context.test_force)
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "H_GS-(T+Z) FORCE (without EXX) (eV/Angstrom)",
                                  force_hamiltgs_relaxed_diff,
                                  false);

        ModuleBase::matrix force_overlap_edm = lr_force.cal_force_overlap_edm(edm_real);
        if (context.test_force)
            ModuleIO::print_force(context.log,
                                  context.ucell,
                                  "OVERLAP-EDM FORCE (eV/Angstrom)",
                                  force_overlap_edm,
                                  false);

#ifdef __EXX
        const double& alpha = context.hybrid_alpha;
        // Exchange is spin-diagonal, so each channel is done independently and summed.
        // `get_exx_Ds_gs` returns the channels unscaled (SPIN_multiple = 1 at nspin=2), unlike
        // the closed-shell `get_exx_Ds_spin1` which returns 0.5*D. Counting the closed-shell
        // `alpha*4.0` back down for both terms:
        //   $D^XD^X$: each slot goes 0.5*D^X_tot -> D^X_is, i.e. x2 each, and the explicit
        //     sum over is adds another x2 -- but $D^X_\text{tot}=\sqrt2 D^X_\sigma$ eats one,
        //     so 4/(2*2) * ... = `alpha`.
        //   $D^\text{gs}(T{+}D^Z)$: the left slot goes 0.5*D_up -> D_is (x2); the right slot
        //     goes 0.5*(T+Z)_tot = (T+Z)_up -> (T+Z)_is (x1, no sqrt2 here); the explicit sum
        //     over is adds x2. So 4/(2*1*2) = `alpha` as well -- NOT `2*alpha`: the earlier
        //     comment forgot that the closed-shell right slot is already the spin SUM, which is
        //     exactly what the `for (is)` loop below now supplies.
        if (LR::exx_kernel_list().count(context.xc_kernel))
        {
            const auto& Ds_trans = LR_Util::get_exx_Ds_gs(dm_trans, context.ucell, context.kv, context.pmat);
            ModuleBase::matrix force_exx_dmtrans(context.ucell.nat, 3);
            for (int is: {0, 1})
            {
                force_exx_dmtrans += lr_force.cal_force_exx_dm_trans(Ds_trans[is], alpha, std::to_string(is));
            }
            if (context.test_force)
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "EXX DMTRANS FORCE (eV/Angstrom)",
                                      force_exx_dmtrans,
                                      false);
            force_hxc_dmtrans += force_exx_dmtrans;
        }
        if (context.ground_is_hybrid)
        {
            const auto& Ds_gs = LR_Util::get_exx_Ds_gs(dm_gs, context.ucell, context.kv, context.pmat);
            const auto& Ds_relaxed_diff
                = LR_Util::get_exx_Ds_gs(relaxed_diff_dm, context.ucell, context.kv, context.pmat);
            ModuleBase::matrix force_exx_gs_relaxed_diff(context.ucell.nat, 3);
            for (int is: {0, 1})
            {
                force_exx_gs_relaxed_diff += lr_force.cal_force_exx_gs_dm_relaxed_diff(Ds_gs[is],
                                                                                       Ds_relaxed_diff[is],
                                                                                       alpha,
                                                                                       std::to_string(is));
            }
            if (context.test_force)
                ModuleIO::print_force(context.log,
                                      context.ucell,
                                      "EXX GS-(T+Z) FORCE (eV/Angstrom)",
                                      force_exx_gs_relaxed_diff,
                                      false);
            force_hamiltgs_relaxed_diff += force_exx_gs_relaxed_diff;
        }
#endif
        forces[istate - ist_begin] = force_hxc_dmtrans + force_hamiltgs_relaxed_diff + force_overlap_edm;
    }
    ModuleBase::timer::end("LR_Gradient", "assemble_open_shell_forces");
    print_force(forces, std::cout, ist_begin);
    print_force(forces, context.log, ist_begin);
    return forces;
}
template std::vector<ModuleBase::matrix> assemble_closed_shell_forces<double>(
    const GradientContext<double>&,
    LR_Force<double>&,
    const elecstate::DensityMatrix<double, double>&,
    const ct::Tensor&,
    const ct::Tensor&,
    int,
    int,
    int);
template std::vector<ModuleBase::matrix> assemble_open_shell_forces<double>(
    const GradientContext<double>&,
    LR_Force<double>&,
    const elecstate::DensityMatrix<double, double>&,
    const ct::Tensor&,
    const ct::Tensor&,
    int,
    int);
template std::vector<ModuleBase::matrix> assemble_closed_shell_forces<std::complex<double>>(
    const GradientContext<std::complex<double>>&,
    LR_Force<std::complex<double>>&,
    const elecstate::DensityMatrix<std::complex<double>, double>&,
    const ct::Tensor&,
    const ct::Tensor&,
    int,
    int,
    int);
template std::vector<ModuleBase::matrix> assemble_open_shell_forces<std::complex<double>>(
    const GradientContext<std::complex<double>>&,
    LR_Force<std::complex<double>>&,
    const elecstate::DensityMatrix<std::complex<double>, double>&,
    const ct::Tensor&,
    const ct::Tensor&,
    int,
    int);
} // namespace LR
