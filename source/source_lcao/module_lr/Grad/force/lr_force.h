#ifndef ABACUS_LR_FORCE_H
#define ABACUS_LR_FORCE_H
#include "force_funcs_lcao.h"
#include "source_lcao/module_lr/Grad/xc/pot_grad_xc.h"
#include "source_lcao/module_lr/potentials/pot_hxc_lrtd.h"

#include <iosfwd>
// free functions, usefull for both ground and excited state
#ifdef __EXX
#include "exx_force_two_dm.h"
#include "source_lcao/module_ri/exx_lri.h"
using TAC = std::pair<int, std::array<int, 3>>;
#endif
namespace LR
{
/** @brief Explicit immutable configuration for the LR force assembly kernels. */
struct ForceOptions
{
    int spin_channels;                  ///< Ground-state density channels, including both spin channels when present.
    bool include_hartree;               ///< Include Hartree in the constructed local response potential.
    bool include_local_pseudopotential; ///< Include the plane-wave local-potential force term.
    bool print_components;              ///< Write decomposed forces to the supplied diagnostic stream.
};

template <typename TK>
class LR_Force
{
  public:
    LR_Force(const UnitCell& ucell,
             const std::vector<ModuleBase::Vector3<double>>& kvec_d,
             const Parallel_Orbitals& pv,
             const ModulePW::PW_Basis& rhodpw,
             const ModulePW::PW_Basis& rhopw,
             const pseudopot_cell_vl& locpp,
             const Structure_Factor& sf,
             const Grid_Driver& gd,
             const TwoCenterBundle& two_center_bundle,
             const ForceOptions& options,
             std::ofstream& log
#ifdef __EXX
             ,
             std::weak_ptr<Exx_LRI<TK>> exx_lri_in,
             const double& alpha
#endif
             ) ///<  for 2-center integrals
        : ucell_(ucell), kvec_d_(kvec_d), pv_(pv), rhodpw_(rhodpw), rhopw_(rhopw), sf_(sf), locpp_(locpp), gd_(gd),
          two_center_bundle_(two_center_bundle), options_(options), log_(log)
#ifdef __EXX
          ,
          exx_lri_(exx_lri_in), alpha_(alpha)
#endif
    {
        ground_channel_factor_ = options_.spin_channels == 1 ? 0.5 : 1.0;
    }

    /// 1. $Tr[H_{GS}^x * (T+D^Z)]$, where GS=groud state and $(T+D^Z)$ is the relaxed difference density matrix
    ModuleBase::matrix cal_force_hamilt_gs_dm_relaxed_diff(const elecstate::DensityMatrix<TK, double>& relaxed_diff_dm,
                                                           const elecstate::DensityMatrix<TK, double>& dm_gs,
                                                           const bool reproduce_gs = false,
                                                           const PotHxcLR* pot_hxc_gs = nullptr);

    /// 2. $Tr[S^x * (EDM)]
    ModuleBase::matrix cal_force_overlap_edm(const elecstate::DensityMatrix<TK, double>& edm);

    /// 3. $\sum_{mnkl}(mn|f_{Hxc}|kl)^x *D^X *D^X$
    ModuleBase::matrix cal_force_hxc_dmtrans(const elecstate::DensityMatrix<TK, double>& dm_trans,
                                             const PotHxcLR& pot_hxc);

    /// 3b. the $g^{xc}$ half of $\partial_x K^{S/T}[D^X]D^X$:
    ///     $\int v^{(2)}[\rho^X,\rho^X](r)\,\partial_x\rho^\text{gs}(r)|_\text{basis}$
    ModuleBase::matrix cal_force_gxc_dmtrans(const elecstate::DensityMatrix<TK, double>& dm_trans,
                                             const elecstate::DensityMatrix<TK, double>& dm_gs,
                                             const PotGradXCLR& pot_grad);

    /// 3b'. open-shell version: $\sum_\tau\int v^{(2)}_\tau[\rho^X,\rho^X]\,
    ///      \partial_x\rho^\text{gs}_\tau|_\text{basis}$. Both transition-density channels
    ///      enter each $v^{(2)}_\tau$, so this cannot be a per-channel loop over the above.
    ModuleBase::matrix cal_force_gxc_dmtrans_openshell(const elecstate::DensityMatrix<TK, double>& dm_trans,
                                                       const elecstate::DensityMatrix<TK, double>& dm_gs,
                                                       const PotGradXCLR& pot_grad);

#ifdef __EXX
    // auto* lrexx_ptr = dynamic_cast<RI::LR<int, std::array<int, 3>, 3, TK>*>(&exx_lri_in.get());
    /// 4. $\alpha \sum_{mnkl}(mk|nl)^x *D^X *D^X$
    ModuleBase::matrix cal_force_exx_dm_trans(const std::map<int, std::map<TAC, RI::Tensor<TK>>>& dm_trans,
                                              const double& alpha,
                                              const std::string& spin_suffix = "");
    ModuleBase::matrix cal_force_exx_gs_dm_relaxed_diff(
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& dm_gs,
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& relaxed_diff_dm,
        const double& alpha,
        const std::string& spin_suffix = "");
#endif

    // test functions
    /// reproduce the force of the ground state
    ModuleBase::matrix reproduce_force_gs(const K_Vectors& kv,
                                          const elecstate::DensityMatrix<TK, double>& dm_gs,
                                          const elecstate::DensityMatrix<TK, double>& edm_gs);

    /// repreduce the ground state local term
    ModuleBase::matrix reproduce_force_gs_loc(const elecstate::DensityMatrix<TK, double>& dm_gs,
                                              const elecstate::Potential& pot_gs);

    /// derivatives of 2-center integrates: dtau(S_ij) and dtau(h_{ij}) (set vh_in_h=0)
    void cal_H2_sz_center2_deriv(const std::vector<double>& orb_cutoffs, const K_Vectors& kv);
    /// 4-center integrates or their derivatives: (ij | kl) or dtau(ij | kl) (set vl_in_h=0)
    void cal_H2_sz_center4(const std::vector<double>& orb_cutoffs, const K_Vectors& kv, const bool is_grad = false);

  protected:
    const UnitCell& ucell_;
    const std::vector<ModuleBase::Vector3<double>>& kvec_d_;
    const Parallel_Orbitals& pv_;
    const ModulePW::PW_Basis& rhodpw_;
    const ModulePW::PW_Basis& rhopw_;
    const pseudopot_cell_vl& locpp_;
    const Structure_Factor& sf_;
    const Grid_Driver& gd_;
    const TwoCenterBundle& two_center_bundle_;
    const ForceOptions options_;
    std::ofstream& log_;
    double ground_channel_factor_ = 1.0; ///< Occupation factor for one closed-shell density channel.
#ifdef __EXX
    std::weak_ptr<Exx_LRI<TK>> exx_lri_;
    const double alpha_;
#endif

    Charge dm_to_charge(const elecstate::DensityMatrix<TK, double>& dm);
    elecstate::Potential dm_to_hxc_potential(const elecstate::DensityMatrix<TK, double>& dm);
    elecstate::Potential local_potential();
};
} // namespace LR

#endif // ABACUS_LR_FORCE_H
