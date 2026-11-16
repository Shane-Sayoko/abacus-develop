#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_POTENTIALS_POT_GRAD_XC_H
#define ABACUS_SOURCE_LCAO_MODULE_LR_POTENTIALS_POT_GRAD_XC_H
#include "source_lcao/module_lr/potentials/pot_lr_base.h"
#include "source_lcao/module_lr/potentials/xc_kernel.h"

namespace LR
{
/// the "potential" contributing to RHS of Z-vector equation
/// from the derivative of xc kernel
class PotGradXCLR : public PotLRBase
{
  public:
    /// constructor for exchange-correlation kernel
    /// `triplet` selects which spin combination of $g^{xc}$ to use. It matters:
    /// the triplet kernel $K^T_{xc}=f_{uu}-f_{ud}$ is NOT zero for a local functional.
    PotGradXCLR(
        const KernelXC &xc_kernel_in,
        const ModulePW::PW_Basis &rho_basis,
        const UnitCell &ucell,
        const int &nrxx,
        const int nspin,
        const bool triplet);
    ~PotGradXCLR() {}
    virtual void cal_v_eff(
        double **rho,
        const UnitCell &ucell,
        ModuleBase::matrix &v_eff,
        const std::vector<int> &ispin_op = {0, 0}) const override;
    /**
     * @brief Derivative of an unrestricted LR quadratic form with respect to GS density.
     * @param[in] rho1 Two borrowed transition-density arrays, each of length nrxx.
     * @param[in] ucell Cell context, consistent with the grid supplied at construction.
     * @param[in,out] v_eff One contiguous output spin channel of nrxx values; accumulated in Ry.
     * @param[in] tau Free output spin, zero for up and one for down.
     * @details Both transition spin channels enter A_tau - div(E_tau).
     * For an unmodified XC kernel this equals the contracted third derivative.
     * Stabilized LR sigma coefficients require their preparation Jacobian only
     * on implicit sigma derivatives; explicit gradient factors remain active.
     * Requires nspin=2 and the kernel's requested raw third-derivative arrays.
     */
    void cal_v_eff_openshell(
        const double *const *const rho1,
        const UnitCell &ucell,
        ModuleBase::matrix &v_eff,
        const int tau) const;
    /// kernel components from PotHxcLR
    const KernelXC &xc_kernel_components_;
    const bool triplet_ = false;

  private:
    /// Scratch, shared by every `PotGradXCLR` and grown on demand. These used to be
    /// allocated (and value-initialized) on every call. Safe to share because
    /// `cal_v_eff` is only ever entered from a single thread (all the OpenMP is inside).
    struct Scratch
    {
        std::vector<ModuleBase::Vector3<double>> drho1[2]; ///< $\nabla\rho^1_\sigma$
        std::vector<ModuleBase::Vector3<double>> gdot;     ///< integrand of the divergence
        std::vector<double> vtmp;                          ///< local part $A$
        std::vector<double> div;                           ///< $-\nabla\cdot\boldsymbol{E}$
        void alloc(const int nrxx, const bool two_channel, const bool gga);
    };
    static Scratch &scratch();
};

} // namespace LR
#endif // ABACUS_SOURCE_LCAO_MODULE_LR_POTENTIALS_POT_GRAD_XC_H
