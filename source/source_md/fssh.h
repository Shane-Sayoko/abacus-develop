#ifndef FSSH_MD_H
#define FSSH_MD_H

#include "md_base.h"
#include "source_lcao/module_operator_lcao/fssh_driver.h"

namespace ModuleESolver
{
template<typename T, typename TR> class ESolver_LR;
}

/**
 * @brief Fewest switches surface hopping on analytic KS plus LR potential surfaces.
 * The electronic driver owns the hopping state; MDCell owns nuclear positions and velocities.
 */
class FsshMD final : public MD_base
{
  public:
    /**
     * @brief Construct a FSSH trajectory attached to a distributed MD cell.
     * @param param_in Input and output parameters for this trajectory.
     * @param mdcell_in Nuclear cell used by the standard MD integrator.
     */
    FsshMD(const Parameter& param_in, MDCell& mdcell_in);

    /**
     * @brief Initialize the LR solver, electronic frame, active surface and force.
     * @param p_esolver Persistent KS plus LR solver.
     * @param global_readin_dir Restart input directory.
     * @param decomp Nuclear domain decomposition used by the MD loop.
     */
    void setup(ModuleESolver::ESolver* p_esolver,
               const std::string& global_readin_dir,
               DomainDecomposition& decomp) override;

    /**
     * @brief Propagate electrons, attempt a hop and update the active analytic force.
     * @param p_esolver Persistent solver also supplied to setup().
     * @param ionic_step Global one-based ionic step for the current geometry.
     */
    void advance_surface(ModuleESolver::ESolver* p_esolver, int ionic_step);

    /**
     * @brief Write the nuclear and electronic restart state.
     * @param global_out_dir Directory receiving restart files.
     */
    void write_restart(const std::string& global_out_dir) override;

  protected:
    /**
     * @brief Restore the electronic hopping state alongside the base MD restart.
     * @param global_readin_dir Directory containing restart files.
     */
    void restart(const std::string& global_readin_dir) override;

  private:
    /**
     * @brief Copy one LR surface energy and force to the owned MD atoms.
     * @param state Active FSSH surface index, with zero representing the ground state.
     */
    void apply_surface(int state);

    const Parameter& param_;
    ModuleESolver::ESolver_LR<double, double>* lr_solver_ = nullptr;
    FsshDriver driver_;
    FsshElectronicFrame previous_frame_;
    bool restored_ = false;
};

#endif // FSSH_MD_H
