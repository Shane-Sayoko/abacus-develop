#ifndef ABACUS_SOURCE_ESOLVER_LR_POSTPROCESS_H
#define ABACUS_SOURCE_ESOLVER_LR_POSTPROCESS_H

#include <string>

namespace ModuleESolver
{

/**
 * @brief Decide whether LR final output needs an all-state force calculation.
 *
 * FSSH evaluates its active surface during MD; relaxation evaluates its target
 * state during the runner. Neither needs the single-point all-state force output.
 * This policy does not suppress spectra or electron-hole density output.
 *
 * @param[in] calculation Validated calculation mode.
 * @param[in] md_type Validated MD integrator name; relevant only for MD.
 * @param[in] cal_force Whether force calculation is requested.
 * @param[in] excited_relax Whether the runner already evaluates the relaxation target.
 * @return True if final output must calculate the forces of all LR states.
 */
bool needs_lr_postprocess_forces(
    const std::string &calculation,
    const std::string &md_type,
    bool cal_force,
    bool excited_relax);

} // namespace ModuleESolver

#endif
