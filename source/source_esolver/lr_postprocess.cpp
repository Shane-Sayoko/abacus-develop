#include "lr_postprocess.h"

namespace ModuleESolver
{

bool needs_lr_postprocess_forces(
    const std::string &calculation,
    const std::string &md_type,
    const bool cal_force,
    const bool excited_relax)
{
    if (!cal_force || excited_relax)
    {
        return false;
    }
    const bool fssh_md = calculation == "md" && md_type == "fssh";
    return !fssh_md;
}

} // namespace ModuleESolver
