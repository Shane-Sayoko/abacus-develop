#include "scf_retry.h"

namespace ModuleESolver
{
bool run_scf_retry(
    const bool enabled,
    const std::function<bool()> &solve,
    const std::function<void(bool)> &configure,
    const std::function<void(bool)> &finish)
{
    bool converged = solve();
    if (converged || !enabled)
    {
        finish(converged);
        return converged;
    }
    try
    {
        configure(true);
        converged = solve();
        finish(converged);
    }
    catch (...)
    {
        configure(false);
        throw;
    }
    configure(false);
    return converged;
}
} // namespace ModuleESolver
