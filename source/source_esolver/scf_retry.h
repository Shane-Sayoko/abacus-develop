#ifndef ABACUS_ESOLVER_SCF_RETRY_H
#define ABACUS_ESOLVER_SCF_RETRY_H

#include <functional>

namespace ModuleESolver
{
/**
 * @brief Execute a baseline SCF and, only on failure, one same-geometry retry.
 * @param[in] enabled Whether to allow the retry.
 * @param[in] solve Run SCF using the current configuration; true means converged.
 * @param[in] configure Select retry (true) or restore baseline (false), clearing history.
 * @param[in] finish Publish the final electronic endpoint once, with its convergence status.
 * @return Final SCF convergence status.
 * Geometry preparation belongs to the caller and is never repeated here.
 * The baseline configuration is restored even when retry solving or publication throws.
 */
bool run_scf_retry(
    bool enabled,
    const std::function<bool()> &solve,
    const std::function<void(bool)> &configure,
    const std::function<void(bool)> &finish);
} // namespace ModuleESolver

#endif
