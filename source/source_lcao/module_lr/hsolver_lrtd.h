#ifndef ABACUS_LR_HSOLVER_LRTD_H
#define ABACUS_LR_HSOLVER_LRTD_H
#include "source_base/parallel_2d.h"
#include "source_base/macros.h"
#include <complex>
#include <string>
#include <vector>
namespace LR
{
template<typename T> class HamiltLR;
template<typename T> class HamiltULR;
template<typename T> using Real = typename GetTypeReal<T>::type;
namespace HSolver
{
/** @brief Explicit eigensolver configuration supplied by the solver composition point. */
struct DiagonalizationOptions
{
    int subspace_dimension; ///< Davidson workspace multiplier.
    int subspace_solver;    ///< 0=LAPACK, 1=ELPA, 2=ScaLAPACK.
    int block_size;         ///< Subspace matrix distribution block size.
    bool audit_operator;    ///< Check dense Hermiticity and phase covariance when states are exported.
};
/**
 * @brief Solve a distributed LR eigenproblem and measure the operator residual.
 * @param[in] hm Hamiltonian with matrix() and collective hPsi() actions in Ry.
 * @param[in,out] psi State-major local amplitudes; caller owns dim*nband entries.
 * @param[in] dim Local particle-hole storage per state.
 * @param[in] nband Number of requested eigenstates.
 * @param[in] nk Number of k points per spin.
 * @param[in] nocc Occupied window per spin.
 * @param[in] nvirt Virtual window per spin.
 * @param[in] pX Particle-hole distributions: virtual rows, occupied columns.
 * @param[out] eig Requested eigenvalues in Ry; caller owns nband entries.
 * @param[in] method Eigensolver name.
 * @param[in] diag_ethr Eigensolver convergence threshold in Ry.
 * @param[in] precondition Local diagonal preconditioner.
 * @param[in] options Explicit workspace and subspace solver settings.
 * @param[in] hermitian Whether the assembled dense problem is Hermitian.
 * @return Global ||HX-E X||/||X|| in Ry for each returned state.
 */

template <typename T, typename THamilt>
std::vector<double> solve(
    const THamilt &hm,
    T *psi,
    const int &dim,   ///< local leading dimension (or nbasis)
    const int &nband, ///< nstates in LR-TDDFT, not (nocc+nvirt)
    const int &nk,
    const std::vector<int> &nocc,
    const std::vector<int> &nvirt,
    const std::vector<Parallel_2D> &pX,
    double *eig,
    const std::string method,
    const Real<T> &diag_ethr, ///< threshold for diagonalization
    const std::vector<Real<T>> &precondition,
    const DiagonalizationOptions &options,
    const bool hermitian);
}
}
#endif
