#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_HSOLVER_LRTD_HPP
#define ABACUS_SOURCE_LCAO_MODULE_LR_HSOLVER_LRTD_HPP
#include "source_base/module_container/ATen/core/tensor_map.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_reduce.h"
#include "source_hsolver/diag_comm_info.h"
#include "source_hsolver/diago_cg.h"
#include "source_hsolver/diago_dav_subspace.h"
#include "source_hsolver/diago_david.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_print.h"

#include <stdexcept>

namespace LR
{
template <typename T>
using Real = typename GetTypeReal<T>::type;

namespace HSolver
{
/** @brief Explicit eigensolver configuration supplied by the solver composition point. */
struct DiagonalizationOptions
{
    int subspace_dimension; ///< Davidson workspace multiplier.
    int subspace_solver;    ///< 0=LAPACK, 1=ELPA, 2=ScaLAPACK.
    int block_size;         ///< Subspace matrix distribution block size.
};
template <typename T>
inline void print_eigs(const std::vector<T>& eigs,
                       const std::string& label = "",
                       const double factor = 1.0,
                       const double precision = 8)
{
    std::streamsize old = std::cout.precision();
    std::cout << label << std::setprecision(precision) << std::endl;
    for (auto& e: eigs)
    {
        std::cout << e * factor << " ";
    }
    std::cout << std::endl;
    std::cout.precision(old);
}

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
std::vector<double> solve(const THamilt& hm,
                          T* psi,
                          const int& dim,   ///< local leading dimension (or nbasis)
                          const int& nband, ///< nstates in LR-TDDFT, not (nocc+nvirt)
                          const int& nk,
                          const std::vector<int>& nocc,
                          const std::vector<int>& nvirt,
                          const std::vector<Parallel_2D>& pX,
                          double* eig,
                          const std::string method,
                          const Real<T>& diag_ethr, ///< threshold for diagonalization
                          const std::vector<Real<T>>& precondition,
                          const DiagonalizationOptions& options,
                          const bool hermitian = true)
{
    ModuleBase::TITLE("HSolverLR", "solve");
    ModuleBase::timer::start("HSolverLR", "solve");
    const std::vector<std::string> spin_types = {"singlet", "triplet"};
    // note: if not TDA, the eigenvalues will be complex
    // then we will need a new constructor of DiagoDavid

    int empty_layouts = dim <= 0 ? 1 : 0;
    for (const auto& layout: pX)
    {
        if (layout.get_local_size() == 0)
        {
            empty_layouts = 1;
        }
    }
    Parallel_Reduce::reduce_all(empty_layouts);
    if (empty_layouts > 0)
    {
        throw std::invalid_argument("LR requires nonempty particle-hole blocks on every rank; reduce MPI ranks");
    }

    // 1. allocate eigenvalue
    std::vector<Real<T>> eigenvalue(nband); // nstates
    // 2. select the method
#ifdef __MPI
    const hsolver::diag_comm_info comm_info = {POOL_WORLD, GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL};
#else
    const hsolver::diag_comm_info comm_info = {GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL};
#endif

    if (method == "lapack")
    {
        std::vector<T> Amat_full = hm.matrix();
        const int gdim = std::sqrt(Amat_full.size());
        eigenvalue.resize(gdim);
        if (hermitian)
        {
            LR_Util::diag_lapack(gdim, Amat_full.data(), eigenvalue.data());
        }
        else
        {
            std::vector<std::complex<double>> eig_complex(gdim);
            LR_Util::diag_lapack_nh(gdim, Amat_full.data(), eig_complex.data());
            print_eigs(eig_complex, "Right eigenvalues: of the non-Hermitian matrix: (Ry)");
            for (int i = 0; i < gdim; i++)
            {
                eigenvalue[i] = eig_complex[i].real();
            }
        }
        bool openshell = std::is_same<THamilt, HamiltULR<T>>::value;
        // copy eigenvectors
#ifdef __MPI
        LR_Util::global2local_X(psi, Amat_full.data(), nband, nk, nocc, nvirt, pX, openshell);
#else
        std::memcpy(psi, Amat_full.data(), sizeof(T) * nband * gdim);
#endif
    }
    else
    {
        // 3. set maxiter and funcs
        const int maxiter = hsolver::DiagoIterAssist<T>::PW_DIAG_NMAX;

        auto hpsi_func
            = [&hm](T* psi_in, T* hpsi, const int ld_psi, const int nvec) { hm.hPsi(psi_in, hpsi, ld_psi, nvec); };
        auto spsi_func = [&hm](const T* psi_in, T* spsi, const int ld_psi, const int nvec) {
            std::memcpy(spsi, psi_in, sizeof(T) * ld_psi * nvec);
        };

        if (method == "dav")
        {
            // Allow 5 tries at most. If ntry > ntry_max = 5, exit diag loop.
            const int ntry_max = 5;
            // In non-self consistent calculation, do until totally converged. Else allow 5 eigenvecs to be NOT
            // converged.
            // do diag and add davidson iteration counts up to avg_iter
            hsolver::DiagoDavid<T> david(precondition.data(), nband, dim, options.subspace_dimension, comm_info);
            std::vector<double> ethr_band(nband, diag_ethr);
            hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(
                david.diag(hpsi_func, spsi_func, dim, psi, eigenvalue.data(), ethr_band, maxiter, ntry_max, 0));
        }
        else if (method == "dav_subspace") // need refactor
        {
            hsolver::Diago_DavSubspace<T> dav_subspace(precondition,
                                                       nband,
                                                       dim,
                                                       options.subspace_dimension,
                                                       diag_ethr,
                                                       maxiter,
                                                       comm_info,
                                                       options.subspace_solver,
                                                       options.block_size);
            std::vector<double> ethr_band(nband, diag_ethr);
            hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(
                dav_subspace.diag(hpsi_func, spsi_func, psi, dim, eigenvalue.data(), ethr_band, false /*scf*/));
        }
        else if (method == "cg")
        {
            ////// `diagH_subspace` needs refactor:
            ////// replace `Hamilt*` with `hpsi_func`
            ////// or I cannot use `is_subspace=true` as my `HamiltLR` does not inherit `Hamilt`.

            // auto subspace_func = [&hm](const ct::Tensor& psi_in, ct::Tensor& psi_out) {
            //     const auto ndim = psi_in.shape().ndim();
            //     REQUIRES_OK(ndim == 2, "dims of psi_in should be less than or equal to 2");
            //     // Convert a Tensor object to a psi::Psi object
            //     auto psi_in_wrapper = psi::Psi<T>(psi_in.data<T>(),
            //         1,
            //         psi_in.shape().dim_size(0),
            //         psi_in.shape().dim_size(1));
            //     auto psi_out_wrapper = psi::Psi<T>(psi_out.data<T>(),
            //         1,
            //         psi_out.shape().dim_size(0),
            //         psi_out.shape().dim_size(1));
            //     auto eigen = ct::Tensor(ct::DataTypeToEnum<Real<T>>::value,
            //         ct::DeviceType::CpuDevice,
            //         ct::TensorShape({ psi_in.shape().dim_size(0) }));
            //     hsolver::DiagoIterAssist<T>::diagH_subspace(hm, psi_in_wrapper, psi_out_wrapper,
            //     eigen.data<Real<T>>());
            //     };

            ////// why diago_cg depends on basis_type?
            // hsolver::DiagoCG<T> cg("lcao", "nscf", true, subspace_func, diag_ethr, maxiter, GlobalV::NPROC_IN_POOL);

            auto subspace_func = [](T* psi_in, T* psi_out, const int ld_psi, const int nband, const bool S_orth) {};
            hsolver::DiagoCG<T> cg("lcao", "nscf", false, subspace_func, diag_ethr, maxiter, GlobalV::NPROC_IN_POOL);

            auto hpsi_func
                = [&hm](T* psi_in, T* hpsi, const int ld_psi, const int nvec) { hm.hPsi(psi_in, hpsi, ld_psi, nvec); };
            auto spsi_func = [](T* psi_in, T* spsi, const int ld_psi, const int nvec) {
                std::memcpy(spsi, psi_in, sizeof(T) * static_cast<size_t>(ld_psi) * static_cast<size_t>(nvec));
            };

            std::vector<double> ethr_band(nband, diag_ethr);
            cg.diag(hpsi_func, spsi_func, dim, nband, dim, psi, eigenvalue.data(), ethr_band, precondition.data());
        }
        else
        {
            throw std::runtime_error("HSolverLR::solve: method not implemented");
        }
    }

    // 5. copy eigenvalues
    for (int ist = 0; ist < nband; ++ist)
    {
        eig[ist] = eigenvalue[ist];
    }

    // 6. output eigenvalues and eigenvectors
    print_eigs(eigenvalue, "eigenvalues: (Ry)");
    print_eigs(eigenvalue, "eigenvalues: (eV)", ModuleBase::Ry_to_eV);

    // Reapply the actual distributed operator. Eigenvalue changes alone do not
    // establish convergence, and a dense solver only consumes one matrix triangle.
    std::vector<T> applied(static_cast<std::size_t>(dim) * nband, T(0));
    hm.hPsi(psi, applied.data(), dim, nband);
    std::vector<double> residuals(nband, 0.0);
    for (int state = 0; state < nband; ++state)
    {
        double residual_square = 0.0;
        double norm_square = 0.0;
        for (int row = 0; row < dim; ++row)
        {
            const std::size_t index = static_cast<std::size_t>(state) * dim + row;
            const double residual = std::abs(applied[index] - eigenvalue[state] * psi[index]);
            const double magnitude = std::abs(psi[index]);
            residual_square += residual * residual;
            norm_square += magnitude * magnitude;
        }
        Parallel_Reduce::reduce_all(residual_square);
        Parallel_Reduce::reduce_all(norm_square);
        if (!std::isfinite(residual_square) || !std::isfinite(norm_square) || norm_square <= 0.0)
        {
            throw std::runtime_error("Invalid LR eigenvector or operator residual");
        }
        residuals[state] = std::sqrt(residual_square / norm_square);
    }
    print_eigs(residuals, "LR normalized eigenpair residuals (Ry)", 1.0, 17);

    // output iters
    std::cout << " Average iterative diagonalization steps: " << hsolver::DiagoIterAssist<T>::avg_iter
              << "; current threshold: " << diag_ethr << std::endl;
    ModuleBase::timer::end("HSolverLR", "solve");
    return residuals;
}
} // namespace HSolver
} // namespace LR

#endif // ABACUS_SOURCE_LCAO_MODULE_LR_HSOLVER_LRTD_HPP
