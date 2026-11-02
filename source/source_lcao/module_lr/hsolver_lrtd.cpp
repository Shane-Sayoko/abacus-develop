#include "hsolver_lrtd.h"
#include "hamilt_casida.h"
#include "hamilt_ulr.h"
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

namespace LR
{


    namespace HSolver
    {
        /// The LR Hamiltonians (HamiltLR, HamiltULR) are not hamilt::Hamilt, so
        /// they get their own hsolver::HSOperator view. S is the identity.
        template <typename T, typename THamilt>
        class LRHSOperator : public hsolver::HSOperator<T>
        {
          public:
            explicit LRHSOperator(const THamilt& hm) : hm_(hm) {}
            void update_k(const int ik) override {}
            void hpsi(const T* x, T* hx, const int ld, const int nvec) const override { hm_.hPsi(x, hx, ld, nvec); }
            void spsi(const T* x, T* sx, const int ld, const int nvec) const override
            {
                std::memcpy(sx, x, sizeof(T) * static_cast<size_t>(ld) * static_cast<size_t>(nvec));
            }
          private:
            const THamilt& hm_;
        };

        /// @brief Real sign used to audit linearity of a real LR operator.
        inline double audit_phase(double) { return -1.0; }
        /// @brief Imaginary phase used to audit complex linearity at a single k point too.
        inline std::complex<double> audit_phase(std::complex<double>) { return {0.0, 1.0}; }

        template <typename T>
        inline void print_eigs(
            const std::vector<T> &eigs,
            const std::string &label = "",
            const double factor = 1.0,
            const int precision = 8)
        {
            std::streamsize old = std::cout.precision();
            std::cout << label << std::setprecision(precision) << std::endl;
            for (auto& e : eigs) { std::cout << e * factor << " "; }
            std::cout << std::endl;
            std::cout.precision(old);
        }

        /// eigensolver for common Hamilt
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
            const bool hermitian)
        {
            ModuleBase::TITLE("HSolverLR", "solve");
            ModuleBase::timer::start("HSolverLR", "solve");
            const std::vector<std::string> spin_types = { "singlet", "triplet" };
            // note: if not TDA, the eigenvalues will be complex
            // then we will need a new constructor of DiagoDavid

            // 1. allocate eigenvalue
            std::vector<Real<T>> eigenvalue(nband);   //nstates
            // 2. select the method
#ifdef __MPI
            const hsolver::diag_comm_info comm_info = { POOL_WORLD, GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL };
#else
            const hsolver::diag_comm_info comm_info = { GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL };
#endif

            if (method == "lapack")
            {
                std::vector<T> Amat_full = hm.matrix();
                const int gdim = std::sqrt(Amat_full.size());
                eigenvalue.resize(gdim);
                if (options.audit_operator && hermitian)
                {
                    double defect = 0.0;
                    for (int row = 0; row < gdim; ++row)
                    {
                        for (int column = 0; column < gdim; ++column)
                        {
                            const auto difference =
                                Amat_full[row * gdim + column] - std::conj(Amat_full[column * gdim + row]);
                            defect = std::max(defect, std::abs(difference));
                        }
                    }
                    print_eigs(
                        std::vector<double>{defect},
                        "LR dense Hermiticity defect (Ry)",
                        1.0,
                        17);
                }
                if (hermitian) { LR_Util::diag_lapack(gdim, Amat_full.data(), eigenvalue.data()); }
                else
                {
                    std::vector<std::complex<double>> eig_complex(gdim);
                    LR_Util::diag_lapack_nh(gdim, Amat_full.data(), eig_complex.data());
                    print_eigs(eig_complex, "Right eigenvalues: of the non-Hermitian matrix: (Ry)");
                    for (int i = 0; i < gdim; i++) { eigenvalue[i] = eig_complex[i].real(); }
                }
                bool openshell = std::is_same<THamilt, HamiltULR<T>>::value;
                // copy eigenvectors
#ifdef __MPI
                LR_Util::global2local_X(psi, Amat_full.data(), nband, nk,
                                        nocc, nvirt, pX, openshell);
#else
                std::memcpy(psi, Amat_full.data(), sizeof(T) * nband * gdim);
#endif
            }
            else
            {
                // 3. set maxiter and the operator
                const int maxiter = hsolver::DiagoIterAssist<T>::PW_DIAG_NMAX;
                const LRHSOperator<T, THamilt> op(hm);
                const int davidson_dimension = options.subspace_dimension;

                if (method == "dav")
                {
                    // Allow 5 tries at most. If ntry > ntry_max = 5, exit diag loop.
                    const int ntry_max = 5;
                    // do diag and add davidson iteration counts up to avg_iter
                    hsolver::DiagoDavid<T> david(precondition.data(),
                                                 nband,
                                                 dim,
                                                 davidson_dimension,
                                                 comm_info);
                    std::vector<double> ethr_band(nband, diag_ethr);
                    hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(david.diag(op,
                        dim, psi, eigenvalue.data(), ethr_band, maxiter, ntry_max, 0));
                }
                else if (method == "dav_subspace") //need refactor
                {
                    hsolver::Diago_DavSubspace<T> dav_subspace(
                        precondition,
                        nband,
                        dim,
                        davidson_dimension,
                        diag_ethr,
                        maxiter,
                        comm_info,
                        options.subspace_solver,
                        options.block_size);
                    std::vector<double> ethr_band(nband, diag_ethr);
                    hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(
                        dav_subspace.diag(op, psi, dim, eigenvalue.data(), ethr_band, false /*scf*/));
                }
                else if (method == "cg")
                {
                    // the subspace rotation of DiagoCG now works on any HSOperator, so it could be
                    // switched on here; it is kept off to leave the LR results unchanged
                    hsolver::DiagoCG<T> cg("lcao", "nscf", false, comm_info, diag_ethr, maxiter);

                    std::vector<double> ethr_band(nband, diag_ethr);
                    cg.diag(op,
                            dim,
                            nband,
                            dim,
                            psi,
                            eigenvalue.data(),
                            ethr_band,
                            precondition.data());
                }
                else { throw std::runtime_error("HSolverLR::solve: method not implemented"); }
            }

            // 5. copy eigenvalues
            for (int ist = 0;ist < nband;++ist) { eig[ist] = eigenvalue[ist]; }

            // 6. output eigenvalues and eigenvectors
            print_eigs(eigenvalue, "eigenvalues: (Ry)");
            print_eigs(eigenvalue, "eigenvalues: (eV)", ModuleBase::Ry_to_eV);

            // Reapply the actual distributed operator. Eigenvalue changes alone do not
            // establish convergence, and a dense solver only consumes one matrix triangle.
            std::vector<T> applied(static_cast<std::size_t>(dim) * nband, T(0));
            hm.hPsi(
                psi,
                applied.data(),
                dim,
                nband);
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
            print_eigs(
                residuals,
                "LR normalized eigenpair residuals (Ry)",
                1.0,
                17);
            if (options.audit_operator)
            {
                const T phase = audit_phase(T(0));
                std::vector<T> rotated(applied.size());
                for (std::size_t index = 0; index < rotated.size(); ++index)
                {
                    rotated[index] = phase * psi[index];
                }
                std::vector<T> rotated_action(applied.size(), T(0));
                hm.hPsi(
                    rotated.data(),
                    rotated_action.data(),
                    dim,
                    nband);
                double defect_square = 0.0;
                double reference_square = 0.0;
                for (std::size_t index = 0; index < rotated.size(); ++index)
                {
                    const double difference = std::abs(rotated_action[index] - phase * applied[index]);
                    const double magnitude = std::abs(applied[index]);
                    defect_square += difference * difference;
                    reference_square += magnitude * magnitude;
                }
                Parallel_Reduce::reduce_all(defect_square);
                Parallel_Reduce::reduce_all(reference_square);
                if (!std::isfinite(defect_square) || !std::isfinite(reference_square))
                {
                    throw std::runtime_error("LR phase covariance diagnostic is non-finite");
                }
                double defect = std::sqrt(defect_square);
                if (reference_square > 0.0)
                {
                    defect /= std::sqrt(reference_square);
                }
                print_eigs(
                    std::vector<double>{defect},
                    "LR relative phase-covariance defect",
                    1.0,
                    17);
            }

            // normalization is already satisfied
            // std::cout << "check normalization of eigenvectors:" << std::endl;
            // for (int ist = 0;ist < nband;++ist)
            // {
            //     double norm2 = 0;
            //     for (int ik = 0;ik < psi.get_nk();++ik)
            //     {
            //         for (int ib = 0;ib < psi.get_nbasis();++ib)
            //         {
            //             norm2 += std::norm(psi(ist, ik, ib));
            //             // std::cout << "norm2_now=" << norm2 << std::endl;
            //         }
            //     }
            //     std::cout << "state " << ist << ", norm2=" << norm2 << std::endl;
            // }

            // output iters
            std::cout << " Average iterative diagonalization steps: " << hsolver::DiagoIterAssist<T>::avg_iter
                << "; current threshold: " << diag_ethr << std::endl;
            ModuleBase::timer::end("HSolverLR", "solve");
            return residuals;
        }
        template std::vector<double> solve<double, HamiltLR<double>>(
            const HamiltLR<double> &,
            double *,
            const int &,
            const int &,
            const int &,
            const std::vector<int> &,
            const std::vector<int> &,
            const std::vector<Parallel_2D> &,
            double *,
            const std::string,
            const double &,
            const std::vector<double> &,
            const DiagonalizationOptions &,
            const bool);
        template std::vector<double> solve<double, HamiltULR<double>>(
            const HamiltULR<double> &,
            double *,
            const int &,
            const int &,
            const int &,
            const std::vector<int> &,
            const std::vector<int> &,
            const std::vector<Parallel_2D> &,
            double *,
            const std::string,
            const double &,
            const std::vector<double> &,
            const DiagonalizationOptions &,
            const bool);
        template std::vector<double> solve<std::complex<double>, HamiltLR<std::complex<double>>>(
            const HamiltLR<std::complex<double>> &,
            std::complex<double> *,
            const int &,
            const int &,
            const int &,
            const std::vector<int> &,
            const std::vector<int> &,
            const std::vector<Parallel_2D> &,
            double *,
            const std::string,
            const double &,
            const std::vector<double> &,
            const DiagonalizationOptions &,
            const bool);
        template std::vector<double> solve<std::complex<double>, HamiltULR<std::complex<double>>>(
            const HamiltULR<std::complex<double>> &,
            std::complex<double> *,
            const int &,
            const int &,
            const int &,
            const std::vector<int> &,
            const std::vector<int> &,
            const std::vector<Parallel_2D> &,
            double *,
            const std::string,
            const double &,
            const std::vector<double> &,
            const DiagonalizationOptions &,
            const bool);
    }
}
