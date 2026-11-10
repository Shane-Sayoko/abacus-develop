#include "fssh_state_overlap.h"

#include "source_base/module_container/ATen/core/tensor.h"
#include "source_base/module_external/blas_connector.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

extern "C"
{
    void dgetrf_(
        const int *m,
        const int *n,
        double *a,
        const int *lda,
        int *ipiv,
        int *info);
    void dgetri_(
        const int *n,
        double *a,
        const int *lda,
        const int *ipiv,
        double *work,
        const int *lwork,
        int *info);
}

namespace LR
{
std::vector<double> contract_fssh_mo_overlap(
    const ModuleBase::ComplexMatrix &coef_old,
    const ModuleBase::ComplexMatrix &coef_new,
    const ModuleBase::ComplexMatrix &s_ao_dense,
    const std::vector<int> &orbital_indices,
    int nbasis)
{
    const int nmo = static_cast<int>(orbital_indices.size());
    std::vector<double> overlap(static_cast<size_t>(nmo) * nmo, 0.0);
    if (nmo == 0 || nbasis == 0)
    {
        return overlap;
    }

    // Orbital rows are contiguous and therefore represent AO-by-MO columns for BLAS.
    container::Tensor old_coefficients(container::DataType::DT_COMPLEX_DOUBLE, {nmo, nbasis});
    container::Tensor new_coefficients(container::DataType::DT_COMPLEX_DOUBLE, {nmo, nbasis});
    old_coefficients.zero();
    new_coefficients.zero();
    auto *old_data = old_coefficients.data<std::complex<double>>();
    auto *new_data = new_coefficients.data<std::complex<double>>();
    for (int p = 0; p < nmo; ++p)
    {
        const int orbital = orbital_indices[p];
        for (int mu = 0; mu < nbasis; ++mu)
        {
            const size_t offset = static_cast<size_t>(p) * nbasis + mu;
            if (orbital >= 0 && orbital < coef_old.nr)
            {
                old_data[offset] = coef_old(orbital, mu);
            }
            if (orbital >= 0 && orbital < coef_new.nr)
            {
                new_data[offset] = coef_new(orbital, mu);
            }
        }
    }

    container::Tensor ao_times_new(container::DataType::DT_COMPLEX_DOUBLE, {nmo, nbasis});
    container::Tensor mo_overlap(container::DataType::DT_COMPLEX_DOUBLE, {nmo, nmo});
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    // S is row-major: transpose its column-major view without conjugating it.
    // U = S C_new^T, followed by M = conj(C_old) U. Keep U complex.
    BlasConnector::gemm_cm(
        'T',
        'N',
        nbasis,
        nmo,
        nbasis,
        one,
        s_ao_dense.c,
        s_ao_dense.nc,
        new_data,
        nbasis,
        zero,
        ao_times_new.data<std::complex<double>>(),
        nbasis);
    BlasConnector::gemm_cm(
        'C',
        'N',
        nmo,
        nmo,
        nbasis,
        one,
        old_data,
        nbasis,
        ao_times_new.data<std::complex<double>>(),
        nbasis,
        zero,
        mo_overlap.data<std::complex<double>>(),
        nmo);
    const auto *mo_data = mo_overlap.data<std::complex<double>>();
    for (int p = 0; p < nmo; ++p)
    {
        for (int q = 0; q < nmo; ++q)
        {
            overlap[static_cast<size_t>(p) * nmo + q] = mo_data[p + static_cast<size_t>(q) * nmo].real();
        }
    }
    return overlap;
}

ModuleBase::ComplexMatrix closed_shell_fssh_overlap(
    const FsshSpinChannel spin_channel,
    const ModuleBase::ComplexMatrix &coef_old,
    const ModuleBase::ComplexMatrix &coef_new,
    const ModuleBase::ComplexMatrix &s_ao_dense,
    const std::vector<CasidaWavefunction> &casida_old,
    const std::vector<CasidaWavefunction> &casida_new,
    int occupied_total,
    int state_count,
    int nocc_lr,
    int nvirt_lr,
    int occ_offset,
    double *determinant)
{
    const int nbasis = coef_old.nc;
    ModuleBase::ComplexMatrix Sigma_lowdin(state_count, state_count);

    const int n_lr = nocc_lr + nvirt_lr;

    // ---- 辅助函数: LR 局部索引 → KS 全局索引 ----
    // 占据: loc ∈ [0, nocc_lr) → KS: occ_offset + loc
    // 虚:   loc ∈ [nocc_lr, n_lr) → KS: occupied_total + (loc - nocc_lr)
    auto local_to_ks = [&](int loc) -> int
    {
        if (loc < nocc_lr)
        {
            return occ_offset + loc;
        }
        else
        {
            return occupied_total + (loc - nocc_lr);
        }
    };

    // ================================================================
    // Step 1: 构建完整 MO 重叠矩阵 S^{MO}_{pq}
    //   p,q 遍历 LR 子空间: [0, nocc_lr) = 占据, [nocc_lr, n_lr) = 虚
    //   S^{MO}_{pq} = Σ_{μν} C*_{p_ks,μ}(t) S^{AO}_{μν} C_{q_ks,ν}(t')
    //   保留 MO 与 Casida 系数的原始共同规范。
    // ================================================================
    std::vector<int> orbital_indices(n_lr);
    for (int p = 0; p < n_lr; ++p)
    {
        orbital_indices[p] = local_to_ks(p);
    }
    const std::vector<double> s_mo = contract_fssh_mo_overlap(
        coef_old,
        coef_new,
        s_ao_dense,
        orbital_indices,
        nbasis);

    // ---- 辅助宏: 从 s_mo 提取子块 ----
    // S^{oo}_{ij} = s_mo[i * n_lr + j]                  i,j ∈ [0, nocc_lr)
    // S^{ov}_{ib} = s_mo[i * n_lr + nocc_lr + b]        i ∈ [0, nocc_lr), b ∈ [0, nvirt_lr)
    // S^{vo}_{aj} = s_mo[(nocc_lr + a) * n_lr + j]      a ∈ [0, nvirt_lr), j ∈ [0, nocc_lr)
    // S^{vv}_{ab} = s_mo[(nocc_lr + a) * n_lr + nocc_lr + b]
    auto Soo = [&](int i, int j) -> double { return s_mo[i * n_lr + j]; };
    auto Sov = [&](int i, int b) -> double { return s_mo[i * n_lr + nocc_lr + b]; };
    auto Svo = [&](int a, int j) -> double { return s_mo[(nocc_lr + a) * n_lr + j]; };
    auto Svv = [&](int a, int b) -> double { return s_mo[(nocc_lr + a) * n_lr + nocc_lr + b]; };

    // ================================================================
    // Step 2: LU 分解 S^{oo}，计算 det(S^{oo}) 和 (S^{oo})^{-1}
    // ================================================================
    std::vector<double> Soo_cm(nocc_lr * nocc_lr); // 列主序 (LAPACK)
    for (int i = 0; i < nocc_lr; ++i)
    {
        for (int j = 0; j < nocc_lr; ++j)
        {
            Soo_cm[i + j * nocc_lr] = Soo(i, j); // col-major: [row + col * lda]
        }
    }

    double norm_soo = 0.0;
    for (int i = 0; i < nocc_lr; ++i)
    {
        double row_sum = 0.0;
        for (int j = 0; j < nocc_lr; ++j)
        {
            row_sum += std::abs(Soo(i, j));
        }
        norm_soo = std::max(norm_soo, row_sum);
    }

    std::vector<int> ipiv(nocc_lr);
    int info = 0;
    int nn = nocc_lr;

    // LU 分解: S^{oo} = P L U
    dgetrf_(
        &nn,
        &nn,
        Soo_cm.data(),
        &nn,
        ipiv.data(),
        &info);

    if (info != 0)
    {
        throw std::runtime_error("Löwdin S^{oo} LU factorization failed");
    }

    // 从 LU 对角线提取行列式
    double det_Soo = 1.0;
    int sign_perm = 0;
    for (int i = 0; i < nocc_lr; ++i)
    {
        det_Soo *= Soo_cm[i + i * nocc_lr]; // U 的对角元
        if (ipiv[i] != i + 1)
        {
            sign_perm++; // LAPACK ipiv 是 1-based
        }
    }
    if (sign_perm % 2 == 1)
    {
        det_Soo = -det_Soo;
    }
    if (determinant)
    {
        *determinant = det_Soo;
    }

    // 求逆: (S^{oo})^{-1}，就地覆盖 Soo_cm
    int lwork = -1;
    double wquery;
    dgetri_(
        &nn,
        Soo_cm.data(),
        &nn,
        ipiv.data(),
        &wquery,
        &lwork,
        &info);
    lwork = static_cast<int>(wquery) + 1;
    std::vector<double> work(lwork);
    dgetri_(
        &nn,
        Soo_cm.data(),
        &nn,
        ipiv.data(),
        work.data(),
        &lwork,
        &info);
    // 现在 Soo_cm = (S^{oo})^{-1}，列主序: Soo_inv(i,j) = Soo_cm[i + j * nocc_lr]

    if (info != 0)
    {
        throw std::runtime_error("Löwdin S^{oo} inversion failed");
    }

    double norm_inv_soo = 0.0;
    for (int i = 0; i < nocc_lr; ++i)
    {
        double row_sum = 0.0;
        for (int j = 0; j < nocc_lr; ++j)
        {
            row_sum += std::abs(Soo_cm[i + j * nocc_lr]);
        }
        norm_inv_soo = std::max(norm_inv_soo, row_sum);
    }
    const double condition_number = norm_soo * norm_inv_soo;
    if (!std::isfinite(condition_number) || condition_number > 1.0e10)
    {
        std::ostringstream message;
        message << "Löwdin S^{oo} is ill-conditioned: kappa_inf=" << condition_number << " exceeds 1e10";
        throw std::runtime_error(message.str());
    }

    auto Soo_inv = [&](int i, int j) -> double { return Soo_cm[i + j * nocc_lr]; };

    // ================================================================
    // Step 3: 计算辅助矩阵
    //   Q_{jb} = [(S^{oo})^{-1} S^{ov}]_{jb} = Σ_k Soo_inv(j,k) * S^{ov}(k,b)
    //   R_{ia} = [S^{vo} (S^{oo})^{-1}]_{ai} → 转置存为 R[i * nvirt + a]
    //          = Σ_j S^{vo}(a,j) * Soo_inv(j,i)
    // ================================================================
    std::vector<double> Q(nocc_lr * nvirt_lr, 0.0); // Q[j * nvirt_lr + b]
    for (int j = 0; j < nocc_lr; ++j)
    {
        for (int b = 0; b < nvirt_lr; ++b)
        {
            for (int k = 0; k < nocc_lr; ++k)
            {
                Q[j * nvirt_lr + b] += Soo_inv(j, k) * Sov(k, b);
            }
        }
    }

    std::vector<double> R(nocc_lr * nvirt_lr, 0.0); // R[i * nvirt_lr + a]
    for (int i = 0; i < nocc_lr; ++i)
    {
        for (int a = 0; a < nvirt_lr; ++a)
        {
            for (int j = 0; j < nocc_lr; ++j)
            {
                R[i * nvirt_lr + a] += Svo(a, j) * Soo_inv(j, i);
            }
        }
    }

    // ================================================================
    // Step 4: 计算 Schur 补 S̃^{vv}_{ab} = S^{vv}_{ab} - Σ_i S^{vo}_{ai} Q_{ib}
    //   = S^{vv}_{ab} - [S^{vo} (S^{oo})^{-1} S^{ov}]_{ab}
    // ================================================================
    std::vector<double> Stilde(nvirt_lr * nvirt_lr, 0.0); // 行主序
    for (int a = 0; a < nvirt_lr; ++a)
    {
        for (int b = 0; b < nvirt_lr; ++b)
        {
            double correction = 0.0;
            for (int i = 0; i < nocc_lr; ++i)
            {
                correction += Svo(a, i) * Q[i * nvirt_lr + b];
            }
            Stilde[a * nvirt_lr + b] = Svv(a, b) - correction;
        }
    }

    // ================================================================
    // Step 5: 构建 Sigma_lowdin 矩阵
    // ================================================================

    const double ground_overlap = det_Soo * det_Soo;
    const bool triplet = spin_channel == FsshSpinChannel::triplet;
    const double ground_excitation_factor = triplet ? 0.0 : std::sqrt(2.0);
    std::vector<double> old_ground_contraction(state_count, 0.0);
    std::vector<double> new_ground_contraction(state_count, 0.0);

    // Both alpha and beta occupied determinants enter the ground overlap.
    Sigma_lowdin(0, 0) = std::complex<double>(ground_overlap, 0.0);

    // Ground-to-singlet overlap: sqrt(2) d^2 v_J.
    for (int J = 1; J < state_count; ++J)
    {
        if (J >= static_cast<int>(casida_new.size()))
        {
            continue;
        }
        const auto &X_J = casida_new[J].X_coeffs;
        double val = 0.0;
        for (int j = 0; j < nocc_lr; ++j)
        {
            for (int b = 0; b < nvirt_lr; ++b)
            {
                int idx = j * nvirt_lr + b;
                if (idx < static_cast<int>(X_J.size()))
                {
                    val += Q[j * nvirt_lr + b] * X_J[idx];
                }
            }
        }
        new_ground_contraction[J] = val;
        Sigma_lowdin(0, J) = std::complex<double>(ground_excitation_factor * ground_overlap * val, 0.0);
    }

    // Singlet-to-ground overlap: sqrt(2) d^2 u_I.
    for (int I = 1; I < state_count; ++I)
    {
        if (I >= static_cast<int>(casida_old.size()))
        {
            continue;
        }
        const auto &X_I = casida_old[I].X_coeffs;
        double val = 0.0;
        for (int i = 0; i < nocc_lr; ++i)
        {
            for (int a = 0; a < nvirt_lr; ++a)
            {
                int idx = i * nvirt_lr + a;
                if (idx < static_cast<int>(X_I.size()))
                {
                    val += X_I[idx] * R[i * nvirt_lr + a];
                }
            }
        }
        old_ground_contraction[I] = val;
        Sigma_lowdin(I, 0) = std::complex<double>(ground_excitation_factor * ground_overlap * val, 0.0);
    }

    // 激发态→激发态:
    //   Sigma_IJ = d^2 [sum_ij Soo_inv(j,i) M^{IJ}_{ij} + 2 u_I v_J] for singlets.
    //   For triplets, opposite-spin determinant products cancel the u_I v_J term.
    //   其中 M^{IJ}_{ij} = Σ_{ab} X^I_{ia} S̃_{ab} X^J_{jb}
    //
    // 优化: 先预计算 T^J_{ja} = Σ_b S̃_{ab} X^J_{jb}
    //        再 M^{IJ}_{ij} = Σ_a X^I_{ia} T^J_{ja}
    for (int J = 1; J < state_count; ++J)
    {
        if (J >= static_cast<int>(casida_new.size()))
        {
            continue;
        }
        const auto &X_J = casida_new[J].X_coeffs;

        // T^J[j * nvirt_lr + a] = Σ_b S̃_{ab} X^J_{jb}
        std::vector<double> TJ(nocc_lr * nvirt_lr, 0.0);
        for (int j = 0; j < nocc_lr; ++j)
        {
            for (int a = 0; a < nvirt_lr; ++a)
            {
                for (int b = 0; b < nvirt_lr; ++b)
                {
                    int idx_jb = j * nvirt_lr + b;
                    if (idx_jb < static_cast<int>(X_J.size()))
                    {
                        TJ[j * nvirt_lr + a] += Stilde[a * nvirt_lr + b] * X_J[idx_jb];
                    }
                }
            }
        }

        for (int I = 1; I < state_count; ++I)
        {
            if (I >= static_cast<int>(casida_old.size()))
            {
                continue;
            }
            const auto &X_I = casida_old[I].X_coeffs;

            // M^{IJ}_{ij} = Σ_a X^I_{ia} T^J_{ja}
            // The Schur term is shared by the two same-spin excitations.
            double sigma_sum = 0.0;
            for (int i = 0; i < nocc_lr; ++i)
            {
                for (int j = 0; j < nocc_lr; ++j)
                {
                    double m_ij = 0.0;
                    for (int a = 0; a < nvirt_lr; ++a)
                    {
                        int idx_ia = i * nvirt_lr + a;
                        if (idx_ia < static_cast<int>(X_I.size()))
                        {
                            m_ij += X_I[idx_ia] * TJ[j * nvirt_lr + a];
                        }
                    }
                    sigma_sum += Soo_inv(j, i) * m_ij;
                }
            }
            // Form the singlet correction directly without dividing by a possibly tiny d^2.
            const double cross_term = triplet ? 0.0 : 2.0 * old_ground_contraction[I] * new_ground_contraction[J];
            Sigma_lowdin(I, J) = std::complex<double>(ground_overlap * (sigma_sum + cross_term), 0.0);
        }
    }

    return Sigma_lowdin;
}
} // namespace LR
