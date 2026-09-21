#ifndef FSSH_DRIVER_H_
#define FSSH_DRIVER_H_

#include <complex>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

#include "source_base/complexmatrix.h"

class UnitCell;
struct FsshDriverTestAccess;

/// @brief Structure to hold LR-TDDFT Casida equation results for a single excitation.
struct CasidaWavefunction {
    double omega = 0.0;           ///< Excitation energy in Hartree
    std::vector<double> X_coeffs; ///< Casida X coefficients (excitations)
    std::vector<double> Y_coeffs; ///< Casida Y coefficients (de-excitations)
    int nocc_lr = 0;              ///< LR module's actual nocc (may differ from KS nocc)
    int nvirt_lr = 0;             ///< LR module's actual nvirt (may differ from KS nvirt)

    CasidaWavefunction() = default;
    CasidaWavefunction(double w, std::vector<double> x, std::vector<double> y,
                       int nolr = 0, int nvlr = 0)
        : omega(w), X_coeffs(std::move(x)), Y_coeffs(std::move(y)),
          nocc_lr(nolr), nvirt_lr(nvlr) {}
};

/// @brief External data for one deterministic FSSH replay step.
/// @details All quantities are in atomic units.  The time-NAC matrix must be
/// anti-Hermitian and expressed in the same continuous adiabatic gauge as the
/// electronic coefficients.  The active-state indices are local indices in
/// the replay state list, not physical excitation labels such as S1 or S3.
struct FsshReplayStepInput {
    std::vector<double> energies;                    ///< State energies in Hartree.
    ModuleBase::ComplexMatrix time_nac;              ///< Time NAC, <Phi_i|d/dt|Phi_j>.
    double kinetic_energy = 0.0;                     ///< Post-hop nuclear kinetic energy in Hartree.
    double random_number = 0.0;                      ///< Prescribed uniform variate in [0, 1).
    int active_state_before = 0;                     ///< Reference active state before hop handling.
    int active_state_after = 0;                      ///< Reference active state after hop handling.
};

/// @brief Diagnostics produced by one deterministic FSSH replay step.
struct FsshReplayStepResult {
    int active_state_before = 0;                     ///< Active state used for propagation and hopping.
    int proposed_state = 0;                          ///< State selected by ABACUS probabilities and random number.
    int active_state_after = 0;                      ///< Reference-controlled state used for decoherence.
    std::vector<double> hopping_probabilities;       ///< Fewest-switches probabilities before decoherence.
    std::vector<std::complex<double>> coefficients_before_decoherence; ///< RK4 result before damping.
    std::vector<std::complex<double>> coefficients_after_decoherence;  ///< Result after damping/renormalization.
    std::vector<double> decoherence_tau;             ///< Inactive-state decoherence times in atomic units.
    std::vector<double> decoherence_damping;         ///< Inactive-state amplitude damping factors.
};

/// @brief Result of the uniform nuclear-velocity rescaling used after a FSSH hop.
struct FsshVelocityRescaleResult {
    bool accepted = false;                    ///< Whether uniform rescaling can represent the hop.
    double scale_factor = 1.0;                ///< Factor applied to every nuclear velocity component.
    double kinetic_energy_after = 0.0;        ///< Kinetic energy after rescaling in Hartree.
    double total_energy_residual = 0.0;       ///< (K' + E_new) - (K + E_old) in Hartree.
};

/// @brief Fewest Switches Surface Hopping (FSSH) driver.
/// Handles the non-adiabatic couplings, wavepacket propagation,
/// and stochastic surface hopping logic.
class FsshDriver {
public:
    FsshDriver();
    ~FsshDriver();

    // [FSSH修改说明] init 函数增加 nocc 和 nks_total 参数
    // 原代码: void init(int nbasis, int nstates, double dt, int current_state_index);
    // 修改原因: TDA NAC 计算需要知道占据轨道数(nocc)和总KS轨道数(nks_total)
    //   以区分占据-虚轨道子空间，并正确索引 Casida X 系数
    /// @brief Initializes the FSSH engine.
    /// @param nocc Number of occupied KS orbitals (needed for TDA NAC).
    /// @param nks_total Total number of KS bands available (occ + virt, needed for full MO overlap).
    /// @param enable_decoherence Whether to apply decoherence correction after
    /// the current-step hop decision and velocity rescaling.
    void init(int nbasis, int nstates, double dt, int current_state_index,
              int nocc = 0, int nks_total = 0, bool enable_decoherence = false,
              unsigned int random_seed = 42, double degen_energy_threshold = 1.0e-3);

    /// @brief Replace the active surface and electronic amplitudes for an external replay segment.
    /// @param active_state Zero-based active-state index at the segment boundary.
    /// @param coefficients Normalized complex amplitudes in the replay-state ordering.
    /// @throw std::runtime_error if the driver is uninitialized, the state index or vector size is invalid,
    /// or the supplied amplitudes are not finite and normalized.
    void set_replay_state(int active_state, const std::vector<std::complex<double>>& coefficients);

    /// @brief Save or restore all electronic FSSH state required for a deterministic restart.
    /// @param stream Checkpoint stream opened by the caller.
    /// @return True if the complete checkpoint was written or read successfully.
    bool save_checkpoint(std::ostream& stream) const;
    bool load_checkpoint(std::istream& stream);

    /// @brief Caches the Casida wavefunction data from current step for use in the next step's TDA NAC.
    void set_casida_cache(const std::vector<CasidaWavefunction>& wfcs);

    /// @brief Executes a single FSSH step with optional TDDFT energy substitution.
    int run_step_advanced(const ModuleBase::ComplexMatrix& coef_old,
                          const ModuleBase::ComplexMatrix& coef_new,
                          const std::string& csr_file,
                          const std::vector<double>& energies,
                          UnitCell& ucell,
                          bool use_tddft = false,
                          const std::vector<double>& tddft_energies = std::vector<double>(),
                          const std::vector<CasidaWavefunction>& casida_wfcs = std::vector<CasidaWavefunction>());

    /// @brief Propagates one FSSH electronic step using externally supplied replay data.
    /// @details This test-only interface bypasses orbital-overlap and electronic-structure
    /// calculations.  It calculates the ABACUS RK4 coefficients and hop probabilities,
    /// then follows @p active_state_after so that the reference nuclear path remains fixed
    /// while the post-hop decoherence operation is compared step by step.
    /// @param input External energies, time-NAC, kinetic energy, random number, and states.
    /// @return Coefficients, probabilities, and decoherence diagnostics for this replay step.
    /// @throw std::runtime_error if dimensions, state continuity, or units implied by the
    /// input contract are invalid.
    FsshReplayStepResult run_step_replay(const FsshReplayStepInput& input);

    /// @brief Evaluate the production uniform velocity-rescaling rule without a unit cell.
    /// @details This scalar interface shares the same acceptance and scale-factor
    /// calculation as production hopping.  It is provided for deterministic energy
    /// conservation audits when the original per-atom pre-hop velocity is unavailable.
    /// A nonzero energy release cannot be represented by uniform scaling from exactly
    /// zero kinetic energy, so that boundary case is rejected rather than producing NaN.
    /// @param kinetic_energy_before Nuclear kinetic energy before the proposed hop in Hartree.
    /// @param old_energy Active-surface energy before the proposed hop in Hartree.
    /// @param new_energy Target-surface energy after the proposed hop in Hartree.
    /// @return Acceptance, scale factor, post-hop kinetic energy and total-energy residual.
    /// @throw std::invalid_argument if an input energy is non-finite or kinetic energy is negative.
    FsshVelocityRescaleResult evaluate_velocity_rescale(double kinetic_energy_before,
                                                         double old_energy,
                                                         double new_energy) const;

    /// @brief Gets the current active electronic state index.
    int get_current_state() const { return current_state_; }

    /// @brief Gets the number of electronic states in the FSSH model.
    int get_nstates() const { return nstates_; }

    // Standard Tully Model Tests
    void test_math_1();
    void test_math_2();
    void test_math_3();
    void test_math_4();
    void test_math_5();
    void test_tully_model_1();
    void test_tully_model_1_scan();
    void test_tully_model_2();
    void test_tully_model_2_scan();
    void test_tully_model_3();
    void test_tully_model_3_scan();

    /// @brief Runs a 6-part Löwdin validation suite and writes one report per scheme.
    /// @details
    ///   The suite cross-validates the Löwdin CIS/TDA overlap implementation against:
    ///   1) explicit determinant reference,
    ///   2) algebraic identities,
    ///   3) limit behavior,
    ///   4) convergence toward diagonal approximation in weak-coupling limit,
    ///   5) phase/gauge robustness,
    ///   6) numerical stability diagnostics.
    /// @param output_prefix Prefix for report filenames.
    void run_lowdin_validation_suite(const std::string& output_prefix = "lowdin_validation");
    void run_lowdin_validation_suite_8_5(const std::string& output_prefix = "lowdin_validation_8_5");

private:
    friend struct FsshDriverTestAccess;

    int nstates_;
    int nbasis_;
    double dt_;
    int current_state_;

    // [FSSH修改说明] 新增 TDA 所需的轨道子空间信息
    // 原代码: (无此成员变量)
    // 修改原因: TDA NAC 需要将 KS 轨道分为占据/虚子空间，
    //   并在全 KS 轨道空间上计算 MO 重叠矩阵
    int nocc_;       ///< Number of occupied KS orbitals
    int nks_total_;  ///< Total number of KS bands (occ + virt)
    bool decoherence_enabled_; ///< Whether to damp inactive-state amplitudes after propagation
    double degen_energy_threshold_; ///< Maximum energy gap (Ha) for Procrustes alignment

    // [FSSH修改说明] 新增 Casida 波函数缓存
    // 原代码: (无此成员变量)
    // 修改原因: TDA NAC 公式需要前后两个时间步的 X 系数:
    //   Σ_{IJ}(t,t') = Σ_i Σ_{ab} X^{I}_{ia}(t) S^{MO}_{a+nocc,b+nocc} X^{J}_{ib}(t')
    //   必须缓存上一步的 casida_wfcs 供下一步使用
    std::vector<CasidaWavefunction> casida_wfcs_old_;

    std::vector<std::complex<double>> electronic_coeffs_;

    std::mt19937 gen_;
    std::uniform_real_distribution<double> rand_dist_;

    /// @brief Parses the asynchronous overlap matrix from a CSR file.
    ModuleBase::ComplexMatrix parse_latest_csr(const std::string& filename);

    // [FSSH修改说明] calculate_nac_from_dense 增加 use_tddft 参数及 Casida 波函数数据
    // 原代码:
    //   void calculate_nac_from_dense(coef_old, coef_new, s_ao_dense, sigma_out);
    // 修改原因: 当 use_tddft=true 时，NAC 不再是 KS 单粒子轨道之间的耦合，
    //   而是多体 TDA 态之间的耦合，需要 Casida X 系数参与计算
    // -----------------------------------------------------------------------
    // 数学公式对比:
    //
    // 【KSDFT NAC】(use_tddft = false)
    //   态 = KS 单粒子轨道 φ_i
    //   S^{MO}_{ij} = Σ_{μν} C*_{iμ}(t) S^{AO}_{μν} C_{jν}(t+Δt)
    //   σ^{KS}_{ij} = [S^{MO}_{ij} - (S^{MO}_{ji})*] / (2Δt)
    //
    // 【TDA NAC】(use_tddft = true)
    //   态 = 多体态: 基态 Φ_0 = |φ_1...φ_{nocc}|,
    //                激发态 Ψ_I = Σ_{ia} X^I_{ia} c†_a c_i Φ_0
    //
    //   Step 1: 计算全 KS 轨道的 MO 重叠 (p,q 遍历所有占据+虚轨道):
    //     S^{MO}_{pq} = Σ_{μν} C*_{pμ}(t) S^{AO}_{μν} C_{qν}(t+Δt)
    //
    //   Step 2: 构造多体态重叠矩阵 Σ_{IJ}:
    //     基态-基态:  Σ_{00} ≈ det(S^{oo}) ≈ 1
    //     基态-激发态: Σ_{0I}(t,t') = Σ_{ia} X^I_{ia}(t') · S^{MO}_{i, nocc+a}
    //     激发态-基态: Σ_{I0}(t,t') = Σ_{ia} X^I_{ia}(t) · S^{MO}_{nocc+a, i}
    //     激发态-激发态 (核心公式):
    //       Σ_{IJ}(t,t') = Σ_i Σ_{ab} X^I_{ia}(t) · S^{MO}_{nocc+a, nocc+b} · X^J_{ib}(t')
    //       (利用 <Φ^a_i|Φ^b_j> ≈ δ_{ij} S^{MO}_{nocc+a,nocc+b} 近似)
    //
    //   Step 3: 反对称化得 NAC:
    //     σ^{TDA}_{IJ} = [Σ_{IJ} - (Σ_{JI})*] / (2Δt)
    //
    // 【关键区别】
    //   KS NAC: 单粒子图像，σ_{ij} 是轨道 φ_i 与 φ_j 间的耦合
    //   TDA NAC: 多体图像，σ_{IJ} 是多体态 Ψ_I 与 Ψ_J 间的耦合，
    //     通过 Casida 系数 X^I_{ia} 将单粒子 MO 重叠"投影"到多体态空间
    //     激发态-激发态耦合仅涉及虚轨道子空间的重叠 S^{MO}_{ab}，
    //     而 KS NAC 直接计算轨道间的全空间重叠
    // -----------------------------------------------------------------------
    /// @brief Calculates Non-Adiabatic Couplings (NAC) from dense overlap matrices.
    /// @param use_tddft If true, compute NAC between TDA many-body states using Casida X coefficients;
    ///                  if false, compute NAC between KS single-particle orbitals (original behavior).
    /// @param casida_new_aligned_out If non-null and use_tddft=true, returns an aligned copy
    ///   of casida_new where the same phase corrections + degenerate-subspace SVD rotations
    ///   used internally to fix Sigma have also been applied to the X coefficients. The
    ///   caller should cache this aligned version (not the raw casida_new) so that the next
    ///   step's "old" cache stays in the same gauge as the propagated electronic coefficients.
    void calculate_nac_from_dense(const ModuleBase::ComplexMatrix& coef_old,
                                  const ModuleBase::ComplexMatrix& coef_new,
                                  const ModuleBase::ComplexMatrix& s_ao_dense,
                                  ModuleBase::ComplexMatrix& sigma_out,
                                  bool use_tddft = false,
                                  const std::vector<CasidaWavefunction>& casida_old = {},
                                  const std::vector<CasidaWavefunction>& casida_new = {},
                                  std::vector<CasidaWavefunction>* casida_new_aligned_out = nullptr);

    /// @brief Computes the time derivative of the electronic coefficients.
    std::vector<std::complex<double>> compute_derivative(
        const std::vector<std::complex<double>>& coeffs,
        const std::vector<double>& energies,
        const ModuleBase::ComplexMatrix& sigma);

    /// @brief Propagates the electronic wave function using the Runge-Kutta 4th order method.
    void propagate_rk4(const ModuleBase::ComplexMatrix& sigma,
                       const std::vector<double>& energies);

    /// @brief Applies an energy-gap-based decoherence correction to inactive electronic states.
    /// @param energies Electronic-state energies in Hartree.
    /// @param ucell Unit cell containing current nuclear velocities and masses.
    void apply_decoherence_correction(const std::vector<double>& energies,
                                      const UnitCell& ucell);

    /// @brief Applies the production energy-based decoherence formula for a supplied kinetic energy.
    /// @param energies Electronic-state energies in Hartree.
    /// @param kinetic_energy_hartree Nuclear kinetic energy after hop handling in Hartree.
    /// @param tau_out Optional inactive-state decoherence times.
    /// @param damping_out Optional inactive-state damping factors.
    void apply_decoherence_with_kinetic_energy(const std::vector<double>& energies,
                                               double kinetic_energy_hartree,
                                               std::vector<double>* tau_out = nullptr,
                                               std::vector<double>* damping_out = nullptr);

    /// @brief Computes the fewest-switches probabilities from the current coefficients.
    std::vector<double> compute_hopping_probabilities(const ModuleBase::ComplexMatrix& sigma) const;

    /// @brief Selects a target state using a prescribed uniform random number.
    int select_hopping_target(const std::vector<double>& probabilities, double random_number) const;

    /// @brief Determines if a surface hop should occur based on fewest-switches criteria.
    int check_hopping(const ModuleBase::ComplexMatrix& sigma);

    /// @brief Rescales atomic velocities to conserve total energy after a successful hop.
    bool rescale_velocity(UnitCell& ucell, int old_state, int new_state, const std::vector<double>& energies);

    // -----------------------------------------------------------------------
    // Löwdin 精确 CIS/TDA 多体态重叠
    // -----------------------------------------------------------------------
    // 精确公式 (Malmqvist 1986, Plasser et al. 2016):
    //   Σ_{IJ} = det(S^{oo}) × Σ_{ij} [(S^{oo})^{-1}]_{ji} × Σ_{ab} X^I_{ia} S̃_{ab} X^J_{jb}
    //   其中 S̃ = S^{vv} - S^{vo} (S^{oo})^{-1} S^{ov} 是 Schur 补
    //
    // 与对角近似的区别:
    //   对角近似: Σ_{IJ} ≈ Σ_i Σ_{ab} X^I_{ia} S^{MO}_{ab} X^J_{ib}  (仅 i=j 项)
    //   Löwdin:   通过 (S^{oo})^{-1}_{ji} 耦合不同占据轨道 i≠j
    //             通过 Schur 补 S̃ 修正虚轨道重叠 (包含占据-虚交叉项)
    //             通过 det(S^{oo}) 修正基态归一化
    // -----------------------------------------------------------------------
    /// @brief Computes the exact Löwdin CIS/TDA overlap matrix between many-body states.
    /// @param Sigma_lowdin [out] The exact CIS overlap matrix (nstates_ x nstates_)
    /// @param occ_phase Legacy occupied-orbital sign input. It is deliberately not
    ///        applied inside this Löwdin contraction: the MO coefficients and CI
    ///        amplitudes must remain in their common gauge.
    /// @param det_Soo_out [out] If non-null, returns det(S^{oo}) for diagnostic purposes.
    void compute_lowdin_sigma(const ModuleBase::ComplexMatrix& coef_old,
                              const ModuleBase::ComplexMatrix& coef_new,
                              const ModuleBase::ComplexMatrix& s_ao_dense,
                              const std::vector<CasidaWavefunction>& casida_old,
                              const std::vector<CasidaWavefunction>& casida_new,
                              int nocc_lr, int nvirt_lr, int occ_offset,
                              const std::vector<double>& occ_phase,
                              ModuleBase::ComplexMatrix& Sigma_lowdin,
                              double* det_Soo_out = nullptr);
};

#endif // FSSH_DRIVER_H_
