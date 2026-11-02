#ifndef FSSH_DRIVER_H_
#define FSSH_DRIVER_H_

#include "source_base/complexmatrix.h"
#include "source_lcao/module_lr/utils/fssh_electronic_frame.h"

#include <complex>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

class MDCell;
struct FsshDriverTestAccess;

/// @brief External data for one deterministic FSSH replay step.
/// @details All quantities are in atomic units.  The time-NAC matrix must be
/// anti-Hermitian and expressed in the same continuous adiabatic gauge as the
/// electronic coefficients.  The active-state indices are local indices in
/// the replay state list, not physical excitation labels such as S1 or S3.
struct FsshReplayStepInput
{
    std::vector<double> energies;       ///< State energies in Hartree.
    ModuleBase::ComplexMatrix time_nac; ///< Time NAC, <Phi_i|d/dt|Phi_j>.
    double kinetic_energy = 0.0;        ///< Post-hop nuclear kinetic energy in Hartree.
    double random_number = 0.0;         ///< Prescribed uniform variate in [0, 1).
    int active_state_before = 0;        ///< Reference active state before hop handling.
    int active_state_after = 0;         ///< Reference active state after hop handling.
};

/// @brief Diagnostics produced by one deterministic FSSH replay step.
struct FsshReplayStepResult
{
    int active_state_before = 0;               ///< Active state used for propagation and hopping.
    int proposed_state = 0;                    ///< State selected by ABACUS probabilities and random number.
    int active_state_after = 0;                ///< Reference-controlled state used for decoherence.
    std::vector<double> hopping_probabilities; ///< Fewest-switches probabilities before decoherence.
    std::vector<std::complex<double>> coefficients_before_decoherence; ///< RK4 result before damping.
    std::vector<std::complex<double>> coefficients_after_decoherence;  ///< Result after damping/renormalization.
    std::vector<double> decoherence_tau;     ///< Inactive-state decoherence times in atomic units.
    std::vector<double> decoherence_damping; ///< Inactive-state amplitude damping factors.
};

/// @brief Result of the uniform nuclear-velocity rescaling used after a FSSH hop.
struct FsshVelocityRescaleResult
{
    bool accepted = false;              ///< Whether uniform rescaling can represent the hop.
    double scale_factor = 1.0;          ///< Factor applied to every nuclear velocity component.
    double kinetic_energy_after = 0.0;  ///< Kinetic energy after rescaling in Hartree.
    double total_energy_residual = 0.0; ///< (K' + E_new) - (K + E_old) in Hartree.
};

/** @brief Physical state-space configuration for a closed-shell singlet FSSH model. */
struct FsshModelConfig
{
    bool ground_state_coupling = true; ///< False retains S0 as an uncoupled, unoccupied placeholder.
};

/// @brief Fewest Switches Surface Hopping (FSSH) driver.
/// Handles the non-adiabatic couplings, wavepacket propagation,
/// and stochastic surface hopping logic.
class FsshDriver
{
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
    void init(
        int nbasis,
        int nstates,
        double dt,
        int current_state_index,
        int nocc = 0,
        int nks_total = 0,
        bool enable_decoherence = false,
        unsigned int random_seed = 42,
        double degen_energy_threshold = 1.0e-3);

    /**
     * @brief Initialize a driver with an explicitly selected electronic state space.
     * @param[in] model Physical coupling configuration; copied into the driver.
     * @param[in] nbasis Number of AO basis functions.
     * @param[in] nstates Number of surfaces including the S0 placeholder.
     * @param[in] dt Nuclear propagation interval in atomic units.
     * @param[in] current_state_index Zero-based initial surface; must be positive when S0 is excluded.
     * @param[in] nocc Number of occupied spatial orbitals.
     * @param[in] nks_total Number of occupied and virtual KS orbitals.
     * @param[in] enable_decoherence Whether to apply the existing energy-based correction.
     * @param[in] random_seed Seed for the existing hopping generator.
     * @param[in] degen_energy_threshold Excited-state alignment window in Hartree.
     */
    void init(
        const FsshModelConfig &model,
        int nbasis,
        int nstates,
        double dt,
        int current_state_index,
        int nocc,
        int nks_total,
        bool enable_decoherence,
        unsigned int random_seed,
        double degen_energy_threshold);

    /// @brief Replace the active surface and electronic amplitudes for an external replay segment.
    /// @param active_state Zero-based active-state index at the segment boundary.
    /// @param coefficients Normalized complex amplitudes in the replay-state ordering.
    /// @throw std::runtime_error if the driver is uninitialized, the state index or vector size is invalid,
    /// or the supplied amplitudes are not finite and normalized.
    void set_replay_state(int active_state, const std::vector<std::complex<double>> &coefficients);

    /// @brief Save or restore all electronic FSSH state required for a deterministic restart.
    /// @param stream Checkpoint stream opened by the caller.
    /// @return True if the complete checkpoint was written or read successfully.
    bool save_checkpoint(std::ostream &stream) const;
    bool load_checkpoint(std::istream &stream);

    /**
     * @brief Restore a checkpoint only when its physical state space matches the request.
     * @param[in,out] stream Electronic checkpoint stream positioned at its header.
     * @param[in] model Requested physical coupling configuration; v1 requires ground-state coupling.
     * @return True for a complete compatible checkpoint; false leaves the driver unchanged.
     */
    bool load_checkpoint(std::istream &stream, const FsshModelConfig &model);

    /// @brief Caches the Casida wavefunction data from current step for use in the next step's TDA NAC.
    void set_casida_cache(const std::vector<CasidaWavefunction> &wfcs);

    /// @brief Executes a single FSSH step with optional TDDFT energy substitution.
    int run_step_advanced(
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const std::string &csr_file,
        const std::vector<double> &energies,
        MDCell &mdcell,
        bool use_tddft = false,
        const std::vector<double> &tddft_energies = std::vector<double>(),
        const std::vector<CasidaWavefunction> &casida_wfcs = std::vector<CasidaWavefunction>(),
        const std::vector<double> &ks_energies_old = std::vector<double>(),
        ModuleBase::ComplexMatrix *coef_new_aligned_out = nullptr);

    /// @brief Propagates one FSSH electronic step using externally supplied replay data.
    /// @details This test-only interface bypasses orbital-overlap and electronic-structure
    /// calculations.  It calculates the ABACUS RK4 coefficients and hop probabilities,
    /// then follows @p active_state_after so that the reference nuclear path remains fixed
    /// while the post-hop decoherence operation is compared step by step.
    /// @param input External energies, time-NAC, kinetic energy, random number, and states.
    /// @return Coefficients, probabilities, and decoherence diagnostics for this replay step.
    /// @throw std::runtime_error if dimensions, state continuity, or units implied by the
    /// input contract are invalid.
    FsshReplayStepResult run_step_replay(const FsshReplayStepInput &input);

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
    FsshVelocityRescaleResult evaluate_velocity_rescale(
        double kinetic_energy_before, double old_energy, double new_energy) const;

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
    void run_lowdin_validation_suite(const std::string &output_prefix = "lowdin_validation");
    void run_lowdin_validation_suite_8_5(const std::string &output_prefix = "lowdin_validation_8_5");

  private:
    friend struct FsshDriverTestAccess;

    /**
     * @brief Contract selected MO overlaps through two complex matrix products.
     * @param[in] coef_old Old coefficients, with orbital rows and AO columns.
     * @param[in] coef_new New coefficients, with orbital rows and AO columns.
     * @param[in] s_ao_dense Inter-time AO overlap; symmetry is not assumed.
     * @param[in] orbital_indices KS indices in the requested MO order; missing rows contribute zero.
     * @param[in] nbasis Number of AO columns to contract.
     * @return Real parts of the selected MO overlaps in row-major order.
     */
    static std::vector<double> contract_mo_overlap(
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const ModuleBase::ComplexMatrix &s_ao_dense,
        const std::vector<int> &orbital_indices,
        int nbasis);

    /**
     * @brief Apply occupied and virtual rotations as two real matrix products.
     * @param[in,out] amplitudes Row-major X or Y amplitudes; absent or mismatched vectors are unchanged.
     * @param[in] occupied_transform Row-major rotation with solver rows and aligned columns.
     * @param[in] virtual_transform Row-major rotation with solver rows and aligned columns.
     * @param[in] nocc Number of occupied orbitals.
     * @param[in] nvirt Number of virtual orbitals.
     */
    static void transform_casida_amplitudes(
        std::vector<double> &amplitudes,
        const std::vector<double> &occupied_transform,
        const std::vector<double> &virtual_transform,
        int nocc,
        int nvirt);

    int nstates_;
    int nbasis_;
    double dt_;
    int current_state_;

    // [FSSH修改说明] 新增 TDA 所需的轨道子空间信息
    // 原代码: (无此成员变量)
    // 修改原因: TDA NAC 需要将 KS 轨道分为占据/虚子空间，
    //   并在全 KS 轨道空间上计算 MO 重叠矩阵
    int nocc_;                      ///< Number of occupied KS orbitals
    int nks_total_;                 ///< Total number of KS bands (occ + virt)
    bool decoherence_enabled_;      ///< Whether to damp inactive-state amplitudes after propagation
    double degen_energy_threshold_; ///< Maximum energy gap (Ha) for Procrustes alignment
    FsshModelConfig model_config_;  ///< Set only by explicit initialization or validated checkpoint restore.

    /**
     * @brief Project an aligned time-NAC onto the selected physical state space.
     * @param[in,out] time_nac Square matrix in atomic units with S0 at index zero.
     */
    void project_ground_state_nac(ModuleBase::ComplexMatrix &time_nac) const;

    // Previous spatial singlet amplitudes, in the same gauge as the cached MO
    // coefficients and propagated electronic coefficients.
    std::vector<CasidaWavefunction> casida_wfcs_old_;

    std::vector<std::complex<double>> electronic_coeffs_;

    std::mt19937 gen_;
    std::uniform_real_distribution<double> rand_dist_;

    /// @brief Parses the asynchronous overlap matrix from a CSR file.
    ModuleBase::ComplexMatrix parse_latest_csr(const std::string &filename);

    // The KS path contracts the full cross-frame AO overlap into MO overlaps.
    // The closed-shell TDA path additionally contracts both spin sectors using
    // normalized spatial singlet amplitudes; see compute_lowdin_sigma below.
    // All occupied orbitals contribute, including those with zero excitation
    // amplitude. The raw ground-state overlap is det(Soo)^2, and the retained
    // ground-plus-singles overlap need not be unitary. State gauge alignment
    // precedes the finite-difference time coupling (Sigma - Sigma^dagger)/(2 dt).
    /// @brief Calculates Non-Adiabatic Couplings (NAC) from dense overlap matrices.
    /// @param use_tddft If true, compute NAC between TDA many-body states using Casida X coefficients;
    ///                  if false, compute NAC between KS single-particle orbitals (original behavior).
    /// @param casida_new_aligned_out If non-null and use_tddft=true, returns an aligned copy
    ///   of casida_new where the same phase corrections + degenerate-subspace SVD rotations
    ///   used internally to fix Sigma have also been applied to the X coefficients. The
    ///   caller should cache this aligned version (not the raw casida_new) so that the next
    ///   step's "old" cache stays in the same gauge as the propagated electronic coefficients.
    /// @param ks_energies_old/new KS eigenvalues in Hartree.  When both cover the LR
    ///   orbital window, exact/near-degenerate occupied and virtual KS blocks are
    ///   parallel-transported before the TDA overlap is evaluated.
    /// @param coef_new_aligned_out Optional aligned KS coefficients.  The caller must
    ///   cache this result for the next step, together with the aligned Casida data.
    void calculate_nac_from_dense(
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const ModuleBase::ComplexMatrix &s_ao_dense,
        ModuleBase::ComplexMatrix &sigma_out,
        bool use_tddft = false,
        const std::vector<CasidaWavefunction> &casida_old = {},
        const std::vector<CasidaWavefunction> &casida_new = {},
        std::vector<CasidaWavefunction> *casida_new_aligned_out = nullptr,
        const std::vector<double> &ks_energies_old = {},
        const std::vector<double> &ks_energies_new = {},
        ModuleBase::ComplexMatrix *coef_new_aligned_out = nullptr);

    /// @brief Computes the time derivative of the electronic coefficients.
    std::vector<std::complex<double>> compute_derivative(
        const std::vector<std::complex<double>> &coeffs, const std::vector<double> &energies,
        const ModuleBase::ComplexMatrix &sigma);

    /// @brief Propagates the electronic wave function using the Runge-Kutta 4th order method.
    void propagate_rk4(const ModuleBase::ComplexMatrix &sigma, const std::vector<double> &energies);

    /// @brief Applies an energy-gap-based decoherence correction to inactive electronic states.
    /// @param energies Electronic-state energies in Hartree.
    /// @param mdcell Distributed cell containing current nuclear velocities and masses.
    void apply_decoherence_correction(const std::vector<double> &energies, const MDCell &mdcell);

    /// @brief Applies the production energy-based decoherence formula for a supplied kinetic energy.
    /// @param energies Electronic-state energies in Hartree.
    /// @param kinetic_energy_hartree Nuclear kinetic energy after hop handling in Hartree.
    /// @param tau_out Optional inactive-state decoherence times.
    /// @param damping_out Optional inactive-state damping factors.
    void apply_decoherence_with_kinetic_energy(
        const std::vector<double> &energies,
        double kinetic_energy_hartree,
        std::vector<double> *tau_out = nullptr,
        std::vector<double> *damping_out = nullptr);

    /// @brief Computes the fewest-switches probabilities from the current coefficients.
    std::vector<double> compute_hopping_probabilities(const ModuleBase::ComplexMatrix &sigma) const;

    /// @brief Selects a target state using a prescribed uniform random number.
    int select_hopping_target(const std::vector<double> &probabilities, double random_number) const;

    /// @brief Determines if a surface hop should occur based on fewest-switches criteria.
    int check_hopping(const ModuleBase::ComplexMatrix &sigma);

    /// @brief Rescales atomic velocities to conserve total energy after a successful hop.
    bool rescale_velocity(
        MDCell &mdcell,
        int old_state,
        int new_state,
        const std::vector<double> &energies);

    /**
     * @brief Computes raw overlaps between restricted closed-shell singlet CIS/TDA states.
     * @details Uses |I> = sum_ia X_ia (|Phi_i_alpha^a_alpha> + |Phi_i_beta^a_beta>)/sqrt(2).
     * Spatial X has unit Euclidean norm for a normalized state; this contraction
     * is bilinear and does not renormalize it. Both occupied spin determinants
     * enter Sigma_00 = det(S_oo)^2. The projected state overlap need not be unitary.
     * Production FSSH supplies real Gamma orbitals, all occupied bands, and Y=0.
     * @param[in] coef_old Old MO coefficients, band rows and AO columns, dimensionless.
     * @param[in] coef_new New MO coefficients in the same gauge as casida_new.
     * @param[in] s_ao_dense Old/new AO overlap, old AO rows and new AO columns.
     * @param[in] casida_old Old states, ground placeholder first, X indexed i*nvirt_lr+a.
     * @param[in] casida_new New states in the corresponding ordering and convention.
     * @param[in] nocc_lr Occupied spatial-orbital count; production requires nocc_.
     * @param[in] nvirt_lr Number of virtual orbitals in the excitation window.
     * @param[in] occ_offset First occupied KS index; production requires zero.
     * @param[in] occ_phase Legacy sign input, deliberately unused: MO and X must
     * remain in their common gauge.
     * @param[out] Sigma_lowdin Dimensionless nstates_ by nstates_ matrix <old I|new J>.
     * @param[out] det_Soo_out Optional spatial determinant diagnostic, not its square.
     * @pre The occupied overlap block must be invertible and well-conditioned.
     */
    void compute_lowdin_sigma(
        const ModuleBase::ComplexMatrix &coef_old,
        const ModuleBase::ComplexMatrix &coef_new,
        const ModuleBase::ComplexMatrix &s_ao_dense,
        const std::vector<CasidaWavefunction> &casida_old,
        const std::vector<CasidaWavefunction> &casida_new,
        int nocc_lr,
        int nvirt_lr,
        int occ_offset,
        const std::vector<double> &occ_phase,
        ModuleBase::ComplexMatrix &Sigma_lowdin,
        double *det_Soo_out = nullptr);
};

#endif // FSSH_DRIVER_H_
