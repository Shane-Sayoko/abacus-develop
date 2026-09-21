#ifndef FSSH_MD_H_
#define FSSH_MD_H_

#include "md_base.h"
#include "source_lcao/module_operator_lcao/fssh_driver.h"
#include "source_io/module_parameter/parameter.h"

// [FSSH修改说明] 增加宏定义以控制是否使用有限差分(FD)受力
// 1: 调用外部 abacus-fd 工具获取当前活性态的数值受力 (物理上更合理但极慢)
// 0: 使用 ESolver 返回的受力 (目前 ABACUS 仅支持基态受力)
#ifndef FSSH_USE_FD_FORCE
#define FSSH_USE_FD_FORCE 1
#endif

// [FSSH修改说明] 增加宏定义以控制是否使用外部工具获取电子态(SCF+LR-TDDFT)
#ifndef FSSH_USE_FD_STATES
#define FSSH_USE_FD_STATES 1
#endif

/// @brief Fewest Switches Surface Hopping (FSSH) Molecular Dynamics Class.
/// Inherits from standard MD_base, utilizing Verlet integration for nuclei
/// combined with stochastic electronic quantum jumps.
class FsshMD : public MD_base {
public:
    FsshMD(const Parameter& param_in, UnitCell& unit_in);
    virtual ~FsshMD();

    /// @brief Restore the FSSH electronic checkpoint before MD setup when requested.
    void setup(ModuleESolver::ESolver* p_esolver, const std::string& global_readin_dir) override;

    /// @brief Persist the FSSH electronic state together with the standard MD restart data.
    void write_restart(const std::string& global_out_dir) override;

    /// @brief Standard Verlet-style first half integration
    void first_half(std::ofstream& ofs) override;

    /// @brief Standard Verlet-style second half integration
    void second_half() override;

    /// @brief The core Quantum-Classical interface hooking into ESolver
    void execute_hopping(ModuleESolver::ESolver* p_esolver, const Parameter& param_in);

private:
    FsshDriver fssh_engine_;                 // The encapsulated FSSH physical engine
    ModuleBase::ComplexMatrix coef_old_;     // Cache for previous electronic wavefunctions
    std::vector<double> ks_energies_cache_;  // KS eigenvalues paired with coef_old_ (Hartree)
    std::vector<double> tddft_energies_cache_; // Cache for absolute electronic energies (Hartree)
    bool fssh_initialized_;                  // Flag for initial setup
    bool skip_hopping_once_ = false;         ///< Skip setup-time propagation after a restored checkpoint
    std::vector<ModuleBase::Vector3<double>> restored_force_; ///< force at checkpoint for the first resumed half-step
    double restored_potential_ = 0.0;
    bool has_restored_force_ = false;

    // Velocity synchronization between MD_base flat arrays and UnitCell structured arrays
    void sync_vel_to_ucell();
    void sync_vel_from_ucell();
    void restart(const std::string& global_readin_dir) override;
    [[noreturn]] void checkpoint_and_fail(const std::string& message, const Parameter& param_in);

    /// @brief Hijacks the states calculation by calling the external abacus-fd tool.
    void update_states_kslr_fd(const Parameter& param_in, ModuleESolver::ESolver* p_esolver,
                               std::vector<double>& tddft_energies,
                               std::vector<CasidaWavefunction>& casida_wfcs);

    /// @brief Hijacks the force calculation by calling the external abacus-fd tool.
    /// @details Dumps current STRU, runs abacus-fd via system call, parses excited_forces.txt,
    ///          and overwrites MD_base::force with the active state's numerical force.
    void update_force_active_state_fd(const Parameter& param_in);
};

#endif // FSSH_MD_H_
