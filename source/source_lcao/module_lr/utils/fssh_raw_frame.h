#ifndef ABACUS_LR_FSSH_RAW_FRAME_H
#define ABACUS_LR_FSSH_RAW_FRAME_H

#include "fssh_electronic_frame.h"
#include "fssh_local_diabatization.h"

#include <array>
#include <string>

namespace LR
{
/** @brief Geometry and basis identity attached to a raw electronic frame or AO overlap endpoint. */
struct FsshFrameGeometry
{
    int frame_id = -1;                  ///< Unique nonnegative frame identifier within a trajectory.
    double time_au = 0.0;               ///< Physical time in atomic units.
    std::string basis_id;               ///< Nonempty caller-supplied identity of species, PP and AO basis.
    std::array<double, 9> cell_bohr{};  ///< Cartesian lattice vectors, row-major, in bohr.
    std::vector<double> positions_bohr; ///< Owned Cartesian coordinates, atom-major xyz, in bohr.
};

/**
 * @brief Owned, validated real-Gamma singlet TDA frame with read-only access.
 * @details Capture directly from the electronic solver before any orbital/state alignment.
 * Solver-root labels are energy-ordered surfaces: 0 is S0, positive r uses LR root r-1.
 * An array permutation may change storage order but must carry every associated field.
 * The type cannot certify that independently supplied MO and X came from the same solve;
 * that provenance is the caller's responsibility. It prevents later cache mutation.
 */
class FsshRawFrame
{
  public:
    /**
     * @brief Validate and copy an unaligned electronic snapshot.
     * @param[in] electronic Raw orbitals, normalized spatial X, Ha energies and residuals.
     * @param[in] solver_roots One solver-root label per stored surface, a permutation of 0..N-1.
     * @param[in] geometry Geometry/basis identity of this solver evaluation; copied and owned.
     * @throw std::runtime_error On invalid dimensions, non-real orbitals, non-TDA data or inconsistent labels.
     */
    FsshRawFrame(
        const FsshElectronicFrame &electronic, const std::vector<int> &solver_roots, const FsshFrameGeometry &geometry);

    /** @return Read-only owned electronic snapshot; valid for this object's lifetime. */
    const FsshElectronicFrame &electronic() const;
    /** @return Read-only solver-root labels in storage order. */
    const std::vector<int> &solver_roots() const;
    /** @return Read-only endpoint geometry. */
    const FsshFrameGeometry &geometry() const;

  private:
    FsshElectronicFrame electronic_;
    std::vector<int> solver_roots_;
    FsshFrameGeometry geometry_;
};

/** @brief Owned AO cross overlap and the actual geometries used to construct its rows and columns. */
struct FsshRawAoOverlap
{
    ModuleBase::ComplexMatrix values; ///< Dimensionless, old AO rows/new AO columns.
    FsshFrameGeometry old_geometry;   ///< Actual bra endpoint, not a guessed geometry from post-hop velocities.
    FsshFrameGeometry new_geometry;   ///< Actual ket endpoint.
};

/** @brief Prepared LD endpoints in common solver-root order, with explicit maps back to raw storage. */
struct FsshLDEndpoints
{
    ModuleBase::ComplexMatrix overlap; ///< Raw retained-state overlap, before polar decomposition.
    std::vector<double> energy_old;    ///< Old surface energies in Ha, common root order.
    std::vector<double> energy_new;    ///< New surface energies in Ha, common root order.
    std::vector<int> solver_roots;     ///< Roots propagated; excludes zero in excited-only mode.
    std::vector<int> old_indices;      ///< Common-root index -> old raw-frame storage index.
    std::vector<int> new_indices;      ///< Common-root index -> new raw-frame storage index.
    int full_state_count = 0;          ///< Full storage size including the S0 placeholder.
    double timestep_au = 0.0;          ///< Positive time difference between the endpoint snapshots.
};

/** @brief Deterministic LD electronic step before random selection, nuclear feedback and decoherence. */
struct FsshLDFrameStep
{
    FsshLD::Propagation propagation;                    ///< Retained block in common solver-root order.
    FsshLD::HoppingProbabilities hopping;               ///< Retained-block probabilities in common solver-root order.
    std::vector<std::complex<double>> coefficients_new; ///< Full new raw-frame order; excluded S0 stays zero.
    std::vector<double> probabilities_new;              ///< Full new raw-frame target order.
    int active_new_index = -1;                          ///< Same active surface in new storage, before hop selection.
};

/**
 * @brief Construct complete LD endpoints without modifying either raw snapshot.
 * @param[in] old_frame Validated old snapshot.
 * @param[in] new_frame Validated new snapshot with unchanged state/orbital spaces and basis.
 * @param[in] ao_overlap Raw AO overlap with matching endpoint identities and geometries.
 * @param[in] ground_state_coupling Include S0; otherwise select the excited block before any polar operation.
 * @return Owned raw overlap, both endpoint energies, time interval and state-order maps.
 * @throw std::runtime_error On stale/wrong geometry, changed windows or invalid overlap.
 */
FsshLDEndpoints prepare_fssh_ld_endpoints(
    const FsshRawFrame &old_frame,
    const FsshRawFrame &new_frame,
    const FsshRawAoOverlap &ao_overlap,
    bool ground_state_coupling);

/**
 * @brief Advance a prepared raw-frame pair using LD and guarded SHARC endpoint probabilities.
 * @param[in] endpoints Prepared data, including old/new storage maps.
 * @param[in] coefficients_old Normalized amplitudes in FULL old raw-frame storage order.
 * @param[in] active_old_index Active surface index in old raw storage; must be in the retained block.
 * @return Owned propagated coefficients and probabilities in full NEW raw order, before decoherence.
 * @throw std::runtime_error On excluded-state population or malformed state-order maps.
 */
FsshLDFrameStep advance_fssh_ld_endpoints(
    const FsshLDEndpoints &endpoints, const std::vector<std::complex<double>> &coefficients_old, int active_old_index);

/**
 * @brief Resolve an unambiguous solver root before requesting an adiabatic force.
 * @param[in] frame New raw frame used by the force solver.
 * @param[in] raw_state_index Selected surface in this frame's storage order.
 * @param[in] ground_state_coupling Whether S0 participates in the model and resolution check.
 * @return Solver surface index: 0 for ground, r>0 for LR gradient root r-1.
 * @throw std::runtime_error If the state is excluded or its energy is unresolved by LR residuals.
 */
int resolved_fssh_force_root(const FsshRawFrame &frame, int raw_state_index, bool ground_state_coupling);
} // namespace LR
#endif
