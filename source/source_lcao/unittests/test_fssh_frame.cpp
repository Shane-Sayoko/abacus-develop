#include "source_base/parallel_cell.h"
#include "source_base/parallel_global.h"
#include "source_lcao/module_lr/utils/fssh_frame.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace
{
/** @brief Minimal distributed frame fixture with contributions on rank zero. */
class FsshFrameTest : public testing::Test
{
  protected:
    ModuleBase::matrix energies;
    ModuleBase::matrix occupations;
    ModuleBase::matrix excitations;
    Parallel_Orbitals orbital_layout;
    Parallel_2D casida_layout;
    std::unique_ptr<psi::Psi<double>> orbitals;
    std::unique_ptr<ct::Tensor> amplitudes;

    void SetUp() override
    {
        energies.create(1, 2);
        occupations.create(1, 2);
        excitations.create(1, 1);
        energies(0, 0) = -0.8;
        energies(0, 1) = 0.2;
        occupations(0, 0) = 2.0;
        excitations(0, 0) = 0.4;
        orbital_layout.set_serial(2, 2);
        casida_layout.set_serial(1, 1);
        orbitals.reset(new psi::Psi<double>(1, 2, 2, 2, true));
        amplitudes.reset(new ct::Tensor(ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {1, 1}));
        amplitudes->zero();
        for (int band = 0; band < 2; ++band)
        {
            for (int ao = 0; ao < 2; ++ao)
            {
                (*orbitals)(0, band, ao) = 0.0;
            }
        }
        if (ModuleBase::world_comm_domain().rank() == 0)
        {
            (*orbitals)(0, 0, 0) = 1.0;
            (*orbitals)(0, 1, 1) = 1.0;
            amplitudes->data<double>()[0] = 1.0;
        }
    }

    FsshElectronicFrame collect(int occupied) const
    {
        return LR::make_fssh_frame(2,
                                   2,
                                   occupied,
                                   1,
                                   1,
                                   -2.0,
                                   energies,
                                   occupations,
                                   excitations,
                                   *orbitals,
                                   orbital_layout,
                                   casida_layout,
                                   *amplitudes,
                                   std::vector<double>{2.0e-12});
    }
};

TEST_F(FsshFrameTest, GathersAcrossRanksAndConvertsRyToHartree)
{
    const FsshElectronicFrame frame = collect(1);
    ASSERT_EQ(frame.surface_energies_hartree.size(), 2u);
    EXPECT_DOUBLE_EQ(frame.eigen_residuals_hartree.at(1), 1.0e-12);
    EXPECT_DOUBLE_EQ(frame.surface_energies_hartree[0], -1.0);
    EXPECT_DOUBLE_EQ(frame.surface_energies_hartree[1], -0.8);
    EXPECT_DOUBLE_EQ(frame.ks_energies_hartree[0], -0.4);
    EXPECT_DOUBLE_EQ(frame.mo_coefficients(0, 0).real(), 1.0);
    EXPECT_DOUBLE_EQ(frame.mo_coefficients(1, 1).real(), 1.0);
    EXPECT_DOUBLE_EQ(frame.casida_states[1].X_coeffs[0], 1.0);
    EXPECT_EQ(frame.occupied_bands, 1);
}

TEST_F(FsshFrameTest, RejectsTruncatedOccupiedSpace)
{
    EXPECT_THROW(collect(0), std::logic_error);
}

TEST_F(FsshFrameTest, RejectsEmptyCasidaState)
{
    amplitudes->zero();
    EXPECT_THROW(collect(1), std::logic_error);
}

TEST(FsshFrameDistributed, ReconstructsBlockCyclicOrbitalsAndTwoCasidaStates)
{
    Parallel_Orbitals pv;
    Parallel_2D px;
#ifdef __MPI
    pv.init(4, 4, 1, ModuleBase::world_comm_domain().communicator());
    px.init(2, 2, 1, ModuleBase::world_comm_domain().communicator());
#else
    pv.set_serial(4, 4);
    px.set_serial(2, 2);
#endif
    ModuleBase::matrix energies(1, 4);
    ModuleBase::matrix occupations(1, 4);
    ModuleBase::matrix excitations(1, 2);
    occupations(0, 0) = 2.0;
    occupations(0, 1) = 2.0;
    excitations(0, 0) = 0.4;
    excitations(0, 1) = 0.6;
    psi::Psi<double> orbitals(1, pv.get_col_size(), pv.get_row_size(), pv.get_row_size(), true);
    ct::Tensor amplitudes(ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {2, px.get_local_size()});
    for (int col = 0; col < pv.get_col_size(); ++col)
    {
        for (int row = 0; row < pv.get_row_size(); ++row)
        {
            orbitals(0, col, row) = 0.01 * (10 * pv.local2global_col(col) + pv.local2global_row(row) + 1);
        }
    }
    for (int state = 0; state < 2; ++state)
    {
        for (int col = 0; col < px.get_col_size(); ++col)
        {
            for (int row = 0; row < px.get_row_size(); ++row)
            {
                const int local_index = state * px.get_local_size() + col * px.get_row_size() + row;
                amplitudes.data<double>()[local_index]
                    = 0.1 * (10 * state + 2 * px.local2global_col(col) + px.local2global_row(row) + 1);
            }
        }
    }
    const FsshElectronicFrame frame = LR::make_fssh_frame(3,
                                                          4,
                                                          2,
                                                          2,
                                                          px.get_local_size(),
                                                          -2.0,
                                                          energies,
                                                          occupations,
                                                          excitations,
                                                          orbitals,
                                                          pv,
                                                          px,
                                                          amplitudes,
                                                          std::vector<double>{2.0e-12, 4.0e-12});
    for (int band = 0; band < 4; ++band)
    {
        for (int ao = 0; ao < 4; ++ao)
        {
            EXPECT_DOUBLE_EQ(frame.mo_coefficients(band, ao).real(), 0.01 * (10 * band + ao + 1));
        }
    }
    for (int state = 0; state < 2; ++state)
    {
        for (int element = 0; element < 4; ++element)
        {
            EXPECT_DOUBLE_EQ(frame.casida_states[state + 1].X_coeffs[element], 0.1 * (10 * state + element + 1));
        }
    }
}
} // namespace

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    int process_count = 1;
    int thread_count = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argc, argv, process_count, thread_count, rank);
#ifdef __MPI
    // The production finalizer owns these communicators. Initialize them through
    // the same wrappers before running a fixture that uses the world domain.
    int group_size = 1;
    int group_rank = 0;
    int group_index = 0;
    int pool_size = 1;
    int pool_rank = 0;
    int pool_index = 0;
    Parallel_Global::init_pools(process_count,
                                rank,
                                1,
                                1,
                                group_size,
                                group_rank,
                                group_index,
                                pool_size,
                                pool_rank,
                                pool_index);
    Parallel_Global::split_diag_world(process_count, process_count, rank, group_rank, group_size, group_index);
    Parallel_Global::split_grid_world(process_count, process_count, rank, group_rank, group_size);
#endif
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
    return result;
}
