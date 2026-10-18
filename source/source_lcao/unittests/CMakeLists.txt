abacus_disable_feature_definitions(__MLALGO __CUDA __ROCM)

if(ENABLE_LCAO)
AddTest(
  TARGET MODULE_LCAO_fssh_frame
  LIBS parameter base psi device container
  SOURCES test_fssh_frame.cpp
    ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
    ${ABACUS_SOURCE_DIR}/source_lcao/module_lr/utils/fssh_frame.cpp
)

AddTest(
  TARGET MODULE_LCAO_fssh_driver
  LIBS parameter base cell device symmetry neighbor_search abacus::link_libs
  SOURCES test_fssh_driver.cpp
    ${ABACUS_SOURCE_DIR}/source_lcao/module_operator_lcao/fssh_driver.cpp
    ${ABACUS_SOURCE_DIR}/source_md/fssh_mdcell.cpp
)
endif()
