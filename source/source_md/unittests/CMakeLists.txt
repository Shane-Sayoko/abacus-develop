abacus_disable_feature_definitions(__MPI __LCAO)
abacus_add_local_feature_definitions(__NORMAL)
include(${CMAKE_CURRENT_LIST_DIR}/../test_support.cmake)

AddTest(
  TARGET MODULE_MD_fssh_mdcell
  LIBS parameter psi device
  SOURCES test_fssh_mdcell.cpp
  ../fssh_mdcell.cpp
  ../../source_cell/magnetism.cpp
  ${depend_files}
)
