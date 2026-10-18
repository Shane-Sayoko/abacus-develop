abacus_disable_feature_definitions(__MPI __LCAO)
abacus_add_local_feature_definitions(__NORMAL)

include(${CMAKE_CURRENT_LIST_DIR}/../test_support.cmake)

AddTest(
  TARGET MODULE_MD_LJ_pot
  LIBS parameter psi device
  SOURCES lj_pot_test.cpp
  ../../source_cell/module_neighlist/test/test_domain_decomposition.cpp
  ${depend_files}
)

AddTest(
  TARGET MODULE_MD_func
  LIBS parameter psi device
  SOURCES md_func_test.cpp
  test_md_func.cpp
  ${depend_files}
)



AddTest(
  TARGET MODULE_MD_run
  LIBS parameter psi device
  SOURCES run_md_test.cpp
  ../run_md.cpp
  ../md_base.cpp
  ../fire.cpp
  ../langevin.cpp
  ../msst.cpp
  ../nhchain.cpp
  ../verlet.cpp
  ../../source_cell/mdcell_reader.cpp
  ../../source_cell/magnetism.cpp
  ${depend_files}
)

AddTest(
  TARGET MODULE_MD_fire
  LIBS parameter psi device
  SOURCES fire_test.cpp
  ../md_base.cpp
  ../fire.cpp
  ${depend_files}
)

AddTest(
  TARGET MODULE_MD_verlet
  LIBS parameter psi device
  SOURCES verlet_test.cpp
  ../md_base.cpp
  ../verlet.cpp
  ${depend_files}
)

AddTest(
  TARGET MODULE_MD_nhc
  LIBS parameter psi device
  SOURCES nhchain_test.cpp
  ../md_base.cpp
  ../nhchain.cpp
  ../../source_base/output.cpp
  ${depend_files}
)

AddTest(
  TARGET MODULE_MD_msst
  LIBS parameter psi device
  SOURCES msst_test.cpp
  ../md_base.cpp
  ../msst.cpp
  ../../source_cell/update_cell.cpp
  ../../source_base/output.cpp
  ${depend_files}
)

AddTest(
  TARGET MODULE_MD_lgv
  LIBS parameter psi device
  SOURCES langevin_test.cpp
  ../md_base.cpp
  ../langevin.cpp
  ${depend_files}
)
