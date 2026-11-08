add_library(
  md OBJECT
  fire.cpp
  langevin.cpp
  md_base.cpp
  md_func.cpp
  msst.cpp
  nhchain.cpp
  run_md.cpp
  verlet.cpp)

if(ENABLE_COVERAGE)
  add_coverage(md)
endif()

add_subdirectory(module_plumed)

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
    add_subdirectory(unittests)
  endif()
endif()

if(ENABLE_LCAO)
  target_sources(md PRIVATE fssh.cpp fssh_restart.cpp fssh_mdcell.cpp fssh_replay.cpp)
endif()

if(ENABLE_LCAO AND ENABLE_MPI)
  add_executable(fssh_replay fssh_replay_main.cpp fssh_replay.cpp fssh_mdcell.cpp
    ../source_lcao/module_operator_lcao/fssh_driver.cpp
    ../source_lcao/module_lr/utils/fssh_state_overlap.cpp)
  target_link_libraries(fssh_replay PRIVATE parameter base cell device symmetry
    neighbor_search neighbor domain_decomposition abacus::link_libs)
endif()
