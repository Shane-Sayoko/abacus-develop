add_library(
    md
    OBJECT
    fire.cpp
    fssh.cpp
    fssh_mdcell.cpp
    fssh_replay.cpp
    langevin.cpp
    md_base.cpp
    md_func.cpp
    msst.cpp
    nhchain.cpp
    run_md.cpp
    verlet.cpp
)

if(ENABLE_COVERAGE)
  add_coverage(md)
endif()

if(ENABLE_LCAO AND ENABLE_MPI)
  add_executable(
      fssh_replay
      fssh_replay_main.cpp
      fssh_replay.cpp
      fssh_mdcell.cpp
  )
  target_link_libraries(fssh_replay PRIVATE
      esolver hsolver hamilt_general elecstate module_pwdft module_ofdft
      module_stodft module_dfpt xc_ vdw relax io_advanced io_basic io_input
      surchem neighbor_search neighbor planewave symmetry cell parameter
      psi_overall_init psi_init psi deltaspin container device base
      hamilt_lcao tddft dftu orb gint hcontainer numerical_atomic_orbitals
      lr lr_grad rdmft)
  if(ENABLE_ELPA)
    target_link_libraries(fssh_replay PRIVATE genelpa)
  endif()
  if(ENABLE_LIBRI)
    target_link_libraries(fssh_replay PRIVATE bse ri module_exx_symmetry)
  endif()
  target_link_libraries(fssh_replay PRIVATE abacus::link_libs)
  if(USE_OPENMP)
    target_link_libraries(fssh_replay PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
  endif()
endif()
