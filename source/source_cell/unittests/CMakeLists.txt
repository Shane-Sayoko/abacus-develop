if(ENABLE_MPI)
  add_executable(
    MODULE_CELL_NEIGHBOR_mdcell_reader
    test_mdcell_reader.cpp
    ../mdcell_reader.cpp
    ../mdcell.cpp
    ../print_cell.cpp
    ../domain_decomposition.cpp
    ../module_neighlist/neighbor_search.cpp
    ../module_neighlist/bin_manager.cpp
    ../module_neighlist/page_allocator.cpp)
  target_include_directories(MODULE_CELL_NEIGHBOR_mdcell_reader
                             PRIVATE ${ABACUS_SOURCE_DIR})
  target_compile_definitions(MODULE_CELL_NEIGHBOR_mdcell_reader
                             PRIVATE __NORMAL)
  target_link_libraries(
    MODULE_CELL_NEIGHBOR_mdcell_reader
    PRIVATE parameter
            base
            device
            Threads::Threads
            MPI::MPI_CXX
            GTest::gtest
            GTest::gmock
            abacus::linalg_libs)
  add_test(
    NAME MODULE_CELL_NEIGHBOR_mdcell_reader_np4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:MODULE_CELL_NEIGHBOR_mdcell_reader>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

endif()
