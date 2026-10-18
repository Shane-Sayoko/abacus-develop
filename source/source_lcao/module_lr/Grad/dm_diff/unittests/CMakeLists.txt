remove_definitions(-DUSE_LIBXC)
AddTest(
  TARGET dm_diff_test
  LIBS psi base parameter ${math_libs} device container
  SOURCES test_dm_diff.cpp ../../../utils/lr_util.cpp
)