remove_definitions(-DUSE_LIBXC)
AddTest(
  TARGET CVCX_test
  LIBS base parameter ${math_libs} container device psi 
  SOURCES test_CVCX.cpp ../../../utils/lr_util.cpp ../CVCX_parallel.cpp ../CVCX_serial.cpp
)