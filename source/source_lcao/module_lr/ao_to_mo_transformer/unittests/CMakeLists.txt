abacus_disable_feature_definitions(__LIBXC)
AddTest(
  TARGET MODULE_LR_ao_to_mo_test
  LIBS parameter base container device psi
  SOURCES test_ao_to_mo.cpp ../../utils/lr_util.cpp ../ao_to_mo_parallel.cpp ../ao_to_mo_serial.cpp
)