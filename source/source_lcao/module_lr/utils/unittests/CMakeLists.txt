abacus_disable_feature_definitions(__LIBXC)
AddTest(
  TARGET MODULE_LR_lr_util_phys_test
  LIBS parameter base device container planewave #for FFT
  SOURCES test_lr_util_physics.cpp test_lr_util_occupations.cpp ../lr_util.cpp

)

AddTest(
  TARGET MODULE_LR_lr_util_algo_test
  LIBS parameter base device psi container planewave #for FFT
  SOURCES test_lr_util_algo.cpp ../lr_util.cpp
)
