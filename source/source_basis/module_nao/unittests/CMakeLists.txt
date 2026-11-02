addtest(
  TARGET MODULE_NAO_smoothgen_test
  LIBS parameter device base
  SOURCES test_smoothgen.cpp ../smoothgen.cpp)

AddTest(
  TARGET MODULE_NAO_legacy_radial_padding
  SOURCES
    test_numerical_radial_legacy.cpp
    test_radial_set_legacy.cpp
    ../numerical_radial.cpp
    ../radial_set.cpp
    ../../module_ao/orb_atomic_lm.cpp
    ../../module_ao/orb_atomic.cpp
  LIBS parameter device base
)
