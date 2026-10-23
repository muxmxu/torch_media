# unit_test/metric/CMakeLists.txt -- torchmedia::metric, one ctest target per area (goldens: gen_golden.py).
foreach(area audio detection image segmentation text)
    add_executable(metric_test_${area} test_${area}.cpp)
    target_link_libraries(metric_test_${area} PRIVATE torch_media)
    add_test(NAME metric_test_${area} COMMAND metric_test_${area})
endforeach()
# Randomised parity sweep driver (develop_log/2026-10-23/progress11 D10; sweep/sweep_gen.py): opt-in, not in ctest.
add_executable(metric_sweep EXCLUDE_FROM_ALL sweep/sweep.cpp)
target_link_libraries(metric_sweep PRIVATE torch_media)
