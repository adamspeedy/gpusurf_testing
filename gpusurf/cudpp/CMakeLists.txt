# Find dependencies
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
SET(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/intermediate)

add_library(
  cudpp SHARED
  cudpp/common/src/cmd_arg_reader.cpp
  cudpp/common/src/bank_checker.cpp
  cudpp/common/src/cutil.cpp
  cudpp/common/src/stopwatch_linux.cpp
  cudpp/common/src/multithreading.cpp
  cudpp/common/src/findFile.cpp
  cudpp/common/src/param.cpp
  cudpp/common/src/stopwatch.cpp
  cudpp/cudpp/src/cudpp.cpp
  cudpp/cudpp/src/cudpp_plan.cpp 
  cudpp/cudpp/src/cudpp_plan_manager.cpp 
  cudpp/cudpp/src/cudpp_maximal_launch.cpp
  cudpp/cudpp/src/app/compact_app.cu 
  cudpp/cudpp/src/app/scan_app.cu 
  cudpp/cudpp/src/app/segmented_scan_app.cu 
  cudpp/cudpp/src/app/spmvmult_app.cu 
  cudpp/cudpp/src/app/radixsort_app.cu 
  cudpp/cudpp/src/app/rand_app.cu 
)
target_include_directories(cudpp PUBLIC cudpp/cudpp/include cudpp/cudpp/src cudpp/cudpp/src/kernel cudpp/common/inc)

# install(
#   DIRECTORY 
#     cudpp/cudpp/include/
#     cudpp/common/inc/
#   DESTINATION include
# )

# install(
#   TARGETS ${PROJECT_NAME}
#   EXPORT export_${PROJECT_NAME}
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
#   INCLUDES DESTINATION include
# )
