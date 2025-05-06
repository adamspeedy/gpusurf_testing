cmake_minimum_required(VERSION 3.17)
project(gpusurf LANGUAGES CXX CUDA)

# Compiler setup (assumed to be GNU)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-march=native -O3)
add_compile_options(-Wno-unused-variable -Wno-unused-parameter -Wno-narrowing -Wno-unused-result -Wno-write-strings -Wno-ignored-qualifiers)

# Find dependencies
add_subdirectory(cudpp)

find_package(CUDAToolkit REQUIRED)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
string(CONCAT CUDA_COMPUTE "compute_" "${CUDA_ARCH_LIST}" )
string(CONCAT CUDA_CODE "sm_" "${CUDA_ARCH_LIST}")
message(WARNING "Autodetected Cuda Version: ${CUDA_ARCH_LIST}.")
#SET(CUDA_NVCC_FLAGS -arch=${CUDA_COMPUTE} -code=${CUDA_CODE} --std c++11 --default-stream per-thread)
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
SET(CUDA_VERBOSE_BUILD OFF)
SET(CUDA_BUILD_TYPE Device)
SET(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/intermediate)
#target_include_directories(${PROJECT_NAME} PUBLIC "${CUDA_INCLUDE_DIRS}")

#set(OpenCV_DIR /usr/local/opencv_cuda/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include/opencv4/")

# Libraries
add_library(
  ${PROJECT_NAME} SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuIntegralImage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuIntegralImageProcessor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuIntegralImage_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuSurfDetector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuSurfDetectorInternal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuSurfFeatures.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuSurfOctave.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpusurf.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/timing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/detector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_globals.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_area.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuSurfStereoDetector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GpuSurfStereoDetectorInternal.cpp
)

#add_executable(gpusurf_run src/main.cpp)

target_link_libraries(${PROJECT_NAME} cudpp CUDA::cublas ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}")
#target_link_libraries(gpusurf_run cudpp CUDA::cublas ${OpenCV_LIBS})

#target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/cudpp")
#target_include_directories(${PROJECT_NAME} PUBLIC "${CUDA_INCLUDE_DIRS}"})
#target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include/opencv4/")

target_include_directories(${PROJECT_NAME} 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# target_include_directories(gpusurf_run
# PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)


install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME} 
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

# export(
#   EXPORT export_${PROJECT_NAME}
#   FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
# )
