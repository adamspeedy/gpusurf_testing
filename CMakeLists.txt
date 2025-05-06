cmake_minimum_required(VERSION 3.17)
project(gpusurf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Eigen3 REQUIRED)
# opencv stuff
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_subdirectory(gpusurf)

add_executable(gpusurf_run src/main.cpp)
add_executable(gpusurfStereo_run src/stereoMatching.cpp)

target_link_libraries(gpusurf_run gpusurf ${OpenCV_LIBS})
target_link_libraries(gpusurfStereo_run gpusurf ${OpenCV_LIBS})
target_include_directories(gpusurf_run PUBLIC include/ src/)
target_include_directories(gpusurfStereo_run PUBLIC include/ src/)
target_include_directories(gpusurf_run PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(gpusurfStereo_run PUBLIC ${EIGEN3_INCLUDE_DIR})
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS gpusurf_run gpusurfStereo_run
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
  )