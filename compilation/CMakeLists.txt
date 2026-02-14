cmake_minimum_required(VERSION 3.18)
project(atr_vmd_cpp_main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    videoio
    video
    calib3d
    features2d
    flann
    xfeatures2d
    cudaarithm
    cudabgsegm
    cudafilters
    cudaimgproc
    cudawarping
)

add_executable(main_cpp
    ${PROJECT_ROOT}/main.cpp
    ${PROJECT_ROOT}/cpp_modules/cpp/kalman.cpp
    ${PROJECT_ROOT}/cpp_utils/cpp/cpp_utils.cpp
)

set_target_properties(main_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/compilation/cpp/bin
)

target_include_directories(main_cpp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_ROOT}
    ${PROJECT_ROOT}/cpp_modules/hpp
    ${PROJECT_ROOT}/cpp_utils/hpp
)

target_link_libraries(main_cpp PRIVATE
    ${OpenCV_LIBS}
)

# Ensure the binary can locate OpenCV shared libs in the same env.
if(DEFINED OpenCV_INSTALL_PATH)
    target_link_directories(main_cpp PRIVATE ${OpenCV_INSTALL_PATH}/lib)
endif()
