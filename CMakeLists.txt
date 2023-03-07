cmake_minimum_required(VERSION 3.15)

project(sjs C CXX)

find_package(CUDA QUIET REQUIRED)
if (${CUDA_FOUND})
    message("-- CUDA Found: TRUE")
    message("-- CUDA VERSION: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
    message("-- CUDA_INCLUDE_DIR: ${CUDA_INCLUDE_DIRS}")
    message("-- CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 75)
    set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -lineinfo")
    add_definitions(-DUSE_CUDA)
endif ()

add_executable(cugemm_v1 cugemm_v1.cu)
set_target_properties(cugemm_v1 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(cugemm_v1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)

add_executable(cugemm_v2 cugemm_v2.cu)
set_target_properties(cugemm_v2 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(cugemm_v2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)

add_executable(cugemm_v3 cugemm_v3.cu)
set_target_properties(cugemm_v3 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(cugemm_v3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        >)
