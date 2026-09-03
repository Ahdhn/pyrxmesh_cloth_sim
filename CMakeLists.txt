cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(massspring LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pyrxmesh.cmake_dir
    OUTPUT_VARIABLE pyrxmesh_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

find_package(pyrxmesh CONFIG REQUIRED PATHS "${pyrxmesh_DIR}" NO_DEFAULT_PATH)
pyrxmesh_add_plugin(_massspring src/massspring.cu)

install(TARGETS _massspring
    LIBRARY DESTINATION massspring COMPONENT python
    RUNTIME DESTINATION massspring COMPONENT python
)
