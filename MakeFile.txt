cmake_minimum_required(VERSION 3.6)
project(Task1)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_STANDARD 17)


add_executable(Task1 Task_1.cpp)
