cmake_minimum_required(VERSION 3.20)
project(SLAE_)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE src *.hpp *.h *.cpp)
add_library(SLAE_ INTERFACE ${src})