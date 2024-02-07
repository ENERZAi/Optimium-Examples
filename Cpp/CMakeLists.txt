cmake_minimum_required(VERSION 3.21)

project(Optimium-Cpp LANGUAGES CXX)

find_package(OptimaV2-Runtime REQUIRED)

add_executable(optimium-cpp main.cpp)
target_link_libraries(optimium-cpp PRIVATE OptimaV2::Runtime)