cmake_minimum_required(VERSION 3.21)

project(Optimium-Cpp LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core highgui videoio imgproc)
find_package(Optimium-Runtime REQUIRED HINTS "$ENV{HOME}/.local")

add_executable(optimium-cpp main.cpp)
target_link_libraries(optimium-cpp PRIVATE Optimium::Runtime opencv_core opencv_imgproc opencv_highgui opencv_videoio)
set_target_properties(optimium-cpp PROPERTIES
                      CXX_STANDARD 20)