cmake_minimum_required(VERSION 3.12)
project(trafficsignnet_zcu104_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(unilog REQUIRED)
find_package(xir REQUIRED)
find_package(vart REQUIRED COMPONENTS runner)
find_package(Threads REQUIRED)

add_executable(benchmark_zcu104 benchmark_zcu104.cpp)

target_compile_options(benchmark_zcu104 PRIVATE -O3 -Wall -Wextra)

target_link_libraries(
  benchmark_zcu104
  PRIVATE
    unilog::unilog
    xir::xir
    vart::runner
    ${OpenCV_LIBS}
    Threads::Threads
)
