cmake_minimum_required(VERSION 3.10)
project(stlog)
set(CMAKE_CXX_STANDARD 11)

add_executable(stlog StLog/main.cpp
  StLog/LogRecord.cpp
  StLog/LogRecord.h
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(stlog PRIVATE Threads::Threads)