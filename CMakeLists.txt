cmake_minimum_required(VERSION 3.10)
project(stlog)
set(CMAKE_CXX_STANDARD 11)

add_executable(stlog StLog/main.cpp
  StLog/LogRecord.cpp
  StLog/LogRecord.h
)
