cmake_minimum_required(VERSION 3.10)
project(PTZ-Calib)

# ----------------------------------------------------------------------------
#  Basic
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS         "-g -Wall -pthread -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(ARCHIVE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_INSTALL_PREFIX})
if(NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE "Release" )
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-DTP_DEBUG=0)
else()
    add_definitions(-DTP_DEBUG=1)
endif()

# ----------------------------------------------------------------------------
#   Dependencies
# ----------------------------------------------------------------------------
include(3rdparty)

# ----------------------------------------------------------------------------
#   Sources
# ----------------------------------------------------------------------------

include_directories(src)
add_subdirectory(src/utils)
add_subdirectory(src/core)
add_subdirectory(src/app)

# ----------------------------------------------------------------------------
#  Display
# ----------------------------------------------------------------------------
include(summary)