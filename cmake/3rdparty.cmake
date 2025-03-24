set(3rdparty_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

# plog
include_directories(${3rdparty_DIR}/plog-1.1.5/include)

# json
include_directories(${3rdparty_DIR}/json-3.9.1/single_include)

# opencv
find_package(OpenCV 4.3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ceres
find_package(Ceres REQUIRED)
include_directories("${CMAKE_INSTALL_PREFIX}/include/eigen3")
include_directories(${CERES_INCLUDE_DIRS})

set(EXTERNAL_LIBRARIES
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)