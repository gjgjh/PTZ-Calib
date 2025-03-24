file(GLOB ptz_core_src "*.c*")
add_library(ptz_core ${ptz_core_src})
target_link_libraries(ptz_core ${EXTERNAL_LIBRARIES})