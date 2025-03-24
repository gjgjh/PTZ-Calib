add_executable(run_ptz_ba run_ptz_ba.cc)
target_link_libraries(run_ptz_ba ptz_utils ptz_core ${EXTERNAL_LIBRARIES})
install(TARGETS run_ptz_ba
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

add_executable(run_ptz_reloc run_ptz_reloc.cc)
target_link_libraries(run_ptz_reloc ptz_utils ptz_core ${EXTERNAL_LIBRARIES})
install(TARGETS run_ptz_reloc
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)