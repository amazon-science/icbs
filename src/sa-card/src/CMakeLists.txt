list(APPEND TESTS sa-card-test)
list(APPEND EXES sa-card)
list(APPEND EXES ${TESTS})

foreach (EXE IN ITEMS ${EXES})
	add_executable(${EXE} ${EXE}.cpp)
endforeach()

foreach (TEST IN ITEMS ${TESTS})
	target_link_libraries(${TEST} PRIVATE Catch2::Catch2WithMain)
	target_link_libraries(${TEST} PRIVATE OpenMP::OpenMP_CXX)
	add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()
