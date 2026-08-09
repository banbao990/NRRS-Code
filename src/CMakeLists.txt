add_subdirectory(kiraray)

# just a temporary workaround for tcnn dont pass definitions to custom cuda 
add_definitions(-DTCNN_MIN_GPU_ARCH=$ENV{CUDA_ARCH} -DTCNN_SHAMPOO)

add_subdirectory(ext)

set ( KRR_EXP_INCLUDE_ALL
	${KRR_INCLUDE_DIRS}
	${KRR_MISC_INCLUDE_ALL}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/render
	${CMAKE_CURRENT_SOURCE_DIR}/ext/tcnn/include
	${CMAKE_CURRENT_SOURCE_DIR}/ext/tcnn/dependencies
)

message("KRR INCLUDE DIRS: ${KRR_EXP_INCLUDE_ALL}")

add_subdirectory("render")

add_executable( testbed main/main.cpp)
target_link_libraries (testbed PUBLIC krr_nrc_bb_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_nrc_bb_lib>)
target_link_libraries (testbed PUBLIC krr_adn_bb_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_adn_bb_lib>)
target_link_libraries (testbed PUBLIC krr_ears_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_ears_lib>)
target_link_libraries (testbed PUBLIC krr_nrrs_bb_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_nrrs_bb_lib>)

target_link_libraries (testbed PUBLIC krr_loadimage_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_loadimage_lib>)

target_link_libraries (testbed PUBLIC krr_simplewpt_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_simplewpt_lib>)
target_link_libraries (testbed PUBLIC krr_common_bb_cuda_libs -WHOLEARCHIVE:$<TARGET_FILE:krr_common_bb_cuda_libs>)
target_link_libraries (testbed PUBLIC krr_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)


# if(TARGET pykrr)
#     message("Linking EXP library to python binding")
#     #TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_npm_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_npm_lib>)
# 
#     TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_ears_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_ears_lib>)
#     TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_nrrs_bb_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_nrrs_bb_lib>)
# 
#     TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_simplewpt_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_testwpt_lib>)
# 
#     TARGET_LINK_LIBRARIES(pykrr PUBLIC krr_common_bb_cuda_libs -WHOLEARCHIVE:$<TARGET_FILE:krr_common_bb_cuda_libs>)
# endif()

###############################################################################
# post build
###############################################################################

macro(copy_post_build target dll)
 add_custom_command(TARGET ${target} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${dll}
    $<TARGET_FILE_DIR:${target}>)
endmacro()

copy_post_build(testbed
      "${KRR_RENDER_SOURCE_DIR}/ext/openvdb/openvdb/bin/$<IF:$<CONFIG:Debug>,openvdb_d,openvdb>.dll")
copy_post_build(testbed
      "${KRR_RENDER_SOURCE_DIR}/ext/openvdb/tbb/lib/$<IF:$<CONFIG:Debug>,debug/tbb_debug.dll,tbb.dll>")
copy_post_build(testbed "$<TARGET_FILE:IlmBase::Half>")