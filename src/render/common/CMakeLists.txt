# SOA

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/commonitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/commonitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/commonitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/commonitem.soa)

set (KRR_COMMON_BB_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/commonitem_soa.h
)

add_custom_target (krr_common_bb_soa_generated DEPENDS ${KRR_COMMON_BB_SOA_GENERATED})

# PTX

include_directories(${KRR_EXP_INCLUDE_ALL})
EXP_CUDA_COMPILE_EMBED(DENOISE_TASK_PTX denoise/device.cu krr-denoisetask krr_common_bb_soa_generated ${KRR_EXP_INCLUDE_ALL})
add_library(KRR_DENOISE_TASK_PTX STATIC
	# peseudo target for SOA generation priority
	${DENOISE_TASK_PTX}
)
add_dependencies(KRR_DENOISE_TASK_PTX krr_common_bb_soa_generated)

###############################################################################
# common cuda libs
###############################################################################

set (KRR_COMMON_BB_CUDA_LIBS_ALL
	krr_lib
	krr_cuda_cfg
	krr_cuda_warning
	KRR_DENOISE_TASK_PTX
)

SET_SOURCE_FILES_PROPERTIES (
	"commoncudautilshost.cpp"
	"tree/treemanip.cpp"
	"denoise/denoisetask.cpp"
	PROPERTIES LANGUAGE CUDA
)

add_library (krr_common_bb_cuda_libs STATIC
    "commoncudautils.cu"
    "commoncudautilshost.cpp"
    "tree/treemanip.cpp"
    "topk/top_k_gpu.cu"
    "denoise/traditional.cu"
    "denoise/denoisetask.cpp"
)

#set_property(TARGET krr_common_bb_cuda_libs PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_COMPILE_DEFINITIONS (krr_common_bb_cuda_libs PUBLIC ${KRR_DEFINITIONS})
set_property(TARGET krr_common_bb_cuda_libs PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_include_directories(krr_common_bb_cuda_libs PUBLIC  ${KRR_EXP_INCLUDE_ALL})
message(STATUS "COMMON: KRR_EXP_LIBS_ALL: ${KRR_EXP_INCLUDE_ALL}")
target_link_libraries (krr_common_bb_cuda_libs PUBLIC ${KRR_COMMON_BB_CUDA_LIBS_ALL})
