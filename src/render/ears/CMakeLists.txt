set ( KRR_EARS_PT
	"integrator.cpp"
	"octree.cpp"
)

SET_SOURCE_FILES_PROPERTIES (
	"integrator.cpp"
	"octree.cpp"
	PROPERTIES LANGUAGE CUDA
)

# SOA

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/workitem.soa)

set (KRR_EARS_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
)

add_custom_target (krr_ears_soa_generated DEPENDS ${KRR_EARS_SOA_GENERATED})

# PTX

include_directories(${KRR_EXP_INCLUDE_ALL})
include (${KRR_EXP_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
EXP_CUDA_COMPILE_EMBED(EARS_PTX device.cu krr-ears krr_ears_soa_generated krr_common_bb_soa_generated ${KRR_EXP_INCLUDE_ALL})
add_library(KRR_EARS_PTX STATIC
	# peseudo target for SOA generation priority
	${EARS_PTX}
)
add_dependencies(KRR_EARS_PTX krr_ears_soa_generated krr_common_bb_soa_generated)

set ( KRR_EARS_LIBS_ALL
	#krr_lib
	krr_cuda_cfg
	krr_cuda_warning
	tiny-cuda-nn
	KRR_EARS_PTX
)

add_library (krr_ears_lib STATIC
	${KRR_EARS_PT}
	# SOA
	${KRR_EARS_SOA_GENERATED}
	# PTX
	${EARS_PTX}
)
add_dependencies(krr_ears_lib krr_ears_soa_generated krr_common_bb_soa_generated) # do need nrc, as we just use headers
set_property(TARGET krr_ears_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
TARGET_COMPILE_DEFINITIONS (krr_ears_lib PUBLIC ${KRR_DEFINITIONS})
target_link_libraries (krr_ears_lib PRIVATE ${KRR_EARS_LIBS_ALL})
target_include_directories(krr_ears_lib PUBLIC ${KRR_EARS_INCLUDE_ALL})