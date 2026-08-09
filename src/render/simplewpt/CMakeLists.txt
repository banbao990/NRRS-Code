set ( KRR_SIMPLEWPT
	"integrator.cpp"
)

SET_SOURCE_FILES_PROPERTIES (
	"integrator.cpp"
	PROPERTIES LANGUAGE CUDA
)

# SOA

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/workitem.soa)

set (KRR_SIMPLEWPT_SOA_GENERATED
	${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
)

add_custom_target (krr_simplewpt_soa_generated DEPENDS ${KRR_SIMPLEWPT_SOA_GENERATED})

# PTX

include_directories(${KRR_EXP_INCLUDE_ALL})
include (${KRR_EXP_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
#include (${CMAKE_CURRENT_SOURCE_DIR}/kiraray/common/cmake/CompilePTX.cmake)
EXP_CUDA_COMPILE_EMBED(SIMPLEWPT_PTX device.cu krr-simplewpt krr_simplewpt_soa_generated ${KRR_EXP_INCLUDE_ALL})
add_library(KRR_SIMPLEWPT_PTX STATIC
	# peseudo target for SOA generation priority
	${SIMPLEWPT_PTX}
)
add_dependencies(KRR_SIMPLEWPT_PTX krr_simplewpt_soa_generated krr_common_bb_soa_generated)

set ( KRR_SIMPLEWPT_LIBS_ALL
	#krr_lib
	krr_cuda_cfg
	krr_cuda_warning
	KRR_SIMPLEWPT_PTX
)
add_library (krr_simplewpt_lib STATIC
	${KRR_SIMPLEWPT}
	# SOA
	${KRR_SIMPLEWPT_SOA_GENERATED_H}
	# PTX
	${SIMPLEWPT_PTX}
)
add_dependencies(krr_simplewpt_lib krr_simplewpt_soa_generated krr_common_bb_soa_generated krr_common_bb_cuda_libs)
set_property(TARGET krr_simplewpt_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_link_libraries (krr_simplewpt_lib PRIVATE ${KRR_SIMPLEWPT_LIBS_ALL})
target_include_directories(krr_simplewpt_lib PUBLIC ${KRR_SIMPLEWPT_INCLUDE_ALL})