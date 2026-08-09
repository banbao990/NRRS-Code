set ( KRR_BB_ADN
	"integrator.cpp"
	"adntrain.cu"
)

set_source_files_properties (
	"integrator.cpp"
	PROPERTIES LANGUAGE CUDA
)

###############################################################################
# automatically creating definitions of structure of arrays (soa)
###############################################################################

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/workitem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/workitem.soa)

set (KRR_ADN_BB_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/workitem_soa.h
)

add_custom_target (krr_adn_bb_soa_generated DEPENDS ${KRR_ADN_BB_SOA_GENERATED})

###############################################################################
# generating PTX code from optix shader routines
###############################################################################
include_directories (${KRR_EXP_INCLUDE_ALL})
include (${KRR_EXP_PROJECT_ROOT}/common/cmake/CompilePTX.cmake)
#include (${CMAKE_CURRENT_SOURCE_DIR}/kiraray/common/cmake/CompilePTX.cmake)
EXP_CUDA_COMPILE_EMBED(ADN_BB_PTX device.cu krr-adn-bb krr_adn_bb_soa_generated krr_common_bb_soa_generated ${KRR_EXP_INCLUDE_ALL})
add_library(KRR_ADN_BB_PTX STATIC
	# peseudo target for SOA generation priority
	${ADN_BB_PTX}
)
add_dependencies(KRR_ADN_BB_PTX krr_adn_bb_soa_generated krr_common_bb_soa_generated)

set ( KRR_EXP_LIBS_ALL
	krr_cuda_cfg
	krr_cuda_warning
	tiny-cuda-nn
	KRR_ADN_BB_PTX
)

add_library (krr_adn_bb_lib
	${KRR_BB_ADN}
	# SOA
	${KRR_ADN_BB_SOA_GENERATED}
	# PTX 
	${ADN_BB_PTX}
)
add_dependencies(krr_adn_bb_lib krr_adn_bb_soa_generated krr_common_bb_soa_generated krr_nrc_bb_lib krr_common_bb_cuda_libs)
#set_property(TARGET krr_adn_bb_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET krr_adn_bb_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_link_libraries (krr_adn_bb_lib PUBLIC ${KRR_EXP_LIBS_ALL})
target_include_directories(krr_adn_bb_lib PUBLIC ${KRR_EXP_INCLUDE_ALL})
