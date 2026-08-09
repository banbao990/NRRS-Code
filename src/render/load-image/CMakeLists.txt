set ( KRR_LOADIMAGE
	"integrator.hpp"
)

SET_SOURCE_FILES_PROPERTIES (
	"integrator.hpp"
	PROPERTIES LANGUAGE CUDA
)

include_directories(${KRR_EXP_INCLUDE_ALL})

set ( KRR_LOADIMAGE_LIBS_ALL
	krr_cuda_cfg
	krr_cuda_warning
)

add_library (krr_loadimage_lib STATIC
	${KRR_LOADIMAGE}
)

set_property(TARGET krr_loadimage_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_link_libraries (krr_loadimage_lib PRIVATE ${KRR_LOADIMAGE_LIBS_ALL})
