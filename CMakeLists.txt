cmake_minimum_required(VERSION 3.18...3.26)
project(PyBlas)
find_package(BLAS REQUIRED)
find_package(PythonExtensions REQUIRED)
message("LINKER FLAGS = " ${BLAS_LIBRARIES})
set(
	BINDINGS
	${CMAKE_CURRENT_SOURCE_DIR}/src/pyblas/level3.cpp
)

# TODO: May not work on some implementations. Seems like cblas is not universal header.
# Example of a generic cmake pattern to find blas
#https://github.com/apache/mxnet/blob/a1b0ff29d97e3b0e865a835412e3c6b224254969/cmake/Modules/FindAccelerate.cmake
set(Accelerate_INCLUDE_SEARCH_PATHS
	/Library/Developer/CommandLineTools/SDKs/MacOSX11.3.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
)
find_path(Accelerate_CBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Accelerate_INCLUDE_SEARCH_PATHS})


message(${Accelerate_CBLAS_INCLUDE_DIR})
add_library(wrappers MODULE ${BINDINGS}) 
include_directories(${Accelerate_CBLAS_INCLUDE_DIR})
target_link_libraries(wrappers ${BLAS_LIBRARIES})
set_target_properties(wrappers PROPERTIES LINKER_LANGUAGE CXX)
python_extension_module(wrappers) # from PythonExtensions
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set_target_properties(wrappers PROPERTIES OUTPUT_NAME "pyblas_core")
set_target_properties(wrappers PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")

install(TARGETS wrappers LIBRARY DESTINATION PyBlas)
