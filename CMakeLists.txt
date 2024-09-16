# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.12)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("YetAnotherDeepLearningLib")

message("=======${CMAKE_SOURCE_DIR}=========================")

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.82.0 REQUIRED COMPONENTS NO_MODULE) 
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:\\CPPLibraries")
# set(Eigen3_DIR "C:\\CPPLibraries\\Eigen3\\build")
find_package(Eigen3 3.3.8 REQUIRED NO_MODULE)
if(Boost_FOUND)
    message(PROJECT_SOURCE_DIR="${Boost_INCLUDE_DIRS}")
endif()


# Include sub-projects.
add_subdirectory ("YetAnotherDeepLearningLib")

#https://cmake.org/cmake/help/latest/module/FindBoost.html
