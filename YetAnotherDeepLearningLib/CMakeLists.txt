
# CMakeList.txt : CMake project for YetAnotherDeepLearningLib, include source and define
# project specific logic here.
#

message("===============CMakeLists.txt in YetAnotherDeepLearningLib=================")

set (SOURCES 
	"YetAnotherDeepLearningLib.cpp")


set (INCLUDES
	"YetAnotherDeepLearningLib.h"
	"Tensor/Tensor.h"
	"Test/Test.h")

#add_library(tensor_lib Tensor/Tensor.cpp Tensor/Tensor.h)
# add_executable (YetAnotherDeepLearningLib ${SOURCES} "Test/Test.cpp")
add_executable (YetAnotherDeepLearningLib ${SOURCES} ${INCLUDES})
target_link_libraries (YetAnotherDeepLearningLib Eigen3::Eigen)
include_directories(${Boost_INCLUDE_DIRS}) 
target_link_libraries(YetAnotherDeepLearningLib ${Boost_LIBRARIES})
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET YetAnotherDeepLearningLib PROPERTY CXX_STANDARD 20)
endif()


