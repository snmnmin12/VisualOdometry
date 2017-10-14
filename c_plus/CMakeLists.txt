cmake_minimum_required(VERSION 2.8)

project (VO)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB HEADERS "*.h")
#file(GLOB SOURCES "*.cpp")

#add_executable(VO ${HEADERS} ${SOURCES}) 
add_executable(VO main.cpp vo.cpp ${HEADERS})
target_link_libraries(VO ${OpenCV_LIBS})