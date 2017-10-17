
find_package( OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# g20 dependency
# add findg2o.cmake
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
FIND_PACKAGE( G2O REQUIRED )
FIND_PACKAGE( Eigen3 REQUIRED )
# CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

ADD_EXECUTABLE(vo main.cpp vo.cpp bundle.cpp bundle.h vo.h)
TARGET_LINK_LIBRARIES( vo
    ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_solver_cholmod g2o_stuff g2o_csparse_extension g2o_types_sba ${CSPARSE_LIBRARY})