#/*********************************************************************** 
# deformable_depth: Copyright (C) 2014 - James Steven Supancic III
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2, or (at your option)
#   any later version.                                                                                                                                      
#                                                                                                                                                           
#   This program is distributed in the hope that it will be useful,                                                                                         
#   but WITHOUT ANY WARRANTY; without even the implied warranty of                                                                                          
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                                                                           
#   GNU General Public License for more details.                                                                                                            
# ***********************************************************************/                                                                                  

cmake_minimum_required(VERSION 2.6.4)
PROJECT(deformable_depth)                                                                                                                                           
                                                                                                                                                            
# Set the C and CXX compiler versions                                                                                                                       
# We require 4.6 for range based for (for-each loop)                                                                                                        
#SET(CMAKE_C_COMPILER "/usr/bin/gcc-4.6.3")                                                                                                                  
#SET(CMAKE_CXX_COMPILER "/usr/bin/g++-4.6.3")                                                                                                                
SET(CMAKE_C_COMPILER "/usr/bin/gcc-4.9.3")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9.3")                                                                                                                
#set( CMAKE_VERBOSE_MAKEFILE on )
             
# tell Cmake where to find OpenCL configurations...                                                                                                         
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})                                                                                                            
# we will be using OpenCV for this project                                                                               
FIND_PACKAGE( OpenCV REQUIRED )                                                                                                                             
# we will also be using Boost (Mostly for filesystem operations)  
FIND_PACKAGE( Boost COMPONENTS REQUIRED system filesystem thread regex)
include(local.cmake)
# I use OpenSSL for hashing data (for caches)
FIND_PACKAGE( OpenSSL REQUIRED )
# We use levelDB for interop with Deep Caffe
FIND_PACKAGE( LevelDB REQUIRED )

# make OpenNI2 dependency (for reading .ONI videos), optional.
IF (DEFINED DD_ENABLE_OPENNI)
  # require OpenNI
  set(OPENNI_INCLUDE_DIR "/home/jsupanci/workspace/OpenNI-Linux-x64-2.2/Include")
  set(OPENNI_LIB_DIR "/home/jsupanci/workspace/OpenNI-Linux-x64-2.2/Redist")
  add_definitions(-DDD_ENABLE_OPENNI)
  set(DD_ONI2_FLAGS "-lOpenNI2")
ENDIF (DEFINED DD_ENABLE_OPENNI)

# if we are rendering new synthetic examples we need the following
# for fast rendering of the model we need: 
find_package(GLUT)
find_package(OpenGL)       
IF (DEFINED DD_ENABLE_HAND_SYNTH)
  # require OpenCL?
  # find_package(OpenCL REQUIRED)
  # This gives a robust model from which to extract training data
  FIND_PACKAGE(LibHand REQUIRED
    PATHS "/home/jsupanci/workspace/libhand-public/hand_cpp/dist"
    NO_MODULE
    NO_DEFAULT_PATH)
  # define a symbol for the compiler to know the behavior is enabled  
  add_definitions(-DDD_ENABLE_HAND_SYNTH)
ENDIF (DEFINED DD_ENABLE_HAND_SYNTH)
# Use Python for plotting
# SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -I/usr/include/python2.7/ -lpython2.7")
      
# check if off-the-shelf MRF Optimization libraries are requested
IF (DEFINED DD_ENABLE_MRF_OPTIMIZATION)
  set(MRF_LIBS " MRF ") 
ENDIF (DEFINED DD_ENABLE_MRF_OPTIMIZATION)

# check if caffe deep learning has been configured
IF (DEFINED DD_ENABLE_DEEP_CAFFE)
   set(CAFFE_LIBS "caffe" "protobuf" "atlcblas")
ENDIF (DEFINED DD_ENABLE_DEEP_CAFFE)

# Setup Qt for clipboard stuff
FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

# we will be using c++0x and showing all warnings.                        
# -O0 no optimization, for debugging clarity
# -O3 fast exec, slow build.
# -Og something between O0 and O3
# -fopenmp enables OpenMP
#  -fno-omit-frame-pointer -fsanitize=address # for debugging memory corruptions
SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${DD_ONI2_FLAGS} ")
SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address ")
SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -DBOOST_DISABLE_ASSERTS ")
SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -ljsoncpp -ltbb -lgmp -fopenmp -Werror -Wreturn-type -mno-avx -O3 -std=gnu++1y")
SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -DDD_CXX11 -g -ggdb -march=native -L. -lgomp -frtti -Wno-deprecated ")
SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -lexiv2 ")
# SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -gdwarf-2 -gstrict-dwarf " )
# GLOB some groups... to compile                                                                                                     
#FILE(GLOB Main_src_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
#FILE(GLOB Main_cpp
# "{util/,MetaData,Trees,Rendering,Models,Analysis,Model,System,Parts,src}*.cpp")
AUX_SOURCE_DIRECTORY(src Main_cpp)
AUX_SOURCE_DIRECTORY(System Main_cpp)
AUX_SOURCE_DIRECTORY(util Main_cpp)
AUX_SOURCE_DIRECTORY(MetaData Main_cpp)
AUX_SOURCE_DIRECTORY(Trees Main_cpp)
AUX_SOURCE_DIRECTORY(Rendering Main_cpp)
AUX_SOURCE_DIRECTORY(Analysis Main_cpp)
AUX_SOURCE_DIRECTORY(Model Main_cpp)
AUX_SOURCE_DIRECTORY(Parts Main_cpp)
AUX_SOURCE_DIRECTORY(Tracking Main_cpp)
AUX_SOURCE_DIRECTORY(IBR Main_cpp)
AUX_SOURCE_DIRECTORY(FLECM Main_cpp)
AUX_SOURCE_DIRECTORY(toytracer Main_cpp)

# libs and incs
include_directories("${PROJECT_SOURCE_DIR}" "${Boost_INCLUDE_DIR}" "/usr/include/ni"
  " ../liblinear-1.92/" ${LibHand_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIR}
  " ../MRF_Minimization/MRF2.2/ " " ../deep_caffe/include " " ../deep_caffe/build/src "
   Tracking System Parts src "${LevelDB_INCLUDE_PATH}"
   toytracer util  MetaData Trees Rendering Analysis Model FLECM IBR ../libhand/hand_cpp/source/
)       
link_directories(${Boost_LIBRARY_DIR} ${OPENNI_LIB_DIR} 
  "../MRF_Minimization/MRF2.2/" "../deep_caffe/build/lib/")
                             
# Disable all optimizations for Kahan summations.
set_source_files_properties(Kahan.cpp PROPERTIES COMPILE_FLAGS -O0)

message("libhand libs = " ${LibHand_LIBRARIES})
message("OpenCV  libs = " ${OpenCV_LIBS})
message("caffe   libs = " ${CAFFE_LIBS})
message("MRF     libs = " ${MRF_LIBS})
message("QT      libs = " ${QT_LIBRARIES})
message("all     libs = " ${OpenCV_LIBS} ${Boost_LIBRARIES}
  ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} 
  ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OPENSSL_LIBRARIES} 
  ${LibHand_LIBRARIES} ${LevelDB_LIBRARY} ${MRF_LIBS} ${CAFFE_LIBS}   
  ${QT_LIBRARIES})

# configure the build
add_executable(deformable_depth ${Main_cpp}) 
TARGET_LINK_LIBRARIES(deformable_depth ${OpenCV_LIBS} ${Boost_LIBRARIES}
  ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} 
  ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OPENSSL_LIBRARIES} 
  ${LibHand_LIBRARIES} ${LevelDB_LIBRARY} ${MRF_LIBS} ${CAFFE_LIBS}   
  ${QT_LIBRARIES}
)                                                              
