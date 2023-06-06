PROJECT(ml_text_reader)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

SET(EXECUTABLE_OUTPUT_PATH ../bin/)

# list of the source files
set(dataRead_files ../FileData.cpp)
set(train_network_files ../Network.cpp)
set(main_function ../main.cpp
# define the executable
if(WIN32)
  add_executable(dataRead WIN32 ${dataRead_files})
  add_executable(train_network WIN32 ${train_network_files})
  add_executable(main WIN32 ${main_function})

else()
  add_executable(dataRead ${dataRead_files})
  add_executable(train_network ${train_network_files})
  add_executable(main ${main_function})
endif()
# in Windows + Visual Studio we need this to make it a console application
if(WIN32)
  if(MSVC)
    set_target_properties(dataRead PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(train_network PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(main PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
  endif(MSVC)
endif(WIN32)
# install application