cmake_minimum_required(VERSION 2.8)

PROJECT(ml_text_reader)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SUPPRESS_REGENERATION true)

SET(EXECUTABLE_OUTPUT_PATH ../bin/)


# list of the source files
set(Digit_Reader_Files ../main.cpp ../Network.cpp ../FileData.cpp ../network_analysis.cpp ../Network.h ../FileData.h ../network_analysis.h)

set(include_files ../eigen)


# define the executable
if(WIN32)
  add_executable(Digit_Reader WIN32 ${Digit_Reader_Files})

else()

  add_executable(Digit_Reader ${Digit_Reader_Files})

endif()
# in Windows + Visual Studio we need this to make it a console application
if(WIN32)
  if(MSVC)
 
    set_target_properties(Digit_Reader PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
  endif(MSVC)
endif(WIN32)

target_include_directories(Digit_Reader PUBLIC ${include_files})

# install application
