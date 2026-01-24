cmake_minimum_required(VERSION 3.20)

project(mcppfa VERSION 0.1.0 LANGUAGES CXX)

add_library(mcppfa INTERFACE)

target_include_directories(mcppfa INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(mcppfa INTERFACE cxx_std_17)

include(CMakePackageConfigHelpers)

install(TARGETS mcppfa EXPORT mcppfaTargets)

install(DIRECTORY include/ DESTINATION include)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mcppfaConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mcppfaConfig.cmake
  INSTALL_DESTINATION lib/cmake/mcppfa
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/mcppfaConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/mcppfaConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/mcppfaConfigVersion.cmake
  DESTINATION lib/cmake/mcppfa
)

install(EXPORT mcppfaTargets
  NAMESPACE mcppfa::
  DESTINATION lib/cmake/mcppfa
)
