cmake_minimum_required(VERSION 3.15)
project(hailo.js)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

set(NODE_ADDON_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/node-addon-api")

# Include directories
include_directories(
        ${NODE_ADDON_API_DIR}
        "/users/nodify/.nvm/versions/node/v20.11.1/include/node"
        "./include/hailo"
        "./include"
        "src/include"
)

# Make sure CLion recognizes standard C++ headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add definitions for node-addon-api
add_definitions(
        -DNAPI_CPP_EXCEPTIONS
        -DDEBUG=1
        -DNODE_API_DEBUG=1
        -D_DEBUG=1
)

# Source files
set(SOURCE_FILES
        "src/addon.cpp"
        "src/hailo_device.cpp"
        "src/detection_utils.cpp"
)

# Add all source files for intellisense
file(GLOB_RECURSE HEADER_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/old/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/old/*.hpp"
)

# Create a library target (not an executable since this is a Node.js addon)
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

# Link against HailoRT
target_link_libraries(${PROJECT_NAME}
        pthread
        dl
)

# Set output name to match binding.gyp
set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME "hailo.js"
        PREFIX ""
        SUFFIX ".node"
)

# Add custom target for running npm build
add_custom_target(npm_build
        COMMAND npm run build
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running npm build..."
        VERBATIM
)