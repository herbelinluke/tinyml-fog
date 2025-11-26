# Pico ML Node - TinyML Anomaly Detection
cmake_minimum_required(VERSION 3.13)

# Initialize Pico SDK (must be before project())
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(pico_ml_node C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the SDK
pico_sdk_init()

# Add executable
add_executable(pico_ml_node
    main.cpp
    ml_model.cpp
)

# Link libraries
target_link_libraries(pico_ml_node
    pico_stdlib
    hardware_uart
)

# Enable USB serial output (for debugging and fog node communication)
pico_enable_stdio_usb(pico_ml_node 1)
pico_enable_stdio_uart(pico_ml_node 0)

# Create UF2 file for easy flashing
pico_add_extra_outputs(pico_ml_node)

