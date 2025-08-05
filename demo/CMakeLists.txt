cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Find and enable OpenMP
find_package(OpenMP REQUIRED)

# Define the executable target first
add_executable(demo_test src/main.cpp src/utils.cpp src/kmeans.cpp src/vamana_graph.cpp src/search.cpp)

# --- Sanity check: Print the include path that will be used ---
message(STATUS "[Demo] Calculated include path: ${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Now, apply properties to the defined target
# Add the main project's include directory to our search path
target_include_directories(demo_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Link against the main diskann library
# Assumes diskann library has been built.
target_link_libraries(demo_test PRIVATE Threads::Threads OpenMP::OpenMP_CXX diskann)

# Add compiler flags for AVX2 and FMA support for performance
target_compile_options(demo_test PRIVATE -mavx2 -mfma)