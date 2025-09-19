cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Find and enable OpenMP
find_package(OpenMP REQUIRED)

# Allow enabling AddressSanitizer for demo binaries (diagnostics only)
option(DEMO_USE_ASAN "Build demo targets with AddressSanitizer" OFF)

# Define the executable target first
add_executable(demo_test src/main.cpp src/utils.cpp src/kmeans.cpp src/vamana_graph.cpp src/search.cpp src/build.cpp src/pipeline.cpp)

# NEW: streaming kmeans executable
add_executable(stream_kmeans src/stream_kmeans.cpp)

# NEW: exbits_cost executable
add_executable(exbits_cost motivation/exbits_cost.cpp)

# --- Sanity check: Print the include path that will be used ---
message(STATUS "[Demo] Calculated include path: ${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Now, apply properties to the defined target
# Add the main project's include directory to our search path
target_include_directories(demo_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(stream_kmeans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(exbits_cost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Link against the main diskann library
# Assumes diskann library has been built.
target_link_libraries(demo_test PRIVATE Threads::Threads OpenMP::OpenMP_CXX diskann)
target_link_libraries(stream_kmeans PRIVATE Threads::Threads OpenMP::OpenMP_CXX diskann)
target_link_libraries(exbits_cost PRIVATE Threads::Threads OpenMP::OpenMP_CXX diskann)

# Ensure we don't pull in unused OpenMP runtimes via transitive deps
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(demo_test PRIVATE -Wl,--as-needed)
  target_link_options(stream_kmeans PRIVATE -Wl,--as-needed)
  target_link_options(exbits_cost PRIVATE -Wl,--as-needed)
endif()

# rabitq headers
include_directories(/home/danbai.wq/DiskANN/rabitq/rabitqlib)

# Add compiler flags for AVX2 and FMA support for performance
if (DEMO_USE_ASAN)
  target_compile_options(demo_test PRIVATE -fsanitize=address -fno-omit-frame-pointer -O1 -g)
  target_link_options(demo_test PRIVATE -fsanitize=address -static-libasan)
  target_compile_options(stream_kmeans PRIVATE -fsanitize=address -fno-omit-frame-pointer -O1 -g)
  target_link_options(stream_kmeans PRIVATE -fsanitize=address -static-libasan)
endif()

# Optional: liburing (for io_uring async I/O)
find_path(URING_INCLUDE_DIR liburing.h PATHS /usr/include /usr/local/include)
find_library(URING_LIB uring PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64)
if (URING_INCLUDE_DIR AND URING_LIB)
    message(STATUS "[Demo] Found liburing: ${URING_LIB}")
    target_compile_definitions(demo_test PRIVATE HAS_IO_URING=1)
    target_include_directories(demo_test PRIVATE ${URING_INCLUDE_DIR})
    target_link_libraries(demo_test PRIVATE ${URING_LIB})
else()
    message(STATUS "[Demo] liburing not found; io_uring disabled")
endif()

# Optional: libaio (Linux AIO)
find_path(LIBAIO_INCLUDE_DIR libaio.h PATHS /usr/include /usr/local/include)
find_library(LIBAIO_LIB aio PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64)
if (LIBAIO_INCLUDE_DIR AND LIBAIO_LIB)
    message(STATUS "[Demo] Found libaio: ${LIBAIO_LIB}")
    target_compile_definitions(demo_test PRIVATE HAS_LIBAIO=1)
    target_include_directories(demo_test PRIVATE ${LIBAIO_INCLUDE_DIR})
    target_link_libraries(demo_test PRIVATE ${LIBAIO_LIB})
else()
    message(STATUS "[Demo] libaio not found; Linux AIO disabled")
endif()