add_executable(titan titan/main.cpp titan/commands.cpp)
target_link_libraries(titan PRIVATE titan_runtime titan_core)
