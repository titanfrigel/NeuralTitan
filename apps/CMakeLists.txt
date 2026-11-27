# titan.manifest sets the UTF-8 code page; MSVC embeds a listed .manifest.
add_executable(titan titan/main.cpp titan/commands.cpp titan/titan.manifest)
target_link_libraries(titan PRIVATE titan_runtime titan_core)
