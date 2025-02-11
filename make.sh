#mkdir build && cd build
#cmake .. -DCMAKE_TOOLCHAIN_FILE=[C:/Users/tljak/projects/vcpkg/scripts/buildsystems/vcpkg.cmake]
#cmake -B build -S . -D SDL2_DIR="C:\Program Files (x86)/SDL2/cmake" -D GLEW_ROOT="C:\Program Files (x86)/glew"
#cmake --build .
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="C:/Users/tljak/projects/vcpkg/scripts/buildsystems/vcpkg.cmake"