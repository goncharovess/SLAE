# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/goncharovess/SLAE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/goncharovess/SLAE/build

# Include any dependencies generated for this target.
include Tests/CMakeFiles/run_test_Tridiagonal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Tests/CMakeFiles/run_test_Tridiagonal.dir/compiler_depend.make

# Include the progress variables for this target.
include Tests/CMakeFiles/run_test_Tridiagonal.dir/progress.make

# Include the compile flags for this target's objects.
include Tests/CMakeFiles/run_test_Tridiagonal.dir/flags.make

Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o: Tests/CMakeFiles/run_test_Tridiagonal.dir/flags.make
Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o: ../Tests/test_Tridiagonal.cpp
Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o: Tests/CMakeFiles/run_test_Tridiagonal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/goncharovess/SLAE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o"
	cd /home/goncharovess/SLAE/build/Tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o -MF CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o.d -o CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o -c /home/goncharovess/SLAE/Tests/test_Tridiagonal.cpp

Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.i"
	cd /home/goncharovess/SLAE/build/Tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/goncharovess/SLAE/Tests/test_Tridiagonal.cpp > CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.i

Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.s"
	cd /home/goncharovess/SLAE/build/Tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/goncharovess/SLAE/Tests/test_Tridiagonal.cpp -o CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.s

# Object files for target run_test_Tridiagonal
run_test_Tridiagonal_OBJECTS = \
"CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o"

# External object files for target run_test_Tridiagonal
run_test_Tridiagonal_EXTERNAL_OBJECTS =

Tests/run_test_Tridiagonal: Tests/CMakeFiles/run_test_Tridiagonal.dir/test_Tridiagonal.cpp.o
Tests/run_test_Tridiagonal: Tests/CMakeFiles/run_test_Tridiagonal.dir/build.make
Tests/run_test_Tridiagonal: /usr/lib/x86_64-linux-gnu/libgtest_main.a
Tests/run_test_Tridiagonal: /usr/lib/x86_64-linux-gnu/libgtest.a
Tests/run_test_Tridiagonal: Tests/CMakeFiles/run_test_Tridiagonal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/goncharovess/SLAE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable run_test_Tridiagonal"
	cd /home/goncharovess/SLAE/build/Tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run_test_Tridiagonal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Tests/CMakeFiles/run_test_Tridiagonal.dir/build: Tests/run_test_Tridiagonal
.PHONY : Tests/CMakeFiles/run_test_Tridiagonal.dir/build

Tests/CMakeFiles/run_test_Tridiagonal.dir/clean:
	cd /home/goncharovess/SLAE/build/Tests && $(CMAKE_COMMAND) -P CMakeFiles/run_test_Tridiagonal.dir/cmake_clean.cmake
.PHONY : Tests/CMakeFiles/run_test_Tridiagonal.dir/clean

Tests/CMakeFiles/run_test_Tridiagonal.dir/depend:
	cd /home/goncharovess/SLAE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/goncharovess/SLAE /home/goncharovess/SLAE/Tests /home/goncharovess/SLAE/build /home/goncharovess/SLAE/build/Tests /home/goncharovess/SLAE/build/Tests/CMakeFiles/run_test_Tridiagonal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Tests/CMakeFiles/run_test_Tridiagonal.dir/depend
