# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_SOURCE_DIR = /home/deepshukla/aopt-exercise6/aopt-exercise6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/deepshukla/aopt-exercise6/aopt-exercise6/build

# Include any dependencies generated for this target.
include GradientDescent/CMakeFiles/GradientDescent.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include GradientDescent/CMakeFiles/GradientDescent.dir/compiler_depend.make

# Include the progress variables for this target.
include GradientDescent/CMakeFiles/GradientDescent.dir/progress.make

# Include the compile flags for this target's objects.
include GradientDescent/CMakeFiles/GradientDescent.dir/flags.make

GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.o: GradientDescent/CMakeFiles/GradientDescent.dir/flags.make
GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.o: /home/deepshukla/aopt-exercise6/aopt-exercise6/GradientDescent/main.cc
GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.o: GradientDescent/CMakeFiles/GradientDescent.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/deepshukla/aopt-exercise6/aopt-exercise6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.o"
	cd /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.o -MF CMakeFiles/GradientDescent.dir/main.cc.o.d -o CMakeFiles/GradientDescent.dir/main.cc.o -c /home/deepshukla/aopt-exercise6/aopt-exercise6/GradientDescent/main.cc

GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GradientDescent.dir/main.cc.i"
	cd /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/deepshukla/aopt-exercise6/aopt-exercise6/GradientDescent/main.cc > CMakeFiles/GradientDescent.dir/main.cc.i

GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GradientDescent.dir/main.cc.s"
	cd /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/deepshukla/aopt-exercise6/aopt-exercise6/GradientDescent/main.cc -o CMakeFiles/GradientDescent.dir/main.cc.s

# Object files for target GradientDescent
GradientDescent_OBJECTS = \
"CMakeFiles/GradientDescent.dir/main.cc.o"

# External object files for target GradientDescent
GradientDescent_EXTERNAL_OBJECTS =

Build/bin/GradientDescent: GradientDescent/CMakeFiles/GradientDescent.dir/main.cc.o
Build/bin/GradientDescent: GradientDescent/CMakeFiles/GradientDescent.dir/build.make
Build/bin/GradientDescent: GradientDescent/CMakeFiles/GradientDescent.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/deepshukla/aopt-exercise6/aopt-exercise6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../Build/bin/GradientDescent"
	cd /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GradientDescent.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
GradientDescent/CMakeFiles/GradientDescent.dir/build: Build/bin/GradientDescent
.PHONY : GradientDescent/CMakeFiles/GradientDescent.dir/build

GradientDescent/CMakeFiles/GradientDescent.dir/clean:
	cd /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent && $(CMAKE_COMMAND) -P CMakeFiles/GradientDescent.dir/cmake_clean.cmake
.PHONY : GradientDescent/CMakeFiles/GradientDescent.dir/clean

GradientDescent/CMakeFiles/GradientDescent.dir/depend:
	cd /home/deepshukla/aopt-exercise6/aopt-exercise6/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/deepshukla/aopt-exercise6/aopt-exercise6 /home/deepshukla/aopt-exercise6/aopt-exercise6/GradientDescent /home/deepshukla/aopt-exercise6/aopt-exercise6/build /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent /home/deepshukla/aopt-exercise6/aopt-exercise6/build/GradientDescent/CMakeFiles/GradientDescent.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : GradientDescent/CMakeFiles/GradientDescent.dir/depend
