TARGET=../lib/boda
CPP=g++
CPPFLAGS=-Wall -O3 -g -std=c++0x -rdynamic -fPIC -I/usr/include/python2.7 -I/usr/include/octave-3.8.1 -I/usr/include/octave-3.8.1/octave -fopenmp -Wall
LDFLAGS=-lboost_system -lboost_filesystem -lboost_iostreams -lboost_regex -lpython2.7 -loctave -loctinterp -fopenmp -lturbojpeg
# generally, there is no need to alter the makefile below this line
VPATH=../src ../src/gen ../src/ext
OBJS=$(shell cat ../src/obj_list | grep -v ^\#) $(shell cat ../src/gen/gen_objs | grep -v ^\#)
ifeq ($(shell test -L makefile ; echo $$? ),1)
all : 
	@echo "makefile should be a symbolic link to avoid accidentally building in the root directory ... attempting to create ./obj,./lib,./run, symlink makefile in ./obj, and recurse make into ./obj"
	-mkdir obj run lib
	-ln -sf ../makefile obj/makefile
	cd obj && $(MAKE)
else
.SUFFIXES:
%.o : %.cc
	$(CPP) $(CPPFLAGS) -MMD -c $<
%.d : %.cc
	@touch $@
%.o : %.cpp
	$(CPP) $(CPPFLAGS) -MMD -c $<
%.d : %.cpp
	@touch $@
DEPENDENCIES = $(OBJS:.o=.d)
# FIXME: unfortunately, make will continue even if there is an error executing prebuild.py (which is bad), and
# somtimes likes to run prebuild.py multiple times (which not great, but okay). the solutions i know of seem
# worse than the problem, though.
$(info py_prebuild_hook:  $(shell python ../pysrc/prebuild.py )) 
LIBTARGET=../lib/libboda.so
all : $(TARGET) # $(LIBTARGET)
$(TARGET): $(OBJS)
	$(CPP) $(CPPFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)
$(LIBTARGET): $(OBJS)
	$(CPP) -shared $(CPPFLAGS) -o $(LIBTARGET) $(OBJS) $(LDFLAGS)
.PHONY : clean
clean:
	-rm -f $(TARGET) $(OBJS) $(DEPENDENCIES)
include $(DEPENDENCIES)
endif
