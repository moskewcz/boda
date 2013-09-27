TARGET=../lib/boda
VPATH=../src ../src/gen
OBJS=str_util.o boda.o pugixml.o results_io.o boda_base.o geom_prim.o pyif.o octif.o model.o
CPP=g++
CPPFLAGS=-Wall -O3 -g -std=c++0x -rdynamic -I/usr/include/python2.7 -I/usr/include/octave-3.6.4 -I/usr/include/octave-3.6.4/octave -fopenmp
LDFLAGS=-lboost_system -lboost_filesystem -lpython2.7 -loctave -loctinterp -fopenmp

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

$(info py_prebuild_hook:  $(shell python ../pysrc/prebuild.py ))

all : $(TARGET) 
$(TARGET): $(OBJS)
	$(CPP) $(CPPFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

.PHONY : clean
clean:
	-rm -f $(TARGET) $(OBJS) $(DEPENDENCIES)

include $(DEPENDENCIES)

endif
