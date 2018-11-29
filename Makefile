CFLAGS+=-I./mpich/include
LDFLAGS+=-L./mpich/lib -Wl,-rpath,./mpich/lib
LDLIBS+=-lmpi