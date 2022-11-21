OBJDIR=./o
# EXEDIR=../test_execs
EXEDIR=.
DBGDIR=.

LDFLAGS=--no-relax

all: HungarianCUDA.cu
	nvcc -lineinfo --generate-code arch=compute_80,code=compute_80 HungarianCUDA.cu \
	-std=c++11 -lcuda -o $(EXEDIR)/LAP19

debug: HungarianCUDA.cu
	nvcc -rdc=true -G -g --generate-code arch=compute_70,code=compute_70 HungarianCUDA.cu \
	-std=c++11 -o $(DBGDIR)/LAP19_dbg

clean:
	rm -f $(EXEDIR)/LAP19
	rm -f $(DBGDIR)/LAP19_dbg
	rm -f *.o