OBJDIR=./o
EXEDIR=../test_execs

all: HungarianCUDA.cu
	nvcc -rdc=true -lineinfo --generate-code arch=compute_70,code=compute_70 HungarianCUDA.cu \
	-std=c++11 -o $(EXEDIR)/LAP19

clean:
	rm -f $(EXEDIR)/LAP19
	rm -f *.o