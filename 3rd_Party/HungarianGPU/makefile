all: HungarianCUDA.cu
	nvcc -rdc=true -lineinfo --generate-code arch=compute_70,code=compute_70 HungarianCUDA.cu \
	-std=c++11 -o LAP19

clean:
	rm -f LAP19
	rm -f *.o