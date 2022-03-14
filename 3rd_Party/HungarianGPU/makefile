all: HungarianCUDA.o
#	nvcc --generate-code arch=compute_35,code=compute_35 HungarianCUDA.cu \-std=c++11 -o ../tests/HungarianCUDA
	g++ HungarianCUDA.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o ../tests/HungarianCUDA
#	-arch=sm_60 -c HungarianCUDA.cu -std=c++11 -o ../tests/HungarianCUDA
HungarianCUDA.o: HungarianCUDA.cu
	nvcc -arch=sm_60 -c HungarianCUDA.cu -o HungarianCUDA.o

clean:
	rm ../tests/HungarianCUDA
	rm *.o