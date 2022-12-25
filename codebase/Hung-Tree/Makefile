CUDAFLAGS = -g -Xcompiler -fopenmp -lineinfo -O3 -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80

all: main.o
	g++ o/main.o -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart -lgomp -o cuLAP

main.o: main.cu
	nvcc ${CUDAFLAGS} -c main.cu -o o/main.o

clean:
	rm -rf o/*.o cuLAP