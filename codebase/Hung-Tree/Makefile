CUDAFLAGS = -g -Xcompiler -fopenmp -lineinfo -O3 -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70

all: main.o
	g++ o/main.o -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart -lgomp -o cuLAP

main.o: main.cu
	nvcc ${CUDAFLAGS} -c main.cu -o o/main.o

clean:
	rm -rf o/*.o cuLAP