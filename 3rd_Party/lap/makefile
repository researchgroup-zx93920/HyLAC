all: main.o culap.o f_culap.o f_cutils.o
	g++ o/main.o o/culap.o o/f_culap.o o/f_cutils.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o ../tests/cuLAP

main.o: main.cu
	nvcc -arch=sm_60 -c main.cu -o o/main.o

culap.o: culap.cu
	nvcc -arch=sm_60 -c culap.cu -o o/culap.o

f_culap.o: f_culap.cu
	nvcc -arch=sm_60 -c f_culap.cu -o o/f_culap.o

f_cutils.o: f_cutils.cu
	nvcc -arch=sm_60 -c f_cutils.cu -o o/f_cutils.o

clean:
	rm ../tests/cuLAP
	rm o/*.o