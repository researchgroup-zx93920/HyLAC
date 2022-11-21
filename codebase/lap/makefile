OBJDIR=./o
EXEDIR=.

all: main.o culap.o f_culap.o f_cutils.o
	g++ $(OBJDIR)/main.o $(OBJDIR)/culap.o $(OBJDIR)/f_culap.o $(OBJDIR)/f_cutils.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o $(EXEDIR)/cuLAP

main.o: main.cu
	nvcc -arch=sm_80 -c main.cu -o $(OBJDIR)/main.o

culap.o: culap.cu
	nvcc -arch=sm_80 -c culap.cu -o $(OBJDIR)/culap.o

f_culap.o: f_culap.cu
	nvcc -arch=sm_80 -c f_culap.cu -o $(OBJDIR)/f_culap.o

f_cutils.o: f_cutils.cu
	nvcc -arch=sm_80 -c f_cutils.cu -o $(OBJDIR)/f_cutils.o

clean:
	rm -f $(EXEDIR)/cuLAP
	rm -f $(OBJDIR)/*.o