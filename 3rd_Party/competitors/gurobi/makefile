OBJDIR=./o
EXEDIR=.

all: LAP_gurobi.o
	g++ -g $(OBJDIR)/LAP_gurobi.o -I/opt/gurobi952/linux64/include/ -L/opt/gurobi952/linux64/lib -lgurobi_c++ -lgurobi95 -o $(EXEDIR)/LAP_gurobi

LAP_gurobi.o: LAP_gurobi.cpp
	g++ -g -c LAP_gurobi.cpp -o $(OBJDIR)/LAP_gurobi.o -I/opt/gurobi952/linux64/include/ -L/opt/gurobi952/linux64/lib -lgurobi_c++ -lgurobi95

clean:
	rm -f $(EXEDIR)/LAP_gurobi
	rm -f $(OBJDIR)/*.o