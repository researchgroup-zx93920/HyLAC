OBJDIR=./o
EXEDIR=.

all: lap.o main.o
	g++ $(OBJDIR)/lap.o $(OBJDIR)/main.o -o $(EXEDIR)/jvc

lap.o: lap.cpp
	g++ -c lap.cpp -o $(OBJDIR)/lap.o

main.o: main.cpp
	g++ -c main.cpp -o $(OBJDIR)/main.o

clean:
	rm -f $(EXEDIR)/jvc
	rm -f $(OBJDIR)/*.o