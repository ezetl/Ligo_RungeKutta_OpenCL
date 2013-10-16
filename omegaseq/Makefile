CC = gcc

CFLAGS = -Wall -lm -fPIC -std=c99

SOURCES = ode45.c pn_rhs.c
OBJECTS = $(SOURCES:.c=.o)
MAIN = evol.c
BIN = pn_cpu
BIN_OMP = $(BIN)_omp

EXTRAS = omega_rhs.inc

all: cpu cpu_omp

#$(OBJECTS): %.o : %.c
#	$(CC) $(CFLAGS) 

cpu: $(OBJECTS)
	$(CC) $(CFLAGS) -o $(BIN) $(MAIN) $(OBJECTS)

cpu_omp ...: CFLAGS += -fopenmp
cpu_omp: $(OBJECTS)
	$(CC) $(CFLAGS) -o $(BIN_OMP) $(MAIN) $(OBJECTS)

clean:
	rm -f $(BIN) $(BIN_OMP) $(OBJECTS)

.PHONY: $(SOURCES)
