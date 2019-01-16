CC=mpicc
CFLAGS=-std=c99 -Werror -pedantic -Wall -Wextra -O3
LDFLAGS=
LIBS=-lm
SOURCES=CP.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=CP

all: $(EXECUTABLE)

$(EXECUTABLE): %: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	rm -frd $(EXECUTABLE) $(OBJECTS)
