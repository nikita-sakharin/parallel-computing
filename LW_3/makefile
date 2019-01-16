CC=gcc
CFLAGS=-std=c99 -Werror -pedantic -Wall -Wextra -O3
LDFLAGS=
LIBS=-lm -fopenmp
SOURCES=LW_3.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=LW_3

all: $(EXECUTABLE)

$(EXECUTABLE): %: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	rm -frd $(EXECUTABLE) $(OBJECTS)
