# Makefile -- libkdl (Kernel Dynamic Linker) build system
#
# Targets:
#   make libkdl.so      -- shared library (no link-time GPU deps; uses dlopen)
#   make bench_dispatch  -- dispatch overhead microbenchmark
#   make all             -- both
#   make clean           -- remove build artifacts

CC      ?= gcc
CFLAGS  := -O2 -Wall -Wextra -Wno-unused-parameter -fPIC -std=c11
LDFLAGS := -ldl -lm -lpthread

SRCDIR  := .
BENCHDIR := ../benchmarks

.PHONY: all clean

all: libkdl.so bench_dispatch runtime_select_poc bench_layers

# -- Shared library (zero link-time GPU dependencies) --
libkdl.so: $(SRCDIR)/kdl.c $(SRCDIR)/kdl.h
	$(CC) $(CFLAGS) -shared -o $@ $(SRCDIR)/kdl.c $(LDFLAGS)

# -- Dispatch overhead benchmark --
bench_dispatch: $(BENCHDIR)/bench_dispatch.c libkdl.so
	$(CC) $(CFLAGS) -I$(SRCDIR) -o $@ $(BENCHDIR)/bench_dispatch.c \
		-L. -lkdl $(LDFLAGS) -Wl,-rpath,'$$ORIGIN'

# -- runtime_select proof-of-concept (standalone, no libkdl dependency) --
runtime_select_poc: $(SRCDIR)/runtime_select_poc.c
	$(CC) $(CFLAGS) -o $@ $(SRCDIR)/runtime_select_poc.c -ldl

# -- Layer-by-layer CUDA driver stack benchmark --
bench_layers: $(SRCDIR)/bench_layers.c
	$(CC) -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 -o bench_layers $(SRCDIR)/bench_layers.c -ldl

clean:
	rm -f libkdl.so bench_dispatch runtime_select_poc bench_layers
