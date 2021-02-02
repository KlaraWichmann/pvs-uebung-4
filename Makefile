.ONESHELL:
GCC_FLAGS = -Wall -fopenmp
GCC_L = -lOpenCL

ASSIGNMENT_GROUP=B
ASSIGNMENT_NUMBER=04
ASSIGNMENT_TITLE=pvs$(ASSIGNMENT_NUMBER)-grp$(ASSIGNMENT_GROUP)

.PHONY: build
build: hello_world matmult

.PHONY: debug
debug: GCC_FLAGS += -g
debug: build

.PHONY: hello_world
hello_world:
	g++ $(GCC_FLAGS) hello_world.cpp -o hello_world $(GCC_L)

.PHONY: matmult
matmult:
	g++ $(GCC_FLAGS) matmult.cpp -o matmult $(GCC_L)


.PHONY: test
test: build
	./hello_world
	./matmult

.PHONY: clean
clean:
	rm hello_world

.PHONY: codeformat
codeformat:
	clang-format -i *.[ch]pp

FILES=Makefile README.md *.[ch]pp

ASSIGNMENT_DIR=$(ASSIGNMENT_TITLE)
TARBALL_NAME=$(ASSIGNMENT_TITLE)-piekarski-wichmann-ruckel.tar.gz
.PHONY: tarball
tarball:
	[ -z "$(TARBALL_NAME)" ] || rm $(TARBALL_NAME)
	mkdir $(ASSIGNMENT_DIR)
	for f in $(FILES); do cp $$f $(ASSIGNMENT_DIR); done
	tar zcvf $(TARBALL_NAME) $(ASSIGNMENT_DIR)
	rm -fr $(ASSIGNMENT_DIR)
