.PHONY: build run

build:
	g++ -g -O0 tester.cpp -o ref_tester
	cargo build -p oklab
	rustc -g --edition=2021 tester.rs --extern oklab=../../target/debug/liboklab.rlib -o our_tester
	rustc -g --edition=2021 checker.rs -o checker

run: build
	./checker

clean:
	rm -f ref_tester our_tester checker
