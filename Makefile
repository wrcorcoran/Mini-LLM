.PHONY: configure build test clean

configure:
	mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Sanitize ..

build:
	cd build && cmake --build .

test: build
	cd build && ctest --output-on-failure

clean:
	rm -rf build