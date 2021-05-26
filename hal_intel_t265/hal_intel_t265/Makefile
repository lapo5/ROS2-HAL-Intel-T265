LIBNAME           = intel_t265

CCFLAGS           = -Wall -Wextra -Wno-narrowing -std=c++17 -shared -fPIC
INCLUDEFLAGS      = -isystem../../dependencies/install/include \
					-isystem../../dependencies/install/include/libserial \
					-isystem./include \
					`python3 -m pybind11 --includes`
OPTIMIZATIONFLAGS = -O3 -mtune=native -march=native -DNDEBUG
PARALLELISMFLAGS  = -fopenmp -ltbb
PYBIND11FLAGS     = -shared -fPIC `python3 -m pybind11 --includes` -o ${LIBNAME}`python3-config --extension-suffix` 

LDFLAGS           = -lblas -llapack -DHAVE_CBLAS=1 -ltbb -lrealsense2


all:
	g++ ${CCFLAGS} ${INCLUDEFLAGS} ${OPTIMIZATIONFLAGS} ${PARALLELISMFLAGS} ${LIBNAME}.cpp -o ${LIBNAME}`python3-config --extension-suffix` ${LDFLAGS}

clean:
	rm -f ${LIBNAME}`python3-config --extension-suffix`
	rm -f test_cpp
