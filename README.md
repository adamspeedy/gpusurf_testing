# GpuSurf Testing
This package is meant to help understand tuning the gpusurf parameters. Mainly to do with the extraction and stereo matching of images.


## Instillation
```
mkdir build 
cd build
cmake ..
make
```
## Usage
once build the executable file in the build folder can be run from the terminal using

```
./gpusurf_run
```

## Dependencies
- Opencv 4.0
- CUDA
- Eigen3

other dependencies like cudpp are linked in the CMakeLists 