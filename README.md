# Skyscrappers puzzle
Program sloving [Skyscrappers](http://wiki.logic-masters.de/index.php?title=Skyscrapers/en "Puzzlewiki") puzzle on GPU for master's thesis.
Generate with ```CMAKE_CUDA_FLAGS``` to set Compute Capability of Your device.
##### Example
```cmake ../ -DCMAKE_CUDA_FLAGS="-arch=sm_30"```
```cmake ~/Skyscrapers_source -DCMAKE_CUDA_FLAGS="-arch=sm_50"```

