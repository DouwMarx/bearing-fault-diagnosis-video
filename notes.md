Options from simple to complex:
- Treat each pixel in video as signal over time and filter out the gear mesh frequency/ retain only the fault frequency and its harmonics/ shaft frequency modulated by fault frequency.
- Use a row of pixels only, which becomes a wave when viewed through time. Do the 2D Fourier transform in time and space. Then, if you want to sharpen the gear, you choose only the (diagonal) elements that represent traveling waves at the wave speed of interest. If you are interested in the fault, you want to eliminate the traveling (on dianoal) elvement, and keep only elements that move straight down
- Use the whole video, and do the 3D fourier transform in time and 2 spatial dimensions. Then, you can design a 3D filter.

Another pretty simple option is to do something like 2d or 3d FFT, but then to take the ratio of the magnitudes of the orignal and faulty video. Then you threshold this ratio and retain only dominant frequencies, which should indicate the wave of the fault. You can then overlay that mask on the original video to see where the fault is located spatially.

It is also possible to phase align the traveling waves quite easily after you had taken the 