cmake --fresh -S . -B build \
-DWarpX_amrex_src=$PWD/amrex \
-DWarpX_picsar_src=$PWD/picsar \
-DWarpX_COMPUTE=CUDA \
-DWarpX_OPENPMD=OFF \
-DWarpX_FFT=ON
