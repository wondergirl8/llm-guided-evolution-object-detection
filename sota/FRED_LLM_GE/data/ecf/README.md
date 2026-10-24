`ecf_codec.cpp`, `ecf_codec.h` and `LICENSE` are copied unchanged from
[prophesee-ai/hdf5_ecf](https://github.com/prophesee-ai/hdf5_ecf) at commit
`8735ce7161a637cacc5f0affef48129a13ca5a3f` (Apache-2.0). FRED's `Event/events.hdf5`
files are compressed with this codec (HDF5 filter 36559).

`ecf_codec_export.h` and `ecf_capi.cpp` are ours. `data/fred_events.py` compiles
them with g++ into `build/libecf.so` on first use.
