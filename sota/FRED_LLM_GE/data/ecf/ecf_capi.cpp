// C entry points so data/fred_events.py can call Prophesee's ECF decoder via ctypes.
#include "ecf_codec.h"

extern "C" {
size_t ecf_decompressed_size(const unsigned char *in) {
    ECF::Decoder decoder;
    return decoder.getDecompressedSize(reinterpret_cast<const std::uint32_t *>(in));
}

size_t ecf_decode(const unsigned char *in, size_t nbytes, void *out) {
    ECF::Decoder decoder;
    return decoder(in, in + nbytes, reinterpret_cast<ECF::EventCD *>(out));
}
}
