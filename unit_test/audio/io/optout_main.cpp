// TORCHMEDIA_NO_*_IMPLEMENTATION: this TU emits torchmedia's codec implementations except dr_wav and stb_vorbis, which
// user_codecs.cpp compiles itself, as a project already using them would. Linking at all is the main check (duplicate
// symbols otherwise); torchmedia must also leave the consumer's STB_VORBIS_HEADER_ONLY defined.
#define TORCHMEDIA_IO_IMPLEMENTATION
#define TORCHMEDIA_NO_DR_WAV_IMPLEMENTATION
#define TORCHMEDIA_NO_STB_VORBIS_IMPLEMENTATION
#define STB_VORBIS_HEADER_ONLY
#include <torchmedia.hpp>
#ifndef STB_VORBIS_HEADER_ONLY
#error "torchmedia undefined the consumer's STB_VORBIS_HEADER_ONLY"
#endif
#include "fixtures.hpp"
#include "test_util.hpp"

int main() {
    for (const auto &f: tm_fixture::fixtures) { // every codec decodes, whichever TU compiled it
        const auto x = torchmedia::audio::io::load_audio_from_memory(f.data, f.size).data;
        TM_CHECK(x.size(0) == f.num_channels && x.size(1) > 0);
    }
    return tm_test::summary("audio_test_io_optout");
}
