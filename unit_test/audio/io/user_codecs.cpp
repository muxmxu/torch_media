// A consumer's own build of dr_wav and stb_vorbis (the same vendored files), compiled outside torchmedia.
#define DR_WAV_IMPLEMENTATION
#include <torchmedia/_audio/_vendor/dr_wav.h>
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wtautological-compare" // as in _io_stb_vorbis_impl.hpp
#endif
#include <torchmedia/_audio/_vendor/stb_vorbis.c>
