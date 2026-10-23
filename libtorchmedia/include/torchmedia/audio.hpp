#ifndef LIBTORCHMEDIA_AUDIO
#define LIBTORCHMEDIA_AUDIO
// clang-format off
// Each header includes what it uses (_functional.hpp pulls in _functional_filtering.hpp for deemphasis and
// loudness), so the order below is only for reading. Keep this block unsorted.
#include "_audio/_functional_filtering.hpp"
#include "_audio/_functional.hpp"
#include "_audio/_functional_beamforming.hpp"
#include "_audio/_functional_alignment.hpp"
#include "_audio/_io.hpp"
#include "_audio/_separation.hpp"
#include "_audio/_transform.hpp"
// clang-format on
#endif // LIBTORCHMEDIA_AUDIO
