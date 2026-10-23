#ifndef LIBTORCHMEDIA_AUDIO
#define LIBTORCHMEDIA_AUDIO
// clang-format off
// Order matters: _functional_filtering.hpp must precede _functional.hpp (the latter uses lfilter
// for deemphasis). Keep this block unsorted.
#include "_audio/_functional_filtering.hpp"
#include "_audio/_functional.hpp"
#include "_audio/_functional_beamforming.hpp"
#include "_audio/_functional_alignment.hpp"
#include "_audio/_io.hpp"
#include "_audio/_separation.hpp"
#include "_audio/_transform.hpp"
// clang-format on
#endif // LIBTORCHMEDIA_AUDIO
