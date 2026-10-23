#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
#define LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
// Aggregator for the torchmedia::vision::transform layer (torchvision 0.20.1 transforms.v2 parity), themed like
// torchvision's modules: misc (shared helpers, containers, _misc / _augment classes), geometry, color.
// See develop_log/2026-10-23/progress07, progress09, progress10.
#include "_transform_color.hpp"
#include "_transform_geometry.hpp"
#include "_transform_misc.hpp"
#endif // LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
