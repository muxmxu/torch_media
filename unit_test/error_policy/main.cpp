// The error policy is the user's choice (develop_log/2026-10-24/progress05). This file builds twice:
// error_policy_test_throw (default) checks that every call below throws the expected exception and records nothing;
// error_policy_test_silent (LIB_TORCH_MEDIA_NO_EXCEPTIONS) checks the same calls return the documented fallback and
// that torchmedia::last_error() names the failure. Silent-only sections cover the composite and constructor paths.
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <filesystem>
#include <fstream>
#include <functional>
#include <thread>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <unistd.h>
#include "test_util.hpp"

namespace fn = torchmedia::audio::functional;
namespace tf = torchmedia::audio::transform;
namespace aio = torchmedia::audio::io;
namespace sep = torchmedia::audio::separation;
namespace kd = torchmedia::audio::compliance::kaldi;
namespace vf = torchmedia::vision::functional;
namespace vt = torchmedia::vision::transform;
namespace vio = torchmedia::vision::io;
namespace mc = torchmedia::metric::classification;
namespace td = torchmedia::data;
namespace fs = std::filesystem;

static const auto missing = fs::path("/nonexistent_torchmedia_error_policy") / "x.wav";

// One failing call. `call` returns whether its result is the silent build's fallback; `expect` is part of the message.
template<class E = std::invalid_argument>
static void check(const char *name, const std::function<bool()> &call, const std::string &expect) {
    torchmedia::clear_last_error(); // a no-op in the throwing build
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
    const bool fallback = call();
    const auto msg = torchmedia::last_error();
    tm_test::record(fallback, name, __FILE__, __LINE__, "not the fallback");
    tm_test::record(msg.find(expect) != std::string::npos, name, __FILE__, __LINE__,
                    "last_error() = \"" + msg + "\", expected \"" + expect + "\"");
#else
    try {
        (void) call();
        tm_test::record(false, name, __FILE__, __LINE__, "did not throw");
    } catch (const E &e) {
        const std::string what = e.what();
        tm_test::record(what.find(expect) != std::string::npos, name, __FILE__, __LINE__,
                        "what() = \"" + what + "\", expected \"" + expect + "\"");
    }
    tm_test::record(torchmedia::last_error().empty(), name, __FILE__, __LINE__,
                    "the throwing build recorded a message");
#endif
}

static auto undef(const torch::Tensor &t) -> bool { return !t.defined(); }

// ---- one representative failing call per public area, in both builds ----
static void test_audio() {
    const auto x = torch::tensor({1.f, 2.f, 3.f});
    check(
            "audio functional", [&] { return undef(fn::convolve(x, x, static_cast<fn::convolve_mode>(7))); },
            "Unrecognized convolve mode");
    check(
            "audio functional pair", [] { return undef(fn::speed(torch::randn({1, 64}), 16000, 0.0).first); },
            "desired frequency should be positive"); // resample's own check, reached through speed
    check(
            "audio transform",
            [&] { return undef(tf::Vol(tf::vol_option().gain(0.0).gain_type(tf::vol_gain_type::power))(x)); },
            "Vol: gain must be positive for gain_type power");
    check(
            "audio transform constructor",
            [] { return !tf::MelScale(tf::mel_scale_option().f_min(9000.0)).fb().defined(); },
            "MelScale: require f_min <= f_max"); // a bool helper: the message must reach the exception too
    // a composite's fallback guard never swallows an undefined input: the op still rejects it in both builds
    check(
            "audio transform undefined input",
            [] { return undef(tf::SpecAugment(tf::spec_augment_option().zero_masking(true))(torch::Tensor())); },
            "Spectrogram must have at least two dimensions");
    check(
            "audio io",
            [] {
                const auto r = aio::load_audio(missing, aio::load_option().frame_offset(-1));
                return undef(r.data) && r.sample_rate == 0;
            },
            "frame_offset must be non-negative");
    check<std::runtime_error>(
            "audio io missing file",
            [] {
                const auto r = aio::load_audio(missing);
                return undef(r.data) && r.sample_rate == 0;
            },
            "File does not exist");
    check<std::runtime_error>(
            "audio io info",
            [] {
                const auto i = aio::info(missing);
                return i.sample_rate == 0 && i.num_frames == 0 && i.encoding.empty();
            },
            "File does not exist");
    check(
            "separation", [] { return undef(sep::auxiva_ip(torch::randn({2, 3, 4}))); },
            "auxiva_ip: X must be a complex64 or complex128 STFT");
    check(
            "kaldi", [] { return undef(kd::mfcc(torch::randn({1, 4000}), kd::mfcc_option().num_ceps(24))); },
            "num_ceps cannot be larger than num_mel_bins");
}

static void test_vision() {
    check(
            "vision functional",
            [] { return undef(vf::normalize(torch::zeros({3, 4, 4}, torch::kUInt8), {0.5}, {0.5})); },
            "normalize: input tensor should be a float tensor");
    check("vision functional vector", [] { return vf::five_crop(torch::rand({3, 4, 4}), {8}).empty(); }, "five_crop");
    check(
            "vision transform",
            [] { return undef(vt::RandomCrop(vt::random_crop_option().size({2}))(torch::rand({4}))); },
            "RandomCrop: expected a tensor with at least 2 dims");
    check(
            "vision container",
            [] {
                const vt::transform_fn crop = vt::RandomCrop(vt::random_crop_option().size({2}));
                const vt::transform_fn flip = vt::RandomHorizontalFlip(vt::random_flip_option().p(1.0));
                return undef(vt::Compose({crop, flip})(torch::rand({4})));
            },
            "RandomCrop: expected a tensor with at least 2 dims"); // silent: the flip never sees the fallback
    check(
            "vision transform undefined input",
            [] { return undef(vt::ColorJitter(vt::color_jitter_option().brightness({0.5}))(torch::Tensor())); },
            "adjust_brightness: expected a [..., 1 or 3, H, W] image");
    check(
            "vision io", [] { return undef(vio::decode_image(torch::empty({0}, torch::kUInt8))); },
            "decode_image: expected a non-empty 1-D uint8 CPU tensor");
    check<std::runtime_error>(
            "vision io read_file", [] { return undef(vio::read_file(missing)); }, "read_file: could not read");
    check<std::runtime_error>(
            "vision io load_image", [] { return undef(vio::load_image(missing)); }, "File does not exist");
}

static void test_metric() {
    namespace ma = torchmedia::metric::audio;
    namespace mt = torchmedia::metric::text;
    namespace md = torchmedia::metric::detection;
    namespace mi = torchmedia::metric::image;
    namespace ms = torchmedia::metric::segmentation;
    const auto p4 = torch::rand({4}), t5 = torch::randint(0, 2, {5});
    check(
            "metric audio", [] { return undef(ma::signal_noise_ratio(torch::rand({4}), torch::rand({5}))); },
            "same shape");
    check("metric text", [] { return undef(mt::word_error_rate({"a b"}, {"a", "b"})); }, "word_error_rate");
    check("metric classification stat", [&] { return undef(mc::binary_accuracy(p4, t5)); }, "same shape");
    check(
            "metric classification curve",
            [&] {
                const auto [a, b, c] = mc::binary_roc(p4, t5);
                return undef(a) && undef(b) && undef(c);
            },
            "same shape");
    check("metric classification misc", [&] { return undef(mc::binary_hinge_loss(p4, t5)); }, "same shape");
    check(
            "metric detection",
            [] {
                return undef(
                        md::box_convert(torch::zeros({3, 5}), md::box_format_mode::xyxy, md::box_format_mode::xywh));
            },
            "box_convert: expected boxes of shape [..., 4]");
    check("metric image", [] { return undef(mi::total_variation(torch::rand({3, 4}))); }, "to be an 4D tensor");
    check(
            "metric segmentation",
            [] {
                return undef(ms::mean_iou(torch::zeros({1, 4, 4}, torch::kLong), torch::zeros({1, 4, 4}, torch::kLong),
                                          ms::mean_iou_option().num_classes(0)));
            },
            "`num_classes` to be a positive integer");
    check(
            "metric class forward",
            [&] {
                mc::BinaryAccuracy acc;
                return undef(acc(p4, t5)); // silent: not the accuracy of an empty batch
            },
            "same shape");
    check(
            "metric detection functional",
            [] {
                const auto r = md::mean_average_precision(std::vector<md::detection_prediction>(1),
                                                          std::vector<md::detection_target>(2));
                return undef(r.map_per_class) && undef(r.mar_100_per_class) && undef(r.classes);
            },
            "to have the same length");
    check(
            "metric class update",
            [&] {
                mc::BinaryAccuracy acc;
                acc.update(p4, t5); // rejected: the state stays untouched
                acc.update(torch::tensor({0.9f, 0.1f}), torch::tensor({1, 0}));
                return acc.compute().item<double>() == 1.0;
            },
            "same shape");
}

static void test_data() {
    const torch::data::datasets::TensorDataset ten(torch::arange(10));
    check("data", [&] { return td::random_split(ten, {3, 3}).empty(); }, "Sum of input lengths");
    check<std::out_of_range>(
            "data index", [&] { return undef(td::Subset(ten, {1, 2}).get(5).data); },
            "Subset: index 5 is out of range");
    check<std::runtime_error>(
            "data folder", [] { return td::ImageFolder(missing).size().value() == 0; }, "find_classes: cannot read");
}

#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
// Consumes the message of the call just made: the fallback, plus a recorded message containing `expect`.
static auto failed(bool fallback, const std::string &expect) -> bool {
    const auto msg = torchmedia::last_error();
    torchmedia::clear_last_error();
    if (!fallback || msg.find(expect) == std::string::npos) {
        std::fprintf(stderr, "    fallback %d, last_error() = \"%s\", expected \"%s\"\n", fallback, msg.c_str(),
                     expect.c_str());
        return false;
    }
    return true;
}

static void test_silent_basics() {
    torchmedia::clear_last_error();
    TM_CHECK(torchmedia::last_error().empty());
    (void) fn::create_dct(4, 8, "foo");
    const auto first = torchmedia::last_error();
    TM_CHECK(!first.empty());
    (void) fn::create_dct(4, 8, "ortho"); // success keeps the message
    TM_CHECK(torchmedia::last_error() == first);
    std::string other;
    std::thread([&] { other = torchmedia::last_error(); }).join(); // thread-local
    TM_CHECK(other.empty());
    torchmedia::clear_last_error();
    TM_CHECK(torchmedia::last_error().empty());
}

// Composites stop at a part's fallback: no computation on it, and the part's message survives.
static void test_silent_audio_composites() {
    const auto zero_d = torch::tensor(0.5f);
    const auto spec_msg = "spectrogram: expected a (..., time) waveform";
    TM_CHECK(failed(undef(fn::spectral_centroid(zero_d, 16000, 0, torch::hann_window(16), 16, 4, 16)), spec_msg));
    fn::mel_spectrogram_option mel;
    TM_CHECK(failed(undef(fn::melspectrogram(zero_d, mel)), spec_msg));
    fn::mfcc_option mfcc;
    TM_CHECK(failed(undef(fn::mfcc(zero_d, mfcc)), spec_msg));
    const auto w = torch::randn({2, 8});
    const auto empty = torch::ones({0});
    TM_CHECK(failed(undef(fn::lfilter(w, empty, empty)), "Expected n_order to be positive"));
    TM_CHECK(failed(undef(fn::filtfilt(w, empty, empty)), "Expected n_order to be positive"));
    TM_CHECK(failed(undef(fn::filtfilt(w, torch::ones({3}), torch::ones({2}))), "Expected coeffs to be the same size"));
    const auto [sp, lengths] = fn::speed(w, 16000, 0.0, torch::tensor({8}));
    TM_CHECK(failed(undef(sp) && !lengths.has_value(), "desired frequency should be positive"));
    TM_CHECK(failed(undef(kd::vtln_warp_mel_freq(10, 7500, 20, 8000, 0.9, torch::linspace(0, 8000, 9))),
                    "vtln_low option"));

    TM_CHECK(failed(undef(tf::MelSpectrogram()(zero_d)), spec_msg));
    TM_CHECK(failed(undef(tf::MFCC()(zero_d)), spec_msg));
    TM_CHECK(failed(undef(tf::LFCC()(zero_d)), spec_msg));
    const auto masks = tf::SpecAugment(tf::spec_augment_option().p(2.0));
    TM_CHECK(failed(undef(masks(torch::rand({1, 8, 8}))), "p must be between 0.0 and 1.0"));
    TM_CHECK(failed(undef(tf::SpecAugment(tf::spec_augment_option().p(2.0).iid_masks(true))(torch::rand({1, 8, 8}))),
                    "p must be between 0.0 and 1.0"));
    const auto spec = torch::complex(torch::randn({2, 3, 4}), torch::randn({2, 3, 4})); // (channel, freq, time)
    const auto bad_mask = torch::rand({5, 4});
    tf::MVDR mvdr;
    TM_CHECK(failed(undef(mvdr(spec, bad_mask)), "The dimensions of mask"));
    const auto c3 = torch::complex(torch::randn({3, 2, 2}), torch::randn({3, 2, 2}));
    TM_CHECK(failed(undef(tf::SoudenMVDR()(spec, torch::real(c3), torch::real(c3), int64_t{0})), "must be complex"));
    TM_CHECK(failed(undef(tf::RTFMVDR()(spec, torch::real(c3[0]), torch::real(c3), int64_t{0})),
                    "rtf and psd_n must be complex"));
    const auto [sw, sl] = tf::Speed(tf::speed_option().factor(0.0))(w, torch::tensor({8}));
    TM_CHECK(failed(undef(sw) && !sl.has_value(), "Resample: invalid arguments"));
}

// A rejected constructor records its message; every later call returns the fallback (audio transforms).
static void test_silent_audio_constructors() {
    const auto spec = torch::rand({1, 201, 10});
    tf::AmplitudeToDB db(tf::amplitude_to_db_option().top_db(-1.0));
    TM_CHECK(failed(true, "AmplitudeToDB: top_db must be positive value"));
    TM_CHECK(failed(undef(db(spec)), "AmplitudeToDB: invalid arguments"));
    tf::MelScale ms(tf::mel_scale_option().f_min(9000.0));
    TM_CHECK(failed(!ms.fb().defined(), "MelScale: require f_min <= f_max"));
    TM_CHECK(failed(undef(ms(spec)), "MelScale: invalid arguments"));
    tf::MelScale bad_norm(tf::mel_scale_option().norm("foo"));
    TM_CHECK(failed(undef(bad_norm(spec)), "MelScale: invalid arguments"));
    tf::InverseMelScale ims(tf::inverse_mel_scale_option().f_min(9000.0));
    TM_CHECK(failed(undef(ims(torch::rand({1, 128, 10}))), "InverseMelScale: invalid arguments"));
    const auto w = torch::randn({1, 4000});
    TM_CHECK(failed(undef(tf::MFCC(tf::mfcc_option().dct_type(3))(w)), "MFCC: invalid arguments"));
    tf::MFCC many(tf::mfcc_option().n_mfcc(200));
    TM_CHECK(failed(true, "MFCC: cannot select more MFCC coefficients"));
    TM_CHECK(failed(undef(many(w)), "MFCC: invalid arguments"));
    TM_CHECK(failed(undef(tf::LFCC(tf::lfcc_option().dct_type(3))(w)), "LFCC: invalid arguments"));
    TM_CHECK(failed(undef(tf::LFCC(tf::lfcc_option().n_lfcc(1000))(w)), "LFCC: invalid arguments"));
    TM_CHECK(failed(undef(tf::Vol(tf::vol_option().gain(-1.0))(w)), "Vol: invalid arguments"));
}

static void test_silent_audio_io() {
    const auto dir = fs::temp_directory_path() / std::format("torchmedia_error_policy_{}", ::getpid());
    fs::create_directories(dir);
    std::ofstream(dir / "bad.wav") << "RIFF0000WAVEjunk";
    const auto r = aio::load_audio(dir / "bad.wav");
    TM_CHECK(failed(undef(r.data) && r.sample_rate == 0, "Could not decode WAV"));
    const std::string bytes = "not audio at all";
    TM_CHECK(failed(aio::info_from_memory(bytes.data(), bytes.size()).encoding.empty(), "Unsupported or unrecognized"));
    fs::remove_all(dir);
}

static void test_silent_vision() {
    TM_CHECK(failed(undef(vf::solarize(torch::ones({1, 2, 2}, torch::kBool), 0.5)), "invert: only floating"));
    TM_CHECK(failed(undef(vf::equalize(torch::ones({1, 2, 2}, torch::kBool))), "to_dtype"));
    const auto line = torch::rand({4}); // < 2 dims: query_chw rejects it in every random class
    TM_CHECK(failed(undef(vt::RandomResizedCrop(vt::random_resized_crop_option().size({2}))(line)),
                    "RandomResizedCrop"));
    TM_CHECK(failed(undef(vt::RandomShortestSize(vt::random_shortest_size_option().min_size({2}))(line)),
                    "RandomShortestSize"));
    TM_CHECK(failed(undef(vt::ScaleJitter(vt::scale_jitter_option().target_size({4, 4}))(line)), "ScaleJitter"));
    TM_CHECK(failed(undef(vt::RandomZoomOut(vt::random_zoom_out_option().p(1.0))(line)), "RandomZoomOut"));
    TM_CHECK(failed(undef(vt::RandomAffine(vt::random_affine_option().degrees({10}))(line)), "RandomAffine"));
    TM_CHECK(failed(undef(vt::RandomPerspective(vt::random_perspective_option().p(1.0))(line)), "RandomPerspective"));
    TM_CHECK(failed(undef(vt::ElasticTransform()(line)), "ElasticTransform"));
    TM_CHECK(failed(undef(vt::RandomErasing(vt::random_erasing_option().p(1.0))(line)), "RandomErasing"));
    TM_CHECK(failed(undef(vt::RandomPhotometricDistort()(line)), "RandomPhotometricDistort"));
    TM_CHECK(failed(undef(vt::RandomChannelPermutation()(line)), "RandomChannelPermutation"));
    // a 2-channel image fails the first colour op; later ops do not run on its fallback
    TM_CHECK(failed(
            undef(vt::RandomPhotometricDistort(vt::random_photometric_distort_option().p(1.0))(torch::rand({2, 4, 4}))),
            "expected a [..., 1 or 3, H, W] image"));
    TM_CHECK(failed(
            undef(vt::ColorJitter(vt::color_jitter_option().brightness({0.5}).contrast({0.5}))(torch::rand({2, 4, 4}))),
            "expected a [..., 1 or 3, H, W] image"));

    // rejected constructor options: a size makes every call fail; others fall back to a default / neutral value
    const auto img = torch::rand({3, 8, 8});
    vt::RandomCrop crop(vt::random_crop_option().size({1, 2, 3}));
    TM_CHECK(failed(true, "RandomCrop: please provide only two dimensions"));
    TM_CHECK(failed(undef(crop(img)), "RandomCrop: invalid arguments"));
    TM_CHECK(failed(undef(vt::RandomResizedCrop(vt::random_resized_crop_option())(img)),
                    "RandomResizedCrop: invalid arguments"));
    vt::LinearTransformation lin(vt::linear_transformation_option()
                                         .transformation_matrix(torch::eye(4).slice(0, 0, 3))
                                         .mean_vector(torch::zeros({4})));
    TM_CHECK(failed(undef(lin(torch::rand({1, 2, 2}))), "LinearTransformation: invalid arguments"));
    vt::RandomZoomOut zoom(vt::random_zoom_out_option().side_range({0.5, 2.0}).p(1.0));
    TM_CHECK(failed(true, "RandomZoomOut: invalid side range"));
    const auto zoomed = zoom(img); // default side range (1, 4): the canvas grows
    TM_CHECK(zoomed.size(-1) >= 8 && zoomed.size(-2) >= 8 && torchmedia::last_error().empty());
    vt::RandomPerspective persp(vt::random_perspective_option().distortion_scale(2.0).p(1.0));
    TM_CHECK(failed(persp(img).sizes() == img.sizes(), "distortion_scale"));
    vt::RandomErasing erase(vt::random_erasing_option().scale({-1.0, 2.0}).p(1.0));
    TM_CHECK(failed(erase(img).sizes() == img.sizes(), "RandomErasing: scale"));
    vt::RandomRotation rot(vt::random_rotation_option().degrees({-10.0})); // (0, 0): no rotation
    TM_CHECK(failed(torch::equal(rot(img), img), "if degrees is a single number, it must be positive"));
    vt::RandomAffine shift(vt::random_affine_option().degrees({0.0}).translate({2.0, 0.5})); // no translation
    TM_CHECK(failed(torch::equal(shift(img), img), "translation values should be between 0 and 1"));
    vt::RandomAffine unscaled(vt::random_affine_option().degrees({0.0}).scale({0.0, 2.0})); // no scaling
    TM_CHECK(failed(true, "RandomAffine: scale values should be positive"));
    TM_CHECK(torch::equal(unscaled(img), img) && torch::equal(unscaled(img), img) && torchmedia::last_error().empty());
    const vt::transform_fn id = [](const torch::Tensor &x) { return x; };
    vt::RandomChoice choice({id, id}, {0.0, 0.0}); // equal weights
    TM_CHECK(failed(torch::equal(choice(img), img), "RandomChoice: p sums to zero"));

    const auto dir = fs::temp_directory_path() / std::format("torchmedia_error_policy_v_{}", ::getpid());
    fs::create_directories(dir);
    std::ofstream(dir / "bad.png") << "\x89PNG\r\n\x1a\n" << std::string(40, 'x');
    std::ofstream(dir / "bad.pgm") << "P2 1 1 255\nx";
    TM_CHECK(failed(undef(vio::read_image(dir / "bad.png")), "decode_image: could not decode"));
    TM_CHECK(failed(undef(vio::read_image(dir / "none.png")), "read_file: could not read"));
    TM_CHECK(failed(undef(vio::load_image(dir / "bad.pgm")), "PNM: expected a decimal integer"));
    TM_CHECK(failed(undef(vio::load_image(dir / "bad.png")), "Could not load image"));
    fs::remove_all(dir);
}

// The vision containers stop at a part's fallback: the part's message survives and no later part (a flip would
// throw c10::Error on an undefined tensor) runs on it.
static void test_silent_vision_containers() {
    const auto line = torch::rand({4});
    const auto two = torch::rand({2, 4, 4}); // Grayscale rejects 2 channels
    const vt::transform_fn crop = vt::RandomCrop(vt::random_crop_option().size({2}));
    const vt::transform_fn hflip = vt::RandomHorizontalFlip(vt::random_flip_option().p(1.0));
    const vt::transform_fn vflip = vt::RandomVerticalFlip(vt::random_flip_option().p(1.0));
    const vt::transform_fn gray = vt::Grayscale();
    const vt::transform_fn norm = vt::Normalize(vt::normalize_option().mean({0.5}).std({0.5}));
    const auto crop_msg = "RandomCrop: expected a tensor with at least 2 dims";
    TM_CHECK(failed(undef(vt::Compose({crop, norm})(line)), crop_msg)); // not normalize's message
    TM_CHECK(failed(undef(vt::RandomApply({gray, hflip}, 1.0)(two)), "rgb_to_grayscale: expected"));
    for (int seed = 0; seed < 4; ++seed) { // both orders occur
        torch::manual_seed(seed);
        TM_CHECK(failed(undef(vt::RandomOrder({gray, vflip})(two)), "rgb_to_grayscale: expected"));
    }
    // an undefined input still reaches the first part, which rejects it
    TM_CHECK(failed(undef(vt::Compose({crop, hflip})(torch::Tensor())), crop_msg));
}

// A rejected batch: forward() / operator() return the value-initialised result (not the metric of no data), the
// accumulated state is untouched; the functional mean_average_precision likewise.
static void test_silent_metric_forward() {
    namespace ma = torchmedia::metric::audio;
    namespace md = torchmedia::metric::detection;
    namespace ms = torchmedia::metric::segmentation;
    const auto p4 = torch::rand({4}), t5 = torch::randint(0, 2, {5});
    mc::BinaryAccuracy acc;
    TM_CHECK(failed(undef(acc.forward(p4, t5)), "same shape"));
    TM_CHECK(failed(undef(acc(p4, t5)), "same shape"));
    TM_CHECK(failed(undef(ma::SignalNoiseRatio()(torch::rand({4}), torch::rand({5}))), "same shape"));
    ms::MeanIoU miou(ms::mean_iou_option().num_classes(2));
    TM_CHECK(failed(undef(miou(torch::zeros({1, 4, 4}, torch::kLong), torch::zeros({1, 5, 5}, torch::kLong))),
                    "same shape"));
    (void) fn::create_dct(4, 8, "foo"); // a valid batch keeps an earlier message and counts alone
    const auto earlier = torchmedia::last_error();
    TM_CHECK(acc(torch::tensor({0.9f}), torch::tensor({1})).item<double>() == 1.0);
    TM_CHECK(failed(acc.compute().item<double>() == 1.0 && torchmedia::last_error() == earlier, "norm must be"));

    std::vector<md::detection_prediction> preds(1);
    std::vector<md::detection_target> target(2);
    const auto map_undef = [](const md::mean_average_precision_result &r) {
        return undef(r.map_per_class) && undef(r.mar_100_per_class) && undef(r.classes);
    };
    md::MeanAveragePrecision map;
    TM_CHECK(failed(map_undef(map(preds, target)), "to have the same length"));
    TM_CHECK(failed(map_undef(md::mean_average_precision(preds, target)), "to have the same length"));
    const std::vector<md::detection_prediction> one_p{
            {torch::tensor({{0.f, 0.f, 2.f, 2.f}}), torch::tensor({0.9f}), torch::tensor({1}), {}}};
    const std::vector<md::detection_target> one_t{
            {torch::tensor({{0.f, 0.f, 2.f, 2.f}}), torch::tensor({1}), {}, {}, {}}};
    TM_CHECK(!map_undef(md::mean_average_precision(one_p, one_t)) && torchmedia::last_error().empty());
    const auto bad_md = md::mean_average_precision_option().max_detection_thresholds({1, 10});
    TM_CHECK(failed(map_undef(md::mean_average_precision(one_p, one_t, bad_md)), "max_detection_thresholds"));
}

// Datasets: a sample whose loader fails has undefined data (its target is kept); the message names the file.
static void test_silent_data() {
    const auto dir = fs::temp_directory_path() / std::format("torchmedia_error_policy_d_{}", ::getpid());
    fs::create_directories(dir / "cls");
    std::ofstream(dir / "cls" / "a.png") << "\x89PNG\r\n\x1a\n" << std::string(40, 'x');
    std::ofstream(dir / "cls" / "b.jpg"); // empty: read_file rejects it
    std::ofstream(dir / "cls" / "c.ppm") << "P3 1 1 255\n1 2";
    std::ofstream(dir / "cls" / "d.wav") << "RIFF0000WAVEjunk";
    td::ImageFolder images(dir, td::dataset_folder_option().transform([](const torch::Tensor &x) { return x * 2; }));
    TM_CHECK(images.size().value() == 2);
    TM_CHECK(failed(undef(images.get(0).data), "a.png: decode_image: could not decode"));
    TM_CHECK(failed(undef(images.get(1).data) && images.get(1).target.item<int64_t>() == 0, "b.jpg"));
    td::ImageFolder loaded(dir, {}, td::image_folder_option().load_image(true));
    TM_CHECK(failed(undef(loaded.get(2).data), "c.ppm"));
    td::AudioFolder audio(dir);
    TM_CHECK(audio.size().value() == 1);
    TM_CHECK(failed(undef(audio.get(0).data), "Could not decode WAV"));
    TM_CHECK(failed(td::ImageFolder(dir / "none").classes().empty(), "find_classes: cannot read"));
    fs::remove_all(dir);
}
#endif

int main() {
    TM_RUN(test_audio);
    TM_RUN(test_vision);
    TM_RUN(test_metric);
    TM_RUN(test_data);
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
    TM_RUN(test_silent_basics);
    TM_RUN(test_silent_audio_composites);
    TM_RUN(test_silent_audio_constructors);
    TM_RUN(test_silent_audio_io);
    TM_RUN(test_silent_vision);
    TM_RUN(test_silent_vision_containers);
    TM_RUN(test_silent_metric_forward);
    TM_RUN(test_silent_data);
    return tm_test::summary("error_policy_test_silent");
#else
    return tm_test::summary("error_policy_test_throw");
#endif
}
