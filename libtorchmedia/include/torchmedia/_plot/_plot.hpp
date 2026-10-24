#pragma once
#ifndef LIB_TORCH_MEDIA_PLOT_HPP
#define LIB_TORCH_MEDIA_PLOT_HPP

#include <string>
#include <vector>
#include <torch/torch.h>
#include <matplot/matplot.h>

#include "../global_include.hpp"

// Optional plotting module (matplot++ / gnuplot backend). Not included by torchmedia.hpp; include
// <torchmedia/plot.hpp> explicitly and link torch_media_plot. See develop_log/2026-05-31/progress05.
namespace torchmedia::plot {

    namespace detail {
        inline auto to_vector(const tensor_t &t) -> std::vector<double> {
            const auto c = t.detach().to(torch::kCPU, torch::kDouble).contiguous().reshape({-1});
            const double *p = c.data_ptr<double>();
            return std::vector<double>(p, p + c.numel());
        }

        inline auto to_matrix(const tensor_t &t) -> std::vector<std::vector<double>> {
            const auto c = t.detach().to(torch::kCPU, torch::kDouble).contiguous(); // [F, T]
            const int64_t rows = c.size(0);
            const int64_t cols = c.size(1);
            const double *p = c.data_ptr<double>();
            std::vector<std::vector<double>> m(static_cast<size_t>(rows), std::vector<double>(static_cast<size_t>(cols)));
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t j = 0; j < cols; ++j)
                    m[static_cast<size_t>(i)][static_cast<size_t>(j)] = p[i * cols + j];
            return m;
        }
    } // namespace detail

    // Chaining plot builder:
    //   Plotter().heatmap(spec).colorbar().title("mel").xlabel("time").save("mel.png");
    //   Plotter().waveform(wav).ylabel("amp").save("wave.png");
    class Plotter {
    public:
        Plotter() { matplot::cla(); }

        // Line plot of a waveform ([T] or the first channel of [C, T]).
        Plotter &waveform(const tensor_t &wav) {
            const auto w = wav.dim() > 1 ? wav.select(0, 0) : wav;
            matplot::plot(detail::to_vector(w));
            return *this;
        }

        // Heatmap of a [F, T] matrix (or the last two dims of a higher-rank tensor).
        Plotter &heatmap(const tensor_t &mat) {
            const auto m = mat.dim() > 2 ? mat.reshape({mat.size(-2), mat.size(-1)}) : mat;
            matplot::imagesc(detail::to_matrix(m));
            return *this;
        }

        Plotter &title(const std::string &t) {
            matplot::title(t);
            return *this;
        }
        Plotter &xlabel(const std::string &s) {
            matplot::xlabel(s);
            return *this;
        }
        Plotter &ylabel(const std::string &s) {
            matplot::ylabel(s);
            return *this;
        }
        Plotter &colorbar() {
            matplot::colorbar();
            return *this;
        }
        Plotter &save(const std::string &path) {
            matplot::save(path);
            return *this;
        }
    };

    // Convenience wrappers built on Plotter.
    inline auto save_waveform(const tensor_t &wav, const std::string &path) -> void {
        Plotter().waveform(wav).xlabel("sample").ylabel("amplitude").save(path);
    }

    inline auto save_spectrogram(const tensor_t &spec, const std::string &path) -> void {
        Plotter().heatmap(spec).colorbar().xlabel("time").ylabel("frequency").save(path);
    }

} // namespace torchmedia::plot
#endif // LIB_TORCH_MEDIA_PLOT_HPP
