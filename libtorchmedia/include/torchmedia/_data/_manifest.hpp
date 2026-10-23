#pragma once
#ifndef LIB_TORCH_MEDIA_DATA_MANIFEST_HPP
#define LIB_TORCH_MEDIA_DATA_MANIFEST_HPP
// ManifestDataset (develop_log/2026-10-24/progress03): samples listed in a text file instead of a directory tree.
#include <charconv>
#include <fstream>
#include <set>
#include "_folder.hpp"

namespace torchmedia::data {
    // The manifest format. Without a header (default) every non-blank line is Kaldi-style `key path [label]`, fields
    // separated by runs of spaces / tabs (so paths cannot hold spaces; a TSV without spaces in its fields reads the
    // same way). With header(true) the file is a TSV whose first non-blank line names the columns: fields are split on
    // every tab (paths may hold spaces) and path_column / label_column pick the fields ("" label_column: unlabeled).
    // Labels are int64 by default; class_labels(true) reads them as class names indexed in sorted order (classes() /
    // class_to_idx(), as DatasetFolder). A missing or empty label gives target -1. Relative paths are resolved against
    // the manifest's directory. A malformed line throws std::invalid_argument naming the manifest and its line number.
    typedef struct manifest_option {
        bool _header = false;
        std::string _path_column = "path";
        std::string _label_column = "label";
        bool _class_labels = false;
        transform_fn _transform;
        transform_fn _target_transform;

        auto header(bool v) -> manifest_option & {
            _header = v;
            return *this;
        }
        auto path_column(std::string v) -> manifest_option & {
            _path_column = std::move(v);
            return *this;
        }
        auto label_column(std::string v) -> manifest_option & {
            _label_column = std::move(v);
            return *this;
        }
        auto class_labels(bool v) -> manifest_option & {
            _class_labels = v;
            return *this;
        }
        auto transform(transform_fn v) -> manifest_option & {
            _transform = std::move(v);
            return *this;
        }
        auto target_transform(transform_fn v) -> manifest_option & {
            _target_transform = std::move(v);
            return *this;
        }
    } manifest_option_t;

    namespace detail {
        // Fields of one line: split on every tab (TSV), or on runs of spaces / tabs (Kaldi).
        inline auto split_fields(std::string_view line, bool tsv) -> std::vector<std::string> {
            std::vector<std::string> out;
            if (tsv) {
                size_t start = 0;
                for (size_t tab; (tab = line.find('\t', start)) != std::string_view::npos; start = tab + 1) {
                    out.emplace_back(line.substr(start, tab - start));
                }
                out.emplace_back(line.substr(start));
                return out;
            }
            for (size_t pos = line.find_first_not_of(" \t"); pos != std::string_view::npos;) {
                const auto end = std::min(line.find_first_of(" \t", pos), line.size());
                out.emplace_back(line.substr(pos, end - pos));
                pos = line.find_first_not_of(" \t", end);
            }
            return out;
        }

        inline auto manifest_list(const path_t &manifest, const loader_fn &loader, const manifest_option &opt)
                -> sample_list {
            if (!loader) {
                return invalid<sample_list>("ManifestDataset: loader must be callable");
            }
            std::ifstream file(manifest);
            if (!file || std::filesystem::is_directory(manifest)) { // a directory opens as an empty stream
                return invalid<sample_list, std::runtime_error>("ManifestDataset: cannot open " + manifest.string());
            }
            const auto error = [&](int64_t line, const std::string &msg) {
                return invalid<sample_list>(std::format("ManifestDataset: {}:{}: {}", manifest.string(), line, msg));
            };
            sample_list list;
            std::vector<std::string> labels, columns;
            int64_t path_col = 1, label_col = 2; // Kaldi: key path [label]
            std::string line;
            for (int64_t number = 1; std::getline(file, line); ++number) {
                if (line.ends_with('\r')) {
                    line.pop_back();
                }
                if (line.find_first_not_of(" \t") == std::string::npos) {
                    continue;
                }
                const auto fields = split_fields(line, opt._header);
                if (opt._header && columns.empty()) { // the header row
                    columns = fields;
                    const auto col = [&](const std::string &name) -> int64_t {
                        const auto it = std::ranges::find(columns, name);
                        return name.empty() || it == columns.end() ? -1 : it - columns.begin();
                    };
                    path_col = col(opt._path_column);
                    label_col = col(opt._label_column);
                    if (path_col < 0 || (label_col < 0 && !opt._label_column.empty())) {
                        return error(number, std::format("the header has no column '{}'",
                                                         path_col < 0 ? opt._path_column : opt._label_column));
                    }
                    continue;
                }
                if (opt._header ? fields.size() != columns.size() : fields.size() < 2 || fields.size() > 3) {
                    return error(number, opt._header ? std::format("expected {} fields like the header, got {}",
                                                                   columns.size(), fields.size())
                                                     : std::format("expected 'key path [label]', got {} fields",
                                                                   fields.size()));
                }
                if (fields[path_col].empty()) { // a TSV field; it would resolve to the manifest's directory
                    return error(number, "empty path");
                }
                const path_t path = fields[path_col];
                list.samples.emplace_back(path.is_relative() ? manifest.parent_path() / path : path, -1);
                list.keys.push_back(fields[opt._header ? path_col : 0]);
                labels.push_back(label_col >= 0 && label_col < std::ssize(fields) ? fields[label_col] : "");
                const auto &label = labels.back();
                if (!opt._class_labels && !label.empty()) {
                    const auto [end, ec] =
                            std::from_chars(label.data(), label.data() + label.size(), list.samples.back().second);
                    if (ec != std::errc() || end != label.data() + label.size()) {
                        return error(number, std::format("label '{}' is not an int64 (class_labels(true) reads class "
                                                         "names)",
                                                         label));
                    }
                }
            }
            if (opt._class_labels) {
                for (const auto &name: std::set<std::string>(labels.begin(), labels.end())) {
                    if (!name.empty()) {
                        list.class_to_idx.emplace(name, std::ssize(list.classes));
                        list.classes.push_back(name);
                    }
                }
                for (size_t i = 0; i < labels.size(); ++i) {
                    list.samples[i].second = labels[i].empty() ? -1 : list.class_to_idx.at(labels[i]);
                }
            }
            for (const auto &s: list.samples) {
                list.targets.push_back(s.second);
            }
            return list;
        }
    } // namespace detail

    // A dataset read from a manifest file (manifest_option above): get(i) loads samples()[i] with `loader` (e.g.
    // image_loader(...) / audio_loader(...)). keys(): the Kaldi key per sample (with a header: the path as written).
    class ManifestDataset : public detail::path_dataset<ManifestDataset> {
    public:
        ManifestDataset(const path_t &manifest, const loader_fn &loader, const manifest_option &opt = {}) :
            path_dataset(detail::manifest_list(manifest, loader, opt), loader, opt._transform, opt._target_transform) {}

        auto keys() const -> const std::vector<std::string> & { return _list->keys; }
    };
} // namespace torchmedia::data

#endif // LIB_TORCH_MEDIA_DATA_MANIFEST_HPP
