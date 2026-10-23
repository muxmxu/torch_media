#pragma once
// Shared pieces of the classification tests: the golden cases of gen_golden_classification.py (text in
// golden_classification.hpp) run through classification_driver.hpp, one metric family per test binary.
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "classification_driver.hpp"
#include "golden_classification.hpp"
#include "metric_test_util.hpp"

struct cls_entry {
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<double> data;
};
struct cls_case {
    std::string name;
    std::map<std::string, cls_entry> in, out;
    std::map<std::string, double> prm;
};

// "case <name>" / "in|out <key> <dtype> <ndim> <shape...> <n> <values...>" / "prm <key> <value>" / "err <key>"
inline auto golden_cases() -> const std::vector<cls_case> & {
    static const auto cases = [] {
        std::string text;
        for (const auto *chunk: golden::cls_text) {
            text += chunk;
        }
        std::istringstream is(text);
        std::vector<cls_case> v;
        std::string tag, key, value;
        while (is >> tag) {
            if (tag == "case") {
                v.emplace_back();
                is >> v.back().name;
                continue;
            }
            is >> key;
            if (tag == "prm") {
                is >> value;
                v.back().prm[key] = std::stod(value);
            } else if (tag == "err") {
                v.back().out[key + "!"] = {};
            } else {
                cls_entry e;
                int64_t ndim = 0, n = 0;
                is >> e.dtype >> ndim;
                e.shape.resize(ndim);
                for (auto &s: e.shape) {
                    is >> s;
                }
                is >> n;
                e.data.resize(n);
                for (auto &d: e.data) {
                    is >> value;
                    d = std::stod(value); // handles nan / inf
                }
                (tag == "in" ? v.back().in : v.back().out)[key] = e;
            }
        }
        return v;
    }();
    return cases;
}

inline auto golden_view(const cls_case &c) -> cls_driver::case_view {
    return {[&c](const std::string &k) {
                const auto &e = c.in.at(k);
                const auto dt = e.dtype == "float32"   ? torch::kFloat32
                                : e.dtype == "float16" ? torch::kFloat16
                                : e.dtype == "float64" ? torch::kFloat64
                                                       : torch::kInt64;
                return G(e).to(dt);
            },
            [&c](const std::string &k) { return c.prm.count(k) > 0 || c.in.count(k) > 0; },
            [&c](const std::string &k) { return c.prm.at(k); }};
}

// Every expected key of the family equals the driver's output (dtype, shape, values; "<key>!" = both raised) and the
// driver gives no other key.
inline void check_golden_cases(const std::string &family, cls_driver::outputs (*run)(const cls_driver::case_view &)) {
    int64_t compared = 0;
    for (const auto &c: golden_cases()) {
        const auto got = run(golden_view(c));
        for (const auto &[k, e]: c.out) {
            if (!k.starts_with(family + "/")) {
                continue;
            }
            const auto name = c.name + ":" + k;
            const auto it = got.find(k);
            if (it == got.end()) {
                tm_test::record(false, name.c_str(), __FILE__, __LINE__, "missing (torchmedia raised?)");
            } else if (!k.ends_with("!")) {
                const auto before = tm_test::g_failures;
                check_golden(it->second, e, name, __LINE__);
                if (tm_test::g_failures != before && std::getenv("CLS_DRIVER_VERBOSE")) { // debugging aid
                    std::cerr << name << " got " << it->second << " expected " << G(e) << std::endl;
                }
                ++compared;
            }
        }
        for (const auto &[k, v]: got) {
            if (!c.out.count(k)) {
                const auto name = c.name + ":" + k;
                tm_test::record(false, name.c_str(), __FILE__, __LINE__, "unexpected output");
            }
        }
    }
    TM_CHECK(compared > 0);
}
