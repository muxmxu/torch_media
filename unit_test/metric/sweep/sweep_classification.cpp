// Sweep driver for the classification metrics: evaluates cases.pt (sweep_classification_gen.py) with
// classification_driver.hpp, writes results.pt ("<key>!" = 1 where torchmedia raised).
#include <fstream>
#include "../classification_driver.hpp"

int main() {
    std::ifstream f("cases.pt", std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
    const auto cases = torch::pickle_load(buf).toList();
    c10::List<c10::Dict<std::string, torch::Tensor>> results;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const auto d = cases.get(ci).toGenericDict();
        const cls_driver::case_view c{[&](const std::string &k) { return d.at(k).toTensor(); },
                                      [&](const std::string &k) { return d.contains("p_" + k) || d.contains(k); },
                                      [&](const std::string &k) { return d.at("p_" + k).toDouble(); }};
        c10::Dict<std::string, torch::Tensor> out;
        for (const auto &part: {cls_driver::run_stat(c), cls_driver::run_curve(c), cls_driver::run_misc(c)}) {
            for (const auto &[k, v]: part) {
                out.insert(k, v.defined() ? v : torch::tensor(1));
            }
        }
        results.push_back(out);
    }
    const auto bytes = torch::pickle_save(c10::IValue(results));
    std::ofstream("results.pt", std::ios::binary).write(bytes.data(), bytes.size());
}
