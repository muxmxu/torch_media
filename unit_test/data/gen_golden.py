"""Goldens for unit_test/data (develop_log/2026-10-24/progress03): torchvision 0.20.1 DatasetFolder / ImageFolder /
find_classes / make_dataset on a directory tree (TREE, rebuilt by the C++ test from golden.hpp) and torch 2.5.1
torch.utils.data.random_split. Run with the repo .venv:  .venv/bin/python unit_test/data/gen_golden.py > golden.hpp
"""
import os
import shutil
import stat
import sys
import tempfile

import torch
from torch.utils.data import random_split
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import find_classes, make_dataset

# (kind, path under the base directory, symlink target). The classes live under root/; ext/ is outside it.
TREE = [
    ("file", "root/Zebra/z1.png", ""),  # upper case sorts before lower case
    ("file", "root/apple/a.png", ""),
    ("file", "root/apple/B.PNG", ""),  # extensions match case-insensitively
    ("file", "root/apple/c.txt", ""),
    ("file", "root/apple/xpng", ""),  # matches the dot-less extension "png"
    ("file", "root/apple/sub/d.jpg", ""),
    ("file", "root/apple/sub/deeper/e.jpeg", ""),
    ("file", "root/apple/sub-x/f.png", ""),  # sorted(os.walk): "sub-x" < "sub.y" < "sub/deeper" by path string
    ("file", "root/apple/sub.y/g.png", ""),
    ("file", "root/apple/dir.png/inner.txt", ""),  # a directory named like an image is not a sample
    ("noperm", "root/apple/noperm", ""),  # unlistable directory: os.walk skips it
    ("file", "root/banana/k.Jpg", ""),
    ("link", "root/banana/link_dir", "../../ext"),  # symlinked directory: followed
    ("link", "root/banana/file_link.png", "../apple/a.png"),  # symlinked file
    ("link", "root/banana/broken.png", "missing.png"),  # broken symlink with an image extension: listed
    ("dir", "root/empty_cls/nested", ""),  # no files at all
    ("file", "root/only_txt/readme.txt", ""),  # no valid file
    ("link", "root/sym_cls", "../ext"),  # symlinked class directory
    ("link", "root/broken_cls", "nowhere"),  # broken symlink: not a class
    ("file", "root/.hidden/i.png", ""),  # hidden directories are classes too
    ("file", "root/über/u.png", ""),  # non-ASCII: code-point (UTF-8 byte) order
    ("file", "root/root_file.png", ""),  # a file directly under root: not a class
    ("file", "ext/h.png", ""),
    ("file", "ext/notes.md", ""),
]


def build(base):
    for kind, rel, target in TREE:
        p = os.path.join(base, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if kind == "file":
            open(p, "wb").write(b"x")
        elif kind in ("dir", "noperm"):
            os.makedirs(p, exist_ok=True)
        else:
            os.symlink(target, p)
    for kind, rel, _ in TREE:
        if kind == "noperm":
            os.chmod(os.path.join(base, rel), 0)


def cstr(s):
    out = '"'
    for b in s.encode("utf-8"):
        c = chr(b)
        out += c if 32 <= b < 127 and c not in '"\\' else ("\\" + c if c in '"\\' else "\\%03o" % b)
    return out + '"'


def main():
    base = tempfile.mkdtemp()
    root = os.path.join(base, "root")
    build(base)
    rel = lambda p: p[len(root) + 1:]
    samples = lambda s: [(rel(p), t) for p, t in s]

    def error(fn):
        try:
            fn()
        except (FileNotFoundError, ValueError) as e:
            return str(e)
        raise AssertionError("no error")

    load = lambda p: p
    g = {}
    g["classes"] = find_classes(root)[0]
    ext3 = (".png", ".jpg", ".jpeg")
    d = DatasetFolder(root, load, extensions=ext3, allow_empty=True)
    assert d.classes == g["classes"] and d.targets == [t for _, t in d.samples]
    g["ext3"] = samples(d.samples)
    g["ext3_error"] = error(lambda: DatasetFolder(root, load, extensions=ext3))
    g["nodot"] = samples(DatasetFolder(root, load, extensions=("png",), allow_empty=True).samples)
    sub = lambda p: "sub" in rel(p)
    g["valid_sub"] = samples(DatasetFolder(root, load, is_valid_file=sub, allow_empty=True).samples)
    g["valid_sub_error"] = error(lambda: DatasetFolder(root, load, is_valid_file=sub))
    g["image_folder"] = samples(ImageFolder(root, allow_empty=True).samples)
    custom = {"apple": 5, "missing": 1, "Zebra": 0}
    g["custom"] = samples(make_dataset(root, custom, extensions=(".png",), allow_empty=True))
    g["custom_error"] = error(lambda: make_dataset(root, custom, extensions=(".png",)))
    g["empty_map_error"] = error(lambda: make_dataset(root, {}, extensions=ext3))
    g["none_error"] = error(lambda: make_dataset(root, custom))
    g["no_class_error"] = error(lambda: find_classes(os.path.join(base, "ext"))).replace(base, "<base>")

    for kind, r, _ in TREE:
        if kind == "noperm":
            os.chmod(os.path.join(base, r), stat.S_IRWXU)
    shutil.rmtree(base)

    # random_split: explicit generators for several seeds, and the global generator after manual_seed.
    # [0.4, 0.05, 0.13, 0.34, 0.08] sums to 1.0000000000000002 left to right; Python 3.12's sum() gives 1.0, so these
    # are fractions there (a naive sum would reject them).
    assert 0.4 + 0.05 + 0.13 + 0.34 + 0.08 > 1 and sum([0.4, 0.05, 0.13, 0.34, 0.08]) == 1
    cases = [(10, [3, 7]), (30, [0.3, 0.3, 0.4]), (17, [0.4, 0.05, 0.13, 0.34, 0.08]), (7, [0.5, 0.5]), (5, [1, 0]),
             (9, [2, 0, 7]), (0, [0.5, 0.5]), (100, [0.8, 0.1, 0.1]), (11, [0.25, 0.25, 0.25, 0.25])]
    splits = []
    for seed in (0, 1, 42, 2024):
        for n, lengths in cases:
            parts = random_split(range(n), lengths, generator=torch.Generator().manual_seed(seed))
            splits.append((n, lengths, seed, False, [list(p.indices) for p in parts]))
    for n, lengths in cases[:4]:
        torch.manual_seed(42)
        parts = random_split(range(n), lengths)
        splits.append((n, lengths, 42, True, [list(p.indices) for p in parts]))
    split_errors = [error(lambda: random_split(range(10), [1.5, -0.5])),
                    error(lambda: random_split(range(10), [3, 3]))]

    w = sys.stdout.write
    w("// Generated by unit_test/data/gen_golden.py (torchvision %s, torch %s). Do not edit.\n"
      % (__import__("torchvision").__version__, torch.__version__))
    w("#pragma once\n#include <cstdint>\n#include <string>\n#include <utility>\n#include <vector>\n\nnamespace golden {\n")
    w("    struct entry {\n        const char *kind, *path, *target;\n    };\n")
    w("    inline const std::vector<entry> tree = {\n")
    for kind, r, t in TREE:
        w("            {%s, %s, %s},\n" % (cstr(kind), cstr(r), cstr(t)))
    w("    };\n")
    w("    inline const std::vector<std::string> classes = {%s};\n" % ", ".join(cstr(c) for c in g["classes"]))
    w("    using samples_t = std::vector<std::pair<std::string, int64_t>>;\n")
    for k in ("ext3", "nodot", "valid_sub", "image_folder", "custom"):
        w("    inline const samples_t %s = {\n" % k)
        for p, t in g[k]:
            w("            {%s, %d},\n" % (cstr(p), t))
        w("    };\n")
    for k in ("ext3_error", "valid_sub_error", "custom_error", "empty_map_error", "none_error", "no_class_error"):
        w("    inline const std::string %s = %s;\n" % (k, cstr(g[k])))
    w("    struct split_case {\n        int64_t n;\n        std::vector<double> lengths;\n        uint64_t seed;\n"
      "        bool global;\n        std::vector<std::vector<size_t>> splits;\n    };\n")
    w("    inline const std::vector<split_case> splits = {\n")
    for n, lengths, seed, glob, parts in splits:
        w("            {%d, {%s}, %d, %s, {%s}},\n" % (
            n, ", ".join(repr(float(x)) for x in lengths), seed, "true" if glob else "false",
            ", ".join("{%s}" % ", ".join(str(i) for i in p) for p in parts)))
    w("    };\n")
    w("    inline const std::vector<std::string> split_errors = {%s};\n" % ", ".join(cstr(e) for e in split_errors))
    w("} // namespace golden\n")


if __name__ == "__main__":
    main()
