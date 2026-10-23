#!/bin/zsh
# Per-line coverage union of the _metric headers over the metric test binaries (llvm-cov's multi-object merge drops
# inline header functions whose hashes differ per binary). Usage: cov_union.sh <build-cov dir> <repo root>, after
# running each metric_test_<area> of a TORCHMEDIA_COVERAGE build with LLVM_PROFILE_FILE=<build-cov dir>/<area>.profraw.
# The policy numbers (progress11 D9): _common.hpp from the union, each _<area>.hpp from its owning binary's report
# (printed last). The union can list comment lines of functions another binary instantiates without running.
C=$1; W=$2
cd $C
for t in audio detection image segmentation text; do
  xcrun llvm-profdata merge -sparse $C/$t.profraw -o $C/$t.profdata
  for h in $W/libtorchmedia/include/torchmedia/_metric/*.hpp; do
    xcrun llvm-cov show ./unit_test/metric/metric_test_$t -instr-profile=$C/$t.profdata $h 2>/dev/null | sed "s|^|$(basename $h) |"
  done
done | python3 -I -c '
import re, sys, collections
cnt = collections.defaultdict(dict)
for line in sys.stdin:
    m = re.match(r"(\S+) +(\d+)\|\s*([0-9.kMG]*)\|", line)
    if not m or not m.group(3): continue
    f, n, c = m.group(1), int(m.group(2)), m.group(3)
    v = float(c[:-1]) * {"k": 1e3, "M": 1e6, "G": 1e9}[c[-1]] if c[-1] in "kMG" else float(c)
    cnt[f][n] = max(cnt[f].get(n, 0), v)
for f, d in sorted(cnt.items()):
    miss = sorted(n for n, v in d.items() if v == 0)
    print(f"{f}: {len(d) - len(miss)}/{len(d)} lines ({100 * (len(d) - len(miss)) / len(d):.2f}%) missed {miss}")
'
for t in audio detection image segmentation text; do
  xcrun llvm-cov report ./unit_test/metric/metric_test_$t -instr-profile=$C/$t.profdata \
    $W/libtorchmedia/include/torchmedia/_metric/_$t.hpp 2>/dev/null | grep '^TOTAL' | sed "s/^TOTAL/_$t.hpp/"
done
