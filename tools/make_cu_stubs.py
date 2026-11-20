"""solutions/<stem>.cu -> mine/<stem>.cu skeletons for the G track.

    python3 tools/make_cu_stubs.py            # write the mine/*.cu files that do not exist yet
    python3 tools/make_cu_stubs.py --force    # overwrite (destroys your work!)
    python3 tools/make_cu_stubs.py --check    # every mine/*.cu still has a stub marker or differs from the solution

Every region between `// >>> SOLUTION (title)` and `// <<< SOLUTION` is replaced by a TODO comment carrying
the title, and `#define DRILL_STUB` is added after the header comment: the Python wrapper refuses to build the
file while that line is present, so an unfinished skeleton fails with NotImplementedError instead of
returning garbage. Delete the line when you start implementing.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"^([ \t]*)// >>> SOLUTION \((.*?)\)\n.*?^[ \t]*// <<< SOLUTION\n", re.S | re.M)


def stub(src: str) -> str:
    out = PAT.sub(lambda m: f"{m.group(1)}// TODO(drill): {m.group(2)}\n{m.group(1)}// (see the hint page; the reference is in solutions/)\n\n", src)
    lines = out.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("//"):
        i += 1
    lines.insert(i, "\n#define DRILL_STUB  // delete this line when you start implementing\n")
    return "".join(lines)


def main(argv):
    force, check = "--force" in argv, "--check" in argv
    bad = 0
    for sol in sorted((ROOT / "solutions").glob("*.cu")):
        mine = ROOT / "mine" / sol.name
        text = stub(sol.read_text())
        if "// >>> SOLUTION" in text or "TODO(drill)" not in text:
            print(f"{sol.name}: no SOLUTION region found")
            bad += 1
            continue
        if check:
            if not mine.exists() or mine.read_text() == sol.read_text():
                print(f"{mine.relative_to(ROOT)}: missing or identical to the solution")
                bad += 1
            continue
        if mine.exists() and not force:
            print("skip  ", mine.relative_to(ROOT))
            continue
        mine.write_text(text)
        print("wrote ", mine.relative_to(ROOT))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
