#!/usr/bin/env python3
"""
Turn recorded folders into the face-crop dataset.

    .venv/bin/python extract_crops.py --src game_04 --name game04
    .venv/bin/python extract_crops.py --src iphone_saved --name game03 --session 879F4984

Writes dataset/<name>/{iphone,android}/*.jpg, index.jsonl and marks.json.
"""
import argparse, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_crops as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame (neighbours are near duplicates)")
    ap.add_argument("--session", help="only iPhone frames of this session id prefix (folder with several games)")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(HERE, args.src)
    since = until = None
    if args.session:
        ts = sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob(f"{src}/*_intrinsics.json")
                    if json.load(open(f)).get("session_id", "").startswith(args.session))
        segs, cur = [], [ts[0]]
        for a, b in zip(ts, ts[1:]):
            if b - a > 20000:
                segs.append(cur); cur = []
            cur.append(b)
        segs.append(cur)
        seg = max(segs, key=len)
        since, until = seg[0] - 2000, seg[-1] + 3000
        print(f"session {args.session}: using the longest segment, {len(seg)} frames, "
              f"{(seg[-1] - seg[0]) / 1000:.0f} s")

    out = os.path.join(HERE, "dataset", args.name)
    w = fc.DatasetWriter(src, out, stride=args.stride, since_ms=since or 0, until_ms=until)
    n = w.step()
    w.close()
    marks = []
    for f in glob.glob(f"{src}/*_mark.json"):
        try:
            m = json.load(open(f))
            if (since is None or m["t_ms"] >= since) and (until is None or m["t_ms"] <= until):
                marks.append(m["t_ms"])
        except Exception:
            pass
    json.dump({"t0_ms": w.t0, "marks_ms": sorted(marks)}, open(os.path.join(out, "marks.json"), "w"))
    print(f"{args.name}: {w.counts['frames']} frames -> {w.counts['crops']} face crops "
          f"({w.counts['skipped_no_pose']} frames skipped: no pose)  -> {out}")


if __name__ == "__main__":
    main()
