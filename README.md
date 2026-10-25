# Jenga tower tracker (LiDAR + camera) — work in progress

An iPhone with LiDAR (and optionally a second, ordinary phone) watches a Jenga tower and shows, on a live
viewer on your Mac, **green / red on every block**: green = may be pulled, red = may not. It works out which
layer was pulled after each move, and calls **game over** when pieces fall.

> **Status: experimental course project.** It works end to end on one tower and one desk setup. The numbers
> below are from a handful of games, not a proper benchmark.

![overlay](docs/overlay_example.png)

## What it does
| Feature | How | How well (honest numbers) |
|---|---|---|
| Red/green per block | Balance rules in `jenga_rules.py`: the weight above a layer must stay over the blocks left in it | Checked by hand-worked cases (`test_jenga_rules.py`) and by playing |
| Which layer was pulled | After each MARK, type `left` / `middle` / `right`; the CV model compares the tower before/after in that column and finds the layer | 6/7, 8/10, 9/9, 11/12 right on four games; a fresh tall (16-layer) game was 9/9. Confident answers were almost always right; "NOT CONFIDENT" ones need a check |
| Loose block warning (amber) | LiDAR depth: a block face ≥ 5 mm closer than at the start of the game | Works on replayed recordings with an injected 8 mm / 5 mm push-out. **Not yet seen on a real loose block** |
| Game over | LiDAR tower-top height: a drop of more than a layer = pieces fell (also: type `fell` in the label box) | Fired on all 3 recorded collapses, silent on the clean game |
| No moves left | Every remaining block is red | Banner on the viewer and both camera pictures |
| Tower height from LiDAR | Shown as "LiDAR height ≈ N layers" (matched 16, 9, 7, 9 on four games) | Needs the tower standing and the app streaming depth |

**Not done / known limits**
* "More than one block removed in one turn" is **not** detected unless the tower's top drops. Normal pulls disturb
  neighbours too much for a block-count test to tell them apart.
* Auto-detect of the layer count (Re-detect button) fails on very tall towers: **type the layer count by hand**
  (the LiDAR height hint helps).
* Depth only comes from the iPhone, and it sees only some faces from a steep angle (see *Check coverage* in the viewer).
* ARKit's own mesh was tried and dropped: ~30 mm triangles can't resolve 15 mm layers.
* The balance model assumes equal block weight and a straight tower; it knows nothing about friction.

## Hardware
* iPhone with LiDAR (tested on a 15 Pro) running the app in `lidar-capture/` (Xcode, UIKit + ARKit)
* Optional second camera on the opposite side: any phone (browser page, or your own app that POSTs to `/android`)
* 4 printed ArUco markers (`aruco_markers_A4.png`: DICT_4X4_50, ids 0-3, **40 mm**), taped in a square ring around the
  tower footprint (ids run around the square; set the half-spacing with env `MARKER_HALF`, default 80 mm).
  Phones on stands, ~45 cm away, both seeing at least 3 markers.
* A Mac on the same Wi-Fi as the phones.

## Quick start
```bash
scripts/setup.sh                      # venv + dependencies + fetches STEPWISE into third_party/
.venv/bin/python iphone_saved.py      # server: prints your Mac's IP
```
1. Open `lidar-capture.xcodeproj` in Xcode, choose **your own signing team**, run on the iPhone, type the Mac's IP.
   (Rebuild after pulling updates: the app now also streams depth for the loose-block / game-over features.)
2. Open `http://localhost:8000/` — live view and controls (START / STOP / MARK with the space bar).
3. Second camera: open `https://<mac-ip>:8443/cam` in Chrome and accept the certificate warning (the server makes its own
   certificate in `certs/`). Calibrate its lens once: `.venv/bin/python calibrate_camera.py --show-board`, then
   `.venv/bin/python calibrate_camera.py --url http://127.0.0.1:8000/android_raw.jpg`.
   The shipped `camera_android.json` is **for the author's phone** — replace it with your own calibration.
4. Optional but recommended once per marker mat: `.venv/bin/python calibrate_markers.py` measures where your markers
   really are and writes `marker_map.json` (setup-specific, not in the repo). Without it the nominal layout is used.

## Playing a game
1. Build a fresh tower. In the viewer's **Tower** panel type the real number of layers (a full tower is 18) and press Apply.
   For a short tower you can try **Re-detect tower** instead.
2. Switch on **Training data** (the layer-finder needs the crops it records).
3. Press **START**. Pull a block. When your hand is out of view, wait ~5 s and press **MARK**.
4. Type `left`, `middle` or `right` (as seen from the iPhone) in the box and press Enter. The answer appears with a
   confidence flag; the block is marked removed automatically and the colours update.
   If it's wrong, type the real layer number to correct it. Corrections are saved in `dataset/<run>/labels.jsonl`.
5. Colours: **green** may be pulled, **amber** may be pulled but is loose, **red** may not.
   **NO MOVES LEFT** / **GAME OVER** banners appear on their own.
6. **New game:** clear the *removed* box in the Tower panel, set the layer count and press Apply (this also clears GAME OVER).

### The rules (`jenga_rules.py`)
* Only below the highest complete layer (official rule). **House rule of this game: never the bottom layer** — one
  switch, `BOTTOM_LAYER_LEGAL`, in `jenga_rules.py`.
* After a pull, the centre of mass of everything above each layer must stay over the blocks left in that layer
  (4 mm margin). So: with two blocks left (edge + middle) the edge may go but not the middle; with just the middle, or
  just two edges, nothing may go; a layer is never emptied; a tower already balancing on a thread freezes everything below it.

## Tests
```bash
.venv/bin/python test_jenga_rules.py     # the pull rules, hand-worked cases (no data needed)
.venv/bin/python test_game_events.py     # synthetic game for the tracker
```
`test_loose_replay.py` and `test_height_replay.py` replay recorded depth frames and need recordings that are not in this repo.

## Layout
| File | Purpose |
|---|---|
| `iphone_saved.py` | Upload server + live viewer + overlay + data capture + layer-finder job + loose/game-over watch |
| `lidar-capture/` | iOS app (capture, preview + depth stream, remote START/STOP/MARK) |
| `jenga_rules.py`, `test_jenga_rules.py` | Red/green pull rules and their tests |
| `move_infer.py`, `label_eval.py` | Which layer was pulled (before/after of a MARK) and scoring against `labels.jsonl` |
| `tower_points.py` | Marker layout, pose solver (tries many starts when only 3 markers are visible), LiDAR-into-tower-frame |
| `tower_auto.py`, `calibrate_markers.py` | Layer count / direction from two cameras; measured marker positions |
| `depth_protrusion.py` | LiDAR face offsets: loose blocks and tower height |
| `face_crops.py`, `extract_crops.py` | Face-crop dataset (`dataset/`, not in the repo); crops follow the tower's real height |
| `game_events.py`, `live_tracker.py`, `rgb_cue.py`, `stepwise_adapter.py` | Earlier per-slot reader and live tracker |
| `tap_report.py`, `geo_report.py`, `align_report.py`, `cluster_crops.py`, `camera_coverage.py` | Unlabelled-data analysis (needs torch for the embedding ones) and best-view advice |
| `calibrate_camera.py`, `android_cam.py` | Second-camera lens calibration / app-free capture |
| `fuse_scan.py`, `tsdf_fuse.py`, `refined_fuse.py`, `process_scan.py` | Early mesh-reconstruction experiments (superseded) |

## Notes
* Recordings and the dataset are deliberately not in this repo (large; they show people). `docs/` has tower-only figures.
* Tower orientation matters: the marker frame is turned so the bottom layer runs along x
  (`tower_points.py`, `LAYOUTS["A"]`). A different build orientation needs that one table changed.
* `tap_report.py` (ResNet18 / DINOv2 fingerprints) needs `torch`, `torchvision`, `transformers`; the rest does not.
* Depth reader and tower geometry come from **STEPWISE / ThirdEye** by Aadit Pani
  (https://github.com/AaditPani-RVU/STEPWISE), fetched by the setup script, not copied here.
