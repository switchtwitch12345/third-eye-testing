# Third Eye: quantitative analysis

Written for: Mitul and the team, as the numbers behind the Results section of the CS3235 paper.
Every figure comes from a script in the repo or from the paper's measurements. None is held out. Scripts: `benchmark_height.py`, `benchmark_pose.py`, `label_eval.py` (all in `/Users/mitul/Desktop/lidar-capture/`).

## 1. Layer finder (31 labelled pulls, 10 games)

Counts were rebuilt from the paper's Table 2 percentages and denominators, so they may differ by one pull from the original tally. 95% Wilson intervals:

| Cameras | Exact | 95% interval | Within 1 layer | 95% interval | n |
|---|---|---|---|---|---|
| iPhone only | 79% (23) | 62–90% | 93% (27) | 78–98% | 29 |
| Android only | 53% (16) | 36–70% | 90% (27) | 74–97% | 30 |
| Both, mean | 77% (24) | 60–89% | 94% (29) | 79–98% | 31 |
| Both, product rule | 81% (25) | 64–91% | 94% (29) | 79–98% | 31 |

- **iPhone vs Android is the one reliable difference.** The intervals only just overlap (62–90 vs 36–70), so Android alone is probably worse.
- **Mean vs product rule (77% vs 81%) is not a real difference.** That's one pull out of 31, and the product rule was chosen after seeing the data.
- **The 31 pulls are not independent:** they come from 10 games, and labels are partly "not corrected, so assumed right". All intervals are optimistic.
- Replay of the current server: 5 of 5 moves reached 95% in one 8-layer game, and 6 of 10 in another. Older 16-layer games: 1 of 7.

## 2. Height estimation (3 towers, 233 hand-free frames)

| Method | Exact layer | 95% interval | Mean error (layers) | Time |
|---|---|---|---|---|
| 98th percentile of height | 98% | 96–99% | 0.02 | 1 ms |
| Plain least-squares box fit | 88% | 83–92% | 0.12 | 51 ms |
| Robust box fit, one scale | 88% | – | 0.13 | 51 ms |
| Our fit (30 to 5 mm schedule) | 88% | 83–92% | 0.13 | about 55 ms |

- The 233 frames come from only 3 towers, so the effective sample is 3, not 233. Intervals above are far too narrow.
- Our fit reads about 2 mm tall. That bias, not noise, causes the 12% wrong layers.
- The first run logged 2.6 s for our fit; that was a one-off slowdown. A repeat on its own gave 55 ms.

## 3. Overlay pose (75 iPhone frames, 3 towers)

| Pose | Mean edge strength | Median |
|---|---|---|
| Markers only | 3.07 | 3.18 |
| Markers + LiDAR fit | 2.95 | 2.86 |
| Markers + LiDAR fit + edge snapping | 8.76 | 8.80 |

- The LiDAR fit beat markers-only on 41% of frames: no improvement shown.
- Edge snapping beat the LiDAR fit on 100% of frames, but it optimises this same score, so the result is circular.
- LiDAR corrections were 4.6–10.4 mm of shift and 2.2–7.1° of tilt per game, which is larger than expected. Unexplained.
- A position error in mm is still unmeasured. It needs hand-clicked tower corners as ground truth.

## 4. Timing and throughput

| Quantity | Value |
|---|---|
| Finder compute per move | 0.18–0.34 s (Mac) |
| Calm window before/after a move | at least 1.6 s (was 4.5 s; derived from settings, not measured) |
| Box fit | 62.2 ms median (laptop; about 55 ms on this Mac in the re-timing) |
| Pose solve | 10.8 ms |
| ArUco at 960 px | 7.1 ms |
| Full pipeline replay | 24–28 ms per frame, 8–10× faster than real time |
| Rule checker scaling | L³ (fitted slope 2.80) |
| Acquisition | 3.00, 2.91, 2.85 Hz; 1,706 frames; 0.78–0.87 MB per frame |

## 5. Collapse detection

One real fall across 3 replayed games (381 s): 1 of 1 detected, 0 false alarms. With a single event, the 95% interval for the detection rate is 21–100%, and no false alarms in 3 games says little. Hand over the lens was only simulated, with a replay switch.

## 6. What the numbers support, and what they don't

- **Supported:** the finder is usable on the iPhone view at 7–10 layers, and the combined confidence lets most moves apply automatically (about 77–81% exact, 94% within a layer). Per-frame cost is small enough to run live.
- **Not supported:** that LiDAR improves pose or height over simple baselines, that the product rule beats the mean, or anything about tall towers, other boards, or other lighting.
- **Next, if time allows:** 20 hand-clicked pose frames; a held-out game scored once with settings frozen; a measured wired-vs-Wi-Fi frame rate and end-to-end latency.
