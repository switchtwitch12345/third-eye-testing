# Third Eye: qualitative analysis

Written for: Mitul and the team, as raw material for the Discussion and Results sections of the CS3235 paper.
Evidence is marked **[measured]** (a script in the repo, with its numbers), **[observed]** (seen during development or in a recorded game, not a controlled test) or **[open]** (not checked). Nothing here is from a held-out test.

## 1. What the system does well

1. **A stable overlay when the markers are visible [observed].** The tower outline stays on the tower while the player's hand is in view, because the pose comes from the fixed ArUco board and not from the tower.
2. **Fast layer-finding [measured].** Finder compute is 0.18–0.34 s per move. The calm window needed before and after a move was cut from 4.5 s to at least 1.6 s.
3. **Honest confidence [observed].** The finder applies an answer automatically only at 90% or above, and asks for a Yes/No otherwise. On the 31 labelled pulls the combined rule was right 81% of the time.
4. **Safe failure on the fall detector [measured].** In the 381 s fall replay it raised 1 true detection and 0 false alarms across 3 games.

## 2. Where it fails, and why

| Failure | What we saw | Cause | Status |
|---|---|---|---|
| iPhone and Android disagree on the layer | Android 53% vs iPhone 79% alone **[measured]** | Android view is lower and more oblique, so faces are smaller and more occluded by neighbouring blocks | Mitigated by weighting; not fixed |
| Layer finder worse on tall towers | Old 16-layer games 1/7 **[measured]** | 32×32 face crops of adjacent layers look almost identical at that resolution | Current games use 7–10 layers |
| Lens covered by a hand | Looked like a fall, because the depth showed a close blob **[observed]** | Fall check could not tell a covered lens from a collapsed tower | Fixed with a covered-lens guard and a corroborating p95 drop; checked in replay only, not with a real hand **[open]** |
| Three visible markers | Several different poses fit the marker centres exactly, so the solver could land on the wrong one **[observed]** | Perspective ambiguity at a low camera angle | Fixed by trying many starting views and letting the corners decide |
| Overlay 38 mm off on one board **[observed]** | Whole frame tilted | Printed marker layout did not match the real sheet | Fixed by measuring the tower with LiDAR and calibrating the board |
| Pulled blocks | Tower height watch fired on legal moves **[observed]** | Real Jenga puts the pulled block on top | Fixed with the "pulled blocks go back on top" flag |

## 3. The LiDAR pose correction: a result we cannot yet defend

`benchmark_pose.py` projects the tower outline into the iPhone image and measures edge strength under it (higher is better), on 75 hand-free frames from 3 games.

| Pose | Mean edge strength |
|---|---|
| Markers only | 3.07 |
| Markers + LiDAR box fit | 2.95 |
| Markers + LiDAR fit + edge snapping | 8.76 |

- The LiDAR fit made the edge score **slightly worse** (better on only 41% of frames). That doesn't show it improves the overlay.
- Edge snapping improved all 75 frames, but it optimises this same score, so the result is circular. It shows the snapper works, not that the overlay is more correct.
- The fitted corrections were not small: 2.2° to 7.1° of tilt and 5–10 mm of shift. The 7.1° run (2 Nov) may predate the board recalibration. If the tilt were real, the marker pose would be badly wrong, which our other checks don't support. The likelier cause is that the fit is misled by the partly visible tower. **[open]**
- For the paper: don't claim the LiDAR fit improves pose accuracy. Say that its value is the height reading and the board calibration, and that pose accuracy needs an independent check (for example, manually clicked tower corners).

## 4. Height estimation: a simpler baseline wins on clean frames

From `benchmark_height.py`, 233 hand-free frames: the 98th percentile of point height gets the right layer 98% of the time. Our box fit gets 88%, because it reads about 2 mm tall. Our fit's advantage, handling a hand or fallen block in view, was not shown, even in a synthetic test. Be upfront about this in the Discussion.

## 5. Comparison with existing approaches (qualitative only; no numbers quoted from other work)

| Approach | Sensors | Strength | Limitation compared with ours |
|---|---|---|---|
| Fazeli et al. 2019, Jenga-playing robot (See, feel, act) | Force/torque sensing plus vision | Decides physical feasibility of a push | Needs a robot arm and a lab; we only watch humans |
| Depth-camera object tracker (Kinect-style) | One RGB-D camera | Simple, one device | One viewpoint, so a hand or the tower itself hides blocks |
| ARKit object detection / RoomPlan | iPhone LiDAR | Built into the phone | Not built for a small, repeating-block structure, and gives no per-block identity **[open]**: untested by us |
| Ours | iPhone LiDAR + RGB, Android RGB, ArUco board | Two views, block-level layer finding, runs on a Mac with phones | Needs a printed board, fixed phones, small data; 6–10 layers |

## 6. Limits to state plainly

- Only about 31 labelled pulls in 10 games; no held-out set.
- One tower, one board, one lighting setup, two phones.
- Wired mode, the Android wired path, and a real hand over the lens are untested live. End-to-end latency is unmeasured.
- Reference entries in the paper were written from memory and need checking.
