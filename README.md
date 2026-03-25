# Golf Swing Analyzer

Analyzes golf swing mechanics from a video using MediaPipe Pose Landmarker.

## Setup

```bash
cd golf_swing_analyzer
pip3 install -r requirements.txt
```

The MediaPipe pose model (~25 MB) is downloaded automatically on first run and cached in `~/.cache/golf_swing_analyzer/`.

## Usage

**Basic analysis (prints report to terminal):**
```bash
python3 golf_swing_analyzer.py swing.mp4
```

**Save annotated video + JSON report:**
```bash
python3 golf_swing_analyzer.py swing.mp4 -o annotated.mp4 -r report.json
```

**Left-handed golfer:**
```bash
python3 golf_swing_analyzer.py swing.mp4 --handed left
```

**Faster processing (skip every other frame):**
```bash
python3 golf_swing_analyzer.py swing.mp4 --skip 1
```

**Most accurate model:**
```bash
python3 golf_swing_analyzer.py swing.mp4 --model heavy
```

## What it measures

| Metric | Phase(s) | Target range |
|---|---|---|
| Spine tilt (forward bend) | Address, Impact | 25–45° |
| Shoulder turn | Top of backswing | 30–80° from horizontal |
| Hip rotation | Impact | 8–50° open |
| Lead knee flex | Address | 140–165° joint angle |
| Trail knee flex | Address | 140–165° joint angle |
| Lead elbow angle | Impact | 155–185° (nearly straight) |
| Head drift X | Mid-swing | < ±6% of frame width |
| Head drift Y | Mid-swing | < ±6% of frame height |

## Swing phases detected

`ADDRESS → TAKEAWAY → BACKSWING → TOP → DOWNSWING → IMPACT → FOLLOW_THROUGH → FINISH`

Phase detection is wrist-height + shoulder-turn based — works best when the golfer is filmed **face-on** (camera pointing at the golfer's chest/belt, perpendicular to the target line).

## Tips for best results

- Film **face-on** (perpendicular to ball-target line) at hip height
- Ensure full body is visible, ideally with a plain background
- Use `--model heavy` for best landmark accuracy
- 60 fps video gives the most accurate phase detection; 30 fps works well too
