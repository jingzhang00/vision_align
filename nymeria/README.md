# Nymeria RGB fisheye rectification

`batch_undistort.py` rectifies every Nymeria episode whose root contains both:

- `video.mp4`
- `recording_head/mps/slam/online_calibration.jsonl`

It reads `camera-rgb` from the first calibration record and implements Project
Aria's 15-parameter `FisheyeRadTanThinPrism` (`Fisheye624`) projection directly.
OpenCV's four-coefficient fisheye model is intentionally not used.

Run the complete dataset:

```bash
.venv/bin/python -m nymeria.batch_undistort \
  --dataset-root dataset/nymeria
```

Run one episode:

```bash
.venv/bin/python -m nymeria.batch_undistort \
  --dataset-root dataset/nymeria \
  --episode 20230814_s1_david_hall_act0_n9k3wo
```

The default output for each episode is:

```text
<episode>/undistorted/
├── _SUCCESS
├── video.mp4
└── camera_intrinsics.json
```

