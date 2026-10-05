"""Interactive demo: streamlit run app.py"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from calibrate import (calibrate, camera_pose, geo_to_local, intrinsics, leave_one_out,
                       load_matches, render_overlay, solve_pose, validate)

ROOT = Path(__file__).parent

st.set_page_config(page_title="Camera Calibration Demo", layout="wide")
st.title("Camera calibration from geo-referenced landmarks")
st.caption("Pixels matched to real-world latitude, longitude and altitude → "
           "the camera's focal length, position and viewing direction.")

datasets = sorted(p.parent.name for p in ROOT.glob("*/matches.csv"))
folder = ROOT / st.sidebar.selectbox("Scene", datasets)

ids, pixels, geo = load_matches(folder / "matches.csv")
problems = validate(geo)
if problems:
    st.error("This scene's matches.csv fails the data check, so it is not calibrated:\n\n"
             + "\n".join(f"- {p}" for p in problems))
    st.stop()

image = cv2.imread(str(folder / f"{folder.name}.jpg"))
height, width = image.shape[:2]

st.sidebar.markdown("**Landmarks used**")
st.sidebar.caption("Untick a landmark to see how the solution and error change.")
use = np.array([st.sidebar.checkbox(f"Point {i}", value=True) for i in ids])
if use.sum() < 5:
    st.warning("Keep at least 5 landmarks: the solver estimates 7 unknowns "
               "(focal length + 3 rotation + 3 translation) and each landmark gives 2 equations.")
    st.stop()

ids_u = [i for i, u in zip(ids, use) if u]
pixels_u, geo_u = pixels[use], geo[use]
origin = (geo_u[:, 0].mean(), geo_u[:, 1].mean(), 0.0)
world = geo_to_local(geo_u, origin)


@st.cache_data
def solve(world, pixels, width, height):
    best = calibrate(world, pixels, width, height)
    return best, leave_one_out(world, pixels, width, height)


best, held_out = solve(world, pixels_u, width, height)
if best is None:
    st.error("No valid camera pose found for these landmarks.")
    st.stop()
(cam_lat, cam_lng, cam_alt), heading = camera_pose(best, origin)
valid = [e for e in held_out if e is not None]

c = st.columns(5)
c[0].metric("Focal length", f"{best['f']:.0f} px")
c[1].metric("Field of view", f"{best['fov']:.1f}°")
c[2].metric("Heading", f"{heading:.0f}°")
c[3].metric("Fit error (RMS)", f"{best['rms']:.1f} px")
c[4].metric("Held-out error (median)", f"{np.median(valid):.1f} px" if valid else "n/a",
            help="Calibrate without a landmark, then predict where it should appear.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Labelled vs reprojected")
    overlay = render_overlay(image, ids_u, pixels_u, best["projected"])
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
with right:
    st.subheader("Where the solver put the camera")
    points = pd.DataFrame({"lat": list(geo_u[:, 0]) + [cam_lat],
                           "lon": list(geo_u[:, 1]) + [cam_lng],
                           "color": ["#e5484d"] * len(ids_u) + ["#2f9e44"],
                           "size": [25] * len(ids_u) + [60]})
    st.map(points, color="color", size="size")
    st.caption(f"Red: landmarks. Green: solved camera at {cam_lat:.5f}, {cam_lng:.5f}, "
               f"about {cam_alt:.0f} m up.")

st.subheader("Choosing the focal length")
st.caption("For each candidate field of view, solve the pose and measure the error. "
           "The lowest point is the focal length the solver picks.")
fovs, rms = [], []
for fov in np.arange(15, 90, 0.5):
    f = (width / 2) / np.tan(np.radians(fov) / 2)
    result = solve_pose(world, pixels_u, intrinsics(f, width, height))
    if result is not None:
        fovs.append(fov)
        rms.append(float(np.sqrt(np.mean(result[3] ** 2))))
st.line_chart(pd.DataFrame({"field of view (deg)": fovs, "RMS error (px)": rms})
              .set_index("field of view (deg)"))

st.subheader("Per-landmark error")
st.dataframe(pd.DataFrame({"point": ids_u,
                           "fit error (px)": np.round(best["errors"], 1),
                           "held-out error (px)": [None if e is None else round(e, 1) for e in held_out]}),
             hide_index=True)
