"""Calibrate a fixed webcam from geo-referenced landmarks.

Given a few image pixels matched to real-world (lat, lng, altitude) points,
estimate the camera's focal length (intrinsics) and its position and
orientation in the world (extrinsics), then check the result by
reprojecting the landmarks back into the image.

Usage:
    python calibrate.py boston_harbor_massachusetts
    python calibrate.py boston_harbor_massachusetts --image boston_harbor_massachusetts.jpg
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

EARTH_RADIUS_M = 6_378_137.0


def load_matches(csv_path):
    """Read id,img_x,img_y,map_lat,map_lng,map_altitude rows."""
    ids, pixels, geo = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            pixels.append([float(row["img_x"]), float(row["img_y"])])
            geo.append([float(row["map_lat"]), float(row["map_lng"]), float(row["map_altitude"])])
    return ids, np.array(pixels), np.array(geo)


def validate(geo, expected_region=None):
    """Catch the data-entry mistakes that silently ruin a calibration."""
    problems = []
    lat, lng = geo[:, 0], geo[:, 1]
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lng) > 180):
        problems.append("lat/lng out of range: columns are probably shifted")
    if len({tuple(p) for p in geo.round(6)}) < len(geo):
        problems.append("duplicate world points: two landmarks share one coordinate")
    if expected_region is not None:
        (lat0, lng0), radius_km = expected_region
        d = haversine_km(lat0, lng0, lat.mean(), lng.mean())
        if d > radius_km:
            problems.append(f"landmarks are {d:.0f} km from the expected location")
    return problems


def haversine_km(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a)) / 1000


def geo_to_local(geo, origin):
    """lat/lng/alt -> local East-North-Up metres around `origin`.

    A flat-earth approximation is accurate to centimetres over the
    few kilometres a webcam can see.
    """
    lat0, lng0, alt0 = origin
    east = np.radians(geo[:, 1] - lng0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    north = np.radians(geo[:, 0] - lat0) * EARTH_RADIUS_M
    up = geo[:, 2] - alt0
    return np.column_stack([east, north, up])


def local_to_geo(xyz, origin):
    lat0, lng0, alt0 = origin
    lat = lat0 + math.degrees(xyz[1] / EARTH_RADIUS_M)
    lng = lng0 + math.degrees(xyz[0] / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return lat, lng, alt0 + xyz[2]


def intrinsics(f, width, height):
    """Pinhole K with square pixels and the principal point at the image centre."""
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def solve_pose(world, pixels, K):
    ok, rvec, tvec = cv2.solvePnP(world, pixels, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(world, pixels, K, None, rvec, tvec)
    projected, _ = cv2.projectPoints(world, rvec, tvec, K, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    # Reject solutions that put landmarks behind the camera.
    R, _ = cv2.Rodrigues(rvec)
    if np.any((R @ world.T + tvec)[2] <= 0):
        return None
    return rvec, tvec, projected.reshape(-1, 2), errors


def calibrate(world, pixels, width, height):
    """Six points are too few to solve every intrinsic, so fix the
    principal point and sweep focal length, keeping the f whose best
    pose reprojects with the lowest RMS error."""
    best = None
    for fov_deg in np.arange(10, 120, 0.25):
        f = (width / 2) / math.tan(math.radians(fov_deg) / 2)
        K = intrinsics(f, width, height)
        result = solve_pose(world, pixels, K)
        if result is None:
            continue
        rms = float(np.sqrt(np.mean(result[3] ** 2)))
        if best is None or rms < best["rms"]:
            best = dict(f=f, fov=fov_deg, K=K, rvec=result[0], tvec=result[1],
                        projected=result[2], errors=result[3], rms=rms)
    return best


def leave_one_out(world, pixels, width, height):
    """Fitting error flatters the model, since every point helped fit it.
    Leave-one-out asks the honest question: how well does it predict a
    landmark it has never seen? Returns one error per point (None if the
    remaining points had no solution)."""
    errors = []
    for k in range(len(pixels)):
        keep = np.arange(len(pixels)) != k
        fit = calibrate(world[keep], pixels[keep], width, height)
        if fit is None:
            errors.append(None)
            continue
        pred, _ = cv2.projectPoints(world[k:k + 1], fit["rvec"], fit["tvec"], fit["K"], None)
        errors.append(float(np.linalg.norm(pred.ravel() - pixels[k])))
    return errors


def camera_pose(best, origin):
    """Camera position (lat, lng, alt) and compass heading from a solved pose."""
    R, _ = cv2.Rodrigues(best["rvec"])
    cam_local = (-R.T @ best["tvec"]).ravel()
    view_dir = R.T @ np.array([0, 0, 1.0])
    heading = (math.degrees(math.atan2(view_dir[0], view_dir[1])) + 360) % 360
    return local_to_geo(cam_local, origin), heading


def pixel_to_ground(u, v, best, plane_up=0.0):
    """Where a pixel lands on a horizontal surface (e.g. water or a table).

    One image gives only a direction per pixel, not a distance. Knowing the
    surface's height pins down where along that direction the point is.
    Returns local East-North-Up metres, or None if the pixel looks above
    the horizon and never meets the surface.
    """
    R, _ = cv2.Rodrigues(best["rvec"])
    center = (-R.T @ best["tvec"]).ravel()
    ray = R.T @ np.linalg.inv(best["K"]) @ np.array([u, v, 1.0])
    if abs(ray[2]) < 1e-9:
        return None
    s = (plane_up - center[2]) / ray[2]
    if s <= 0:
        return None
    return center + s * ray


def render_overlay(image, ids, pixels, projected):
    vis = image.copy()
    for i, (p, q) in enumerate(zip(pixels, projected)):
        p, q = tuple(int(v) for v in p), tuple(int(v) for v in q)
        cv2.circle(vis, p, 10, (0, 0, 255), 2)          # red: labelled pixel
        cv2.drawMarker(vis, q, (0, 255, 0), cv2.MARKER_CROSS, 22, 2)  # green: reprojected
        cv2.line(vis, p, q, (0, 255, 255), 1)
        cv2.putText(vis, ids[i], (p[0] + 12, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis, "red = labelled   green = reprojected", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--image", help="image file inside the folder (default: <folder>.jpg)")
    ap.add_argument("--expect", nargs=3, type=float, metavar=("LAT", "LNG", "RADIUS_KM"),
                    help="sanity check that landmarks are near this location")
    args = ap.parse_args()

    ids, pixels, geo = load_matches(args.folder / "matches.csv")
    region = ((args.expect[0], args.expect[1]), args.expect[2]) if args.expect else None
    problems = validate(geo, region)
    if problems:
        print(f"Data check FAILED for {args.folder}:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    image_path = args.folder / (args.image or f"{args.folder.name}.jpg")
    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"could not read {image_path}")
    height, width = image.shape[:2]

    origin = (geo[:, 0].mean(), geo[:, 1].mean(), 0.0)
    world = geo_to_local(geo, origin)
    best = calibrate(world, pixels, width, height)
    if best is None:
        sys.exit("no valid pose found")
    (cam_lat, cam_lng, cam_alt), heading = camera_pose(best, origin)

    print(f"Image {image_path.name}: {width}x{height}, {len(ids)} landmarks\n")
    print("Intrinsics")
    print(f"  focal length       {best['f']:.0f} px  (horizontal FOV {best['fov']:.1f} deg)")
    print(f"  K =\n{np.array2string(best['K'], precision=1, suppress_small=True, prefix='  ')}")
    print("\nExtrinsics")
    print(f"  camera position    {cam_lat:.6f}, {cam_lng:.6f}, {cam_alt:.0f} m")
    print(f"  looking toward     {heading:.0f} deg (0 = north, 90 = east)")
    print("\nReprojection error (px)")
    for i, e in zip(ids, best["errors"]):
        print(f"  point {i}: {e:6.1f}")
    print(f"  RMS:     {best['rms']:6.1f}")

    print("\nLeave-one-out error (px): calibrate on the other points, predict this one")
    held_out = leave_one_out(world, pixels, width, height)
    for i, e in zip(ids, held_out):
        print(f"  point {i}: " + ("no solution" if e is None else f"{e:6.1f}"))
    valid = [e for e in held_out if e is not None]
    if valid:
        print(f"  median:  {np.median(valid):6.1f}")

    out = args.folder / f"{args.folder.name}_reprojected.png"
    cv2.imwrite(str(out), render_overlay(image, ids, pixels, best["projected"]))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
