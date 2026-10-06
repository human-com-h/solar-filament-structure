import numpy as np
import cv2
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.morphology import skeletonize, dilation, diamond

def scaled_polygon_points(ann, sx, sy):
    polys = []
    for poly in ann.get("segmentation", []):
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(arr) >= 3:
            arr[:, 0] *= sx
            arr[:, 1] *= sy
            polys.append(arr)
    return polys

def branch_components_from_instance(ann, width, height, resolution):
    """Return global (y,x) pixel coordinates for spine-anchored lateral skeleton branches.

    Principal path = shortest path on the instance skeleton between skeleton pixels
    nearest to the two endpoints of the manual spine polyline.
    Branch proxies = skeleton pixels outside a 1-pixel dilation of that principal path.
    """
    raw_spine = ann.get("spine", [])
    if not raw_spine or len(raw_spine) < 4:
        return []

    sx, sy = resolution / float(width), resolution / float(height)
    polys = scaled_polygon_points(ann, sx, sy)
    if not polys:
        return []

    allp = np.concatenate(polys, axis=0)
    x0 = max(0, int(np.floor(allp[:, 0].min())) - 2)
    y0 = max(0, int(np.floor(allp[:, 1].min())) - 2)
    x1 = min(resolution, int(np.ceil(allp[:, 0].max())) + 3)
    y1 = min(resolution, int(np.ceil(allp[:, 1].max())) + 3)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return []

    mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    for arr in polys:
        pts = np.rint(arr - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, mask.shape[1]-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, mask.shape[0]-1)
        cv2.fillPoly(mask, [pts], 1)

    sk = skeletonize(mask.astype(bool))
    coords = np.argwhere(sk)  # (y,x)
    if len(coords) < 2:
        return []

    spine = np.asarray(raw_spine, dtype=np.float32).reshape(-1, 2)
    spine[:, 0] *= sx
    spine[:, 1] *= sy
    p0 = np.array([spine[0, 1]-y0, spine[0, 0]-x0], dtype=np.float32)
    p1 = np.array([spine[-1, 1]-y0, spine[-1, 0]-x0], dtype=np.float32)

    tree = cKDTree(coords.astype(np.float32))
    _, i0 = tree.query(p0)
    _, i1 = tree.query(p1)
    start = tuple(int(v) for v in coords[int(i0)])
    end = tuple(int(v) for v in coords[int(i1)])

    if start == end:
        return []

    # Skeleton is connected for valid single-piece filament masks.
    # High off-skeleton cost prevents shortcutting away from the skeleton.
    cost = np.where(sk, 1.0, 1e5).astype(np.float32)
    route, route_cost = route_through_array(
        cost, start, end, fully_connected=True, geometric=True
    )
    route = np.asarray(route, dtype=np.int32)
    if len(route) < 2 or not np.all(sk[route[:,0], route[:,1]]):
        return []

    trunk = np.zeros_like(sk, dtype=bool)
    trunk[route[:,0], route[:,1]] = True

    # Remove the principal path and its immediate junction neighborhood.
    # This creates disconnected lateral skeleton components without introducing
    # an arbitrary physical length threshold.
    branch = sk & ~dilation(trunk, diamond(1))
    nlab, lab = cv2.connectedComponents(branch.astype(np.uint8), connectivity=8)

    comps = []
    for k in range(1, nlab):
        yy, xx = np.where(lab == k)
        if len(yy) == 0:
            continue
        gy = yy + y0
        gx = xx + x0
        comps.append(np.stack([gy, gx], axis=1))
    return comps
