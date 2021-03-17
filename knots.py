
from PIL import Image
import numpy as np

from write_tube import tube, save_sdl
from from_sketch import knot_path

# knot-theoretically valid trefoil
knots0 = {
    "test": np.array([
        (0, 0, 0),
        (3, 0, 0),
        (3, 2, 0),
        (2, 2, 0),
        (2, 1, 0),
        (1, 1, 0),
        (1, 2, 0),
        (0, 2, 0),
    
    ]),
    "3_1": np.array([
        (-1, 2, 0),
        (1, 2, 1),
        (0.5, 0, 1),
        (0.5, 0, 0),
        (0, -1, 0),
        (-2, 0, 1),
        (0, 0.5, 1),
        (0, 0.5, 0),
        (1, 1, 0),
        (2, 0, 0),
        (1, -1, 1),
        (0, 0, 1),
        (0, 0, 0),
    ]),
    "4_1": np.array([
        (1, 3, 0),
        (-1, 3, 1),
        (-1, 1, 1),
        (-1, 1, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, -1, 1),
        (-2, -1, 1),
        (-2, 2, 0.5),
        (2, 2, 0.5),
        (2, -1, 0),
        (-1, -1, 0),
        (-1, -1, 1),
        (1, 1, 1),
        (1, 1, 0),
    ]),
    "5_2": np.array([
        (1, 4, 0),
        (-1, 4, 1),
        (-1, 2, 1),
        (-1, 2, 0),
        (1, 1, 0),
        (1, 1, 1),
        (-1, 0, 1),
        (-1, 0, 0),
        (-1, -1, 0),
        (2, -1, 0),
        (2, 3, 0.5),
        (-2, 3, 0.5),
        (-2, -1, 1),
        (1, -1, 1),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, 1, 1),
        (1, 2, 1),
        (1, 2, 0),
    ]),
    "6_1": np.array([
        (1, -1, 0),
        (-0.5, -0.5, 0),
        (-0.5, -0.5, 1),
        (-3, 0, 1),
        (-3, 0, 0),
        (0, 0, 0),
        (0, 0, 2),
        (3, 0, 2),
        (3, 0, 1),
        (-1, -1, 1),
        (-1, -1, 0),
        (1, -2, 0),
        (2, -2, 0),
        (2, -2, 1.5),
        (2, 1, 1.5),
        (-2, 1, 0.5),
        (-2, -2, 0.5),
        (-1, -2, 1),
        (1, -1, 1),
    ]),
    "7_5": knot_path("knot_sketches/7_5.png")
}

myname = "6_1"
myknot = knots0[myname]

subsample = 5
rad = 1
points_in_circle = 40
min_d = 2 * np.pi / points_in_circle
far = points_in_circle // 3

def l2(diff):
    return np.sqrt(np.sum(diff**2))

points0 = np.concatenate(
    [[
        t1 * a + t0 * (1-a) for a in np.arange(0, 1, 1 / (0.001 + l2(t1-t0) * subsample))
    ] for t0, t1 in zip(myknot, np.roll(myknot, -1, axis=0))],
    axis=0,
)
points = points0

def get_far_points(points, i, far):
    if (i > far and i < len(points) - far):
        return np.concatenate([points[:i-far], points[i+far+1:]], axis=0)
    if i <= far:
        return points[i+far+1:len(points) - (far-i)]
    return points[(i + far) - (len(points) - 1): i-far]

def elastic_fn(a, b, c, d, e, close):
    if l2(a - b) < close and l2(c - b) < close:
        return np.array([0, 0, 0])
        print("too close")
    return 0.5 * (a - c) + (b - c) + (d - c) + 0.5 * (e - c)
    
def repulse_fn(diff, rad):
    norm = 1/np.sqrt(np.sum(np.abs(diff)**2))
    if norm < 1/(2 * rad):
        #print("too far")
        return np.array([0, 0, 0])
    if norm > 10:
        #print("one")
        norm = 1
    #print(l2(diff * (norm**2)))
    return diff * (norm**2)

def forces(points, elasticity=0.1, repulsion=4/len(points), far=far, close = 0.1/subsample, justdamp=False):
    return np.array([
        elasticity * (
            elastic_fn(points[i-2], points[i-1], points[i], points[(i+1) % len(points)], points[(i+2) % len(points)], close)
        ) + (
            np.array([0, 0, 0]) if justdamp else (
                repulsion * sum(
                    repulse_fn(points[i] - q, rad=rad)
                    for q in get_far_points(points, i, far)
                )
            )
        )
        for i in range(len(points))
    ])

def cull_and_generate(points):
    new_points = [points[0]]
    for point in points:
        q = new_points[-1]
        if l2(point - q) > 3 * min_d:
            new_points.append(((point[0] + q[0])/2, (point[1] + q[1])/2, (point[2] + q[2])/2))
        if l2(point - q) > min_d:
            new_points.append(point)
    return np.array(new_points)

def double_points(points):
    new_points = [points[0]]
    for point in points:
        q = new_points[-1]
        new_points.append(((point[0] + q[0])/2, (point[1] + q[1])/2, (point[2] + q[2])/2))
        new_points.append(point)
    return np.array(new_points)

iters = 100

for i in range(iters):
    f = forces(points)
    #print(f)
    points = cull_and_generate(points + f)

    im = np.zeros((100, 100, 3))
    mi = np.min(points)
    ma = np.max(points)
    for p in points:
        im[
            int((99 / (ma - mi)) * (p[0] - mi)),
            int((99 / (ma - mi)) * (p[1] - mi)),
            :
        ] = np.array([
            255,
            int((255 / (ma - mi)) * (p[2] - mi)),
            255,
        ]) 

    if i % 5 == 0:
        Image.fromarray(im.astype("uint8")).save(f"iterations/{i}.png")

dampiters = 10
points = double_points(points)

for i in range(dampiters):
    f = forces(points, justdamp=True)
    #print(f)
    points = cull_and_generate(points + f)

    im = np.zeros((100, 100, 3))
    mi = np.min(points)
    ma = np.max(points)
    for p in points:
        im[
            int((99 / (ma - mi)) * (p[0] - mi)),
            int((99 / (ma - mi)) * (p[1] - mi)),
            :
        ] = np.array([
            255,
            int((255 / (ma - mi)) * (p[2] - mi)),
            255,
        ]) 

    Image.fromarray(im.astype("uint8")).save(f"iterations/{iters}_{i}.png")

save_sdl(tube(points, k=20, r=(ma-mi)/15), name=myname)
