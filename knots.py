
from PIL import Image
import numpy as np

from write_tube import tube, save_sdl
from from_sketch import knot_path
from webscraped_knot_data import eights, nines, tens

# initial specification of knots
knots0 = {**{
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
        "5_1": knot_path("knot_sketches/5_1.png"),
        "5_2": knot_path("knot_sketches/5_2.png"),
        "5_2_old": np.array([ # this one is more symmetric so keeps its twist
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
        "6_2": knot_path("knot_sketches/6_2.png"),
        "6_3": knot_path("knot_sketches/6_3.png"),
        "7_1": knot_path("knot_sketches/7_1.png"),
        "7_2": knot_path("knot_sketches/7_2.png"),    
        "7_3": knot_path("knot_sketches/7_3.png"),
        "7_4": knot_path("knot_sketches/7_4.png"),
        "7_5": knot_path("knot_sketches/7_5.png"),
        "7_6": knot_path("knot_sketches/7_6.png"),
        "7_7": knot_path("knot_sketches/7_7.png"),
    }, **{ # from Arc Presentation on katlas.org (todo: this [:-4] is because I constructed them wrong)
        f"8_{i}": np.array(eight)[:-4] for i, eight in enumerate(eights)
    }, **{
        f"9_{i}": np.array(nine)[:-4] for i, nine in enumerate(nines)
    }, **{
        f"10_{i}": np.array(ten)[:4] for i, ten in enumerate(tens)
    }
}


myname = "8_1"
myknot = knots0[myname]

rad = 1
points_in_circle = 40
min_d = 2 * np.pi / points_in_circle
far = points_in_circle // 3
subsample = points_in_circle  / (2 * np.pi)

def l2(diff):
    return np.sqrt(np.sum(diff**2))

points0 = np.concatenate(
    [[
        t1 * a + t0 * (1-a) for a in np.arange(0, 1, 1 / (0.001 + l2(t1-t0) * subsample))
    ] for t0, t1 in zip(myknot, np.roll(myknot, -1, axis=0))],
    axis=0,
)
points = points0

print("generated points")

def get_far_points(points, i, far):
    if (i > far and i < len(points) - far):
        return np.concatenate([points[:i-far], points[i+far+1:]], axis=0)
    if i <= far:
        return points[i+far+1:len(points) - (far-i)]
    return points[(i + far) - (len(points) - 1): i-far]

def get_near_points(points, i, near):
    if i  - near >= 0 and i + near < len(points) - 1:
        return points[i - near: i + near]
    if i - near < 0:
        return np.concatenate([points[i-near:], points[:i + near]])
    return np.concatenate([points[i - near:], points[:i + near - (len(points) - 1)]])

def elastic_fn(point, neighbours):
    return np.mean(neighbours - point, axis=0)
    
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
            elastic_fn(points[i], get_near_points(points, i, 3))
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
        if l2(point - q) > 5 * min_d:
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

def points_to_image(points, file_name):
    im = np.zeros((100, 100, 3))
    ma = np.max(points)
    mi = np.min(points)
    miz = np.min(points[:, 2])
    maz = np.max(points[:, 2])
    for p in points:
        im[
            int((99 / (ma - mi)) * (p[0] - mi)),
            int((99 / (ma - mi)) * (p[1] - mi)),
            :
        ] = np.array([
            255,
            int((255 / (maz - miz)) * (p[2] - miz)),
            255,
        ])
    Image.fromarray(im.astype("uint8")).save(file_name)

iters = 300

for i in range(iters):
    print(i, len(points))
    f = forces(points)
    #print(f)
    points = cull_and_generate(points + f)

    if i % 5 == 0:
        points_to_image(points, f"iterations/{i}.png")

dampiters = 10
points = double_points(points)

for i in range(dampiters):
    f = forces(points, justdamp=True)
    points = cull_and_generate(points + f)
    if i % 5 == 0:
        points_to_image(points, f"iterations/{iters}_{i}.png")

points = points * 10
mi = np.min(points)
ma = np.max(points)

save_sdl(tube(points, k=20, r=(ma-mi)/15), name=myname)
