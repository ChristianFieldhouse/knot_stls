
from PIL import Image
import numpy as np
import itertools

from write_tube import tube, save_sdl
from from_sketch import knot_path
from webscraped_knot_data import leq_sevens, eights, nines, tens
from legacy_knot_specifications import legacy_knots0

knot_counts = [1, 0, 0, 1, 1, 2, 3, 7]
def name_from_index(i):
    c = i + 1
    for crossing in range(8):
        for index in range(1, knot_counts[crossing] + 1):
            c -= 1
            if c == 0:
                return f"{crossing}_{index}"
            

# initial specification of knots
knots0 = { # from Arc Presentation on katlas.org (todo: this [:-4] is because I constructed them wrong)
    **{
        name_from_index(i): 2 * np.array(leq_seven)[:-4] for i, leq_seven in enumerate(leq_sevens)
    }, **{
        f"8_{i}": 2 * np.array(eight)[:-4] for i, eight in enumerate(eights)
    }, **{
        f"9_{i}": 2 * np.array(nine)[:-4] for i, nine in enumerate(nines)
    }, **{
        f"10_{i}": 2 * np.array(ten)[:-4] for i, ten in enumerate(tens)
    },
    **legacy_knots0
}

rad = 1
points_in_circle = 20
min_d = 2 * np.pi / points_in_circle
far = points_in_circle // 3
subsample = points_in_circle  / (2 * np.pi)

def l2(diff):
    return np.sqrt(np.sum(diff**2))

def cull_and_generate(points):
    new_points = [points[0]]
    for point in points:
        q = new_points[-1]
        if l2(point - q) > 2 * min_d:
            new_points.append(((point[0] + q[0])/2, (point[1] + q[1])/2, (point[2] + q[2])/2))
        if l2(point - q) > min_d:
            new_points.append(point)
    return np.array(new_points)

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
    n = diff/norm
    def step_force(d):
        if d < rad / 4:
            print("link broken ?")
            return 0
        if d > 2.5 * rad:
            return 0
        return rad * (1/d - 1/(2.5 * rad))
    return rad * n * step_force(l2(diff))

def forces(points, elasticity=0.5, repulsion=0.1, far=far, close = 0.1/subsample, justdamp=False):
    return np.array([
        elasticity * (
            elastic_fn(points[i], get_near_points(points, i, points_in_circle // 4))
        ) + (
            np.array([0, 0, 0]) if justdamp else (
                repulsion * sum(
                    repulse_fn(points[i] - q, rad=rad)
                    for q in get_far_points(points, i, far)
                    if (l2(points[i] - q) < 3 * rad)
                )
            )
        )
        for i in range(len(points))
    ])

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
    if maz == miz:
        maz = 1
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

def get_path(knot0, iters=500, dampiters=1, upscale=10):

    points0 = np.concatenate(
        [[
            t1 * a + t0 * (1-a) for a in np.arange(0, 1, 1 / (0.001 + l2(t1-t0) * subsample))
        ] for t0, t1 in zip(knot0, np.roll(knot0, -1, axis=0))],
        axis=0,
    )

    points = cull_and_generate(points0)

    print_step = 1
    for i in range(iters):
        if i % print_step == 0:
            points_to_image(points, f"iterations/{i}.png")
        print(i, len(points))
        f = forces(points)
        #print(f)
        points = cull_and_generate(points + f)

    damped_points = double_points(points)

    for i in range(dampiters):
        f = forces(damped_points, justdamp=True)
        points = cull_and_generate(damped_points + f)
        if i % 5 == 0:
            points_to_image(damped_points, f"iterations/{iters}_{i}.png")

    damped_points = double_points(damped_points * upscale)

    save_points = [damped_points[0]]

    for p in damped_points:
        if np.any(p != save_points[-1]):
            save_points.append(p)
    
    return save_points

eights_paths = []
for i in range(len(leq_sevens)):
    myname = name_from_index(i)
    upscale=10
    path = get_path(knots0[myname], upscale=upscale)
    eights_paths.append(path)
    save_sdl(tube(path, k=20, r=rad * upscale / 2), name=myname)
