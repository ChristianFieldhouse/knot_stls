from PIL import Image
import numpy as np
import itertools

def knot_path(file_name, make_ims=False):
    im = np.array(Image.open(file_name))

    height, width = im.shape[:2]

    start = np.argmax(im[:, :, 3]) # 2d coord of a point on the knot
    y0, x0 = start//im.shape[1], start % im.shape[1]
    #print((y0, x0))
    points = [(y0, x0)]

    r_max = 4 # radius to search for next point

    def search_list(y, x, d, r=r_max):
        d = np.array(d)
        d = d/np.sum(d**2)
        return [
            (j, i) for j, i in itertools.product(
                range(max(y - r, 0), min(y + r, height)),
                range(max(x - r, 0), min(x + r, width)),
            ) if ((j - y) * d[0] + (i - x)*d[1] > 0.6)
        ]

    d = None # stores direction we're going/searching in

    for d0 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        for j, i in search_list(y0, x0, d0):
            if im[j, i, 3] > 0:
                d = d0
                points.append((j, i))
                break
        if d is not None:
            break

    is_on_top = True #stores whether on top / black

    while len(points) < 20 or ((points[-1][0] - y0)**2 + (points[-1][1] - x0)**2 > r_max):
        #print(points)
        y, x = points[-1]
        
        found_point = False
        if is_on_top:
            radius = 0
            while not found_point and radius < r_max: # search for same colour
                radius += 1
                for j, i in search_list(y, x, d, radius):
                    #print(im[j, i, 3], im[j, i, 0], im[j, i, 3] > 40 and im[j, i , 0] == 0)
                    if im[j, i, 3] > 40 and im[j, i , 0] == 0:
                        d = (j - y, i - x)
                        points.append((j, i))
                        found_point = True
                        break
            radius = 0
            while not found_point and radius < r_max: # search for other colour
                radius += 1
                for j, i in search_list(y, x, d, radius):
                    #print("extra : ", im[j, i, 3], im[j, i, 0])
                    if im[j, i, 3] > 40:
                        d = (j - y, i - x)
                        points.append((j, i))
                        found_point = True
                        is_on_top = False
                        break
        else:
            radius = 0
            while not found_point and radius < r_max:
                radius += 1
                for j, i in search_list(y, x, d, radius):
                    if im[j, i, 3] > 40 and im[j, i, 0] > 40:
                        d = (j - y, i - x)
                        points.append((j, i))
                        found_point = True
                        break
            radius = 0
            while not found_point and radius < r_max:
                radius += 1
                for j, i in search_list(y, x, d, radius):
                    if im[j, i, 3] > 40:
                        d = (j - y, i - x)
                        points.append((j, i))
                        found_point = True
                        is_on_top = True
                        break
        
        if not found_point:
            print(points, d, is_on_top)
            print(search_list(y, x, d))
            print([im[q[0], q[1], :] for q in search_list(y, x, d)])
            raise Exception("dead end")
        
        if make_ims:
            out = np.zeros((100, 100, 3))
            for q in search_list(y, x, d):
                out[q[0], q[1], (1 if is_on_top else 0)] = 255
            for p in points:
                out[p[0], p[1], :] = 255

            Image.fromarray(out.astype("uint8")).save(f"iterations/read_{len(points)}.png")

        if len(points) > 1000:
            break

    scale = 3/max(width, height)
    return np.array([(p[1] * scale, p[0] * scale, im[p[0], p[1], 0] == 0) for p in points])

if __name__ == "__main__":
    points_3d = knot_path("knot_sketches/7_5.png", make_ims=True)
