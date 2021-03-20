
import numpy as np

def triangle_ascii(triangle):
    v1, v2, v3 = triangle
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / sum(normal**2)
    return (
        f"facet normal {normal[0]} {normal[1]} {normal[2]} \n\touter loop\n" +
        f"\t\tvertex {v1[0]} {v1[1]} {v1[2]}\n" +
        f"\t\tvertex {v2[0]} {v2[1]} {v2[2]}\n" +
        f"\t\tvertex {v3[0]} {v3[1]} {v3[2]}\n" +
        f"\tendloop\n" + f"endfacet\n"
    )

def vertex_ring(p, n, r, k=10):
    v3 = (0, 0, 1) # gives stable sequence of circles except when vertical (where we have to rotate them)
    b1 = np.cross(v3, n)
    b1 = b1 / np.sqrt(np.sum(b1**2))
    b2 = np.cross(b1, n)
    b2 = b2 / np.sqrt(np.sum(b2**2))
    return [
        p + r * (b1 * np.sin(theta) + b2 * np.cos(theta))
        for theta in np.arange(0, 2*np.pi, 2*np.pi/k)
    ]


def tube(path, r=1, k=10):
    vertex_rings = [
        vertex_ring(path[i], path[i+1] - path[i-1], r, k)
        for i in range(len(path) - 1)
    ] + [
        vertex_ring(path[-1], path[0] - path[-2], r, k)    
    ]
    
    triangles = []
    vertex_rings = vertex_rings + [vertex_rings[0]]
    for i in range(len(path)):
        r1, r2 = vertex_rings[i], vertex_rings[i+1]
        rotate = np.argmin(np.sum((r2 - r1[0])**2, axis=1))
        r2 = np.roll(r2, -rotate, axis=0)
        r1, r2 = r1 + [r1[0]], list(r2) + [r2[0]]
        for j in range(k):
            triangles.append((r1[j], r1[j+1], r2[j]))
            triangles.append((r1[j + 1], r2[j+1], r2[j]))
    return triangles

def circle_path(r, k):
    x = np.array((1, 0, 0))
    y = np.array((0, 1, 1))
    return [
        r * (x * np.sin(theta) + y * np.cos(theta))
        for theta in np.arange(0, 2*np.pi, 2*np.pi/k)
    ]

def save_sdl(triangles, name="new"):
    with open(name + ".stl", "w") as f:
        f.write("solid " + name + "\n")
        for triangle in triangles:
            f.write(triangle_ascii(triangle))
        f.write("endsolid")

if __name__ == "__main__":
    save_sdl(tube(circle_path(5, 10), k=20))
