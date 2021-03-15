
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
	v3 = (0, 0, 1) # fine as long as tube isn't vertical todo: make this smart #(np.pi, np.exp(1), np.sqrt(2))
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
		vertex_ring(path[i], path[i+1] - path[i], r, k)
		for i in range(len(path) - 1)
	] + [
		vertex_ring(path[-1], path[0] - path[-1], r, k)	
	]
	
	triangles = []
	for i in range(len(path) - 1):
		for j in range(k - 1):
			triangles.append((
				vertex_rings[i][j],
				vertex_rings[i][j+1],
				vertex_rings[i+1][j],	
			))
			triangles.append((
				vertex_rings[i][j+1],
				vertex_rings[i+1][j+1],
				vertex_rings[i+1][j],	
			))
		triangles.append((
			vertex_rings[i][-1],
			vertex_rings[i][0],
			vertex_rings[i+1][-1],
		))
		triangles.append((
			vertex_rings[i][0],
			vertex_rings[i+1][0],
			vertex_rings[i+1][-1],	
		))
	for j in range(k - 1):
		triangles.append((
			vertex_rings[-1][j],
			vertex_rings[-1][j+1],
			vertex_rings[0][j],	
		))
		triangles.append((
			vertex_rings[-1][j+1],
			vertex_rings[0][j+1],
			vertex_rings[0][j],	
		))
	triangles.append((
		vertex_rings[-1][-1],
		vertex_rings[-1][0],
		vertex_rings[0][-1],
	))
	triangles.append((
		vertex_rings[-1][0],
		vertex_rings[0][0],
		vertex_rings[0][-1],	
	))
	return triangles

def circle_path(r, k):
	x = np.array((1, 0, 0))
	y = np.array((0, 1, 0))
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
    save_sdl(tube(circle_path(5, 50), k=20))
