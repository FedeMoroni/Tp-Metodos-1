import numpy as np

with open ("sensores.csv", "r") as archivo:
    archivo.readline()
    contenido = archivo.readlines()
    sensores = {}
    for linea in contenido:
        linea = linea.strip().split(",")
        sensores[linea[0]] = (float(linea[1]), float(linea[2]))

mediciones = []

with open("mediciones_trayectoria.csv", "r") as archivo:
    archivo.readline()
    for linea in archivo:
        linea = linea.strip().split(",")
        t = int(linea[0])
        d1 = float(linea[1])
        d2 = float(linea[2])
        d3 = float(linea[3])

        mediciones.append((t, d1, d2, d3))

x1, y1 = sensores["Sensor 1"]
x2, y2 = sensores["Sensor 2"]
x3, y3 = sensores["Sensor 3"]

def generar_semillas(d1, d2, d3, x_ant, y_ant):
    semillas = []
    semillas.append((x_ant, y_ant, "Punto anterior")) # pruebo desde el punto anterior

    # veo cual sensor esta mas cerca
    if d1 <= d2 and d1 <= d3:
        xCercano, yCercano = x1, y1
        r = d1
    elif d2 <= d1 and d2 <= d3:
        xCercano, yCercano = x2, y2
        r = d2
    else:
        xCercano, yCercano = x3, y3
        r = d3

    # agarro puntos sobre ese circulo
    for i in range(15):
        ang = 2 * np.pi * i / 15 # divido a 2pi en 15 partes
        x0 = xCercano + r * np.cos(ang)
        y0 = yCercano + r * np.sin(ang)
        semillas.append((x0, y0, "Punto en circulo mas cercano"))

    # algunos random cerca del anterior por si acaso
    for _ in range(5):
        x0 = x_ant + np.random.uniform(-1, 1)
        y0 = y_ant + np.random.uniform(-1, 1)
        semillas.append((x0, y0, "Punto random"))

    return semillas

def F(x, y, d1, d2):
    return np.array([
        (x - x1)**2 + (y - y1)**2 - d1**2,
        (x - x2)**2 + (y - y2)**2 - d2**2
    ])

def F3(x, y, d3):
    return (x - x3)**2 + (y - y3)**2 - d3**2

def J(x, y):
    return np.array([
        [2 * (x - x1), 2 * (y - y1)],
        [2 * (x - x2), 2 * (y - y2)]
    ])

def definir_x_e_y(d1, d2, d3, x0, y0, max_iter=100):
    p = np.array([x0, y0], dtype=float)
    for _ in range(max_iter):
        x, y = p[0], p[1]
        f = F(x, y, d1, d2)
        jacobiano = J(x, y)
        jacobiano_inv = np.linalg.inv(jacobiano)
        p = p - jacobiano_inv @ f
        v = F(p[0], p[1], d1, d2)
        if np.linalg.norm(v) < 1e-6 and np.abs(F3(p[0], p[1], d3)) < 1e-6:
            return p, _
    return None

trayectoria = []
x, y = 0.0, 0.0

for t, d1, d2, d3 in mediciones:
    semillas = generar_semillas(d1, d2, d3, x, y)
    solucion = None
    semilla = ""

    for x0, y0, descripcion in semillas:
        res = definir_x_e_y(d1, d2, d3, x0, y0)

        if res is not None:
            solucion, iteraciones = res
            semilla = descripcion
            break

    if solucion is None:
        raise ValueError("No converge con ninguna semilla")

    x, y = solucion
    trayectoria.append((t, x, y, semilla, iteraciones))

for punto in trayectoria:
    print(f"t = {punto[0]}, x = {punto[1]:.4f}, y = {punto[2]:.4f}, semilla = {punto[3]}, iteraciones = {punto[4]}")