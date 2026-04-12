import numpy as np

def ubicacion_dron(cast=np.float64, reescalado=1):
    REAL = cast   # o np.float64, o np.float16 si querés probar
    L = REAL(reescalado)   # si querés sin reescalado: L = REAL(1.0)

    def F(x, y, d1, d2): # f1 y f2
        return np.array([
            (x - x1)**2 + (y - y1)**2 - d1**2,
            (x - x2)**2 + (y - y2)**2 - d2**2
        ], dtype=REAL)

    def F3(x, y, d3):
        return REAL((x - x3)**2 + (y - y3)**2 - d3**2)

    def J(x, y): # jacobiano de F (f1 y f2)
        return np.array([
            [REAL(2) * (x - x1), REAL(2) * (y - y1)],
            [REAL(2) * (x - x2), REAL(2) * (y - y2)]
        ], dtype=REAL)

    def generar_semillas(d1, d2, d3, x_ant, y_ant):
        semillas = []
        semillas.append((REAL(x_ant), REAL(y_ant), "Punto anterior")) # pruebo desde el punto anterior

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

        for i in range(15): # agarro puntos sobre ese circulo, a ver si alguno converge
            ang = REAL(2 * np.pi * i / 15) # divido el circulo en 15 partes
            x0 = REAL(xCercano + r * REAL(np.cos(ang)))
            y0 = REAL(yCercano + r * REAL(np.sin(ang)))
            semillas.append((x0, y0, "Punto en circulo mas cercano"))

        # algunos random cerca del anterior por si acaso
        for _ in range(5):
            x0 = REAL(x_ant + np.random.uniform(-1, 1))
            y0 = REAL(y_ant + np.random.uniform(-1, 1))
            semillas.append((x0, y0, "Punto random"))

        return semillas

    def definir_x_e_y(d1, d2, d3, x0, y0, max_iter=100):
        p = np.array([x0, y0], dtype=REAL)

        for i in range(max_iter):
            x, y = p[0], p[1]
            f = F(x, y, d1, d2)
            jacobiano = J(x, y)
            jacobiano_inv = np.linalg.inv(jacobiano)
            p = p - jacobiano_inv @ f
            v = F(p[0], p[1], d1, d2)

            if np.linalg.norm(v) < 1e-6 and np.abs(F3(p[0], p[1], d3)) < 1e-6:
                return p, i
            
        return None

    def resumir(trayectoria):
        convergentes = [p for p in trayectoria if p[1] is not None]
        no_convergentes = [p for p in trayectoria if p[1] is None]

        cantidad_convergen = len(convergentes)
        cantidad_no_convergen = len(no_convergentes)

        if cantidad_convergen > 0:
            promedio_iter = sum(p[4] for p in convergentes) / cantidad_convergen
        else:
            promedio_iter = None

        print(f"Convergen: {cantidad_convergen}")
        print(f"No convergen: {cantidad_no_convergen}")
        if promedio_iter is not None:
            print(f"Promedio de iteraciones: {promedio_iter:.2f}")

    sensores = {}

    with open ("sensores.csv", "r") as archivo:
        archivo.readline()
        contenido = archivo.readlines()

        for linea in contenido:
            linea = linea.strip().split(",")
            sensores[linea[0]] = (REAL(linea[1]) / L, REAL(linea[2]) / L)

    mediciones = []

    with open("mediciones_trayectoria.csv", "r") as archivo:
        archivo.readline()

        for linea in archivo:
            linea = linea.strip().split(",")
            t = int(linea[0])
            d1 = REAL(linea[1]) / L
            d2 = REAL(linea[2]) / L
            d3 = REAL(linea[3]) / L
            mediciones.append((t, d1, d2, d3))

    x1, y1 = sensores["Sensor 1"]
    x2, y2 = sensores["Sensor 2"]
    x3, y3 = sensores["Sensor 3"]

    trayectoria = []
    x, y = REAL(0.0), REAL(0.0)

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
            trayectoria.append((t, None, None, "No converge", None))
            continue

        x, y = solucion
        trayectoria.append((t, x * L, y * L, semilla, iteraciones))

    for punto in trayectoria:
        if punto[1] is None:
            print(f"t = {punto[0]}: no converge")
        else:
            print(f"t = {punto[0]}, x = {punto[1]:.4f}, y = {punto[2]:.4f}, semilla = {punto[3]}, iteraciones = {punto[4]}")
    print("\nResumen de resultados:")
    resumir(trayectoria)

print("Resultados con np.float64 y reescalado 1:\n")
ubicacion_dron(np.float64, 1)

print("\nResultados con np.float64 y reescalado 100:\n")
ubicacion_dron(np.float64, 100)

print("\nResultados con np.float32 y reescalado 1:\n")
ubicacion_dron(np.float32, 1)

print("\nResultados con np.float32 y reescalado 100:\n")
ubicacion_dron(np.float32, 100)

print("\nResultados con np.float32 y reescalado 1000:\n")
ubicacion_dron(np.float32, 1000)

print("\nResultados con np.float32 y reescalado 10000:\n")
ubicacion_dron(np.float32, 10000)