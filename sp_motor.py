"""Laboratorio SP - motor de calculo (sin interfaz).

Modelo directo de potencial espontaneo para cuerpos de geometria regular
(Yungul, 1950; Biswas, 2017), sumado para varios cuerpos por superposicion:

    V(x) = sum_i k_i [(x - x0_i) cos(th_i) + h_i sin(th_i)] / [(x - x0_i)^2 + h_i^2]^q_i

    q = 1.5 esfera (h al centro) | 1.0 cilindro horizontal (h al eje)
    q = 0.5 cilindro vertical (h al techo)

Un cuerpo es un dict con claves q, x0, h, th (grados) y k.
"""
import itertools

import numpy as np

try:
    from scipy.optimize import least_squares
    HAY_SCIPY = True
except ImportError:          # el modelo directo funciona igual sin scipy
    HAY_SCIPY = False

# k cambia de unidades con q (mV*m^(2q-1)), por eso cada forma tiene su rango
FORMAS = {
    0.5: dict(nombre="Cilindro vertical", corto="Cil. vertical", kmax=400, kini=-80,
              ref="techo"),
    1.0: dict(nombre="Cilindro horizontal", corto="Cil. horizontal", kmax=8000, kini=-1200,
              ref="eje"),
    1.5: dict(nombre="Esfera", corto="Esfera", kmax=120000, kini=-15000, ref="centro"),
}
LIMITES = dict(x0=(0.0, 300.0), h=(1.0, 55.0), th=(-90.0, 90.0))
PERFIL = (0.0, 300.0)


# ======================================================================
# Modelo directo
# ======================================================================
def sp_cuerpo(b, x):
    """Anomalia SP (mV) de un cuerpo en los puntos x."""
    th = np.radians(b["th"])
    num = (x - b["x0"]) * np.cos(th) + b["h"] * np.sin(th)
    den = ((x - b["x0"]) ** 2 + b["h"] ** 2) ** b["q"]
    return b["k"] * num / den


def sp_total(cuerpos, x):
    """Suma de las anomalias de todos los cuerpos (superposicion)."""
    V = np.zeros_like(np.asarray(x, dtype=float))
    for b in cuerpos:
        V = V + sp_cuerpo(b, x)
    return V


def estaciones(dx):
    return np.arange(PERFIL[0], PERFIL[1] + 1e-9, float(dx))


# ======================================================================
# Datos sinteticos ("observados")
# ======================================================================
def cuerpo_aleatorio(rng, q, x0):
    """Cuerpo con profundidad y angulo al azar y amplitud de pico entre 55 y 140 mV."""
    lo, hi = {0.5: (4, 18), 1.0: (6, 22), 1.5: (6, 16)}[q]
    b = dict(q=float(q), x0=float(x0), h=round(float(rng.uniform(lo, hi)), 1),
             th=float(round(rng.uniform(-30, 90))), k=1.0)
    m = np.abs(sp_cuerpo(b, np.linspace(*PERFIL, 601))).max()
    objetivo = rng.uniform(55, 140)
    km = FORMAS[q]["kmax"]
    b["k"] = float("%.3g" % np.clip(-objetivo / m, -km, km))
    return b


def posiciones_separadas(rng, n, margen=30.0, separacion=45.0):
    """n posiciones x0 dentro del perfil, separadas al menos 'separacion' m."""
    for _ in range(2000):
        x = np.sort(rng.uniform(PERFIL[0] + margen, PERFIL[1] - margen, n))
        if n == 1 or np.min(np.diff(x)) >= separacion:
            return [float(round(v)) for v in x]
    return [float(round(v)) for v in np.linspace(PERFIL[0] + margen, PERFIL[1] - margen, n)]


def nuevo_reto(n_cuerpos=1, semilla=None, formas=None):
    """Lista de cuerpos 'verdaderos'. formas: lista de q (o None = al azar)."""
    rng = np.random.default_rng(semilla)
    xs = posiciones_separadas(rng, n_cuerpos)
    qs = formas or [float(rng.choice([0.5, 1.0, 1.5])) for _ in range(n_cuerpos)]
    return [cuerpo_aleatorio(rng, q, x) for q, x in zip(qs, xs)]


def observar(verdad, x, sigma, semilla):
    """Datos = modelo verdadero + ruido gaussiano de desviacion sigma (mV)."""
    rng = np.random.default_rng(semilla)
    return sp_total(verdad, x) + sigma * rng.standard_normal(len(x))


# ======================================================================
# Metricas
# ======================================================================
def n_parametros(cuerpos, k_solo=False):
    return len(cuerpos) * (1 if k_solo else 4)


def metricas(obs, mod, sigma, n_par):
    """Metricas de ajuste. chi2_red usa N - p grados de libertad (p = parametros ajustados)."""
    r = obs - mod
    n = len(obs)
    ssr, sst = float(np.sum(r ** 2)), float(np.sum((obs - obs.mean()) ** 2))
    rms = np.sqrt(ssr / n)
    rango = float(obs.max() - obs.min())
    gl = max(n - n_par, 1)
    return dict(r=r, rms=rms, mae=float(np.mean(np.abs(r))), maxr=float(np.max(np.abs(r))),
                nrms=100 * rms / rango if rango > 0 else np.nan,
                r2=1 - ssr / sst if sst > 0 else np.nan,
                chi2_red=ssr / gl / sigma ** 2 if sigma > 0 else np.nan,
                gl=gl, n=n, p=n_par)


# ======================================================================
# Ajuste lineal de k
# ======================================================================
def optimizar_k(cuerpos, x, obs):
    """Con la geometria fija V es lineal en k: el mejor k sale por minimos cuadrados."""
    G = np.column_stack([sp_cuerpo({**b, "k": 1.0}, x) for b in cuerpos])
    k, _, rango, _ = np.linalg.lstsq(G, obs, rcond=None)
    if rango < len(cuerpos):
        return False, "No se pudo: hay cuerpos casi iguales. Sepáralos y prueba de nuevo."
    tope = False
    for b, kk in zip(cuerpos, k):
        km = FORMAS[b["q"]]["kmax"]
        tope |= abs(kk) > km
        b["k"] = float("%.4g" % np.clip(kk, -km, km))
    return True, ("k ajustado, pero alguno llegó al límite: revisa la geometría." if tope
                  else "k ajustado por mínimos cuadrados.")


# ======================================================================
# Inversion no lineal con varios puntos de partida e incertidumbre
# ======================================================================
def _empaquetar(cuerpos):
    p0, lo, hi = [], [], []
    for b in cuerpos:
        km = FORMAS[b["q"]]["kmax"]
        p0 += [b["x0"], b["h"], b["th"], b["k"]]
        lo += [LIMITES["x0"][0], LIMITES["h"][0], LIMITES["th"][0], -km]
        hi += [LIMITES["x0"][1], LIMITES["h"][1], LIMITES["th"][1], km]
    lo, hi = np.array(lo), np.array(hi)
    return np.clip(np.array(p0, float), lo + 1e-9, hi - 1e-9), lo, hi


def _desempaquetar(cuerpos, p):
    return [dict(q=b["q"], x0=p[4 * i], h=p[4 * i + 1], th=p[4 * i + 2], k=p[4 * i + 3])
            for i, b in enumerate(cuerpos)]


def invertir(cuerpos, x, obs, sigma, n_inicios=8, semilla=0, dispersion=None):
    """Ajusta x0, h, th y k de todos los cuerpos (q fija) por minimos cuadrados no lineales.

    Arranca desde el modelo actual y desde n_inicios-1 perturbaciones de el; se queda con
    la de menor error. Devuelve un dict con la solucion, su incertidumbre (1 sigma, a
    partir del jacobiano) y cuantos arranques llegaron a la misma solucion.
    """
    if not HAY_SCIPY:
        raise RuntimeError("Falta scipy: instálalo con  %pip install scipy")
    p0, lo, hi = _empaquetar(cuerpos)
    rng = np.random.default_rng(semilla)
    disp = dispersion or dict(x0=25.0, h=8.0, th=40.0)

    def residuos(p):
        return sp_total(_desempaquetar(cuerpos, p), x) - obs

    arranques = [p0]
    for _ in range(max(n_inicios - 1, 0)):
        p = p0.copy()
        for i in range(len(cuerpos)):
            p[4 * i] += rng.normal(0, disp["x0"])
            p[4 * i + 1] *= np.exp(rng.normal(0, 0.5))
            p[4 * i + 2] += rng.normal(0, disp["th"])
        arranques.append(np.clip(p, lo + 1e-9, hi - 1e-9))

    soluciones = []
    for p in arranques:
        s = least_squares(residuos, p, bounds=(lo, hi), x_scale="jac")
        soluciones.append(s)
    costos = np.array([2 * s.cost for s in soluciones])            # = suma de residuos^2
    mejor = soluciones[int(np.argmin(costos))]
    iguales = int(np.sum(costos <= costos.min() * 1.02 + 1e-9))

    # incertidumbre: cov = s^2 (J^T J)^-1, con s = sigma si se conoce el ruido
    J = mejor.jac
    n, p_ = len(obs), len(mejor.x)
    s2 = sigma ** 2 if sigma > 0 else 2 * mejor.cost / max(n - p_, 1)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(err, err)          # nan si algun error es 0
    except np.linalg.LinAlgError:
        err = np.full(p_, np.nan)
        corr = np.full((p_, p_), np.nan)
    en_limite = [bool(np.isclose(v, lo[i], rtol=0, atol=1e-6 * max(1, abs(lo[i])))
                      or np.isclose(v, hi[i], rtol=0, atol=1e-6 * max(1, abs(hi[i]))))
                 for i, v in enumerate(mejor.x)]
    return dict(cuerpos=_desempaquetar(cuerpos, mejor.x), err=err, corr=corr,
                en_limite=en_limite, rms_ini=float(np.sqrt(np.mean(residuos(p0) ** 2))),
                rms=float(np.sqrt(np.mean(mejor.fun ** 2))), iguales=iguales,
                n_inicios=len(arranques), nfev=int(sum(s.nfev for s in soluciones)))


def nombres_parametros(n_cuerpos):
    return ["%s%d" % (p, i + 1) for i in range(n_cuerpos) for p in ("x0", "h", "θ", "k")]


def correlaciones_fuertes(corr, n_cuerpos, umbral=0.9):
    """Pares de parametros con |correlacion| > umbral (ambiguedad)."""
    nom = nombres_parametros(n_cuerpos)
    pares = []
    for i in range(len(nom)):
        for j in range(i + 1, len(nom)):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) > umbral:
                pares.append((nom[i], nom[j], float(corr[i, j])))
    return pares


def probar_formas(cuerpos, x, obs, sigma, n_inicios=6, max_cuerpos=3):
    """Invierte con todas las combinaciones de forma (q) y compara el ajuste.

    Devuelve una lista ordenada de (formas, chi2_red, rms, cuerpos_invertidos).
    """
    if len(cuerpos) > max_cuerpos:
        raise ValueError("Probar formas admite hasta %d cuerpos (%d combinaciones)."
                         % (max_cuerpos, 3 ** max_cuerpos))
    resultados = []
    for combo in itertools.product([0.5, 1.0, 1.5], repeat=len(cuerpos)):
        prueba = []
        for b, q in zip(cuerpos, combo):
            nb = dict(b)
            if q != b["q"]:          # conserva la amplitud en x0 al cambiar de forma
                nb["k"] = float(np.clip(b["k"] * b["h"] ** (2 * (q - b["q"])),
                                        -FORMAS[q]["kmax"], FORMAS[q]["kmax"]))
            nb["q"] = q
            prueba.append(nb)
        optimizar_k(prueba, x, obs)
        r = invertir(prueba, x, obs, sigma, n_inicios=n_inicios)
        M = metricas(obs, sp_total(r["cuerpos"], x), sigma, n_parametros(prueba))
        resultados.append((combo, M["chi2_red"], M["rms"], r["cuerpos"]))
    resultados.sort(key=lambda t: t[2])
    return resultados


def margen_chi2(gl):
    """Dos desviaciones estandar del chi2 reducido por azar: 2 sqrt(2 / gl).

    Dos modelos cuyo chi2_red difiere menos que esto ajustan igual de bien:
    los datos no permiten preferir uno sobre el otro.
    """
    return 2 * np.sqrt(2.0 / max(gl, 1))


# ======================================================================
# Propiedades analiticas (para verificar el modelo)
# ======================================================================
def amplitud_en_x0(b):
    """V(x0) = k sin(th) h^(1 - 2q)."""
    return b["k"] * np.sin(np.radians(b["th"])) * b["h"] ** (1 - 2 * b["q"])


def ancho_media_altura(q, h):
    """Ancho a media altura de la anomalia con th = 90: 2 h sqrt(2^(1/q) - 1)."""
    return 2 * h * np.sqrt(2 ** (1 / q) - 1)
