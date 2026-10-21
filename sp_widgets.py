"""Laboratorio SP - interfaz interactiva (ipywidgets) para Jupyter / Binder.

Uso en un notebook:
    %matplotlib inline
    from sp_widgets import lanzar
    lab = lanzar()
"""
import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import clear_output, display

import sp_motor as sp

AZ, NJ, GR, RO, MO = "#0a6c73", "#c0561b", "#5a6a66", "#b3261e", "#6f3c99"
TIERRA, ESTRATO = "#e5dccb", "#cfc2a9"
MAX_CUERPOS = 6
EST = {"description_width": "130px"}
LY = w.Layout(width="430px")


class LaboratorioSP:
    def __init__(self):
        self.cuerpos, self.sel, self.nid = [], None, 1
        self.verdad, self.obs, self.propio = None, None, False
        self.sigma, self.dx, self.semilla_ruido = 3.0, 5.0, 1
        self.revelar, self.mejor = False, np.inf
        self.ultima_inv, self.formas_res = None, None
        self._sinc = False
        self._construir()
        self.nuevo_reto()

    # ================= estado =================
    @property
    def x(self):
        return sp.estaciones(self.dx)

    def nuevo_cuerpo(self, q, x0=150.0, h=12.0, th=45.0, k=None):
        b = dict(id=self.nid, q=float(q), x0=float(x0), h=float(h), th=float(th),
                 k=float(sp.FORMAS[q]["kini"] if k is None else k))
        self.nid += 1
        return b

    def generar_observados(self):
        self.obs = (None if self.verdad is None
                    else sp.observar(self.verdad, self.x, self.sigma, self.semilla_ruido))
        self.mejor, self.ultima_inv, self.formas_res = np.inf, None, None

    def nuevo_reto(self, semilla=None):
        n = self.w_nocultos.value
        self.verdad = sp.nuevo_reto(n, semilla=semilla)
        self.semilla_ruido = int(np.random.default_rng(semilla).integers(1e9))
        self.propio, self.revelar = False, False
        self.generar_observados()
        self.cuerpos = [self.nuevo_cuerpo(1.5)]
        self.sel = self.cuerpos[0]["id"]
        self.cargar_editor()
        self.aviso("Nuevo reto: %d cuerpo(s) oculto(s)." % n)
        self.redibujar()

    def metricas(self):
        if self.obs is None:
            return None
        return sp.metricas(self.obs, sp.sp_total(self.cuerpos, self.x), self.sigma,
                           sp.n_parametros(self.cuerpos))

    def cuerpo_sel(self):
        return next((b for b in self.cuerpos if b["id"] == self.sel), None)

    # ================= controles =================
    def _construir(self):
        # --- datos ---
        self.w_nocultos = w.Dropdown(options=[1, 2, 3, 4], value=1,
                                     description="Cuerpos ocultos", style=EST, layout=LY)
        self.w_pista = w.Checkbox(value=True, description="Mostrar las formas (pista)",
                                  indent=False)
        self.w_sigma = w.FloatSlider(value=3, min=0, max=15, step=0.5, description="Ruido σ (mV)",
                                     style=EST, layout=LY, continuous_update=False)
        self.w_dx = w.Dropdown(options=[1.0, 2.5, 5.0, 10.0, 20.0], value=5.0,
                               description="Estaciones cada (m)", style=EST, layout=LY)
        b_nuevo = w.Button(description="Nuevo reto", button_style="primary", icon="refresh")
        self.b_revela = w.Button(description="Revelar solución", icon="eye")
        b_propio = w.Button(description="Usar mi modelo como datos", icon="copy",
                            layout=w.Layout(width="auto"))
        b_quitar = w.Button(description="Quitar datos", icon="trash")
        self.h_pista, self.h_sol = w.HTML(), w.HTML()

        # --- cuerpos ---
        b_add = {q: w.Button(description="+ " + sp.FORMAS[q]["corto"],
                             layout=w.Layout(width="auto")) for q in (1.5, 1.0, 0.5)}
        self.w_sel = w.Dropdown(description="Cuerpo", style=EST, layout=w.Layout(width="300px"))
        self.b_del = w.Button(description="Eliminar", icon="minus", button_style="danger")
        self.w_q = w.ToggleButtons(options=[(sp.FORMAS[q]["corto"], q) for q in (1.5, 1.0, 0.5)],
                                   description="Forma", style=EST)
        mk = lambda d, a, b, s: w.FloatSlider(description=d, min=a, max=b, step=s, style=EST,
                                              layout=LY, continuous_update=False)
        self.w_k = mk("k", -120000, 120000, 100)
        self.w_k.readout_format = ".4g"
        self.w_h = mk("h  prof. (m)", 1, 55, 0.1)
        self.w_th = mk("θ  ángulo (°)", -90, 90, 1)
        self.w_x0 = mk("x₀  posición (m)", 0, 300, 0.5)
        self.b_optk = w.Button(description="Optimizar k", icon="magic", button_style="info")
        self.b_inv = w.Button(description="Inversión automática", icon="cogs",
                              button_style="warning", layout=w.Layout(width="auto"),
                              disabled=not sp.HAY_SCIPY)
        self.w_inicios = w.IntSlider(value=8, min=1, max=30, description="Puntos de partida",
                                     style=EST, layout=LY, continuous_update=False)
        self.b_formas = w.Button(description="Probar formas", icon="shapes",
                                 layout=w.Layout(width="auto"), disabled=not sp.HAY_SCIPY)
        self.h_msg = w.HTML()

        self.salida = w.Output()
        self.h_met, self.h_inv = w.HTML(), w.HTML()

        # --- eventos ---
        b_nuevo.on_click(lambda _: self.nuevo_reto())
        self.w_nocultos.observe(lambda _: self.nuevo_reto(), "value")
        self.w_pista.observe(lambda _: self.redibujar(), "value")
        self.w_sigma.observe(self._al_sigma, "value")
        self.w_dx.observe(self._al_dx, "value")
        self.b_revela.on_click(self._al_revelar)
        b_propio.on_click(self._al_propio)
        b_quitar.on_click(self._al_quitar)
        for q, bt in b_add.items():
            bt.on_click(lambda _, q=q: self.agregar(q))
        self.b_del.on_click(self._al_eliminar)
        for campo, wd in (("k", self.w_k), ("h", self.w_h), ("th", self.w_th), ("x0", self.w_x0)):
            wd.observe(lambda c, campo=campo: self._al_mover(c, campo), "value")
        self.w_q.observe(self._al_forma, "value")
        self.w_sel.observe(self._al_elegir, "value")
        self.b_optk.on_click(self._al_optk)
        self.b_inv.on_click(self._al_invertir)
        self.b_formas.on_click(self._al_formas)

        tit = lambda t: w.HTML("<b style='font-size:13px;letter-spacing:.04em;color:%s'>%s</b>"
                               % (GR, t))
        panel_datos = w.VBox([tit("DATOS OBSERVADOS"), w.HBox([self.w_nocultos, b_nuevo]),
                              self.w_pista, self.w_sigma, self.w_dx, self.h_pista,
                              w.HBox([self.b_revela, b_propio, b_quitar]), self.h_sol])
        panel_cuerpos = w.VBox([tit("CUERPOS DEL MODELO"), w.HBox(list(b_add.values())),
                                w.HBox([self.w_sel, self.b_del]), self.w_q, self.w_k, self.w_h,
                                self.w_th, self.w_x0, self.w_inicios,
                                w.HBox([self.b_optk, self.b_inv, self.b_formas]), self.h_msg])
        panel_der = w.VBox([tit("MÉTRICAS DE AJUSTE"), self.h_met,
                            tit("RESULTADO DE LA INVERSIÓN"), self.h_inv],
                           layout=w.Layout(width="360px"))
        self.vista = w.VBox([w.HBox([panel_cuerpos, panel_datos]),
                             w.HBox([self.salida, panel_der])])

    # ================= editor =================
    def cargar_editor(self):
        self._sinc = True
        self.w_sel.options = [("%d · %s" % (i + 1, sp.FORMAS[b["q"]]["corto"]), b["id"])
                              for i, b in enumerate(self.cuerpos)]
        b = self.cuerpo_sel()
        hay = b is not None
        for wd in (self.w_sel, self.w_q, self.w_k, self.w_h, self.w_th, self.w_x0, self.b_del):
            wd.disabled = not hay
        if hay:
            km = sp.FORMAS[b["q"]]["kmax"]
            self.w_sel.value, self.w_q.value = b["id"], b["q"]
            self.w_k.min, self.w_k.max = -1e9, 1e9
            self.w_k.max, self.w_k.min, self.w_k.step = km, -km, km / 1000
            self.w_k.value, self.w_h.value = b["k"], b["h"]
            self.w_th.value, self.w_x0.value = b["th"], b["x0"]
            self.w_h.description = "h  al %s (m)" % sp.FORMAS[b["q"]]["ref"]
        self._sinc = False

    def aviso(self, t):
        self.h_msg.value = "<span style='color:%s'><b>%s</b></span>" % (AZ, t)

    def _al_mover(self, c, campo):
        if self._sinc:
            return
        b = self.cuerpo_sel()
        if b is not None:
            b[campo] = float(c["new"])
            self.ultima_inv = None
            self.redibujar()

    def _al_forma(self, c):
        if self._sinc:
            return
        b = self.cuerpo_sel()
        if b is None or c["new"] == b["q"]:
            return
        qn = c["new"]       # conserva la amplitud en x0
        b["k"] = float(np.clip(b["k"] * b["h"] ** (2 * (qn - b["q"])),
                               -sp.FORMAS[qn]["kmax"], sp.FORMAS[qn]["kmax"]))
        b["q"] = qn
        self.ultima_inv = None
        self.cargar_editor()
        self.aviso("Cambié la forma conservando la amplitud en x₀.")
        self.redibujar()

    def _al_elegir(self, c):
        if self._sinc or c["new"] is None:
            return
        self.sel = c["new"]
        self.cargar_editor()
        self.redibujar()

    def agregar(self, q):
        if len(self.cuerpos) >= MAX_CUERPOS:
            self.aviso("Máximo %d cuerpos." % MAX_CUERPOS)
            return
        usados = [b["x0"] for b in self.cuerpos]
        x = next((c for c in (150, 90, 210, 50, 250, 120, 180, 30, 270)
                  if all(abs(u - c) > 25 for u in usados)), 150)
        b = self.nuevo_cuerpo(q, x0=x)
        self.cuerpos.append(b)
        self.sel = b["id"]
        self.ultima_inv = None
        self.cargar_editor()
        self.aviso("Agregué un %s en x₀ = %d m." % (sp.FORMAS[q]["nombre"].lower(), x))
        self.redibujar()

    def _al_eliminar(self, _):
        self.cuerpos = [b for b in self.cuerpos if b["id"] != self.sel]
        self.sel = self.cuerpos[-1]["id"] if self.cuerpos else None
        self.ultima_inv = None
        self.cargar_editor()
        self.redibujar()

    # ================= datos =================
    def _al_sigma(self, c):
        self.sigma = float(c["new"])
        self.generar_observados()
        self.redibujar()

    def _al_dx(self, c):
        self.dx = float(c["new"])
        self.generar_observados()
        self.redibujar()

    def _al_revelar(self, _):
        self.revelar = not self.revelar
        self.redibujar()

    def _al_propio(self, _):
        if not self.cuerpos:
            self.aviso("Agrega al menos un cuerpo primero.")
            return
        self.verdad = [{k: b[k] for k in ("q", "x0", "h", "th", "k")} for b in self.cuerpos]
        self.propio, self.revelar = True, False
        self.semilla_ruido = int(np.random.default_rng().integers(1e9))
        self.generar_observados()
        self.aviso("Tu modelo ahora son los datos. Mueve algo y mira el error.")
        self.redibujar()

    def _al_quitar(self, _):
        self.verdad = self.obs = None
        self.revelar = self.propio = False
        self.ultima_inv = self.formas_res = None
        self.redibujar()

    # ================= ajuste =================
    def _actualizar_cuerpos(self, nuevos):
        for b, n in zip(self.cuerpos, nuevos):
            b.update({k: float(n[k]) for k in ("x0", "h", "th", "k")})

    def _al_optk(self, _):
        if self.obs is None or not self.cuerpos:
            return
        _, msg = sp.optimizar_k(self.cuerpos, self.x, self.obs)
        self.ultima_inv = None
        self.aviso(msg)
        self.cargar_editor()
        self.redibujar()

    def invertir(self):
        if self.obs is None or not self.cuerpos:
            self.aviso("No hay datos o no hay cuerpos.")
            return None
        self.aviso("Invirtiendo desde %d puntos de partida..." % self.w_inicios.value)
        r = sp.invertir(self.cuerpos, self.x, self.obs, self.sigma,
                        n_inicios=self.w_inicios.value)
        self._actualizar_cuerpos(r["cuerpos"])
        self.ultima_inv = r
        self.aviso("Inversión: RMS %.2f → %.2f mV (%d evaluaciones del modelo)."
                   % (r["rms_ini"], r["rms"], r["nfev"]))
        self.cargar_editor()
        self.redibujar()
        return r

    def _al_invertir(self, _):
        self.invertir()

    def probar_formas(self):
        if self.obs is None or not self.cuerpos:
            self.aviso("No hay datos o no hay cuerpos.")
            return None
        if len(self.cuerpos) > 3:
            self.aviso("Probar formas admite hasta 3 cuerpos (27 combinaciones).")
            return None
        self.aviso("Probando %d combinaciones de forma..." % 3 ** len(self.cuerpos))
        self.formas_res = sp.probar_formas(self.cuerpos, self.x, self.obs, self.sigma,
                                           n_inicios=4)
        mejor = self.formas_res[0]
        for b, q, n in zip(self.cuerpos, mejor[0], mejor[3]):
            b["q"] = float(q)
            b.update({k: float(n[k]) for k in ("x0", "h", "th", "k")})
        self.aviso("Apliqué la combinación de formas con menor error. Mira la tabla.")
        self.cargar_editor()
        self.invertir()             # para tener incertidumbres de la mejor combinacion
        return self.formas_res

    def _al_formas(self, _):
        self.probar_formas()

    # ================= paneles HTML =================
    def html_metricas(self, M):
        if M is None:
            return ("<div style='padding:10px;border-radius:6px;background:#eef1ee;color:%s'>"
                    "<b>Sin datos.</b> Crea un reto o usa tu modelo como datos.</div>" % GR)
        if self.sigma > 0:
            c = M["chi2_red"]
            est = (("Ajustado", "#257a3e", "#d9eedd", "El error ya es del tamaño del ruido.")
                   if c <= 1 + sp.margen_chi2(M["gl"]) else
                   ("Cerca", "#a86b00", "#f6e7c8", "Falta poco: afina h, θ o k.") if c <= 5 else
                   ("Lejos", RO, "#f6d7d4", "El modelo todavía no explica los datos."))
        else:
            c = M["nrms"]
            est = (("Ajustado", "#257a3e", "#d9eedd", "Sin ruido: ajuste casi exacto.") if c <= 1
                   else ("Cerca", "#a86b00", "#f6e7c8", "Falta poco.") if c <= 8 else
                   ("Lejos", RO, "#f6d7d4", "El modelo todavía no explica los datos."))
        self.mejor = min(self.mejor, M["rms"])
        f = lambda v, d: "—" if not np.isfinite(v) else "%.*f" % (d, v)
        fila = lambda n, v, e: ("<tr><td style='padding:3px 6px'><b>%s</b><br><span style='"
                                "color:%s;font-size:11px'>%s</span></td><td style='padding:3px 6px;"
                                "text-align:right;font-family:monospace;font-size:13px;"
                                "white-space:nowrap'>%s</td></tr>" % (n, GR, e, v))
        return ("<div style='padding:8px 10px;border-radius:6px;background:%s;color:%s;"
                "margin-bottom:6px'><b style='background:%s;color:white;padding:2px 8px;"
                "border-radius:10px;font-size:11px'>%s</b> <b>%s</b></div>"
                "<table style='width:100%%;border-collapse:collapse'>"
                % (est[2], est[1], est[1], est[0].upper(), est[3])
                + fila("χ² reducido", f(M["chi2_red"], 2),
                       "Σ residuo² / σ² / (N − p), con N = %d datos y p = %d parámetros. Meta ≈ 1 "
                       "(± %.2f por azar)." % (M["n"], M["p"], sp.margen_chi2(M["gl"])))
                + fila("RMS", f(M["rms"], 2) + " mV", "Error típico por estación (meta ≈ σ).")
                + fila("NRMS", f(M["nrms"], 1) + " %", "RMS relativo al rango de los datos.")
                + fila("R²", f(M["r2"], 3), "Fracción de la variación explicada.")
                + fila("MAE", f(M["mae"], 2) + " mV", "Error absoluto medio.")
                + fila("Máx |residuo|", f(M["maxr"], 2) + " mV", "El peor punto del perfil.")
                + "</table><div style='color:%s;font-size:12px;margin-top:6px'>Mejor RMS en "
                  "estos datos: <b>%s mV</b></div>" % (GR, f(self.mejor, 2)))

    def html_inversion(self):
        partes = []
        r = self.ultima_inv
        if r is not None:
            nom = ["x₀ (m)", "h (m)", "θ (°)", "k"]
            filas = ""
            for i, b in enumerate(r["cuerpos"]):
                celdas = []
                for j, campo in enumerate(("x0", "h", "th", "k")):
                    v, e = b[campo], r["err"][4 * i + j]
                    lim = " ⚠" if r["en_limite"][4 * i + j] else ""
                    celdas.append("%.4g ± %.2g%s" % (v, e, lim) if np.isfinite(e)
                                  else "%.4g%s" % (v, lim))
                filas += ("<tr><td><b>%d</b> %s</td>%s</tr>"
                          % (i + 1, sp.FORMAS[b["q"]]["corto"],
                             "".join("<td style='text-align:right'>%s</td>" % c for c in celdas)))
            partes.append("<table style='border-collapse:collapse;font-size:12px;width:100%%'>"
                          "<tr><th></th>%s</tr>%s</table>"
                          % ("".join("<th>%s</th>" % n for n in nom), filas))
            estable = r["iguales"] >= max(2, r["n_inicios"] // 2) or r["n_inicios"] == 1
            M = self.metricas()
            ajusta = (M is not None and self.sigma > 0
                      and M["chi2_red"] <= 1 + sp.margen_chi2(M["gl"]))
            if estable:
                extra = ""
            elif ajusta:
                extra = (" La mejor ajusta los datos, pero el problema tiene mínimos locales: "
                         "desde un arranque lejano es fácil no encontrarla. Parte de una "
                         "ubicación cercana a ojo.")
            else:
                extra = (" Ninguna ajusta bien todavía: revisa cuántos cuerpos hay y dónde, "
                         "o usa más puntos de partida.")
            partes.append("<div style='font-size:12px;color:%s'>%s %d de %d puntos de partida "
                          "llegaron a la mejor solución.%s</div>"
                          % ("#257a3e" if estable else "#a86b00" if ajusta else RO,
                             "✔" if estable else "⚠", r["iguales"], r["n_inicios"], extra))
            if any(r["en_limite"]):
                partes.append("<div style='font-size:12px;color:%s'>⚠ Algún parámetro quedó en "
                              "su límite: la solución puede no ser física.</div>" % RO)
            pares = sp.correlaciones_fuertes(r["corr"], len(r["cuerpos"]))
            if pares:
                partes.append("<div style='font-size:12px;color:#a86b00'>Parámetros acoplados "
                              "(|r| > 0.9), los datos no los separan bien: %s</div>"
                              % ", ".join("%s–%s (%.2f)" % p for p in pares[:6]))
            partes.append("<div style='font-size:11px;color:%s'>± = 1σ a partir del jacobiano "
                          "y del ruido σ.</div>" % GR)
        if self.formas_res:
            M = self.metricas()
            marg = sp.margen_chi2(M["gl"]) if M else 0
            best = self.formas_res[0][1]
            filas = "".join(
                "<tr style='%s'><td>%s</td><td style='text-align:right'>%.2f</td><td>%s</td></tr>"
                % ("font-weight:bold" if i == 0 else "",
                   " + ".join(sp.FORMAS[q]["corto"] for q in combo), chi,
                   "compatible" if chi - best <= marg else "")
                for i, (combo, chi, rms, _) in enumerate(self.formas_res[:8]))
            partes.append("<div style='margin-top:8px'><b>Probar formas</b> (mejores 8)</div>"
                          "<table style='font-size:12px;border-collapse:collapse;width:100%%'>"
                          "<tr><th>Formas</th><th>χ² red</th><th></th></tr>%s</table>"
                          "<div style='font-size:11px;color:%s'>«compatible»: su χ² difiere del "
                          "mejor en menos de %.2f, lo que el azar del ruido puede explicar. Si hay "
                          "varias compatibles, los datos no permiten decidir la forma.</div>"
                          % (filas, GR, marg))
        return "".join(partes) or ("<div style='color:%s;font-size:12px'>Usa <b>Inversión "
                                   "automática</b> o <b>Probar formas</b>.</div>" % GR)

    def html_solucion(self):
        if not (self.revelar and self.verdad):
            return ""
        filas = "".join("<tr><td>%s</td><td>%.4g</td><td>%.1f</td><td>%.0f°</td><td>%.0f</td></tr>"
                        % (sp.FORMAS[b["q"]]["corto"], b["k"], b["h"], b["th"], b["x0"])
                        for b in self.verdad)
        return ("<table style='border-collapse:collapse;font-size:12px'><tr><th>Cuerpo</th>"
                "<th>k</th><th>h</th><th>θ</th><th>x₀</th></tr>" + filas + "</table>")

    # ================= grafica =================
    def graficar(self):
        cs, obs, x = self.cuerpos, self.obs, self.x
        XD = np.linspace(*sp.PERFIL, 601)
        M = self.metricas()
        fig = plt.figure(figsize=(11.5, 9.6))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.15, 2.1], hspace=0.38)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.06)
        a1 = fig.add_subplot(gs[0])
        a2 = fig.add_subplot(gs[1], sharex=a1)
        a3 = fig.add_subplot(gs[2], sharex=a1)

        if len(cs) > 1:
            for i, b in enumerate(cs):
                a1.plot(XD, sp.sp_cuerpo(b, XD), color=NJ, lw=1.2, ls="--", alpha=.6,
                        label="cada cuerpo" if i == 0 else None)
        if self.revelar and self.verdad:
            a1.plot(XD, sp.sp_total(self.verdad, XD), color=MO, lw=2, ls="--", label="solución")
        if obs is not None:
            a1.plot(x, obs, "o", color="#1b2826", ms=4, label="observado")
        a1.plot(XD, sp.sp_total(cs, XD), color=AZ, lw=2.6, label="modelo")
        for b in cs:
            a1.axvline(b["x0"], color=NJ, ls=":", lw=1, alpha=.7)
        a1.axhline(0, color="0.7", lw=.8)
        a1.set_ylabel("V (mV)")
        a1.grid(alpha=.25)
        a1.legend(fontsize=8.5, frameon=False, loc="best", ncol=2)
        titulo = "Perfil de potencial espontáneo"
        if M:
            titulo += "   ·   RMS = %.2f mV" % M["rms"]
            if self.sigma > 0:
                titulo += "   ·   χ² reducido = %.2f" % M["chi2_red"]
        a1.set_title(titulo, fontsize=11, loc="left")

        if M:
            s = self.sigma
            if s > 0:
                a2.axhspan(-s, s, color="#d5ebec", zorder=0)
            fuera = np.abs(M["r"]) > s if s > 0 else np.zeros_like(M["r"], bool)
            a2.bar(x, M["r"], width=min(3, 0.6 * self.dx), color=np.where(fuera, RO, AZ))
            a2.axhline(0, color="0.5", lw=.8)
            lim = max(M["maxr"], 2 * s, 1) * 1.1
            a2.set_ylim(-lim, lim)
            a2.set_title("Residuos (observado − modelo) · rojo: fuera de la banda ±σ del ruido",
                         fontsize=9.5, loc="left")
        else:
            a2.text(150, 0, "Sin datos observados", ha="center", va="center", color=GR)
            a2.set_ylim(-1, 1)
        a2.set_ylabel("mV")
        a2.grid(alpha=.25)

        zmin = -60
        a3.axhspan(zmin, 0, color=TIERRA, zorder=0)
        for zz in np.linspace(-8, -52, 5):
            a3.plot(XD, zz + 1.2 * np.sin(XD / 45 + zz), color=ESTRATO, lw=1, zorder=1)
        a3.plot(x, np.full_like(x, 1.6), "v", color=AZ, ms=4, zorder=3)
        a3.axhline(0, color="#1b2826", lw=1.3)
        a3.set_xlim(*sp.PERFIL)
        a3.set_ylim(zmin, 4)
        a3.set_xlabel("x (m)")
        a3.set_ylabel("z (m)")
        a3.set_title("Sección · el tamaño dibujado es ilustrativo (en el modelo va dentro de k)",
                     fontsize=9.5, loc="left")

        def dibujar_cuerpo(b, real):
            sel = b.get("id") == self.sel
            if b["q"] == 0.5:
                kw = (dict(fill=False, ls="--", ec=MO, lw=2) if real else
                      dict(fc=NJ, ec="#1b2826" if sel else NJ, lw=2 if sel else 1, alpha=.85))
                a3.add_patch(Rectangle((b["x0"] - 4, zmin), 8, -zmin - b["h"], zorder=4, **kw))
            elif real:
                a3.scatter([b["x0"]], [-b["h"]], s=520, facecolors="none", edgecolors=MO,
                           linewidths=2, linestyle="--", zorder=4)
            else:
                a3.scatter([b["x0"]], [-b["h"]], s=520, color=NJ,
                           edgecolors="#1b2826" if sel else NJ, linewidths=2.2 if sel else 1,
                           alpha=.85, zorder=4)
                if b["q"] == 1.0:
                    a3.text(b["x0"] + 6, -b["h"] + 3, "⊗ eje", fontsize=8, color=NJ, zorder=5)

        if self.revelar and self.verdad:
            for b in self.verdad:
                dibujar_cuerpo(b, True)
        fig.canvas.draw()          # fija la geometria antes de dibujar flechas en pixeles
        r = self.ultima_inv
        for i, b in enumerate(cs):
            dibujar_cuerpo(b, False)
            zc = -b["h"] - 5 if b["q"] == 0.5 else -b["h"]
            a3.text(b["x0"], -b["h"] - (2.5 if b["q"] == 0.5 else 0), str(i + 1), ha="center",
                    va="center" if b["q"] != 0.5 else "top", color="white", fontsize=9,
                    weight="bold", zorder=6)
            if r is not None:      # barras de error de la ultima inversion (1 sigma)
                ex, eh = r["err"][4 * i], r["err"][4 * i + 1]
                if np.isfinite(ex) and np.isfinite(eh):
                    a3.errorbar(b["x0"], -b["h"], xerr=ex, yerr=eh, fmt="none", ecolor="k",
                                elinewidth=1.2, capsize=3, zorder=8)
            T = a3.transData
            c = T.transform((b["x0"], zc))
            d = np.array([np.cos(np.radians(b["th"])), np.sin(np.radians(b["th"]))]) * 24
            tail, tip = T.inverted().transform(c - d), T.inverted().transform(c + d)
            a3.annotate("", xy=tip, xytext=tail, zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=AZ,
                                        lw=2 if b["id"] == self.sel else 1.2))
            if b["id"] == self.sel:
                xd = b["x0"] - 12
                a3.annotate("", xy=(xd, -b["h"]), xytext=(xd, 0),
                            arrowprops=dict(arrowstyle="<->", color=RO, lw=1.3))
                a3.text(xd - 1.5, -b["h"] / 2, "h=%.1f" % b["h"], color=RO, fontsize=8.5,
                        ha="right", va="center")
        return fig

    def redibujar(self):
        if self._sinc:
            return
        M = self.metricas()
        self.h_met.value = self.html_metricas(M)
        self.h_inv.value = self.html_inversion()
        if self.obs is None:
            self.h_pista.value = "<i>No hay datos cargados.</i>"
        elif self.propio:
            self.h_pista.value = ("<i>Los datos son tu propio modelo con ruido: cambia algo y "
                                  "mira cómo crece el error.</i>")
        elif self.w_pista.value:
            self.h_pista.value = ("<i>Pista: %s.</i>" % ", ".join(
                sp.FORMAS[b["q"]]["nombre"].lower() for b in self.verdad))
        else:
            self.h_pista.value = ("<i>%d cuerpo(s) oculto(s); la forma la tienes que deducir.</i>"
                                  % len(self.verdad))
        self.h_sol.value = self.html_solucion()
        self.b_revela.description = "Ocultar solución" if self.revelar else "Revelar solución"
        self.b_revela.disabled = self.verdad is None
        sin = self.obs is None or not self.cuerpos
        self.b_optk.disabled = sin
        self.b_inv.disabled = self.b_formas.disabled = sin or not sp.HAY_SCIPY
        fig = self.graficar()
        with self.salida:
            clear_output(wait=True)
            display(fig)
        plt.close(fig)


def lanzar():
    lab = LaboratorioSP()
    display(lab.vista)
    return lab
