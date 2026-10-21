# Laboratorio SP — simulador de anomalías de potencial espontáneo

Simulador interactivo para entender cómo se forma una anomalía de potencial espontáneo (SP),
cómo se ajusta un modelo a los datos y **cuánto se puede confiar** en lo que se concluye.

Maestría en Ciencias – Geofísica · Universidad Nacional de Colombia  
Milton Daniel Díaz Barrera · Daniela Liseth Saballet Franco

## Abrir en el navegador (Binder, sin instalar nada)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NinjaSaurio120/simulador_sp/HEAD?urlpath=lab/tree/SP_interactivo.ipynb)

1. Clic en el botón **launch binder** de arriba, o abre este enlace:
   https://mybinder.org/v2/gh/NinjaSaurio120/simulador_sp/HEAD?urlpath=lab/tree/SP_interactivo.ipynb
   (la primera vez tarda unos minutos en preparar el entorno).
2. Se abre `SP_interactivo.ipynb`: ejecuta la celda y usa los controles.
3. Para comprobar que el simulador calcula bien, abre `SP_verificacion.ipynb` y ejecútalo.

Binder borra la sesión tras unos 10 minutos sin uso.

## Qué hace

- **Datos sintéticos** con 1 a 4 cuerpos ocultos (esfera, cilindro horizontal o vertical) y
  ruido gaussiano σ; muestreo de estaciones configurable.
- **Modelo** con hasta 6 cuerpos, editables con deslizadores.
- **Ajuste:** k por mínimos cuadrados (lineal), inversión no lineal de x₀, h, θ y k desde
  varios puntos de partida, y comparación automática de formas.
- **Confianza:** χ² reducido con N − p grados de libertad, incertidumbre ±1σ de cada
  parámetro, correlaciones entre parámetros y aviso de mínimos locales.

## Archivos

| Archivo | Qué es |
|---|---|
| `SP_interactivo.ipynb` | El simulador (una celda) con guía de uso y ejercicios |
| `SP_verificacion.ipynb` | Pruebas de la física y del método (incluye un Monte Carlo) |
| `sp_widgets.py` | Interfaz interactiva (ipywidgets) |
| `sp_motor.py` | Modelo directo, datos, métricas, inversión e incertidumbre |
| `requirements.txt`, `runtime.txt` | Lo que instala Binder |

## Modelo

$$V(x) = \sum_i k_i\,\frac{(x-x_{0,i})\cos\theta_i + h_i\sin\theta_i}{[(x-x_{0,i})^2 + h_i^2]^{q_i}}$$

con q = 1.5 (esfera), 1.0 (cilindro horizontal) y 0.5 (cilindro vertical).

## Referencias

- Yüngül, S. (1950). Interpretation of spontaneous polarization anomalies caused by
  spheroidal orebodies. *Geophysics*, 15(2), 237–246.
- Biswas, A. (2017). A review on modeling, inversion and interpretation of self-potential in
  mineral exploration and tracing paleo-shear zones. *Ore Geology Reviews*, 91, 21–56.
