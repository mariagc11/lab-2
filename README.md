# lab 2 
## Introducción. 
Uno de los elementos tratados en la practica a continuación, es la convolución, para esto usamos los digitos del codigo universitario y el documento de identificación de cada una de las integrantes, este proceso se realiza de manera manual y se implementa esto mismo de manera digital por medio de programación en Phyton mostrando la convolución de manera gráfica, con estas mismas señales se calcula la correlación. Por otro lado, se escogió una señal ECG de apnea del sueño, un trastorno caracterizado por pausas en la respiración durante el sueño, y su detección mediante señales electrocardiográficas (ECG). Se basa en la base de datos Apnea-ECG de PhysioNet, que contiene registros de ECG para el desarrollo de métodos automatizados de diagnóstico. El contenido ha sido elaborado por  Dr. Thomas Penzel de la Universidad Phillips, Marburgo, Alemania, con el objetivo de proporcionar una visión técnica sobre la apnea y su análisis a través de ECG. A partir de esta señal, aplicaremos la transformada de Fourier para analizar señales en el dominio de la frecuencia, lo que nos permitirá extraer información clave sobre su comportamiento en el dominio del tiempo (media, mediana, desviación estandar, máximos y mínimos) y el dominio de la frecuencia (la frecuencia media, frecuencia mediana y desviación etsandar de la frecuencia).


## Marco Teórico

### Convolución de señales discretas 
la convolución es una operación matemática fundamental en el análisis de sistemas lineales e invariantes en el tiempo (LTI). Dónde consiste en calcular la salida de un sistema cuando se conoce su respuesta al impulso h[n], y la señal de entrada x[n].
La definición discreta de la convolución es:

<img width="428" height="86" alt="image" src="https://github.com/user-attachments/assets/0f0e2a74-f595-4169-8d4d-f0f4a19e0fa7" />

Cada muestra de la entrada multiplica la respuesta la impulso y la desplaza en el tiempo. La salida es la suma de todas esas respuestas [n].

---

### Correlación cruzada
La correlación cruzada es una operación matemática que mide la similitud entre dos señales en función de un desfase o retardo (lag). Dadas dos señales discretas x[n] y y[n].
La correlación cruzada se define como:

![Imagen de WhatsApp 2025-09-03 a las 18 20 01_960447c1](https://github.com/user-attachments/assets/63976c2f-b70f-4514-8cf1-0da810eb436e)

Dónde: 
* R_x_y [m] es la correlación cruzada en el desfase m 
* m puede ser positivo o negativo (adelantado o atrasado )
* Parecida a la convolucion pero sin invertir una de las señales

---

### Señales sinusoidales
El coseno y el seno son señales fundamentales en el análisis de la transformadfa de Fourier las cuales se pueden utilizar de diversas formas en las tomas y análisis de señales.
Son funcines bases de la transformada de Fourier

<img width="245" height="42" alt="image" src="https://github.com/user-attachments/assets/5de8452b-75af-4cb1-9863-eecdb270ff35" />

<img width="234" height="38" alt="image" src="https://github.com/user-attachments/assets/e001b2ee-b18a-495d-8884-005cbbd25ec0" />

Ambas tienen la misma frecuencia, pero están desfasadas 90° (pi/2 radianes). Por esta razón, se consideran ortogonales y su producto promedio en un periodo completo es cero.  

---
## Señales Biomedicas: El ECG 
El electroccardiograma (ECG) es una señal biomedica que representa la actvidad eléctrica del corazón. 
Los componentes que se deben tener en cuenta principalmente:

Onda P: despolarización auricular.

Complejo QRS: despolarización ventricular (mayor amplitud y frecuencia).

Onda T: repolarización ventricular.

Características generales:

Banda de frecuencia útil: 0.5–40 Hz.

Valores típicos: amplitud de milivoltios y duración de cada ciclo ≈ 0.6–1 s (frecuencia cardíaca 60–100 latidos/min).

El análisis del ECG en el tiempo permite identificar ritmos y morfología, mientras que en el dominio de la frecuencia se evalúa la energía espectral y se aplican filtros para eliminar ruido.

---
## Transformada de Fourier 
La transformada de Fourier discreta (DFT) permite representar una señal en el dominio de la frecuencia. Su versión computacional es la transformada rápida de Fourier (FFT).
Es definida como:

<img width="278" height="94" alt="image" src="https://github.com/user-attachments/assets/152d5100-a57e-4c1e-a30d-61d0a807c095" />

Convierte una señal en el tiempo en sus componentes frecuenciales.

Permite analizar en qué frecuencias se concentra la energía de la señal.

En ECG, muestra que la mayor parte de la energía está en bajas frecuencias (QRS en torno a 10–25 Hz).

---

### Coeficiente de correlación de Pearson
El coeficiente de correlación de Pearson (r) mide el grado de relación lineal entre dos variables o señales donde definimos r como :

![Imagen de WhatsApp 2025-09-03 a las 17 45 55_e01d58fe](https://github.com/user-attachments/assets/59c607d3-1885-4fcf-a389-ed76f4812371)

r=1 → correlación perfecta positiva.

r=−1 → correlación perfecta negativa.

r=0 → no hay correlación lineal.

Original vs reconstruida (IFFT): r ≈ 1.

Original vs filtrada: r alto pero < 1 (muestra similitud con reducción de ruido).

Pearson vs lag: permite ver si el filtrado generó algún desfase.

---
## Procedimiento practica # 2 

Primero se realiza la definición de las señales estableciendo el sistema discreto h[n] a partir de los dígitos del codigo estudiantil de cada integramnte del grupo, asimismo la señal de entrada x[n] a partir de los digitos de la cédula de los mismo integrantes del grupo. 
Luego se realiza el cálculo de la convolución manual, donde se realiza la operación de la convolución y[n]=x[n]*h[n] mediante sumatorias representando graficamente los resultados obtenidos, como se presentan a continuación:

# Calculos de Maria Jose 

![Imagen de WhatsApp 2025-08-28 a las 10 31 04_e1ff1398](https://github.com/user-attachments/assets/2823db13-5902-432b-90e9-096799a75d06)
![Imagen de WhatsApp 2025-08-28 a las 10 31 38_9e54a34d](https://github.com/user-attachments/assets/35ba6f39-49d5-4574-9c53-fca6162ccab5)

# Calculos de Martin
![Imagen de WhatsApp 2025-09-03 a las 17 47 00_a52bfd0b](https://github.com/user-attachments/assets/e43ac639-52e1-4966-b2be-94fcb1c06010)

# Calculos de Gabriela
<img width="1200" height="1600" alt="image" src="https://github.com/user-attachments/assets/32def052-d6a5-434c-bef5-8115d1c78527" />
<img width="1200" height="1600" alt="image" src="https://github.com/user-attachments/assets/6e6c3258-f641-47c8-8783-72e4c3ae3a17" />
<img width="1200" height="1600" alt="image" src="https://github.com/user-attachments/assets/39db854d-5013-4a12-91e3-774392091d49" />
<img width="1200" height="1600" alt="image" src="https://github.com/user-attachments/assets/18876fff-f783-40eb-a9dd-49158c4337b7" />
  
### Programación:

Inicialmente agregamos las librerias:
```  python
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import scipy.signal as signal
import os

```
- **NumPy** ( np) nos ayuda con el manejo de matrices y operaciones matemáticas.
- **Matplotlib** ( plt) se usa para graficar datos.
- **WFDB** ( wfdb) nos permite trabajar con señales fisiológicas como el ECG.
- **SciPy Signal** ( signal) se usa para procesar señales (aunque en este código no se usa mucho).
- **OS** ( os) nos permite manejar archivos y rutas en el sistema operativo.

## Convolución de secuencias discretas

El código define las señales \( h[n] \) y \( x[n] \) para tres personas diferentes: Gaby, María José y Martin. Estas señales son representadas como arreglos de NumPy, donde \( h[n] \) corresponde al sistema y \( x[n] \) a la señal de entrada. Cada persona tiene su propio conjunto de datos, posiblemente para comparar cómo interactúan las señales en distintas situaciones.
```
# Gaby
h_gaby = np.array([5, 6, 0, 0, 8, 7, 7])  # Sistema h[n]
x_gaby = np.array([1, 0, 0, 0, 8, 1, 0, 4, 5, 6])  # Señal x[n]

# María José
h_maria = np.array([1, 0, 1, 9, 6, 0, 2, 1, 4, 8])
x_maria = np.array([5, 6, 0, 0, 4, 3, 5 ])

# martin
h_martin = np.array([1, 0, 1, 6, 5, 9, 2, 6, 7, 7])
x_martin = np.array([5, 6, 0, 0, 5, 1, 1])

```
## Convolución (Salida del sistema)
```
def calcular_convolucion(h, x):
    return np.convolve(h, x, mode='full')
```
Efectúa <img width="384" height="38" alt="image" src="https://github.com/user-attachments/assets/7ca60661-e83f-4d78-872f-653289c7eef9" />

```
y_gaby   = calcular_convolucion(h_gaby,   x_gaby)
y_maria  = calcular_convolucion(h_maria,  x_maria)
y_martin = calcular_convolucion(h_martin, x_martin)

```

## Grafico:

Este fragmento de código usa *Matplotlib* para visualizar las señales convolucionadas de Gaby, María José y Martin en una sola figura con tres subgráficos. Primero, se crea una figura de tamaño 12x6. Luego, se generan tres gráficos de tipo *stem*, que representan las señales discretas con líneas y marcadores.  

- En el primer *subplot*, se grafica la convolución de Gaby en color amarillo (`y-` para las líneas y `yo` para los marcadores).  
- En el segundo, se muestra la convolución de María José en rojo (`r-` y `ro`).  
- En el tercero, se grafica la convolución de Martin en verde (`g-` y `go`).  

Cada gráfico tiene su título, etiquetas para los ejes \( n \) y \( y[n] \), y *plt.tight_layout()* ajusta los espacios para que no se sobrepongan los elementos. Finalmente, *plt.show()* muestra la figura con las tres gráficas.

```

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(y_gaby, linefmt='y-', markerfmt='yo', basefmt='k-')
plt.title("Convolución de Gaby")
plt.xlabel("n")
plt.ylabel("y[n]")

plt.subplot(3, 1, 2)
plt.stem(y_maria, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title("Convolución de María José")
plt.xlabel("n")
plt.ylabel("y[n]")

plt.subplot(3, 1, 3)
plt.stem(y_Martin, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title("Convolución de Martin")
plt.xlabel("n")
plt.ylabel("y[n]")

plt.tight_layout()
plt.show()

```


### Parte B 
## Cálculo de correlación:

El código calcula la correlación cruzada entre dos señales, un coseno y un seno de 100 Hz. Primero, define el tiempo de muestreo (`Ts = 1.25 ms`) y genera 9 muestras (`n = np.arange(9)`). Luego, crea las señales `x1` y `x2` usando funciones coseno y seno, respectivamente. Finalmente, con `np.correlate(x1, x2, mode='full')`, mide la similitud entre ambas en distintos desfases, lo que permite analizar su relación temporal.

```
Ts = 1.25e-3  
n = np.arange(9) 
f = 100  # Frecuencia en Hz


x1 = np.cos(2 * np.pi * f * n * Ts)  # Señal x1[n]
x2 = np.sin(2 * np.pi * f * n * Ts)  # Señal x2[n]


correlacion = np.correlate(x1, x2, mode='full')
```
## Correlación:

Ts = 1.25e-3  # Tiempo de muestreo en segundos
n = np.arange(9)  # Nueve muestras según el documento
f = 100  # Frecuencia en Hz

Definición de las señales:
x1 = np.cos(2 * np.pi * f * n * Ts)  # Señal x1[n]
x2 = np.sin(2 * np.pi * f * n * Ts)  # Señal x2[n]

Correlación cruzada entre ambas señales:
correlacion = np.correlate(x1, x2, mode='full')

## Gráfico: 
Este código genera una gráfica de la correlación cruzada entre las señales \( x_1[n] \) y \( x_2[n] \). Primero, se crea una figura de tamaño 8x4. Luego, con `plt.stem()`, se representa la correlación cruzada en un gráfico de líneas y marcadores azules (`b-` y `bo`), con una línea base negra (`k-`). El eje x muestra los desplazamientos de la correlación, que van desde `-len(x1) + 1` hasta `len(x1)`, indicando cómo varía la similitud entre las señales a diferentes desfases. Se añaden título, etiquetas y una cuadrícula para mejorar la visualización. Finalmente, `plt.show()` muestra el gráfico.

```
plt.figure(figsize=(8, 4))
plt.stem(np.arange(-len(x1)+1, len(x1)), correlacion, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title("Correlación cruzada entre x1[n] y x2[n]")
plt.xlabel("Desplazamiento (n)")
plt.ylabel("Correlación")
plt.grid(True)
plt.show()

```


### Segundo Punto:

Se selecciona la señal EMG por medio de Physionet [link Physionet](https://physionet.org/)
- Guardar los archivos .hea, .data, .apn en una misma carpeta junto con la señal
- Abrir Python, nombrar el archivo y guardarlo en la misma carpeta donde se encuentran los archivos .hea .data y apn.
- Abrir de nuevo python y iniciar con la programación que explicaremos a continuación:

Primero, definimos la ruta y el nombre del archivo (`record_name = "a01"`) donde está almacenada la señal. Luego, usa la librería `wfdb` para leer tanto la señal (`wfdb.rdrecord()`) como sus anotaciones (`wfdb.rdann()`). Se extrae la señal del primer canal (`record.p_signal[:, 0]`) y la frecuencia de muestreo (`record.fs`). A partir de esto, se calcula el tiempo en segundos dividiendo el número de muestras por la frecuencia de muestreo. Finalmente, se obtienen estadísticas descriptivas de la señal, como la media, mediana, desviación estándar, valor máximo y mínimo, para analizar su comportamiento.

```
record_name = "a01"  # Nombre del archivo sin extensión
record_path = r"C:\Users\majo1\OneDrive\Escritorio\señales\lab señales\lab 2"


record = wfdb.rdrecord(os.path.join(record_path, record_name))
annotation = wfdb.rdann(os.path.join(record_path, record_name), "apn")


signal_data = record.p_signal[:, 0]  # Primer canal (ECG)
sampling_rate = record.fs  # Frecuencia de muestreo

time = np.arange(len(signal_data)) / sampling_rate

```



## Estadisticas Descriptivas:

Primero, obtiene la media (np.mean), mediana (np.median), desviación estándar (np.std), valor máximo (np.max) y valor mínimo (np.min) de los datos de la señal. Luego, imprime estos valores en pantalla con un formato claro, permitiendo analizar su distribución y variabilidad. Esto es útil para entender el comportamiento general de la señal antes de aplicar otros análisis más complejos.
```

time_mean = np.mean(signal_data)
time_median = np.median(signal_data)
time_std = np.std(signal_data)
time_max = np.max(signal_data)
time_min = np.min(signal_data)

print("Estadísticas en el dominio del tiempo:")
print(f"Media: {time_mean}")
print(f"Mediana: {time_median}")
print(f"Desviación estándar: {time_std}")
print(f"Máximo: {time_max}")
print(f"Mínimo: {time_min}")

```

## Graficar la señal en el dominio del tiempo:

El código grafica la señal de ECG en función del tiempo. Primero, crea una figura de 10x4 y luego dibuja la señal con `plt.plot()`, etiquetando los ejes y agregando un título. También incluye una leyenda para identificar la señal y, al final, muestra la gráfica con `plt.show()`, permitiendo visualizar cómo varía la señal a lo largo del tiempo.

```

plt.figure(figsize=(10, 4))
plt.plot(time, signal_data, label="ECG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Señal ECG en el dominio del tiempo")
plt.legend()
plt.show()

```


## Transformada de Fourier y Densidad Espectral:

El código aplica la Transformada de Fourier a la señal de ECG para analizar sus frecuencias. Primero, calcula las frecuencias (`freqs`), obtiene la transformada (`fft_values`) y luego la densidad espectral de potencia (`power_spectrum`). Después, genera dos gráficas: una con la magnitud de la transformada para ver qué frecuencias están presentes y otra con la densidad espectral para visualizar cómo se distribuye la energía. Ambos gráficos incluyen etiquetas y una cuadrícula para facilitar su interpretación.

```

freqs = np.fft.rfftfreq(len(signal_data), d=1/sampling_rate)
fft_values = np.fft.rfft(signal_data)
power_spectrum = np.abs(fft_values) ** 2

-Graficar la transformada de Fourier
plt.figure(figsize=(10, 4))
plt.plot(freqs, np.abs(fft_values))
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Transformada de Fourier de la señal ECG")
plt.grid()
plt.show()

```


```

Graficar la densidad espectral de potencia
plt.figure(figsize=(10, 4))
plt.plot(freqs, power_spectrum)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia")
plt.title("Densidad espectral de la señal ECG")
plt.grid()
plt.show()

```


## Cálculo de estadísticas en el dominio de la frecuencia:

Acá se calcula y analiza estadísticas en el dominio de la frecuencia para la señal de ECG. Primero, obtiene la **frecuencia media** (`freq_mean`), ponderando cada frecuencia con su densidad espectral de potencia y dividiendo por la suma total de la potencia. Luego, calcula la **frecuencia mediana** (`freq_median`), ordenando la densidad espectral y seleccionando la frecuencia en la posición central. La **desviación estándar** (`freq_std`) se obtiene midiendo la dispersión de las frecuencias respecto a la media, también ponderada por la potencia.  

Después, imprime estos valores y genera un **histograma de frecuencias**, donde `plt.hist()` divide el rango de frecuencias en 50 bins y pondera cada una según su potencia, permitiendo visualizar cómo se distribuye la energía en el espectro de la señal. Finalmente, se añaden etiquetas y una cuadrícula para mejorar la interpretación antes de mostrar la gráfica con `plt.show()`.

```
freq_mean = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
freq_median = freqs[np.argsort(power_spectrum)[len(power_spectrum)//2]]
freq_std = np.sqrt(np.sum((freqs - freq_mean)**2 * power_spectrum) / np.sum(power_spectrum))

print("Estadísticas en el dominio de la frecuencia:")
print(f"Frecuencia media: {freq_mean}")
print(f"Frecuencia mediana: {freq_median}")
print(f"Desviación estándar de la frecuencia: {freq_std}")

# Histograma de frecuencias
plt.figure(figsize=(10, 4))
plt.hist(freqs, bins=50, weights=power_spectrum)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")
plt.title("Histograma de frecuencias de la señal ECG")
plt.grid()
plt.show()

```
### Coeficiente de correlación de Pearson

## Original vs reconstruida
```
signal_reconstructed = np.fft.irfft(fft_values, n=len(signal_data))
pearson_ecg_ifft = np.corrcoef(signal_data, signal_reconstructed)[0, 1]

```
Reconstruimos la señal con irfft.
Pearson ≈ 1 → indica que la reconstrucción es idéntica a la original.

## Original vs filtrada (0.5–40 Hz)

```
ecg_filtered = bandpass(signal_data, sampling_rate, 0.5, 40.0, order=4)
pearson_ecg_filtered = np.corrcoef(signal_data, ecg_filtered)[0, 1]

```
Filtrado pasa banda elimina ruido de baja frecuencia (deriva) y alta frecuencia (interferencia muscular o de red).

Pearson < 1 pero alto (ej. 0.8–0.95) → la señal mantiene su forma general.

## Pearson vs desfase
```
lags = np.arange(-200, 201)
pearson_vs_lag = [pearson_lag(signal_data, ecg_filtered, lag=int(k)) for k in lags]
```
Compara la correlación variando el desfase.

Si el máximo ocurre en lag = 0 → no hay retardo (gracias a filtfilt que es de fase cero).

El máximo debe estar en lag=0 con r alto, indicando que el filtrado mantiene la alineación de la señal.


### Analisis y resultados 

## 1) Convolución 

<img width="1280" height="635" alt="image" src="https://github.com/user-attachments/assets/26a06b5a-c626-4aa8-8a81-4ec63ebc7dbd" />

Estas tres gráficas muestran la convolución, que básicamente nos dice cómo responde un sistema cuando le damos una señal de entrada. En el caso de Gaby, la salida tiene picos grandes entre las posiciones 8 y 12 porque en su entrada había valores altos que activaron al sistema. En María José, la salida va creciendo poco a poco hasta llegar a un máximo alrededor de la posición 9 y luego empieza a bajar. En Martín, también hay un máximo en la misma zona, pero la forma es más corta y concentrada. En pocas palabras: cada señal de entrada genera una salida distinta, y la convolución nos muestra cómo el sistema “mezcla” la forma de la respuesta con los valores de la entrada.

## 2) Correlación cruzada

<img width="978" height="499" alt="image" src="https://github.com/user-attachments/assets/dd73c3a2-5d6d-4a6a-a267-ce01a1f91c80" />

Esta gráfica muestra la correlación cruzada entre una señal coseno y una seno. Como estas dos señales están desfasadas 90°, en teoría deberían ser ortogonales, es decir, su correlación en cero debería ser cercana a 0. Eso es lo que vemos aquí: en el centro la correlación es casi nula. Los valores positivos y negativos que aparecen en otros desplazamientos (lags) son producto de que usamos pocas muestras y no cubrimos un periodo completo. En resumen: la gráfica confirma que coseno y seno son casi independientes, aunque en la práctica aparecen pequeños valores residuales por el muestreo limitado.


## 3) ECG en el dominio del tiempo 

<img width="1233" height="498" alt="image" src="https://github.com/user-attachments/assets/37db0bf6-665c-451a-9aa6-207784274704" />

Aquí vemos la señal de ECG en el dominio del tiempo. El eje horizontal representa el tiempo en segundos (casi 30 000, que son varias horas de registro) y el eje vertical la amplitud de la señal en milivoltios. Lo que se nota es que la señal oscila alrededor de cero, lo cual es normal porque el corazón genera impulsos eléctricos positivos y negativos. Los picos hacia arriba corresponden a los complejos QRS (los latidos), mientras que las caídas bruscas o valores atípicos se deben a ruido o artefactos durante la captura. Al verlo completo parece una banda sólida, porque hay millones de puntos, pero si hacemos zoom podríamos distinguir la forma típica del ECG (ondas P, QRS y T).

👉 En resumen: esta gráfica confirma que la señal está centrada en cero, con amplitudes normales para un ECG real, aunque a esta escala no se distinguen los detalles de cada latido.

## 4)Estadísticas en el dominio del tiempo

<img width="425" height="124" alt="image" src="https://github.com/user-attachments/assets/01ee583c-0311-42cd-ad04-337c41780cd2" />

Estas son las estadísticas de la señal ECG en el dominio del tiempo. La media es prácticamente 0, lo cual indica que la señal está bien centrada alrededor de la línea base. La mediana es −0.03, muy cercana a cero, confirmando la simetría de los datos. La desviación estándar es 0.24, lo que muestra una variabilidad moderada típica en el ECG. El valor máximo es 2.44 mV y el mínimo es −1.65 mV, que corresponden a los picos positivos y negativos de los latidos.

👉 En resumen: la señal está centrada en cero, con amplitudes y variabilidad normales para un ECG real.

## 5) Transformada de Fourier de la señal ECG

<img width="1234" height="480" alt="image" src="https://github.com/user-attachments/assets/88f7d2b8-6984-46cd-948c-220d60429f6d" />

Esta es la Transformada de Fourier del ECG, que nos muestra la señal en el dominio de la frecuencia. Aquí se ve que la mayor parte de la energía del ECG está concentrada entre 1 y 25 Hz, que es donde aparecen las ondas P, QRS y T. También se nota un pico muy cerca de 0 Hz, que corresponde a la deriva de la línea base. A partir de 30–40 Hz, la magnitud disminuye casi a cero, lo que confirma que el ECG es principalmente una señal de baja frecuencia.

👉 En resumen: el espectro confirma que el ECG tiene su información útil en bajas frecuencias, mientras que lo demás es principalmente ruido o componentes indeseados.

## 6) Densidad espectral de la señal ECG

<img width="1224" height="487" alt="image" src="https://github.com/user-attachments/assets/101bd143-1354-4c52-aba9-1aed2478a103" />

Esta gráfica muestra la densidad espectral de potencia del ECG, es decir, cuánta energía tiene la señal en cada frecuencia. Se observa que la mayor concentración de potencia está en las bajas frecuencias (0 a 20 Hz), especialmente entre 5 y 15 Hz, que es donde se encuentran las ondas P, QRS y T. También aparece un pico cerca de 0 Hz, que corresponde a la deriva de la línea base. A partir de los 30 Hz, la densidad cae casi a cero, lo que confirma que el ECG es una señal de baja frecuencia.

👉 En resumen: la gráfica demuestra que la energía útil del ECG está en bajas frecuencias, mientras que lo que aparece en frecuencias muy altas suele ser ruido.

## 7) Estadísticas en el dominio de la frecuencia

<img width="839" height="118" alt="image" src="https://github.com/user-attachments/assets/9e452da9-6c3d-4839-871b-e9dd7eefef92" />

Estas son las estadísticas del ECG en el dominio de la frecuencia. La frecuencia media es de aproximadamente 12.1 Hz, lo que refleja que la mayor parte de la energía de la señal se concentra en la zona típica del complejo QRS. La frecuencia mediana aparece en 30.8 Hz, aunque este valor está un poco alto porque depende del método de cálculo, y en la práctica debería ubicarse más cerca de 10–15 Hz. Finalmente, la desviación estándar es de 7.19 Hz, lo que indica que la energía de la señal está distribuida de forma relativamente compacta en bajas frecuencias.

👉 En resumen: el ECG concentra su energía en torno a los 10–20 Hz, lo cual coincide con lo esperado fisiológicamente.

## 8) Histograma de frecuencias ponderado por potencia

<img width="1219" height="496" alt="image" src="https://github.com/user-attachments/assets/7a8c3b80-c171-4335-9773-9944e4eef56e" />

Esta gráfica muestra el histograma de frecuencias del ECG, es decir, cómo se distribuye la energía de la señal a lo largo de las distintas frecuencias. Se observa que la mayor parte de la potencia está concentrada entre 5 y 15 Hz, que corresponde a la actividad principal del corazón (complejos QRS y ondas asociadas). A medida que la frecuencia aumenta, la potencia disminuye rápidamente, casi desapareciendo después de los 30 Hz.

👉 En resumen: el histograma confirma que la energía útil del ECG está en bajas frecuencias, mientras que las altas frecuencias tienen poca contribución y suelen corresponder a ruido.


## 9) Comparacion ECG original vs filtrada 

<img width="1461" height="495" alt="image" src="https://github.com/user-attachments/assets/eb6226ba-1dd7-47ed-a381-4457b510010a" />

Esta gráfica compara la señal ECG original con la ECG filtrada entre 0.5 y 40 Hz. A simple vista parecen casi iguales, lo que significa que el filtrado conserva la forma de la señal. La diferencia está en que la señal filtrada elimina la deriva de baja frecuencia y el ruido de alta frecuencia, quedándose solo con la información útil del corazón.

👉 En resumen: el filtrado limpia la señal sin distorsionar su morfología, asegurando que los latidos se conserven intactos.


## 11) Person 

<img width="644" height="68" alt="image" src="https://github.com/user-attachments/assets/069e7afb-e259-47bd-aba8-ad5ab24eb3bb" />

Estos son los resultados del coeficiente de correlación de Pearson. El valor entre la señal original y la reconstruida con la IFFT es 1.000000, lo que significa que son idénticas. Entre la señal original y la filtrada (0.5–40 Hz) la correlación es de 0.995556, un valor muy cercano a 1, lo que indica que son prácticamente iguales. Además, el máximo Pearson ocurre en lag = 0, confirmando que el filtrado no introdujo desfase en la señal.

👉 En resumen: el ECG filtrado mantiene casi toda la información de la señal original, solo eliminando ruido, y lo hace sin alterar la alineación en el tiempo.

## Correlación de Pearson ECG original vs filtrada según lag

<img width="985" height="495" alt="image" src="https://github.com/user-attachments/assets/dfe79b14-ba32-46d1-87b2-a07a44afebab" />
Esta gráfica muestra cómo cambia la correlación de Pearson entre el ECG original y el filtrado cuando se aplica un desfase (lag). El valor máximo está en lag = 0 con r ≈ 1, lo que significa que las señales son prácticamente iguales y están alineadas. A medida que se aumenta el desfase, la correlación disminuye hasta acercarse a cero o valores negativos, mostrando que ya no coinciden.

👉 En resumen: el filtrado no introduce retraso y mantiene la forma de la señal, solo eliminando ruido.



## ✅ Cierre general

- La **convolución** muestra claramente cómo un sistema LTI transforma entradas discretas mediante **superposición**.  
- La **correlación** entre coseno y seno valida la **ortogonalidad** (con pequeñas desviaciones por pocas muestras).  
- El **ECG** es de **baja frecuencia** (0.5–40 Hz); el espectro y la “potencia” lo evidencian.  
- El **filtrado** limpia la señal sin **desfase** ni pérdida de morfología (Pearson ~0.996).  
- Para métricas espectrales más precisas, usa **Welch** y **mediana energética** por **CDF**.

---

## Conclusión.

En este laboratorio, exploramos la convolución y la correlación como herramientas clave en el procesamiento digital de señales. A través de ejercicios prácticos, comprendimos cómo la convolución permite analizar la respuesta de un sistema a una señal de entrada, mientras que la correlación nos ayuda a medir la similitud entre señales en distintos momentos. Además, aplicamos la Transformada de Fourier para examinar la señal en el dominio de la frecuencia, calculando su densidad espectral de potencia y sus estadísticas descriptivas.

La implementación en Python facilitó el análisis y la visualización de los resultados, permitiéndonos interpretar mejor la información contenida en las señales. Esto es fundamental en aplicaciones como el procesamiento de señales biomédicas, donde la correcta identificación de patrones en ECG u otras señales fisiológicas puede mejorar el diagnóstico y la toma de decisiones clínicas. En general, este laboratorio reforzó la importancia de estas técnicas en el análisis y manipulación de señales digitales.

Al hacer la convolucion en físico aprendemos con más práctica como funciona y que es, viendo como se transforma y se hace una nueva señal a partir de dos señales distintas donde no dependemos de una máquina nomás que con un codigo corto nos resuelve todo el trabajo, ya que así vamos más a fondo al comprender el tema visto 

Además de afianzar los fundamentos matemáticos del procesamiento de señales, este laboratorio permitió evidenciar la relevancia práctica de dichas técnicas en contextos reales como el análisis de señales biomédicas. La aplicación de convolución, correlación y transformada de Fourier no solo aportó al entendimiento de cómo se comportan las señales en diferentes dominios, sino que también mostró el potencial de estas herramientas para la detección de patrones clínicamente relevantes. Esto resalta la importancia de integrar la teoría con la experimentación computacional, fomentando la capacidad de aplicar métodos digitales al diagnóstico y monitoreo en ingeniería biomédica.

## Referencias.
- Oppenheim, AV, y Willsky, AS (1996). Señales y sistemas (2.ª ed.). Prentice Hall.
- Proakis, JG, y Manolakis, DG (2007). Procesamiento de señales digitales: principios, algoritmos y aplicaciones (4.ª ed.). Pearson
- Cohen, L. (1995). Análisis de tiempo-frecuencia . Prentice Hall.
- Clifford, GD, Azuaje, F., y McSharry, PE (2006). Métodos y herramientas avanzad
