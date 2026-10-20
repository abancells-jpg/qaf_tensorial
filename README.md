# qaf_tensorial
Implementación de QAF mediante redes tensoriales

## Estructura del Repositorio

# Libro completo en latex y pdf

# 1_QAF_BENCHMARK

Esta carpeta contiene el archivo contiene el paper QAF_Benchmark_Validation_Juan_Manuel_Goenaga.pdf y el código de Python qaf_benchmar.py, que contiene el banco de pruebas (*benchmark*) sintético y reproducible para evaluar si la representación contextual del **Quantum Analysis Framework (QAF)** mejora la toma de decisiones en problemas de asignación de recursos crítica ante la presencia de incertidumbre y señuelos (*decoys*) engañosos. El código evalúa y compara siete formulaciones de optimización y decisión clasificadas en tres familias principales:

1. **Métodos basados en QUBO (Quadratic Unconstrained Binary Optimization):**
   * **`QUBO-SA`**: Formulación QUBO estándar resuelta mediante Recocido Simulado (*Simulated Annealing* con Numba).
   * **`QAF-QUBO-SA`**: Proxy contextual de QAF que añade términos de acoplamiento negativo para penalizar la asignación simultánea de amenazas mutuamente excluyentes o señuelos.
   * **`Hybrid-BQM-local`**: Adaptador surrogate local con semilla *Greedy* y reparación rápida.

2. **Baselines Clásicos y Ablaciones Contextuales:**
   * **`Greedy`**: Algoritmo de asignación secuencial basado en valor esperado.
   * **`MILP`**: Programación lineal entera mixta exacta utilizando SciPy / HiGHS.
   * **`Context-MILP`**: Ablación del modelo que aplica la misma actualización de creencias contextuales de QAF, pero resuelve la asignación con el solver exacto MILP.

3. **Baseline Probabilístico:**
   * **`Particle-POMDP`**: Planificador por partículas de horizonte único que modela exclusividad contextual mediante muestreo.

Los resultados agregados muestran que la inclusión de **relaciones contextuales** mejora la calidad de decisión y reduce sustancialmente la asignación a señuelos:

| Método | Calidad (% daño evitado) | Desperdicio en Señuelos (%) | Latencia Mediana (ms) | Cumplimiento < 500 ms (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy** | ~86.84% | ~14.78% | ~0.07 ms | 100% |
| **MILP** | ~87.49% | ~15.26% | ~10.64 ms | 100% |
| **QUBO-SA** | ~86.73% | ~15.15% | ~30.87 ms | 100% |
| **Context-MILP** | ~91.44% | ~10.59% | ~10.66 ms | 100% |
| **QAF-QUBO-SA** | ~91.55% | ~5.93% | ~32.51 ms | ~86.67% |
| **Hybrid-BQM-local** | ~86.70% | ~15.15% | ~31.94 ms | 100% |
| **Particle-POMDP** | **~95.16%** | **~4.48%** | ~12.90 ms | 100% |

**Conclusión clave:** La representación contextual de QAF reduce el desperdicio en señuelos del ~15% a menos del 6%. Sin embargo, el solver por Simulated Annealing incrementa la latencia en instancias grandes, siendo superado en eficiencia global por baselines probabilísticos como *Particle-POMDP*.

Antes de poderlo ejecutar, asegúrate de tener instalado Python 3.9 o superior e instala las bibliotecas requeridas ejecutando en la terminal:

pip install numpy pandas scipy matplotlib numba

Para ejecutar el benchmark, hay que introducir, en la carpeta donde está el codigo:

python3 qaf_benchmark.py --outdir ./resultados_benchmark

De este modo, se crea la carpeta resultados_benchmark con una subcarpeta que contiene figuras y otra que contiene tablas en formato csv para poder ser analizada.

# 2_CONSUMO_TOKENS

Esta carpeta contiene el archivo del paper `QAF_Validacion_Consumo_Tokens_Juan_Manuel_Goenaga.pdf` y el código de Python `qaf_consumo_tokens.py`, que contiene el banco de pruebas (*benchmark*) sintético y reproducible para evaluar si la capa de enrutamiento metacognitivo e información dirigida del **Quantum Analysis Framework (QAF)** reduce el consumo de tokens en arquitecturas multiagente sin comprometer la suficiencia decisional. El código evalúa y compara cuatro arquitecturas de comunicación sobre 750 tareas sintéticas de distinta complejidad:

1. **Arquitectura Exhaustiva Convencional:**
   * **`AllAgents`**: Límite superior de costo que activa los 6 agentes (`memory`, `retrieval`, `specialist`, `verification`, `planner`, `reporter`) transmitiendo el contexto completo a cada uno.

2. **Arquitecturas de Enrutamiento:**
   * **`RulesRouter`**: Enrutador por reglas simples que aplica umbrales univariados sobre las variables observadas para seleccionar agentes, enviando contexto completo.
   * **`QAFRouter`**: Enrutador metacognitivo basado en QAF que evalúa términos cruzados (incertidumbre, contradicción, riesgo, brecha de información) y aplica compresión y selección contextual dirigida por agente (*targeted context*).

3. **Límite Teórico:**
   * **`Oracle`**: Límite inferior de costo que conoce el conjunto latente mínimo de agentes requeridos por cada tarea.

Los resultados agregados sobre las 750 tareas demuestran que la combinación de enrutamiento metacognitivo y contexto dirigido logra un ahorro masivo de tokens manteniendo un alto nivel de cobertura y éxito en la ruta[cite: 28, 33]:

| Arquitectura | Tokens Medios / Tarea | Reducción vs. AllAgents (%) | Llamadas a Agentes | Recall de Agentes (%) | Tasa de Ruta Completa (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AllAgents** | ~12,401.38 | 0.00% | 6.00 | 100.00% | 100.00% |
| **RulesRouter** | ~6,898.35 | 44.37% | 2.89 | 72.57% | 29.60% |
| **QAFRouter** | **~4,604.02** | **62.87%** | **4.79** | **97.25%** | **91.07%** |
| **Oracle** | ~3,455.62 | 72.14% | 3.95 | 100.00% | 100.00% |

Desglose de comportamiento adaptativo de **`QAFRouter`** según el nivel de complejidad[cite: 34]:
* **Complejidad Baja (`low`):** ~2,504.54 tokens | 3.40 llamadas a agentes | 84.00% éxito de ruta.
* **Complejidad Media (`medium`):** ~4,722.80 tokens | 5.12 llamadas a agentes | 89.20% éxito de ruta.
* **Complejidad Alta (`high`):** ~6,584.72 tokens | 5.86 llamadas a agentes | 100.00% éxito de ruta.

**Conclusión clave:** El enrutador metacognitivo `QAFRouter` reduce el consumo global de tokens en un **62.87%** frente a la activación exhaustiva, con un ahorro predominantemente enfocado en los tokens de entrada debido a la selección del contexto (64.7% de reducción en entrada). A diferencia de reglas simples (`RulesRouter`), que fallan al omitir agentes necesarios en el 70% de las tareas, `QAFRouter` conserva un **97.25% de recall** y escala dinámicamente su gasto alcanzando el 100% de éxito en tareas de alta complejidad.

Antes de poderlo ejecutar, asegúrate de tener instalado Python 3.9 o superior e instala las bibliotecas requeridas ejecutando en la terminal:

pip install sentencepiece pandas numpy matplotlib

Para ejecutar el benchmark, hay que introducir, en la carpeta donde está el codigo:

python3 QAF_consumo_tokens.py --outdir ./qaf_token_results

De este modo, se crea la carpeta qaf_tokens_results con subcarpetas para el tokenizador BPE entrenado (tokenizer), las figuras gráficas exportadas (figures) y los informes resumidos en tablas CSV (results).

# 3_ESCALABILIDAD_1M_50M

Esta carpeta contiene el archivo del paper `QAF_Escalabilidad_1M_50M_Juan_Manuel_Goenaga.pdf` y el código de Python `qaf_scalability.py`, que contiene la prueba de escalabilidad y análisis de límites de representación del **Quantum Analysis Framework (QAF)** al escalar de 1 a 50 millones de variables binarias ($K \in \{1\text{M}, 5\text{M}, 10\text{M}, 25\text{M}, 50\text{M}\}$).

El código realiza dos tipos de análisis complementarios:

1. **Cálculo Exacto de Leyes de Escalado de Memoria (Teórico):**
   * **QUBO Denso ($\mathcal{O}(K^2)$):** Representación explícita donde todas las variables interactúan entre sí.
   * **QUBO Bandeado CSR ($\mathcal{O}(K \cdot b)$):** Representación dispersa con conectividad acotada localmente a un ancho de banda $b$.
   * **MPS / Tensor Train ($\mathcal{O}(K \cdot d \cdot \chi^2)$):** Representación basada en estados de producto matricial comprimidos gobernados por la dimensión de enlace ($\chi$).

2. **Benchmark Empírico de Compilación Streaming ($\mathcal{O}(K \cdot b)$):**
   * Un proxy de compilación por bloques de tamaño fijo que procesa las $K$ variables en streaming calculando puntuaciones contextuales e interacciones locales sin materializar una matriz de $K \times K$ en memoria.

Aquí están los resultados comparativos de memoria: 

| Variables ($K$) | QUBO Denso ($\mathcal{O}(K^2)$) | QUBO Bandeado ($b=8$) | MPS / Tensor Train ($\chi=16$) |
| :--- | :---: | :---: | :---: |
| **1,000,000 (1M)** | 8 TB | 280 MB | 4.1 GB |
| **5,000,000 (5M)** | 200 TB | 1.4 GB | 20.5 GB |
| **10,000,000 (10M)** | 800 TB | 2.8 GB | 41 GB |
| **25,000,000 (25M)** | 5 PB | 7 GB | 102 GB |
| **50,000,000 (50M)** | **20 PB** | **14 GB** | **205 GB** |

También de han calculado los datos de rendimiento del Proxy Streaming Empírico

* **Throughput Promedio Global:** ~45.5 millones de variables por segundo.
* **Tiempo de Procesamiento ($K = 50\text{M}$):** ~1.07 segundos.
* **Uso Adicional de Memoria RAM (RSS):** ~47.9 MB (permanece constante independientemente de $K$ gracias al procesamiento por bloques).
* **Ajuste de Escalabilidad Temporal:** $R^2 = 0.998$ (comportamiento estrictamente lineal en $K$).

A manera de conclusión, la escala nominal del problema ($K$) no es el único factor determinante para la viabilidad computacional. Representar un sistema de 50 millones de variables en formato QUBO denso requeriría unos **20 Petabytes** de memoria, lo cual es inabordable. Sin embargo, si la estructura del problema conserva conectividad local o de bajo rango, la representación se reduce a **14 GB en formato bandeado** o **205 GB en MPS ($\chi=16$)**, permitiendo procesar el sistema en tiempo lineal y memoria residente acotada.

Para poder ejecutar este código, asegúrate de instalar los paquetes requeridos ejecutando en la terminal:

pip install numpy pandas psutil

Para ejecutar el estudio de escalabilidad y generar las tablas teóricas y empíricas:

python3 QAF_Escalabilidad.py --out ./escalabilidad_resultados

De este modo, se creará la carpeta escalabilidad_resultados conteniendo los siguientes informes CSV:

    scaling_theory.csv: Comparativa teórica exacta de memoria y operaciones para todas las combinaciones de K, b y χ.

    streaming_benchmark_raw.csv: Datos crudos de tiempo y rendimiento medidos en cada repetición.

    streaming_benchmark_summary.csv: Resumen estadístico agregando tiempo medio, throughput y pico de memoria.

# 4_COORDINACION_MULTIAGENTE

Esta carpeta contiene el archivo del paper `QAF_Validacion_Coordinacion_Juan_Manuel_Goenaga.pdf` y el código de Python `qaf_Coordinacion_multiagente.py`, que implementa un banco de pruebas (*benchmark*) sintético y reproducible para evaluar si un coordinador metacognitivo basado en el **Quantum Analysis Framework (QAF)** mejora la coordinación multiagente bajo incertidumbre, observaciones ruidosas y limitaciones de capacidad.

El benchmark evalúa y compara 6 arquitecturas de coordinación sobre 360 escenarios independientes (combinando 4, 8, 16 y 32 agentes con niveles de incertidumbre del 10%, 35% y 60%, con 30 repeticiones cada uno, sumando 2.160 ejecuciones arquitectónicas):

1. **`Broadcast-All`**: Transmite todas las tareas a todos los agentes (máxima saturación de red).
2. **`Independent`**: Cada agente decide localmente si participa en una tarea basándose en su observación ruidosa.
3. **`Rule-Router`**: Reglas fijas asignan tareas al agente con mejor puntuación local.
4. **`Global-Assignment`**: Baseline centralizado fuerte que utiliza todas las observaciones, fusiona la evidencia por promedio simple y asigna tareas mediante el algoritmo húngaro bajo restricciones de capacidad.
5. **`QAF-Coordinator`**: Coordinador metacognitivo que realiza una consulta selectiva a fuentes informativamente diversas (evitando la redundancia de fuentes correlacionadas), aplica fusión contextual y resuelve la asignación global.
6. **`Oracle`**: Cota superior teórica de enrutamiento que conoce el conjunto mínimo real de capacidades requeridas.

Los resultados medios obtenidos tras las 2.160 ejecuciones del benchmark son los siguientes:

| Arquitectura | Éxito Conjunto (%) | Cobertura Capacidades (%) | Mensajes Medios / Escenario | Latencia Mediana (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Broadcast-All** | 8.21% | 9.48% | 680.00 | 0.05 ms |
| **Independent** | 23.67% | 29.50% | 875.82 | 1.67 ms |
| **Rule-Router** | 32.57% | 39.63% | 719.23 | 0.26 ms |
| **Global-Assignment** | 63.08% | 72.70% | 709.97 | 1.75 ms |
| **QAF-Coordinator** | **66.64%** | **76.41%** | **213.88** | 15.10 ms |
| **Oracle** | 66.74% | 76.60% | 29.98 | 0.93 ms |

Si se realiiza una comparativa estadística pareada (`QAF-Coordinator` vs `Global-Assignment`):
* **Mejora Absoluta de Éxito:** +3.56 puntos porcentuales (66.64% vs 63.08%, $p = 3.49 \times 10^{-21}$).
* **Mejora Relativa de Éxito:** **+5.64%** global (escalando hasta **+17.88%** bajo alta incertidumbre del 60%).
* **Reducción de Tráfico de Mensajes:** **-69.87%** (de 709.97 a 213.88 mensajes por escenario).
* **Reducción de Conflictos de Evidencia:** **-24.11%**.
* **Reducción de Asignaciones Innecesarias:** **-63.85%**.

`QAF-Coordinator` demuestra que una coordinación efectiva no requiere difundir toda la información a todos los agentes. Mediante la selección metacognitiva de fuentes y el filtrado de consensos engañosos producidos por agentes correlacionados, logra prácticamente la misma tasa de éxito que el `Oracle` (66.64% vs 66.74%) reduciendo cerca del 70% del tráfico de red. Su ventaja frente a la asignación global clásica es especialmente pronunciada a mayor nivel de ruido e incertidumbre.

Asegúrate de tener instalados los paquetes requeridos ejecutando en la terminal:

pip install numpy pandas scipy matplotlib

Para ejecutar el benchmark de coordinación multiagente:

python3 QAF_Coordinacion_multiagente.py

De este modo, se crearán automáticamente dos carpetas:

    results/: Archivos CSV con métricas detalladas (raw_results.csv, summary.csv, uncertainty_summary.csv, qaf_vs_global_statistics.csv).

    figures/: Las 10 gráficas de resultados exportadas en PNG (fig01_architecture.png a fig10_heatmap.png).

# 5_PATRONES_OCULTOS

Esta carpeta contiene el archivo del paper `QAF_Validacion_Patrones_Ocultos_Juan_Manuel_Goenaga.pdf` y el código de Python `QAF_Patrones_ocultos.py`, que implementa el banco de pruebas (*benchmark*) para evaluar si el **Quantum Analysis Framework (QAF)** puede identificar **patrones relacionales ocultos** cuya información reside en relaciones contextuales de fase y no en las variables observables de primer orden.

El código evalúa tres pistas de pruebas (*tracks*) combinando algoritmos tradicionales, baselines relacionales y representaciones de coherencia QAF:

1. **Track A (Recuperación no supervisada de regímenes latentes):** Se evalúan 3 familias donde las distribuciones marginales gaussianas de las variables son idénticas (`phase4`, `phase_gain`, `context_decoy`).
2. **Track B (Detección de anomalías de coherencia):** Identificación no supervisada de observaciones que rompen la coherencia o rotación de fase esperada.
3. **Track C (Control visible / Control negativo):** Clusters con separación explícita de medias (`visible_control`) para comprobar que el benchmark no favorece a QAF por construcción en datos estándar.

Los resultados obtenidos tras la ejecución completa del benchmark (448 ejecuciones de clustering y 80 de anomalías) son los siguientes:

1. Recuperación de Patrones Ocultos (Track A)

| Método | ARI (Rand Index) | NMI (Info. Mutua) | Exactitud () | Latencia (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **QAF-Coherence** | **0.7672** | **0.7355** | **90.12%** | 26.86 ms |
| **QAF-NoPhase** (Ablación sin fase) | 0.4341 | 0.4564 | 63.08% | 28.58 ms |
| **Context-Relational** (Baseline fuerte) | 0.2513 | 0.2873 | 50.84% | 32.64 ms |
| **QAF-NoContext** (Ablación sin contexto) | 0.0586 | 0.0718 | 40.09% | 24.98 ms |
| **Spectral-Raw** | 0.0011 | 0.0080 | 29.12% | 124.29 ms |
| **Raw-KMeans** | 0.0008 | 0.0076 | 28.73% | 27.18 ms |
| **PCA-KMeans** | -0.0005 | 0.0061 | 28.54% | 112.07 ms |

2. Detección de Anomalías de Coherencia (Track B)

| Método | ROC-AUC | Precision Media (AP) | Latencia (ms) |
| :--- | :---: | :---: | :---: |
| **QAF-Coherence-GMM** | **0.8218** | **0.3400** | 226.19 ms |
| **IsolationForest-Relational** | 0.6362 | 0.1162 | 203.87 ms |
| **LOF-Raw** | 0.5109 | 0.0834 | 12.48 ms |
| **OneClassSVM-Raw** | 0.5079 | 0.0830 | 38.18 ms |
| **IsolationForest-Raw** | 0.5043 | 0.0839 | 191.86 ms |

3. Control Visibilidad / Control Negativo (Track C)

| Método | ARI | NMI | Exactitud () | Latencia (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **PCA-KMeans** | **0.8921** | **0.8725** | **95.68%** | 113.54 ms |
| **Raw-KMeans** | 0.6437 | 0.6558 | 74.36% | 24.53 ms |
| **Spectral-Raw** | 0.0442 | 0.0533 | 35.69% | 128.46 ms |
| **QAF-Coherence** | 0.0014 | 0.0084 | 29.00% | 29.95 ms |

Se han obtenido las siguientes estadísticas primarias de validación:

* **Ventaja Relativa en Clustering:** **+205.32%** de mejora en ARI respecto al baseline relacional fuerte (0.7672 vs 0.2513).
* **Tasa de Victoria en Patrones Ocultos:** **100.00%** de los casos pareados ganados por QAF ($p = 3.55 \times 10^{-15}$, test de Wilcoxon).
* **Ventaja en Detección de Anomalías:** **+0.1855** en ROC-AUC (0.8218 vs 0.6362) con un **100.00%** de tasa de victoria ($p = 1.53 \times 10^{-5}$).

Se han obtenido las siguientes conclusiones de este experimento:

* **Importancia de la Fase ($c_I$):** Eliminar la componente antisimétrica de orientación (`QAF-NoPhase`) degrada el ARI de **0.7672 a 0.4341**, demostrando que el sentido de transformación contiene la información latente del patrón.
* **Importancia del Contexto:** Eliminar la selección contextual (`QAF-NoContext`) provoca el colapso del ARI a **0.0586**, ya que el promediado indiscriminado mezcla subespacios irrelevantes o engañosos.
* **Validez del Control Negativo:** En patrones con medias separadas explícitamente, `PCA-KMeans` obtiene un ARI de **0.8921** mientras QAF cae a **0.0014**, confirmando que QAF no está diseñado para ganar universalmente en datos simples, sino específicamente en estructura relacional de fase.


Asegúrate de instalar los paquetes requeridos ejecutando en la terminal[cite: 58]:

pip install numpy pandas scipy scikit-learn matplotlib

Para ejecutar la suite completa de pruebas de patrones ocultos:   

Bashpython3 QAF_Patrones_ocultos.py

El script creará automáticamente dos subcarpetas en el directorio actual:   results/: CSV con los datos crudos y resúmenes estadísticos (clustering_raw.csv, anomaly_raw.csv, hidden_pattern_summary.csv, primary_statistics.csv).   figures/: Las 10 gráficas de visualización exportadas en alta resolución (fig01_architecture.png a fig10_heatmap.png).
