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

# 3_ESCALABILIDAD


