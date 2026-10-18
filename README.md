# qaf_tensorial
Implementación de QAF mediante redes tensoriales

## Estructura del Repositorio

# Libro completo en latex y pdf

# QAF Benchmark (qaf_benchmark.py): Validación Preliminar de Asignación de Recursos bajo Incertidumbre

Este archivo contiene el banco de pruebas (*benchmark*) sintético y reproducible para evaluar si la representación contextual del **Quantum Analysis Framework (QAF)** mejora la toma de decisiones en problemas de asignación de recursos crítica ante la presencia de incertidumbre y señuelos (*decoys*) engañosos. El código evalúa y compara siete formulaciones de optimización y decisión clasificadas en tres familias principales:

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

Asegúrate de tener instalado Python 3.9 o superior e instala las bibliotecas requeridas ejecutando en la terminal:

pip install numpy pandas scipy matplotlib *numba*

Para ejecutar el benchmark, hay que introducir, en la carpeta donde está el codigo:

python3 qaf_benchmark.py --outdir ./resultados_benchmark

De este modo, se crea la carpeta resultados_benchmark con una subcarpeta que contiene figuras y otra que contiene tables en formato csv para poder ser analizada.



