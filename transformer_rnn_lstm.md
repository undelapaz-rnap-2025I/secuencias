
# Modelado de Secuencias en PLN: De las RNN a los Transformers

Este documento presenta una introducción al modelado de secuencias mediante redes neuronales recurrentes (RNN), Long Short-Term Memory (LSTM) y Transformers, con aplicaciones en procesamiento de lenguaje natural (PLN).

---

## 🧠 Red Neuronal Recurrente (RNN)

Una red neuronal recurrente procesa secuencias entrada por entrada. Cada elemento de la secuencia $x_t$ es procesado junto con un estado oculto anterior $h_{t-1}$, y produce una salida $y_t$ y un nuevo estado oculto $h_t$.

![RNN](figs/rnn.png)  
*Fuente: Figura adaptada de [MDPI Information](https://www.mdpi.com/2078-2489/15/9/517)*

### 📐 Ecuaciones de la RNN

$$
h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)
$$
$$
y_t = W_{hy} h_t + b_y
$$

donde:
- $x_t \in \mathbb{R}^d$: entrada en el tiempo $t$,
- $h_t \in \mathbb{R}^h$: estado oculto,
- $y_t \in \mathbb{R}^o$: salida,
- $W_{hx}, W_{hh}, W_{hy}$: matrices de pesos,
- $b_h, b_y$: vectores de sesgo.

---

## 🔁 Long Short-Term Memory (LSTM)

Las celdas LSTM permiten preservar información durante lapsos de tiempo largos, mitigando los problemas de las RNN estándar.

![LSTM](figs/lstm.png)  
*Fuente: Figura adaptada de [MDPI Information](https://www.mdpi.com/2078-2489/15/9/517)*

### 🔣 Ecuaciones de la celda LSTM

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

donde:
- $\sigma$: función sigmoide,
- $\odot$: producto elemento a elemento.

---

## ✨ Transformers

Los Transformers eliminan completamente la recurrencia y permiten el paralelismo total, basándose en mecanismos de atención.

![Transformer](figs/transformer.png)  
*Fuente: Figura adaptada de [Attention is All You Need](https://arxiv.org/pdf/1706.03762)*

### 🔍 Atención Escalar de Producto

Dado un conjunto de consultas $Q$, claves $K$ y valores $V$, la atención se calcula como:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

donde $d_k$ es la dimensión de las claves.

### ⚙️ Arquitectura General

- **Codificador (Encoder):** Consiste en capas de atención multi-cabeza + redes feedforward.
- **Decodificador (Decoder):** Agrega atención enmascarada y conexiones con la salida del codificador.
- **Positional Encoding:** Suma a los embeddings para incluir el orden secuencial.

Los Transformers son la base de modelos modernos de lenguaje como BERT y GPT.

---

