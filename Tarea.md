### ✍️ **Tarea**
---

#### Parte 1 – Ejercicios a mano

**Instrucciones:** Responde las siguientes preguntas con desarrollo completo y utilizando notación matemática clara. Puedes usar esquemas si lo consideras necesario. Sube las fotografías de la solución en una carpeta llamada `parte_1`.

1. Sea una RNN definida por la ecuación:

   $$
   h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)
   $$

   Supón que $W_{hx} \in \mathbb{R}^{2 \times 1}$, $W_{hh} \in \mathbb{R}^{2 \times 2}$, $b_h = 0$, y que los valores iniciales son:

   $$
   h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad x_1 = 1, \quad x_2 = 2
   $$

   con:

   $$
   W_{hx} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.5 & 0.1 \\\\ 0.2 & 0.4 \end{bmatrix}
   $$

   a) Calcula $h_1$ y $h_2$, utilizando la función tangente hiperbólica redondeada a 3 cifras decimales.

   b) Explica en tus palabras qué significa que la red “comparte pesos a lo largo del tiempo”.

---

2. Considera las siguientes fórmulas de una celda LSTM:

   $$
   \begin{aligned}
   f_t &= \sigma(W_f x_t + U_f h_{t-1}) \\\\
   i_t &= \sigma(W_i x_t + U_i h_{t-1}) \\\\
   \tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1}) \\\\
   c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\\\
   o_t &= \sigma(W_o x_t + U_o h_{t-1}) \\\\
   h_t &= o_t \odot \tanh(c_t)
   \end{aligned}
   $$

   Supón que $x_t = 1$, $h_{t-1} = 0$, $c_{t-1} = 0$, y que todos los pesos son escalares e iguales a 1. Calcula manualmente $f_t, i_t, \tilde{c}_t, c_t, o_t, h_t$. Usa:

   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
   $$

   Redondea los resultados a 3 cifras decimales. ¿Qué observas respecto a la activación final?

---
3. Usando la fórmula de atención escalar de producto:

   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$

   Supón que:

   $$
   Q = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad
   K = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}, \quad
   V = \begin{bmatrix} 10 \\\\ 20 \end{bmatrix}, \quad d_k = 2
   $$

   a) Calcula $QK^T$, luego $\frac{QK^T}{\sqrt{2}}$

   b) Aplica la función softmax y multiplica por $V$ para obtener la salida de la atención.

---
#### Parte 2 – Código

Diseña un Jupyter Notebook que ilustre, mediante ejemplos implementados en PyTorch, el funcionamiento de las arquitecturas RNN, LSTM y Transformer. Guarda el notebook y todos los archivos relacionados en una carpeta llamada `parte_2`.
