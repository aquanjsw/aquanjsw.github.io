---
title: Deriving Backpropagation with einsum
---

We use $\frac{\partial \mathcal L}{\partial \mathbf W^l}$ as an example.

Given:

$$
\begin{aligned}
\mathcal L &= \mathcal L(\mathbf z^L) \\
\mathbf z^l &= \mathbf W^l \mathbf a^{l-1} + \mathbf b^l \\
\mathbf a^l &= f(\mathbf z^l)
\end{aligned}
$$

in which, $L$ is the final layer index, $l\in \{1, 2, \dots, L\}$, $f$ is an element-wise activation function.

and further:

$$
\begin{aligned}
z_i^l &= w_{ij}^l a_j^{l-1} + b_i^l \\
\frac{\partial a_j^l}{\partial z_i^l} &= \delta_{ij}f'(z_i^l) \\
\frac{\partial z_i^{l+1}}{\partial a_j^l} &= w_{ij}^{l+1}\\
(\frac{\partial \mathcal L}{\partial \mathbf z^l})_m
&= \frac{\partial \mathcal L}{\partial z_m^l} \\
&= \frac{\partial a_j^l}{\partial z_m^l}\frac{\partial z_i^{l+1}}{\partial a_j^l} \frac{\partial \mathcal L}{\partial z_i^{l+1}} \\
&= \delta_{mj}f'(z_m^l)w_{ij}^{l+1} \frac{\partial \mathcal L}{\partial z_i^{l+1}} \\
&= f'(z_m^l)w_{im}^{l+1}\frac{\partial \mathcal L}{\partial z_i^{l+1}}
\end{aligned}
$$

we have:

$$
\begin{aligned}
(\frac{\partial \mathcal L}{\partial \mathbf W^l})_{ij}
&= \frac{\partial \mathcal L}{\partial w_{ij}^l} \\
&= \frac{\partial z_m^l}{\partial w_{ij}^l} \frac{\partial \mathcal L}{\partial z_m^l} \\
&= \frac{\partial (w_{mn}^la_n^{l-1}+b_m^l)}{\partial w_{ij}^l}f'(z_m^l)w_{om}^{l+1}\frac{\partial \mathcal L}{\partial z_o^{l+1}} \\
&= \delta_{im}\delta_{jn}a_n^{l-1} f'(z_m^l)w_{om}^{l+1}\frac{\partial \mathcal L}{\partial z_o^{l+1}} \\
&= a_j^{l-1}f'(z_i^l)w_{oi}^{l+1} \frac{\partial \mathcal L}{\partial z_o^{l+1}} \\
&= a_j^{l-1}\frac{\partial \mathcal L}{\partial z_i^l}
\end{aligned}
$$

then:

$$
\frac{\partial \mathcal L}{\partial \mathbf W^l} = \frac{\partial \mathcal L}{\partial \mathbf z^l}(\mathbf a^{l-1})^T
$$
