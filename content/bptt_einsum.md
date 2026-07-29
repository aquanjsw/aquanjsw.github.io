---
title: Derivation of BPTT using Einsum
---

Same as the analysis in standard back propagation, to compute $\frac{\partial \mathcal L_t}{\partial \mathbf U}$, we compute partial derivatives $\frac{\partial \mathcal L_t}{\partial u_{ij}}$ for the elements of $\mathbf U$ first:

$$
\frac{\partial \mathcal L_t}{\partial u_{ij}}
= \sum_{k=1}^t \frac{\partial \mathbf z^k}{\partial u_{ij}} \frac{\partial \mathcal L^t}{\partial \mathbf z^k}
$$

| Compared to the original representation $\mathcal L_t$ and $\mathbf z_k$, we use $\mathcal L^t$ and $\mathbf z^k$ to avoid conflict with the subscripts for einsum.

Analyzing $\frac{\partial \mathbf z^k}{\partial u_{ij}}$ and $\frac{\partial \mathcal L^t}{\partial \mathbf z^k}$ separatly:

---

For $\frac{\partial \mathbf z^k}{\partial u_{ij}}$:

Given:

$$
\mathbf z^k = \mathbf U \mathbf h^{k-1} + \mathbf W \mathbf x^k + \mathbf b
$$

We can naturally use einsum here:

$$
\begin{align}
\frac{\partial \mathbf z^k}{\partial u_{ij}}
&= \frac{\partial (\mathbf U \mathbf h^{k-1} + \mathbf W \mathbf x^k + \mathbf b)}{\partial u_{ij}} \\
&= \frac{\partial \mathbf U\mathbf h^{k-1}}{\partial u_{ij}} \\
&= \frac{\partial u_{mn}h^{k-1}_n}{\partial u_{ij}}\\
&= \delta_{mi}\delta_{nj}h_n^{k-1} \\
&= \delta_{mi}h_j^{k-1}
\end{align}
$$

where (3) ~ (5) is in einsum representation, some explanations:

- A mark with subscripts like $u_{ij}$ is essentially a scaler, so of course it can be used in non-einsum representations like (2) as usual.
  Such marks also have exactly the same meaning in einsum representations, except that there exist an extra **convention**: when some another mark with similar subscripts is multiplied, they do einsum, just like (3).
-

\frac{\partial \mathcal L*t}{\partial \mathbf U} \overset{\text{einsum}}{=}
\frac{\partial \mathcal L^t}{\partial u*{ij}} = \sum*{k=1}^{t}
\frac{\partial z_m^k}{\partial u*{ij}} \frac{\partial \mathcal L^t}{\partial z*m^k}
= \sum*{k=1}^t \frac{\partial \mathbf z}{\partial u\_{ij}}
= \

$$


---

Given：


$$

z*m^k = u*{mn}h*n^{k-1}+w*{mn}x_n+b_m

$$

We get:


$$

\begin{align}
\frac{\partial z*m^k}{\partial u*{ij}}
&= \frac{\partial (u*{mn}h_n^{k-1}+w*{mn}x*n+b_m)}{\partial u*{ij}} \\
&= \frac{\partial (u*{mn}h_n^{k-1})}{\partial u*{ij}} \\
&= \frac{\partial u*{mn}}{\partial u*{ij}}h*n^{k-1} \\
&= \delta*{mi}\delta*{nj}h_n^{k-1} \\
&= \delta*{mi}h_j^{k-1}
\end{align}

$$

, $z_m^k\in\mathbb R^{D\times 1}$, $u_{ij}\in\mathbb R^{D\times D}$, 所以$\delta_{mi}h_j^{k-1}\in\mathbb R^{1\times D}$.
$\delta_{mi}$为[克罗内克函数](https://zh.wikipedia.org/zh-hans/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%CE%B4%E5%87%BD%E6%95%B0), 所以输出的行向量只有第$m=i$个是非零元素
$$
