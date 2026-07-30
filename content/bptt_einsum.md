---
title: Derivation of BPTT using Einstein Notation
---

Similarly as in the standard backpropagation derivation, to compute $\frac{\partial \mathcal L^t}{\partial \mathbf U}$, we first compute partial derivatives $\frac{\partial \mathcal L^t}{\partial u_{ij}}$ for the elements of $\mathbf U$:

$$
\begin{equation}
\frac{\partial \mathcal L^t}{\partial u_{ij}}
= \sum_{k=1}^t \frac{\partial \mathbf z^k}{\partial u_{ij}} \frac{\partial \mathcal L^t}{\partial \mathbf z^k}
\label{eq:root}
\end{equation}
$$

> Compared to the original representation $\mathcal L^t$ and $\mathbf z_k$, we use $\mathcal L^t$ and $\mathbf z^k$ to avoid conflicting with the subscripts for using einsum.

Analyze $\frac{\partial \mathbf z^k}{\partial u_{ij}}$ and $\frac{\partial \mathcal L^t}{\partial \mathbf z^k}$ separately:

---

For $\frac{\partial \mathbf z^k}{\partial u_{ij}}$:

Given:

$$
\begin{equation}
\mathbf z^k = \mathbf U \mathbf h^{k-1} + \mathbf W \mathbf x^k + \mathbf b \label{eq:z}
\end{equation}
$$

We can naturally use einsum here:

$$
\begin{align}
\frac{\partial \mathbf z^k}{\partial u_{ij}}
&= \frac{\partial (\mathbf U \mathbf h^{k-1} + \mathbf W \mathbf x^k + \mathbf b)}{\partial u_{ij}} \\
&= \frac{\partial \mathbf U\mathbf h^{k-1}}{\partial u_{ij}} \label{eq:a1} \\
&= \text{einsum}(\frac{\partial u_{mn}h^{k-1}_n}{\partial u_{ij}}) \label{eq:a2} \\
&= \text{einsum}(\frac{\partial u_{mn}}{\partial u_{ij}} h^{k-1}_n) \label{eq:a3} \\
&= \text{einsum}(\delta_{mi}\delta_{nj}h_n^{k-1}) \label{eq:a4} \\
&= \text{einsum}(\delta_{mi}h_j^{k-1}) \label{eq:a5}
\end{align}
$$

in which, $\text{einsum}(*)$ means using the explanation of Einstein summation convention (**einsum explanation** for short). $\delta$ is the [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta) function.

Note that **there is no einsum representation**, the formula inside $\text{einsum}(*)$ has no special meaning.

For example, the inner formula of $\eqref{eq:a2}$, i.e. $\frac{\partial u_{mn}h^{k-1}_n}{\partial u_{ij}}$, can just be explained as a normal derivative that differentiate a scalar $u_{mn}$ with coefficient $h^{k-1}_n$ w.r.t. scalar $u_{ij}$, the result is of course a scalar.

But such a formula has an **extra** explanation in terms of einsum: we do exactly the same operation above for each _term_ of $\mathbf U\mathbf h^{k-1}$.

> Note that we use _term_ instead of _element_, that's because an _element_ (which is in fact each component of the result column vector, i.e. $\sum_{n=1}^Du_{mn}h_{n}^{k-1}$, in which $D$ is the dimension of vector $\mathbf h$ or vector $\mathbf z$) consists of multiple _terms_.

And during einsum usage, the explanations for each components of the formula are not rigidly fixed, but rather adapt dynamically to the context. For example, the numerator of $\eqref{eq:a2}$ follows einsum explanation, i.e. a term of matrix, while the denominator is just a scalar.

After we simplified the _term_ formula to some understanding-friendly form, e.g. $\eqref{eq:a5}$ here, we can then deduce the matrix form in reverse easily according to some simple clues:

- The result is a row vector (denominator layout) with index $m$ according to $\eqref{eq:a1}$ and $\eqref{eq:a2}$;
- Only the $m=i$-th column has possible non-zero value $h_j^{k-1}$ according to $\eqref{eq:a5}$.

So the final result is a one-hot row vector with the $m=i$-th element being the only possible non-zero value $h_j^{k-1}$:

$$
\begin{equation}
(\frac{\partial \mathbf z^k}{\partial u_{ij}})_m
= \delta_{mi}h_j^{k-1} \label{eq:p1}
\end{equation}
$$

---

For $\frac{\partial \mathcal L^t}{\partial \mathbf z^k}$:

As loss function can be arbitrary, we will focus on uncovering the iterative pattern.

Given $\eqref{eq:z}$ and:

$$
\mathbf h^k = f(\mathbf z^k)
$$

> Note that $f$ is an element-wise mapping, i.e.
>
> $$
> h_i^k = f(z_i^k)
> $$

we have:

$$
\begin{align*}
\epsilon^{t,k}=
\frac{\partial \mathcal L^t}{\partial \mathbf z^k}
&= \frac{\partial \mathbf h^k}{\partial \mathbf z^k} \frac{\partial \mathbf z^{k+1}}{\partial \mathbf h^k} \frac{\partial \mathcal L^t}{\partial \mathbf z^{k+1}} \\
&= \text{einsum}(\frac{\partial h_m^k}{\partial z_i^k} u_{nm} \epsilon_{n}^{t,k+1}) \\
&= \text{einsum}\left(\delta_{mi}f'(z_i^k)u_{nm}\epsilon_{n}^{t,k+1}\right) \\
&= \text{einsum}\left(f'(z_i^k)\sum_n u_{ni}\epsilon_n^{t,k+1}\right)
\end{align*}
$$

in which, $\text{einsum}\left(\delta_{mi}f'(z_i^k)\right)$ is a diagonal matrix with entries $f'(\mathbf z^k)$.

Similar to $\eqref{eq:p1}$, rewrite the above equation to:

$$
\begin{equation}
\epsilon_i^{t,k} = f'(z_i^k)\sum_n u_{ni}\epsilon_n^{k+1} \label{eq:epsilon}
\end{equation}
$$

---

With $\eqref{eq:p1}$ and $\eqref{eq:epsilon}$, we can substitute them into $\eqref{eq:root}$:

$$
\begin{align}
\frac{\partial \mathcal L^t}{\partial u_{ij}}
&= \sum_{k=1}^t \text{einsum}\left((\frac{\partial \mathbf z^k}{\partial u_{ij}})_m \epsilon_m^{t,k}\right) \label{eq:b1} \\
&= \sum_{k=1}^t \text{einsum}\left(\delta_{mi}h_j^{k-1}\epsilon_m^{t,k}\right) \\
&= \sum_{k=1}^t \text{einsum}\left(h_j^{k-1}\epsilon_i^{t,k}\right)
\end{align}
$$

Note that we use index $i$ in $\eqref{eq:epsilon}$ for $\epsilon^{t,k}$ while using index $m$ in $\eqref{eq:b1}$, that is because an index is just a dimensional index notation for the tensor it attached to, an einsum notation can use different letter indices for the same dimension in different equations as long as the logic is consistent. In $\eqref{eq:b1}$, $\frac{\partial \mathbf z^k}{\partial u_{ij}}$ is a row vector while $\epsilon^{t,k}$ is a column vector, so the index must match.

As a result, we have:

$$
\frac{\partial \mathcal L^t}{\partial \mathbf U} = \sum_{k=1}^t \epsilon^{t,k}(\mathbf h^{k-1})^T
$$
