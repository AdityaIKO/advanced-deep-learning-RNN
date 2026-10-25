# Lecture analysis: "Recurrent Neural Network (RNN) & LSTM" (60 slides)

## 1. How the deck is built

| Slides | Block | Key idea |
|---|---|---|
| 1–7 | **Motivation & RNN definition** | Feed-forward nets assume independent inputs. Language does not ("the man who wore a wig on his head went inside": *who* went inside?). An RNN keeps a hidden state $s_t = f(Ux_t + Ws_{t-1})$ with $s_{-1}=0$, output $o_t=\mathrm{softmax}(Vs_t)$. The same $U, V, W$ are shared over time, so there are far fewer parameters. |
| 7 | **I/O topologies** (Karpathy) | one-to-one (classification), one-to-many (captioning), many-to-one (sentiment), many-to-many delayed (**translation**, which is the assignment), many-to-many synced (frame labelling). |
| 8–13 | **Applications** | language modelling and text generation ($o_t = x_{t+1}$), machine translation ("Echt dicke Kiste" → "Awesome sauce"), speech, image captioning. |
| 14–22 | **Training: BPTT** | Loss $E=\sum_t -y_t\log\hat y_t$. $\partial E_3/\partial V = (\hat y_3-y_3)\otimes s_3$ is local, but $\partial E_3/\partial W=\sum_{k=0}^{3}\frac{\partial E_3}{\partial \hat y_3}\frac{\partial \hat y_3}{\partial s_3}\frac{\partial s_3}{\partial s_k}\frac{\partial s_k}{\partial W}$ needs the chain rule back through every earlier step. |
| 23–25, 29–33 | **Vanishing / exploding gradients** | $\frac{\partial s_3}{\partial s_k}=\prod_{j=k+1}^{3}\frac{\partial s_j}{\partial s_{j-1}}$. Each Jacobian contains $\tanh' \le 1$ and $W$. If its norm is below 1 the product shrinks exponentially (vanishing, Pascanu et al. 2013). If it is above 1 the product grows (exploding, fixed by **gradient clipping**). Result: "no significant learning" of long-range dependencies. |
| 26–27 | **Practical: TensorFlow char-RNN** | "hell" → "ello" with a 4-letter vocabulary and 3 hidden units. Train the right output (green) to be high. |
| 28, 34–43 | **LSTM** | cell state $C_t$ plus three sigmoid gates: forget $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$, input $i_t$ with candidate $\tilde C_t=\tanh(\cdot)$, output $o_t$. $C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t$ and $h_t=o_t\odot\tanh(C_t)$. |
| 44–49 | **Why LSTM fixes it (BPTT in LSTM)** | $\partial C_t/\partial C_{t-1}\approx f_t$ (plus smaller terms). The gradient product becomes a product of forget-gate activations, which the network can keep ≈ 1. That is an **additive path** through time (the "constant error carousel"), so gradients do not vanish. |
| 50–52 | **GRU** (Cho et al. 2014) | two gates: update $z$ (how much old memory to keep) and reset $r$ (how to mix the new input with the old state). No separate cell state. Fewer parameters (3 weight blocks vs 4), so it is faster. Performance is comparable to LSTM. LSTM may win when there is plenty of data. |
| 53–57 | **Seq2Seq** | **Encoder** $h_t=f(W^{hh}h_{t-1}+W^{hx}x_t)$ reads the source. The **encoder vector** (the final hidden state) is the decoder's initial state. **Decoder** $h_t=f(W^{hh}h_{t-1})$ with $y_t=\mathrm{softmax}(W^S h_t)$. It maps sequences of **different lengths**. |
| 58 | **BiLSTM** | forward plus backward LSTM, outputs combined, so every position sees both past and future context. |
| 59 | **Assignment** | MT on WMT13 News Commentary DE-EN with seq2seq. Compare RNN / LSTM / GRU on BLEU, convergence (loss graphs) and computation time. |

## 2. The connecting thread
1. Sequences need **memory**, so an RNN with weights shared over time.
2. Training it with BPTT multiplies many Jacobians, so the gradient **vanishes or explodes**.
3. Exploding is fixed by clipping. Vanishing needs an **architectural** fix: gates and an additive cell path (**LSTM**, simplified as the **GRU**).
4. With cells that remember, you can build **encoder-decoder** systems that map one sequence to another (**translation**).
5. The assignment closes the loop. Build (4) and check empirically whether (2) and (3) show up in BLEU, loss curves and training time.

## 3. Points the slides leave implicit (useful for the report and the exam)
* **Notation drift.** Slides 5–22 use $U,W,V,s_t$, slides 29–33 use $W=[W_c,W_x]$ and $C_t$ for the RNN state, and slides 35–49 use $C_t$ for the LSTM cell. Say which convention you use.
* **Why the RNN Jacobian is ≤ 1.** $\partial s_j/\partial s_{j-1}=\mathrm{diag}(\tanh'(z_j))\,W$ with $\tanh'\in(0,1]$. If the largest singular value of $W$ is $< 1/\gamma$ (where $\gamma$ bounds $\tanh'$), the product shrinks exponentially, which is the sufficient condition in Pascanu et al.
* **The LSTM does not remove vanishing completely.** It makes it *controllable*: if $f_t \to 0$ the gradient is still cut, by design (the network chose to forget).
* **Information bottleneck.** The slide-55 encoder vector is a fixed size whatever the sentence length, so plain seq2seq gets worse on long sentences. This motivated **attention** (Bahdanau 2015), which the deck does not cover and which is the natural "future work" in the report.
* **Teacher forcing vs inference.** During training the decoder gets the true previous word. At test time it gets its own prediction (exposure bias), so greedy or beam search is needed.
* **Parameter count per layer.** RNN $H(E+H+1)$, GRU $3H(E+H+1)$, LSTM $4H(E+H+1)$. This predicts the ordering of compute time in the assignment.

## 4. Likely exam / viva questions
1. Derive $\partial E_3/\partial W$ for a simple RNN and show where the vanishing term comes from.
2. Why does gradient clipping fix exploding but not vanishing gradients?
3. Write the LSTM equations and explain each gate's job. Show that $\partial C_t/\partial C_{t-1}\approx f_t$.
4. GRU vs LSTM: differences, parameter counts, and when to prefer each.
5. Draw a seq2seq model. What is the encoder vector? What are its limitations, and how does attention address them?
6. What is BLEU? Why is it computed at corpus level, and what does the brevity penalty do?
