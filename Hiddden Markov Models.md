# Hiddden Markov Models

#### REF : https://www.youtube.com/watch?v=P02Lws57gqM

## Parameters

확률을 담고 있는 행렬

$A\left(a_{i j}\right)$ hidden state가 전이될 확률

$B\left(b_{j k}\right)$ 특정 hidden state에서 특정 관측치를 방출할 확률

$\pi\left(\pi_{i}\right)$ 각 state가 시작될 확률



**State transition probability**

$$
a_{i j}=\mathrm{p}\left(q_{t+1}=s_{j} | q_{t}=s_{i}\right), 1 \leq i, j \leq n
$$

t 시점에서 i번쨰 상태에서 j번째  상태로 전이될 확률

$\sum_{j=1}^{n} a_{i j}=1$ 



**Emission probability**

$b_{j}\left(v_{k}\right)$ = 은닉 상태 $b_{j}$ 에서 관측치 $v_{k}$가 도출될 확률

$$
b_{j}\left(v_{k}\right)=P\left(o_{t}=v_{k} | q_{t}=s_{j}\right), 1 \leq j \leq n, 1 \leq k \leq m
$$

t 시점에서 hidden state j에 있는데, $v_k$를 관측할 확률

$\sum_{j=1}^{n} b_{j}\left(v_{k}\right)=1$ 



**Initial state probability**

$\pi_{i}$ = $s_i$에서 시작할 확률

$\sum_{i=1}^{n} \pi_{i}=1$



## Main Problems

**Evaluation Problem**

HMM 모델$\left(\lambda^{*}\right)$과 시퀸스 O를 주었을 때, 시퀸스 O가 관측될 확률을 구하는 문제.

`Forward algorithm`, `Backward algorithm` 사용



### Decoding problem

모델이 주어지고, 관측가능한 시퀸스O가 주어졌을 때, 가장 그럴싸한 hidden state S를 예측하는 것

`Viterbi algorithm` 사용

람다는 hmm 파라메터

q =  hidden state sequence, o = observable state sequence



### Viterbi Algorithm

관측된 사건들의 순서를 야기한 가장 가능성 높은 은닉 상태들의 순서를 찾기 위한 동적 계획법 알고리즘

![gif](https://t1.daumcdn.net/cfile/tistory/235BB84B512F358118)

$$
v_{t}(i)=_{q_{1}, q_{2}, \cdots, q_{t-1}} \max _{p}\left(o_{1}, o_{2}, \cdots, o_{t}, q_{1}, q_{2}, \cdots, q_{t-1}, q_{t}=s_{i} | \lambda\right)
$$

파라메터$λ$ 가 주어졌을 때 q와 o를 최대화 하는 hidden state를 찾는 것

$$
=\left[\begin{array}{c}

\max \\
1 \leq j \leq n
\end{array} v_{t-1}(j) a_{j i}\right] \cdot b_{i}\left(o_{t}\right)
$$

$$
v_{1}(i)=\pi_{i} b_{i}\left(o_{1}\right)
$$

$$
v_{t}(i)=\left[\begin{array}{c}
\left.\max _{1 \leq j \leq n} v_{t-1}(j) a_{j i}\right] \cdot b_{i}\left(o_{1}\right), 2 \leq t \leq T, 1 \leq i \leq n

\end{array}\right.
$$

$$
\tau_{t}(i)=\underset{1 \leq j \leq n}{\operatorname{argmax}}\left[v_{t-1}(j) a_{j i}\right], 2 \leq t \leq T, 1 \leq i \leq n
$$

각 t 시점에서 `argmax`를 취해서 어떤 시점의 viterbi probability가 컸는지 찾는다.

$$
\hat{q}_{T}=\underset{1 \leq j \leq n}{\operatorname{argmax}} \, v_{T}(j)
$$

$$
\hat{q}_{t}=\tau_{t+1}\left(\hat{q}_{t+1}\right), t=T-1 . T-2, \cdots, 1
$$

$$
\hat{Q}_{t}=\left(\hat{q}_{1}, \hat{q}_{2}, \cdots, \hat{q}_{t}\right)
$$

Forward algorithm는 가능한 모든 경우의 확률의 합

$$
\alpha_{t}(i)=\left[\sum_{j=1}^{n} \alpha_{t-1}(j) a_{j i}\right] \cdot b_{i}\left(o_{t}\right)
$$

Viterbi algorithm은 가능한 모든 경우의 확률의 최대

$$
v_{t}(i)=\left[\begin{array}{c}
\max \\
1 \leq j \leq n
\end{array} \chi_{t-1}(j) a_{j i}\right] \cdot b_{i}\left(o_{t}\right)
$$

### Learning problem

1. 관측 벡터 O의 확률을 최대로 하는 파라메터를 찾자.

2. 여러 개 관측스퀸스를 주면 최적의 파라메터를 찾자.

$$
\mathrm{HMM}\left(\lambda^{*}\right)=\operatorname{argmax}_{\lambda} \mathrm{P}(\mathrm{O} | \lambda)
$$

**Maximum likehood method**

데이터를 기반으로 확률변수의 파라메터를 구하는 방법

어떤 모수가 주어졌을 때, 원하는 값들이 나올 확률을 최대로 만드는 파라메터를 선택



input = HMM 아키텍처

output = 파라메터 (A,B,pie)



**Baum-Welch Algorithm**

파라메터(a,b,pie)에서 감마, 크사이를 구하는 E-step과 감마와 크사이로 a,b,pie를 업데이트 하는 M-step을 반복하며 학습 

![img](C:\Users\root\Desktop\1.PNG)

출처 - ppt 일부 중 발췌

$\gamma_{t}(i)$ = t 시점 상태가 $s_i$일 확률

$\xi_{t}(i, j)$ = t시점 상태가 $s_i$, t+1 시점 상태가 $s_j$일 확률



### E-step

$a_t(i) = \left[\sum_{j=1}^{n} \alpha_{\mathrm{t}-1}(j) a_{j i}\right] \cdot b_{i}\left(o_{t}\right)$

$a_t(i)$ = forward prob

$b_t(i)=\left[\sum_{j=1}^{n} a_{i j} b_{i}\left(o_{t}\right) \beta_{t}(j)\right]$

$b_t(i)$ = backward prob

$a_t(i)b_t(i)$ = t 번째 시점에서 상태 i를 지나는 모든 경로의 해당하는 확률의 합



$\gamma_{t}(i) = p\left(q_{t}=s_{i} | \mathrm{O}, \lambda\right)$

$$
\gamma_{t}(i) = \frac{\alpha_{\mathrm{t}}(i) \beta_{t}(i)}{\sum_{j=1}^{n} \alpha_{\mathrm{t}}(j) \beta_{t}(j)}
$$

즉 $s_i$일 확률 / $S_{1 \sim n}$ 일 확률을 모두 더한 값이다.

$$
\xi_{t}(i, j)=\frac{\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}
$$

i와 j상태를 연결한 $a_{i j} b_{j}\left(o_{t+1}\right)$를 갖는다.

![img](C:\Users\root\Desktop\2.PNG)



### M-step

$\boldsymbol{\pi}_{\boldsymbol{i}}^{\boldsymbol{n e w}}$ = t가 1일 때(시작 할 때), $s_i$에 있을 확률 

즉, $\gamma_{t=1}(i)$



$\boldsymbol{a}_{i j}^{\boldsymbol{n e w}}$ = $s_i$에서 $s_j$로 전이할 기대값 / $s_i$에서 전이할 기대값

$$
\boldsymbol{a}_{i j}^{\boldsymbol{n e w}}=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)}
$$

$\boldsymbol{b}_{i}\left(\boldsymbol{v}_{\boldsymbol{k}}\right)^{\text {new }}$ = $s_i$에서 $v_k$를 관측할 확률 / $s_i$에 있을 확률

$$
\boldsymbol{b}_{i}\left(\boldsymbol{v}_{\boldsymbol{k}}\right)^{\text {new }}=\frac{\sum_{t=1, s t . o_{t}=v_{t}}^{T} \gamma_{t}(i)}{\sum_{t=1}^{T} \gamma_{t}(i)}
$$


