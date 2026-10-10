# . CNN: 이미지의 Local Pattern을 학습하다

## 1.1 CNN은 왜 등장했을까?

이미지를 일반적인 fully connected network에 바로 넣는다고 생각해보자.

예를 들어

$$
224\times224\times3
$$

RGB 이미지가 있다.

전체 input dimension은

$$
224\times224\times3=150,528
$$

이다.

이걸 hidden neuron 4096개와 fully connected로 연결하면 weight 수가

$$
150,528\times4096
$$

가 된다.

약

$$
616\text{ million}
$$

개의 weight가 필요하다.

즉 parameter가 지나치게 많다.

하지만 이미지는 중요한 특징이 있다.

고양이 귀를 찾기 위해 이미지 전체 pixel을 한 번에 볼 필요가 없다.

먼저 주변 pixel들의

$$
\boxed{\text{Local Pattern}}
$$

을 보면 된다.

예를 들어

- edge
- corner
- texture
- curve

같은 것들이다.

이러한 이미지의 spatial structure를 활용하도록 만든 것이 CNN이다.

논문은 CNN을 feedforward neural network로 설명하며, 낮은 layer에서는 기본 edge를, 깊은 layer에서는 object 수준의 특징을 계층적으로 추출한다고 설명한다. 또한 MLP와 비교했을 때 weight sharing과 pooling이 중요한 차이라고 정리한다.     1. A survey on deep learning fu…

---

# 1.2 CNN의 핵심: Convolution

CNN의 가장 중요한 operation은

$$
\boxed{\text{Convolution}}
$$

이다.

입력 이미지 위에 작은 kernel을 이동시키면서 local pattern을 찾는다.

예를 들어 input image가

$$
5\times5
$$

이고 kernel이

$$
3\times3
$$

이라고 하자.

kernel:

$$
K= \begin{bmatrix} k_{11}&k_{12}&k_{13}\\ k_{21}&k_{22}&k_{23}\\ k_{31}&k_{32}&k_{33} \end{bmatrix}
$$

입력의 local 영역:

$$
X= \begin{bmatrix} x_{11}&x_{12}&x_{13}\\ x_{21}&x_{22}&x_{23}\\ x_{31}&x_{32}&x_{33} \end{bmatrix}
$$

이라면 convolution 결과 하나는 개념적으로

$$
y= \sum_{i=1}^{3} \sum_{j=1}^{3} k_{ij}x_{ij} +b
$$

가 된다.

그리고 kernel을 옆으로 이동시키면서 같은 연산을 반복한다.

---

# 1.3 Weight Sharing

CNN의 아주 중요한 특징이다.

같은 kernel을 이미지 전체에서 사용한다.

즉 왼쪽 위를 볼 때 사용한

$$
3\times3
$$

kernel과 오른쪽 아래를 볼 때 사용하는 kernel이 같다.

이를

$$
\boxed{\text{Weight Sharing}}
$$

이라고 한다.

논문도 CNN이 local region에서 학습된 weight를 이미지 전체에 공유하여 parameter 수를 줄인다고 설명한다.     1. A survey on deep learning fu…

이게 중요한 이유는 간단하다.

예를 들어 vertical edge를 찾는 filter가 있다고 하자.

vertical edge는 이미지 왼쪽에 나타나든 오른쪽에 나타나든 같은 edge다.

따라서 위치마다 다른 filter를 만들 필요가 없다.

---

# 1.4 Feature Map

하나의 kernel을 이미지 전체에 적용하면

$$
\boxed{\text{Feature Map}}
$$

이 생성된다.

예를 들어

- Kernel 1 → vertical edge
- Kernel 2 → horizontal edge
- Kernel 3 → texture
- Kernel 4 → curve

처럼 서로 다른 pattern에 반응하는 여러 feature map이 만들어질 수 있다.

물론 실제 CNN에서는 사람이 kernel을 직접 설계하는 것이 아니다.

$$
\boxed{\text{Kernel weight 자체를 학습한다}}
$$

즉 backpropagation을 통해

$$
K \leftarrow K-\eta \frac{\partial L}{\partial K}
$$

로 업데이트된다.

---

# 1.5 Channel

이미지는 기본적으로

$$
H\times W\times C
$$

형태다.

RGB image에서는

$$
C=3
$$

이다.

convolution을 거치면 channel 수는 kernel 수에 따라 달라진다.

예를 들어

$$
224\times224\times3
$$

입력에 output channel 64개를 만들면

$$
224\times224\times64
$$

같은 feature tensor가 나온다.

즉 channel 하나하나가 서로 다른 종류의 learned feature를 나타낸다고 생각할 수 있다.

---

# 1.6 CNN의 Hierarchical Feature Learning

CNN을 여러 layer 쌓으면 feature가 점점 추상적으로 변한다.

초기 layer:

$$
\boxed{\text{Edge / Texture}}
$$

중간 layer:

$$
\boxed{\text{Shape / Pattern}}
$$

깊은 layer:

$$
\boxed{\text{Object Part}}
$$

최종 layer:

$$
\boxed{\text{Semantic Object}}
$$

예를 들어 얼굴 recognition이라면

$$
\text{Edge}
$$

↓

$$
\text{Eye / Nose shape}
$$

↓

$$
\text{Face structure}
$$

↓

$$
\text{Identity}
$$

처럼 변한다.

그래서 CNN의 본질은

$$
\boxed{ \text{Local Feature} \rightarrow \text{Hierarchical Feature} \rightarrow \text{Semantic Representation} }
$$

이라고 볼 수 있다.

---

# 2. AlexNet

현대 CNN 발전에서 중요한 모델이

$$
\boxed{\text{AlexNet}}
$$

이다.

논문은 AlexNet이 ILSVRC-2012에서 우승한 대표적인 CNN이며,

$$
\boxed{\text{5 convolution layers + 3 fully connected layers}}
$$

구조를 사용한다고 설명한다.     1. A survey on deep learning fu…

AlexNet의 중요성은 단순히 layer를 많이 쌓았다는 것만이 아니다.

여러 training technique을 함께 적용했다.

---

## 2.1 ReLU

당시에는 sigmoid나 tanh 같은 activation도 많이 사용됐다.

AlexNet은

$$
\boxed{\text{ReLU}}
$$

를 적극적으로 사용했다.

$$
ReLU(x)=\max(0,x)
$$

ReLU를 사용하면 positive 영역에서 gradient가

$$
1
$$

이므로 sigmoid보다 deep network 학습에 유리하다.

논문 역시 AlexNet의 성공 요인 중 하나로 ReLU를 언급한다.     1. A survey on deep learning fu…

---

## 2.2 Dropout

AlexNet에서는 overfitting을 줄이기 위해

$$
\boxed{\text{Dropout}}
$$

을 사용했다.

training 과정에서 일정 probability로 neuron output을 0으로 만든다.

예를 들어

$$
h= [1.2,\ 0.5,\ 2.1,\ 0.9]
$$

가 있다면 dropout을 적용해서

$$
[1.2,\ 0,\ 2.1,\ 0]
$$

처럼 일부 neuron을 제거한다.

이렇게 하면 특정 neuron에 지나치게 의존하지 않고 더 robust한 feature를 학습하도록 유도할 수 있다.

논문은 AlexNet에서 hidden neuron을 확률적으로 제거하는 dropout을 통해 overfitting을 줄였다고 설명한다.     1. A survey on deep learning fu…

---

## 2.3 Data Augmentation

training image를 그대로 사용하는 것이 아니라

- rotation
- flip
- crop
- color/light adjustment

등으로 변형한다.

즉

$$
x \rightarrow T(x)
$$

형태로 새로운 training sample을 만든다.

논문도 AlexNet의 성공 요인 중 하나로 data augmentation을 설명한다.     1. A survey on deep learning fu…

---

## 2.4 Overlapping Pooling

기존 pooling에서는 window가 서로 겹치지 않는 경우가 많았다.

AlexNet에서는

$$
\text{window size}>\text{stride}
$$

인 overlapping pooling을 사용했다.

논문은 이것이 feature precision과 generalization에 도움이 됐다고 설명한다.     1. A survey on deep learning fu…

---

## 2.5 Multi-GPU

당시 GPU memory가 제한적이어서 network를 두 GPU에 나누어 계산했다.

논문은 AlexNet이 두 개의 GPU를 사용해 network의 neuron을 나누어 처리했다고 설명한다.     1. A survey on deep learning fu…

여기서 중요한 점은 이것이 오늘날 MobileNet에서 사용하는

$$
\boxed{\text{Depthwise Convolution}}
$$

과는 전혀 다른 개념이라는 것이다.

AlexNet의 grouped convolution은 당시 hardware limitation 때문에 두 GPU로 network를 나눈 결과에 가깝다.

---

# 3. VGG

AlexNet 이후에는

> network를 더 깊게 만들면 어떨까?

라는 방향이 강해진다.

대표적인 것이

$$
\boxed{\text{VGG}}
$$

이다.

VGG의 핵심은 복잡한 kernel을 다양하게 쓰기보다는 작은

$$
\boxed{3\times3}
$$

convolution을 반복해서 쌓는 것이다.

논문은 VGG가 depth를 늘려 더 깊은 feature를 학습하는 CNN이라고 설명한다.     1. A survey on deep learning fu…

---

# 3.1 왜 3×3 Convolution을 반복할까?

예를 들어

$$
5\times5
$$

convolution 한 번 대신

$$
3\times3
$$

convolution 두 번을 쓸 수 있다.

두 개의 3×3 convolution을 연속으로 쌓으면 effective receptive field는 대략

$$
5\times5
$$

가 된다.

그런데 parameter 수를 비교해보자.

input channel과 output channel이 동일하게 $C$라고 단순화하면,

5×5 convolution은

$$
25C^2
$$

parameter가 필요하다.

3×3 두 개는

$$
9C^2+9C^2 = 18C^2
$$

이다.

즉 parameter를 줄일 수 있다.

게다가 activation도 두 번 들어간다.

$$
Conv \rightarrow ReLU \rightarrow Conv \rightarrow ReLU
$$

그래서 nonlinearity도 증가한다.

---

# 4. GoogLeNet / Inception

VGG가

$$
\boxed{\text{Depth}}
$$

에 집중했다면 GoogLeNet은

$$
\boxed{\text{Multi-scale Feature}}
$$

를 강조한다.

핵심이

$$
\boxed{\text{Inception Module}}
$$

이다.

하나의 input feature map에 대해 동시에

$$
1\times1
$$

$$
3\times3
$$

$$
5\times5
$$

$$
Pooling
$$

을 적용한다.

그리고 결과를 channel 방향으로 concatenate한다.

즉

$$
X
$$

에서 여러 branch가 갈라진다.

$$
\begin{array}{c} 1\times1 Conv\\ 3\times3 Conv\\ 5\times5 Conv\\ 3\times3 MaxPool \end{array}
$$

↓

$$
Concat
$$

이렇게 한다.

논문도 GoogLeNet이 1×1, 3×3, 5×5 convolution과 pooling을 병렬로 사용해서 여러 scale의 feature를 얻는다고 설명한다.     1. A survey on deep learning fu…

---

# 4.1 왜 1×1 Convolution을 쓸까?

1×1 convolution이 처음 보면 이상하다.

공간적으로는 한 pixel만 보기 때문이다.

하지만 channel dimension을 바꿀 수 있다.

예를 들어

$$
H\times W\times256
$$

feature가 있다고 하자.

3×3 convolution을 바로 적용하면 계산량이 크다.

먼저 1×1 convolution으로

$$
256 \rightarrow 64
$$

channel을 줄인다.

$$
H\times W\times256
$$

↓

$$
1\times1 Conv
$$

↓

$$
H\times W\times64
$$

그 다음

$$
3\times3
$$

convolution을 수행한다.

즉

$$
\boxed{\text{Channel Bottleneck}}
$$

역할이다.

논문도 GoogLeNet이 큰 kernel 전에 1×1 convolution을 사용해서 dimension과 parameter를 줄인다고 설명한다.     1. A survey on deep learning fu…

---

# 5. ResNet

network를 계속 깊게 만들면 무조건 좋아질까?

그렇지 않다.

깊어지면서

$$
\boxed{\text{Vanishing Gradient}}
$$

또는

$$
\boxed{\text{Optimization Difficulty}}
$$

가 발생한다.

그래서 등장한 것이

$$
\boxed{\text{ResNet}}
$$

이다.

핵심은

$$
\boxed{\text{Residual Connection}}
$$

이다.

기존 network는

$$
x \rightarrow F(x)
$$

였다.

ResNet은

$$
\boxed{ y=F(x)+x }
$$

로 만든다.

논문도 ResNet residual block을

$$
f(x)+x
$$

형태로 설명한다.     1. A survey on deep learning fu…

---

# 5.1 왜 Skip Connection이 중요한가?

만약 어떤 layer가 굳이 새로운 transformation을 학습할 필요가 없다면,

$$
F(x)=0
$$

으로 만들 수 있다.

그러면

$$
y=x
$$

이다.

즉 identity mapping을 쉽게 만들 수 있다.

또 gradient도

$$
y=F(x)+x
$$

이므로

$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x}+1
$$

이 된다.

직접적인

$$
+1
$$

path가 생기기 때문에 gradient가 앞쪽 layer까지 전달되기 쉬워진다.

즉

$$
\boxed{ \text{Residual Connection} \Rightarrow \text{Deep Network 학습 용이} }
$$

라고 이해하면 된다.

---

# 6. CNN 전체 발전 흐름

CNN의 발전을 매우 단순하게 정리하면

$$
\boxed{\text{AlexNet}}
$$

ReLU / Dropout / GPU로 deep CNN 가능성을 보여줌

↓

$$
\boxed{\text{VGG}}
$$

작은 3×3 convolution을 반복하며 depth 확대

↓

$$
\boxed{\text{GoogLeNet}}
$$

multi-scale branch와 1×1 bottleneck

↓

$$
\boxed{\text{ResNet}}
$$

residual connection으로 아주 깊은 network 학습

이라고 볼 수 있다.

---

# 7. CNN의 한계

CNN은 이미지에 매우 강력하지만 기본적으로

$$
\boxed{\text{Local Receptive Field}}
$$

를 기반으로 한다.

멀리 떨어진 두 pixel의 관계를 직접 보려면 여러 convolution layer가 필요하다.

예를 들어 이미지의 왼쪽 위에 있는 object와 오른쪽 아래에 있는 object의 관계를 이해하려면 deep layer를 거쳐야 한다.

그래서 이후에는

$$
\boxed{\text{Global Relationship}}
$$

을 직접 학습하는 attention mechanism이 중요해진다.

하지만 그 전에 생성 모델의 큰 흐름인 GAN을 보자.

---

# 8. GAN: "분류"가 아니라 "생성"

CNN의 대표적인 목적은

$$
\text{Image} \rightarrow \text{Feature} \rightarrow \text{Class}
$$

처럼 입력을 분석하는 것이다.

그런데 새로운 질문이 생긴다.

> 학습 데이터와 비슷한 새로운 이미지를 직접 만들 수는 없을까?

이를 위한 대표적인 generative model이

$$
\boxed{\text{GAN}}
$$

이다.

논문은 GAN을 target data distribution을 모방하는 sample을 생성하는 generative modeling framework로 설명한다.     1. A survey on deep learning fu…

---

# 9. GAN의 두 Network

GAN은 두 network가 경쟁한다.

$$
\boxed{\text{Generator }G}
$$

와

$$
\boxed{\text{Discriminator }D}
$$

이다.

Generator:

$$
z \rightarrow G(z)
$$

random noise에서 fake sample을 만든다.

Discriminator:

$$
x \rightarrow D(x)
$$

입력 sample이 real인지 fake인지 판별한다.

논문도 GAN이 random noise에서 synthetic data를 만드는 generator와 real/fake probability를 판단하는 discriminator로 구성된다고 설명한다.     1. A survey on deep learning fu…

---

# 10. Generator

Generator의 입력은 보통 random latent vector다.

$$
z\sim p(z)
$$

예를 들어

$$
z\sim\mathcal N(0,I)
$$

라고 하자.

Generator가 이를 이미지로 변환한다.

$$
\boxed{ x_{\text{fake}}=G(z) }
$$

초기에는 random noise에 가까운 이상한 이미지가 생성된다.

하지만 training하면서 점점 real data와 비슷해진다.

---

# 11. Discriminator

Discriminator는 입력이 실제 training data인지 Generator가 만든 fake인지 구분한다.

$$
D(x) \in[0,1]
$$

예를 들어

$$
D(x)=0.95
$$

라면 real일 가능성이 높다고 판단하는 식이다.

그래서

$$
\text{Real Image} \rightarrow D \rightarrow 1
$$

$$
\text{Fake Image} \rightarrow D \rightarrow 0
$$

이 되도록 학습한다.

---

# 12. GAN의 Adversarial Training

GAN의 핵심은

$$
\boxed{\text{Adversarial}}
$$

즉 서로 경쟁하는 학습이다.

Discriminator는

> fake를 더 잘 찾아내야지.

Generator는

> discriminator를 속일 정도로 진짜 같은 sample을 만들어야지.

라는 목적을 갖는다.

대표적인 objective는

$$
\min_G\max_D V(D,G)
$$

$$
= \mathbb{E}_{x\sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z\sim p_z} [\log(1-D(G(z)))]
$$

이다.

---

# 13. GAN 학습을 단계별로 보면

먼저 real image를 준비한다.

$$
x_{\text{real}}
$$

그리고 random noise를 뽑는다.

$$
z
$$

Generator가 fake image를 만든다.

$$
x_{\text{fake}}=G(z)
$$

Discriminator가 둘을 판별한다.

$$
D(x_{\text{real}})
$$

$$
D(x_{\text{fake}})
$$

Discriminator는

$$
D(x_{\text{real}})\rightarrow1
$$

$$
D(x_{\text{fake}})\rightarrow0
$$

이 되도록 학습한다.

반대로 Generator는

$$
D(G(z))\rightarrow1
$$

이 되도록 학습한다.

즉 fake인데 real이라고 착각하게 만드는 것이다.

---

# 14. GAN에서 중요한 개념: Data Distribution

GAN은 특정 image 하나를 외우는 것이 목적이 아니다.

실제 data가

$$
p_{\text{data}}(x)
$$

라는 distribution을 가진다고 생각한다.

Generator도

$$
p_g(x)
$$

라는 distribution을 만든다.

training 목표는 결국

$$
\boxed{ p_g(x) \approx p_{\text{data}}(x) }
$$

가 되게 하는 것이다.

그래서 새로운 random $z$를 넣더라도 실제 data와 비슷하지만 새로운 sample이 나올 수 있다.

---

# 15. GAN의 활용

논문은 GAN이 high-dimensional data generation, 특히 image synthesis에 강력하며

$$
\boxed{\text{Image Inpainting}}
$$

$$
\boxed{\text{Style Transfer}}
$$

같은 task에 활용됐다고 설명한다.     1. A survey on deep learning fu…

예를 들어 inpainting은 이미지 일부가 사라졌을 때

$$
\text{Damaged Image}
$$

↓

$$
G
$$

↓

$$
\text{Completed Image}
$$

를 생성하는 것이다.

---

# 16. GAN의 문제: Training Instability

GAN은 두 network를 동시에 학습해야 한다.

Generator가 너무 강해져도 문제고,

Discriminator가 너무 강해져도 문제다.

예를 들어 Discriminator가 너무 쉽게 fake를 구분하면

$$
D(G(z))\approx0
$$

이 되어 Generator가 유용한 gradient를 얻기 어려워질 수 있다.

즉

$$
\boxed{\text{G와 D의 Balance}}
$$

가 중요하다.

논문도 generator와 discriminator를 균형 있게 학습해야 해서 training instability가 발생한다고 지적한다.     1. A survey on deep learning fu…

---

# 17. Mode Collapse

GAN에서 아주 유명한 문제다.

실제 dataset에

- 개
- 고양이
- 자동차
- 사람

처럼 다양한 mode가 있다고 해보자.

그런데 Generator가 특정 종류만 잘 만들게 되는 경우가 있다.

예를 들어 noise를 아무리 바꿔도 비슷한 얼굴만 생성하는 것이다.

이를

$$
\boxed{\text{Mode Collapse}}
$$

라고 한다.

즉

$$
z_1,z_2,z_3
$$

가 달라도

$$
G(z_1)\approx G(z_2)\approx G(z_3)
$$

가 되어버린다.

논문도 mode collapse를 target distribution의 다양성을 제대로 반영하지 못하는 문제로 설명한다.     1. A survey on deep learning fu…

이 문제를 해결하기 위한 대표적 발전으로 논문에서는 WGAN과 Progressive GAN 등을 언급한다.     1. A survey on deep learning fu…

---

# 18. 이제 Transformer: Local에서 Global로

CNN은 local pattern을 매우 잘 학습한다.

하지만 다음과 같은 문제가 있다.

문장:

> The animal didn't cross the street because **it** was tired.

여기서 `"it"`이 무엇을 의미하는지를 이해하려면 멀리 떨어진 word 사이 관계를 알아야 한다.

이미지에서도 마찬가지다.

왼쪽 위의 object와 오른쪽 아래의 object가 어떤 관계인지 이해해야 할 수 있다.

그래서 등장한 핵심 아이디어가

$$
\boxed{\text{Attention}}
$$

이다.

---

# 19. Attention의 핵심 직관

attention의 핵심 질문은 단순하다.

> 현재 정보를 처리할 때 다른 입력 중 어디를 얼마나 참고해야 하는가?

예를 들어

$$
\text{"I ate an apple because it was delicious."}
$$

에서 `"it"`을 처리할 때

$$
\text{"apple"}
$$

에 높은 attention을 줄 수 있다.

즉 모든 token을 동일하게 보는 게 아니라

$$
\boxed{\text{Relevant Information에 높은 weight}}
$$

를 주는 것이다.

---

# 20. Query, Key, Value

Transformer의 attention에서 가장 중요한 세 요소가

$$
\boxed{Q,\ K,\ V}
$$

이다.

각 token representation $x$에서 세 가지 vector를 만든다.

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

여기서

$$
Q=\text{Query}
$$

$$
K=\text{Key}
$$

$$
V=\text{Value}
$$

이다.

---

# 21. Query와 Key는 "누구를 볼지" 결정한다

현재 token의 Query와 다른 token들의 Key를 비교한다.

$$
QK^T
$$

값이 크다면 서로 관련이 높다는 뜻이다.

예를 들어 token A의 Query가 token B의 Key와 잘 맞으면

$$
q_A\cdot k_B
$$

가 커진다.

그러면 A가 B를 많이 참고하게 된다.

---

# 22. Attention Score

논문의 Transformer section에서 핵심 식은

$$
\boxed{ Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V }
$$

이다.     1. A survey on deep learning fu…

순서대로 보면

$$
QK^T
$$

으로 similarity를 계산한다.

그리고

$$
\frac{QK^T}{\sqrt{d_k}}
$$

로 scale한다.

그다음

$$
softmax
$$

를 적용해서 attention weight를 만든다.

마지막으로

$$
V
$$

를 weighted sum한다.

---

# 23. 왜 $\sqrt{d_k}$로 나눌까?

vector dimension이 커지면 dot product도 커질 가능성이 높다.

그러면 softmax에 너무 큰 값이 들어가서 distribution이 지나치게 sharp해질 수 있다.

예를 들어

$$
[1,2,3]
$$

보다

$$
[100,200,300]
$$

에 softmax를 적용하면 거의 한 위치에 probability가 몰린다.

그렇게 되면 gradient가 불안정해질 수 있다.

그래서

$$
\boxed{\sqrt{d_k}}
$$

로 나누어 scale을 안정화한다.

논문도 이를 gradient 안정화를 위한 scaling factor로 설명한다.     1. A survey on deep learning fu…

---

# 24. Value는 실제 전달되는 정보

Query와 Key는

$$
\boxed{\text{누구를 얼마나 볼까?}}
$$

를 정한다.

Value는

$$
\boxed{\text{실제로 가져올 정보}}
$$

다.

예를 들어 attention weights가

$$
[0.1,\ 0.7,\ 0.2]
$$

라면,

$$
Output = 0.1V_1 + 0.7V_2 + 0.2V_3
$$

가 된다.

즉 두 번째 token의 정보를 가장 많이 가져온다.

---

# 25. Self-Attention

Q, K, V가 모두 같은 input sequence에서 나오면

$$
\boxed{\text{Self-Attention}}
$$

이라고 한다.

$$
X \rightarrow Q,K,V
$$

즉 문장 안에서 모든 token이 서로를 보는 것이다.

논문은 Transformer가 self-attention으로 input의 global dependency를 잡으며 local과 long-range interaction을 모두 모델링한다고 설명한다.     1. A survey on deep learning fu…

---

# 26. CNN과 Self-Attention의 차이

CNN은 기본적으로 local이다.

$$
3\times3
$$

kernel이면 주변 3×3만 본다.

멀리 있는 pixel과 관계를 만들려면 여러 layer를 지나야 한다.

반면 self-attention은 한 layer에서도

$$
Token_i \leftrightarrow Token_j
$$

를 직접 연결할 수 있다.

즉

$$
\boxed{ \text{CNN: Local Interaction} }
$$

$$
\boxed{ \text{Transformer: Global Interaction} }
$$

이라고 볼 수 있다.

논문도 CNN의 localized receptive field와 달리 Transformer에서는 모든 token이 한 단계에서 global interaction을 할 수 있다고 설명한다.     1. A survey on deep learning fu…

---

# 27. Multi-Head Attention

하나의 attention만 사용하면 한 종류의 관계만 학습할 수 있다.

그래서 여러 attention을 병렬로 사용한다.

$$
head_i = Attention(Q_i,K_i,V_i)
$$

각 head는 서로 다른 relation을 학습할 수 있다.

예를 들어 language라면 어떤 head는

$$
\text{subject-verb}
$$

관계를 보고,

다른 head는

$$
\text{pronoun-reference}
$$

관계를 볼 수 있다.

그 결과를 concatenate한다.

$$
MultiHead = Concat(head_1,\dots,head_h)W_O
$$

참고로 **이 multi-head 세부 식은 해당 survey의 이 구간에 직접 전개되어 있지는 않고, Transformer를 이해하기 위한 일반적인 보충 설명**이다.

---

# 28. Transformer Encoder

논문은 전형적인 Transformer가 encoder-decoder 구조를 갖는다고 설명한다.     1. A survey on deep learning fu…

Encoder에서는 대략

$$
\text{Input Tokens}
$$

↓

$$
\text{Self-Attention}
$$

↓

$$
\text{Feed Forward Network}
$$

↓

$$
\text{Output Representation}
$$

형태가 반복된다.

실제 Transformer에서는 residual connection과 normalization도 사용한다.

즉 개념적으로

$$
X' = X+Attention(X)
$$

그리고

$$
Y = X'+FFN(X')
$$

같은 구조다.

---

# 29. Transformer Decoder

Decoder는 output token을 하나씩 생성해야 한다.

예를 들어 번역이라면

입력:

> 나는 학교에 간다

출력:

> I go to school

을 생성한다.

decoder에서는 이전에 생성된 token과 encoder output을 사용한다.

논문도 decoder가 encoder output과 자신의 sequence를 attention하여 output을 생성한다고 설명한다.     1. A survey on deep learning fu…

---

# 30. Positional Encoding

Transformer는 convolution도 없고 recurrence도 없다.

그래서 단순히 token embedding만 주면 순서를 자동으로 알 수 없다.

예를 들어

> dog bites man

과

> man bites dog

은 token set만 보면 동일하다.

그래서 position 정보를 추가한다.

$$
z_i = x_i+p_i
$$

여기서

$$
p_i
$$

가 position embedding이다.

논문의 ViT 설명에서도 patch embedding에

$$
E_{pos}
$$

를 더해 spatial position을 encoding한다고 설명한다.     1. A survey on deep learning fu…

---

# 31. Vision Transformer

Transformer가 원래 NLP를 위해 개발됐지만 image에도 적용할 수 있다.

대표적인 것이

$$
\boxed{\text{Vision Transformer, ViT}}
$$

이다.

논문은 이미지를 fixed-size patch로 잘라 token처럼 처리한다고 설명한다.     1. A survey on deep learning fu…

---

# 32. Image를 Patch로 나누기

예를 들어 이미지가

$$
224\times224\times3
$$

이고 patch size가

$$
16\times16
$$

이라고 하자.

한 축에 patch가

$$
224/16=14
$$

개 생긴다.

그래서 전체 patch 수는

$$
14\times14=196
$$

개다.

각 patch는

$$
16\times16\times3
$$

이므로 flatten하면

$$
768
$$

dimension이다.

즉 image가

$$
\boxed{196\text{개의 token}}
$$

으로 변환된다.

---

# 33. ViT 입력

논문에서는 이를

$$
z_0 = [p_1E;p_2E;\cdots;p_NE] + E_{pos}
$$

형태로 표현한다.     1. A survey on deep learning fu…

여기서

$$
p_i
$$

는 image patch,

$$
E
$$

는 learnable projection,

$$
E_{pos}
$$

는 positional embedding이다.

즉

$$
\text{Image}
$$

↓

$$
\text{Patches}
$$

↓

$$
\text{Patch Embeddings}
$$

↓

$$
\text{Positional Embeddings 추가}
$$

↓

$$
\text{Transformer Encoder}
$$

이다.

---

# 34. CNN vs ViT

CNN:

$$
\boxed{ \text{Locality} + \text{Weight Sharing} }
$$

이라는 strong inductive bias가 있다.

ViT:

$$
\boxed{ \text{Global Self-Attention} }
$$

을 중심으로 한다.

CNN은 local pattern 학습이 자연스럽고,

Transformer는 멀리 떨어진 patch 사이 관계를 직접 학습하기 쉽다.

논문은 ViT가 global patch interaction을 통해 spatial/contextual relationship을 포착한다고 설명한다.     1. A survey on deep learning fu…

---

# 35. Transformer의 장점

가장 중요한 장점은

$$
\boxed{\text{Long-range Dependency}}
$$

다.

또 RNN처럼 token을 반드시 순차적으로 하나씩 처리할 필요 없이 training 단계에서 sequence의 token 관계를 병렬 계산할 수 있다.

따라서 large-scale model 학습에 적합하다.

그리고

$$
\text{Text}
$$

뿐 아니라

$$
\text{Image}
$$

$$
\text{Video}
$$

$$
\text{Audio}
$$

$$
\text{3D}
$$

등으로 확장되었다.

논문도 Transformer가 NLP에서 출발해 ViT를 통해 computer vision까지 큰 영향을 줬다고 설명한다.     1. A survey on deep learning fu…

---

# 36. Transformer의 단점

standard self-attention에서는 sequence length가 $N$이라면

$$
QK^T
$$

크기가

$$
N\times N
$$

이다.

따라서 attention computation은 대략

$$
O(N^2)
$$

에 해당한다.

sequence가 길어질수록 memory와 computation이 크게 증가한다.

이 계산 복잡도 설명은 survey 해당 문단에 직접 명시되지는 않고, Transformer의 일반적인 구조에서 따라오는 보충 설명이다.

---

# 37. 이제 Diffusion Model

GAN은 뛰어난 생성 성능을 보였지만 앞서 봤듯

$$
\text{Training Instability}
$$

$$
\text{Mode Collapse}
$$

같은 문제가 있다.

이와 다른 방식으로 generative modeling을 수행하는 대표적 방법이

$$
\boxed{\text{Diffusion Model}}
$$

이다.

논문은 diffusion model의 중요한 특징으로 **안정성과 high-quality generation**을 들고, data에 점진적으로 noise를 넣은 다음 그 과정을 역으로 학습한다고 설명한다.     1. A survey on deep learning fu…

---

# 38. Diffusion의 핵심 아이디어

Diffusion을 한 문장으로 표현하면

$$
\boxed{ \text{Data를 Noise로 만들고, Noise를 다시 Data로 복원하는 법을 학습} }
$$

이다.

두 과정이 있다.

$$
\boxed{\text{Forward Process}}
$$

과

$$
\boxed{\text{Reverse Process}}
$$

이다.

---

# 39. Forward Diffusion Process

원본 image를

$$
x_0
$$

라고 하자.

조금 noise를 넣으면

$$
x_1
$$

더 넣으면

$$
x_2
$$

계속 넣으면

$$
x_T
$$

가 된다.

흐름은

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
$$

이다.

마지막에는 거의

$$
\boxed{\text{Gaussian Noise}}
$$

가 된다.

---

# 40. Forward Process 수식

논문은 forward transition을

$$
\boxed{ q(x_t|x_{t-1}) = \mathcal N \left( x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I \right) }
$$

로 설명한다.     1. A survey on deep learning fu…

직관적으로 보면

$$
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon
$$

형태라고 생각할 수 있다.

여기서

$$
\epsilon\sim\mathcal N(0,I)
$$

이다.

즉 기존 image signal 일부를 유지하면서 Gaussian noise를 조금씩 넣는다.

---

# 41. Noise Schedule

$$
\alpha_t
$$

는 각 timestep에서 signal을 얼마나 유지할지를 결정한다.

초기에는 noise가 조금 들어가고,

후반에는 noise가 많이 들어간다.

즉

$$
t\uparrow
$$

할수록

$$
\text{Original Information}\downarrow
$$

$$
\text{Noise}\uparrow
$$

가 된다.

---

# 42. Reverse Process

우리가 실제로 배우고 싶은 것은 반대 방향이다.

$$
x_T
$$

즉 random noise에서 시작해서

$$
x_{T-1}
$$

$$
x_{T-2}
$$

$$
\cdots
$$

$$
x_0
$$

로 돌아가는 것이다.

즉

$$
\boxed{ \text{Noise} \rightarrow \text{Image} }
$$

이다.

논문은 reverse transition을

$$
\boxed{ p_\theta(x_{t-1}|x_t) = \mathcal N \left( x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t) \right) }
$$

로 표현한다.     1. A survey on deep learning fu…

---

# 43. Network가 무엇을 학습하는가?

논문 표현에서는 network가 reverse Gaussian transition의

$$
\mu_\theta(x_t,t)
$$

와

$$
\Sigma_\theta(x_t,t)
$$

를 parameterize한다고 설명한다.     1. A survey on deep learning fu…

쉽게 말하면

> 현재 noisy sample $x_t$에서 어떻게 움직여야 조금 더 clean한 $x_{t-1}$이 되는가?

를 배우는 것이다.

현대 DDPM 설명에서는 흔히 noise

$$
\epsilon
$$

을 직접 예측하는 형태로 많이 설명하지만, **이 survey는 reverse distribution의 mean/variance 관점으로 설명하고 있다.**

---

# 44. Diffusion Training

training에서는 clean image $x_0$가 있다.

random timestep $t$를 뽑고 noise를 추가한다.

$$
x_0 \rightarrow x_t
$$

그 다음 neural network가 reverse direction을 예측한다.

즉

$$
\boxed{ (x_t,t) \rightarrow \text{Network} \rightarrow \text{Denoising Direction} }
$$

이 된다.

논문은 true forward process와 predicted reverse process 사이 discrepancy를 줄이는 방식으로 학습하고, VLB를 objective로 사용할 수 있다고 설명한다.     1. A survey on deep learning fu…

---

# 45. Diffusion Inference

이 부분이 매우 중요하다.

training할 때는 실제 이미지가 있다.

하지만 generation 때는 image가 없다.

random Gaussian noise부터 시작한다.

$$
x_T \sim \mathcal N(0,I)
$$

그리고

$$
x_T \rightarrow x_{T-1} \rightarrow x_{T-2} \rightarrow \cdots \rightarrow x_0
$$

순서로 반복해서 denoise한다.

논문도 trained diffusion model이 Gaussian noise에서 시작해 learned reverse transition을 반복하면서 data를 생성한다고 설명한다.     1. A survey on deep learning fu…

즉 diffusion generation은

$$
\boxed{ \text{Random Noise} \rightarrow \text{Gradual Denoising} \rightarrow \text{Image} }
$$

이다.

---

# 46. 왜 한 번에 Image를 만들지 않을까?

GAN은

$$
z \rightarrow G(z) \rightarrow x
$$

처럼 거의 한 번의 forward generation으로 sample을 만든다.

Diffusion은

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

처럼 여러 단계로 만든다.

이 점진적 refinement가 diffusion의 중요한 특징이다.

논문은 iterative refinement가 fine detail과 coherent synthesis를 만드는 데 효과적이라고 설명한다.     1. A survey on deep learning fu…

---

# 47. GAN vs Diffusion

둘 다 generative model이지만 방식이 완전히 다르다.

GAN:

$$
\boxed{ \text{Generator vs Discriminator} }
$$

Diffusion:

$$
\boxed{ \text{Noise Addition vs Denoising} }
$$

GAN의 generation:

$$
z \rightarrow G(z) \rightarrow x
$$

Diffusion의 generation:

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
$$

GAN은 inference가 빠른 편이지만 adversarial training이 불안정할 수 있다.

Diffusion은 training이 상대적으로 안정적인 대신 generation에서 여러 denoising step이 필요하다.

논문은 diffusion model이 GAN의 adversarial instability 문제를 완화하면서 high-quality output을 생성하는 안정적인 generative alternative라고 설명한다.     1. A survey on deep learning fu…

---

# 48. DDPM

대표적인 diffusion model이

$$
\boxed{\text{DDPM}}
$$

이다.

Denoising Diffusion Probabilistic Model의 약자다.

핵심 구조는 지금 설명한 그대로다.

$$
\text{Forward Gaussian Noise}
$$

↓

$$
\text{Learn Reverse Denoising}
$$

↓

$$
\text{Generate from Noise}
$$

논문은 DDPM을 high-resolution generation의 대표적인 diffusion model로 언급한다.     1. A survey on deep learning fu…

---

# 49. Diffusion을 Super-Resolution에 적용

Diffusion은 단순 image generation뿐 아니라

$$
\boxed{\text{Super-Resolution}}
$$

에도 적용된다.

예를 들어 low-resolution image를 condition으로 준다.

$$
x_{\text{LR}}
$$

그리고 noise에서 시작해서 high-resolution image를 복원한다.

$$
x_T + x_{\text{LR}}
$$

↓

$$
Denoising
$$

↓

$$
x_{\text{HR}}
$$

논문은 대표적으로

$$
\boxed{\text{SR3}}
$$

를 언급하며 blurry/low-quality image를 점진적으로 refine하여 high-frequency detail을 복구한다고 설명한다.     1. A survey on deep learning fu…

---

# 50. Latent Diffusion

일반 diffusion을 pixel space에서 돌리면 computation이 매우 크다.

예를 들어

$$
512\times512\times3
$$

pixel 전체에서 diffusion을 반복하면 비싸다.

그래서 image를 latent space로 압축한 뒤 diffusion을 수행하는 방법이 있다.

$$
Image
$$

↓

$$
Encoder
$$

↓

$$
Latent\ z
$$

↓

$$
Diffusion
$$

↓

$$
Denoised\ Latent
$$

↓

$$
Decoder
$$

↓

$$
Image
$$

이것이

$$
\boxed{\text{Latent Diffusion}}
$$

의 핵심 아이디어다.

논문도 LDM이 latent representation을 사용해 high-resolution synthesis를 보다 scalable하게 수행한다고 설명한다.     1. A survey on deep learning fu…

---

# 51. CNN / GAN / Transformer / Diffusion을 역할로 구분

이제 네 모델을 한 번에 비교해보자.

| 모델          | 핵심 목적              | 핵심 아이디어                     |
| ----------- | ------------------ | --------------------------- |
| CNN         | Feature Extraction | Local convolution           |
| GAN         | Data Generation    | Generator vs Discriminator  |
| Transformer | Relation Modeling  | Self-Attention              |
| Diffusion   | Data Generation    | Noise → Iterative Denoising |

조금 더 직관적으로 보면:

$$
\boxed{\text{CNN}}
$$

> 주변을 보면서 특징을 찾자.

$$
\boxed{\text{GAN}}
$$

> 생성자와 판별자를 경쟁시켜 진짜 같은 데이터를 만들자.

$$
\boxed{\text{Transformer}}
$$

> 입력 전체에서 서로 어떤 관계인지 직접 보자.

$$
\boxed{\text{Diffusion}}
$$

> noise를 조금씩 제거하는 법을 배워 데이터를 만들자.

---

# 52. 입력과 출력으로 비교

## CNN

$$
\text{Image} \rightarrow \text{Feature} \rightarrow \text{Prediction}
$$

예:

$$
\text{Cat Image} \rightarrow \text{CNN} \rightarrow \text{Cat}
$$

---

## GAN

$$
\text{Random Noise} \rightarrow G \rightarrow \text{Fake Image}
$$

training에는

$$
D
$$

도 사용한다.

---

## Transformer

$$
\text{Tokens} \rightarrow \text{Self-Attention} \rightarrow \text{Contextualized Tokens}
$$

예:

$$
\text{Sentence} \rightarrow \text{Transformer} \rightarrow \text{Language Representation}
$$

또는 ViT에서는

$$
\text{Image Patches} \rightarrow \text{Transformer} \rightarrow \text{Visual Representation}
$$

---

## Diffusion

$$
\text{Gaussian Noise}
$$

↓

$$
\text{Repeated Denoising}
$$

↓

$$
\text{Generated Image}
$$

---

# 53. 학습 방식으로 비교

## CNN

보통 supervised learning에서

$$
X \rightarrow CNN \rightarrow \hat Y
$$

$$
L(\hat Y,Y)
$$

를 계산해서 backpropagation한다.

---

## GAN

두 개의 network를 번갈아 학습한다.

$$
D
$$

는

$$
\text{Real vs Fake}
$$

를 구분하고,

$$
G
$$

는

$$
\text{D를 속이도록}
$$

학습한다.

---

## Transformer

Transformer도 기본적으로는 task에 따라 loss를 정한다.

classification이면

$$
CrossEntropy
$$

language modeling이면

$$
-\log P(x_t|x_{\<t})
$$

같은 objective를 사용할 수 있다.

핵심적인 architecture 차이가 self-attention이다.

---

## Diffusion

noisy sample을 만들고 reverse process를 학습한다.

$$
x_0 \rightarrow x_t
$$

그리고

$$
x_t \rightarrow x_{t-1}
$$

방향을 예측하도록 학습한다.

---

# 54. Inference 관점에서 비교

CNN:

$$
\boxed{\text{한 번 Forward}}
$$

GAN Generator:

$$
\boxed{\text{한 번 Forward}}
$$

Autoregressive Transformer:

$$
\boxed{\text{Token을 순차 생성}}
$$

Diffusion:

$$
\boxed{\text{여러 Denoising Step}}
$$

그래서 inference 특성도 매우 다르다.

---

# 55. CNN에서 Transformer로

이 흐름을 representation 관점에서 보면 흥미롭다.

CNN:

$$
\boxed{\text{Local Inductive Bias}}
$$

즉 locality를 강하게 가정한다.

Transformer:

$$
\boxed{\text{Global Relation Learning}}
$$

즉 어떤 token끼리 관련 있는지를 attention으로 학습한다.

CNN에서는

$$
3\times3
$$

주변이 자연스럽게 중요하다고 구조적으로 가정하지만,

Transformer에서는 attention을 통해

$$
\text{어디를 볼지}
$$

자체를 학습한다.

---

# 56. GAN에서 Diffusion으로

generation 관점에서도 변화가 있다.

GAN은

$$
\boxed{\text{Adversarial Learning}}
$$

을 통해 data distribution을 학습한다.

하지만

$$
\text{Mode Collapse}
$$

$$
\text{Training Instability}
$$

문제가 있다.

Diffusion은

$$
\boxed{\text{Denoising Learning}}
$$

으로 generation을 수행한다.

논문은 diffusion이 GAN 계열의 adversarial instability를 피하면서 안정적으로 high-quality output을 만들 수 있다는 점을 강조한다.     1. A survey on deep learning fu…

---

# 57. 네 Architecture가 서로 배타적인 것도 아니다

중요한 점이 하나 있다.

CNN, GAN, Transformer, Diffusion이 항상 서로 독립적인 모델인 것은 아니다.

예를 들어 GAN의 Generator 안에 CNN을 사용할 수 있다.

$$
\text{GAN} = \text{CNN Generator} + \text{CNN Discriminator}
$$

처럼 말이다.

Diffusion의 denoising network도 convolutional architecture를 사용할 수 있다.

Transformer를 diffusion model 안에 사용할 수도 있다.

즉

$$
\boxed{ \text{CNN / Transformer} }
$$

는 주로 **feature processing architecture** 관점이고,

$$
\boxed{ \text{GAN / Diffusion} }
$$

은 **generative modeling framework** 관점이라는 차이가 있다.

이 구분이 정말 중요하다.

---

# 58. CNN과 GAN을 같은 레벨로 보면 왜 헷갈릴까?

CNN은

$$
\boxed{\text{Network Architecture}}
$$

에 가깝다.

GAN은

$$
\boxed{\text{Training Framework}}
$$

에 가깝다.

GAN 안에는 어떤 network architecture든 들어갈 수 있다.

예를 들어

$$
G=\text{CNN}
$$

$$
D=\text{CNN}
$$

일 수 있다.

Transformer 기반 Generator / Discriminator도 가능하다.

따라서

$$
\text{CNN vs GAN}
$$

을 완전히 같은 종류의 model 비교로 보면 안 된다.

---

# 59. Transformer와 Diffusion도 마찬가지

Transformer는

$$
\boxed{\text{Attention-based Architecture}}
$$

이다.

Diffusion은

$$
\boxed{\text{Probabilistic Generative Process}}
$$

이다.

Diffusion의 denoising network 자체를 Transformer로 만들 수도 있다.

따라서 현대 모델에서는

$$
\boxed{ \text{Transformer} + \text{Diffusion} }
$$

조합도 가능하다.

---

# 60. 이 네 가지를 가장 정확하게 분류하면

$$
\boxed{\text{CNN}}
$$

Local spatial representation architecture

$$
\boxed{\text{Transformer}}
$$

Attention-based global representation architecture

$$
\boxed{\text{GAN}}
$$

Adversarial generative learning framework

$$
\boxed{\text{Diffusion}}
$$

Iterative probabilistic generative framework

이라고 보는 것이 가장 깔끔하다.

---

# 61. 전체 발전 흐름

딥러닝 발전을 아주 큰 흐름으로 보면 다음과 같이 정리할 수 있다.

초기 neural network:

$$
\text{MLP}
$$

↓

이미지 spatial structure 활용:

$$
\boxed{\text{CNN}}
$$

↓

더 깊은 CNN:

$$
\text{AlexNet} \rightarrow \text{VGG} \rightarrow \text{GoogLeNet} \rightarrow \text{ResNet}
$$

↓

새로운 data 생성:

$$
\boxed{\text{GAN}}
$$

↓

global relation modeling:

$$
\boxed{\text{Attention / Transformer}}
$$

↓

image를 patch token으로 처리:

$$
\boxed{\text{ViT}}
$$

↓

더 안정적인 generative modeling:

$$
\boxed{\text{Diffusion}}
$$

이라고 볼 수 있다.

물론 실제 연구 역사는 이렇게 완전히 일렬로 대체된 것은 아니다. 이들은 서로 다른 시기에 발전하며 현재도 함께 사용된다. 여기서의 흐름은 **각 architecture가 어떤 문제를 해결하려 했는지 이해하기 위한 개념적 정리**다.

---

# 62. 한 단계 더 깊게: CNN은 어떤 prior를 갖는가?

CNN에는 강한

$$
\boxed{\text{Inductive Bias}}
$$

가 있다.

대표적으로

$$
\text{Locality}
$$

와

$$
\text{Translation-related Weight Sharing}
$$

이다.

즉

> 가까이 있는 pixel은 서로 관련 있을 가능성이 높다.

> 같은 pattern은 이미지 어디에서 나타나든 같은 filter로 찾을 수 있다.

라는 가정을 architecture 자체에 넣는다.

그래서 상대적으로 적은 data에서도 image structure를 효율적으로 학습할 수 있다.

이 inductive bias라는 표현은 해당 survey CNN 문단에서 직접 강조한 용어라기보다, weight sharing과 local feature extraction을 현대적으로 해석한 보충 설명이다.

---

# 63. Transformer는 prior를 덜 강하게 넣는다

Transformer는 CNN처럼

$$
3\times3
$$

neighbor만 먼저 보라고 강제하지 않는다.

모든 token 사이의 attention을 학습할 수 있다.

즉

$$
\boxed{\text{누가 중요한지 data에서 학습}}
$$

한다.

그 대신 image에 대한 locality prior가 CNN보다 약하기 때문에 많은 data와 large-scale pretraining이 중요해질 수 있다.

---

# 64. GAN은 Explicit Likelihood 없이 생성 가능

GAN의 독특한 점은 직접

$$
p(x)
$$

를 명시적으로 계산하지 않고도 generator를 학습할 수 있다는 것이다.

Discriminator라는 상대를 이용해서

$$
p_g
$$

가

$$
p_{\text{data}}
$$

에 가까워지도록 한다.

이 때문에 매우 선명한 image generation으로 큰 성공을 거뒀다.

하지만 objective가 두 player의 game 형태가 되기 때문에 안정적으로 optimize하기 어렵다.

---

# 65. Diffusion은 Markov Chain 관점

Diffusion은

$$
x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T
$$

라는 stochastic process를 만든다.

각 단계가 이전 단계에 의존한다.

forward process:

$$
q(x_t|x_{t-1})
$$

reverse process:

$$
p_\theta(x_{t-1}|x_t)
$$

즉 확률적인 transition을 연속해서 모델링한다.

그래서 GAN과는 매우 다른 수학적 관점에서 generation을 수행한다.

---

# 66. 실제 논문을 읽을 때 어떻게 구분하면 좋은가?

앞으로 논문을 읽다가 새로운 모델이 나오면 먼저 다음 질문을 해보면 된다.

### 첫째, input representation은 무엇인가?

$$
\text{Pixels?}
$$

$$
\text{Tokens?}
$$

$$
\text{Latent?}
$$

$$
\text{Point Cloud?}
$$

---

### 둘째, feature를 어떻게 처리하는가?

$$
\text{Convolution?}
$$

$$
\text{Attention?}
$$

$$
\text{Graph?}
$$

---

### 셋째, objective가 무엇인가?

$$
\text{Classification Loss?}
$$

$$
\text{Reconstruction Loss?}
$$

$$
\text{Adversarial Loss?}
$$

$$
\text{Denoising Objective?}
$$

---

### 넷째, inference는 어떻게 하는가?

한 번 forward하는가?

$$
x\rightarrow y
$$

token을 반복 생성하는가?

$$
y_1\rightarrow y_2\rightarrow\cdots
$$

denoising을 반복하는가?

$$
x_T\rightarrow\cdots\rightarrow x_0
$$

이 네 가지를 보면 대부분 모델의 큰 구조를 빠르게 잡을 수 있다.

---

# 67. CNN / GAN / Transformer / Diffusion 최종 비교

| 구분      | CNN                  | GAN                        | Transformer                   | Diffusion                       |
| ------- | -------------------- | -------------------------- | ----------------------------- | ------------------------------- |
| 핵심 아이디어 | Local convolution    | Adversarial learning       | Self-attention                | Iterative denoising             |
| 주요 목적   | Feature extraction   | Generation                 | Global relation modeling      | Generation                      |
| 대표 입력   | Image                | Random latent $z$      | Token sequence                | Gaussian noise                  |
| 주요 학습   | Task loss            | G vs D                     | Task / language modeling loss | Reverse denoising               |
| 장점      | 효율적 local feature    | Sharp generation           | Long-range dependency         | Stable, high-quality generation |
| 단점      | Global relation에 제한  | Mode collapse, instability | Attention cost 증가             | Sampling 느림                     |
| 대표 발전   | AlexNet, VGG, ResNet | WGAN, Progressive GAN      | ViT                           | DDPM, SR3, LDM                  |

---

# 68. 이 네 개를 한 줄씩 기억하면

CNN:

$$
\boxed{ \text{작은 kernel로 local feature를 계층적으로 학습} }
$$

GAN:

$$
\boxed{ \text{Generator와 Discriminator의 경쟁으로 data distribution을 학습} }
$$

Transformer:

$$
\boxed{ \text{Self-Attention으로 모든 token 간 관계를 학습} }
$$

Diffusion:

$$
\boxed{ \text{Noise를 점진적으로 제거하는 reverse process를 학습} }
$$

이다.

---

# 69. 전체 구조를 하나의 그림으로

$$
\boxed{ \text{Deep Learning} }
$$

↓

### Representation Learning

$$
\boxed{\text{CNN}}
$$

Local spatial feature

$$
\boxed{\text{Transformer}}
$$

Global relation / context

↓

### Generative Modeling

$$
\boxed{\text{GAN}}
$$

Adversarial generation

$$
\boxed{\text{Diffusion}}
$$

Iterative denoising generation

이렇게 분리해서 이해하면 훨씬 덜 헷갈린다.

---

# 70. 마무리

CNN, GAN, Transformer, Diffusion은 모두 현대 딥러닝에서 중요한 위치를 차지하지만, 서로 같은 역할을 수행하는 모델은 아니다.

CNN은

$$
\boxed{\text{local feature extraction}}
$$

이라는 문제를 매우 효과적으로 해결했고,

AlexNet, VGG, GoogLeNet, ResNet을 거치며 더 깊고 효율적인 visual representation learning이 가능해졌다. 논문 역시 이 CNN 발전 과정을 대표 architecture별로 정리하고 있다.     1. A survey on deep learning fu…

GAN은

$$
\boxed{\text{data generation}}
$$

을 adversarial game으로 바꾸어 매우 현실적인 sample을 생성할 수 있게 만들었지만, training instability와 mode collapse가 주요 문제로 남았다.     1. A survey on deep learning fu…

Transformer는

$$
\boxed{\text{self-attention}}
$$

을 통해 local receptive field에 묶이지 않고 global dependency를 직접 학습할 수 있게 했으며, NLP에서 시작해 ViT와 같은 computer vision model로 확대됐다.     1. A survey on deep learning fu…

Diffusion model은

$$
\boxed{ \text{Forward Noise Addition} + \text{Reverse Denoising} }
$$

이라는 확률적 과정을 기반으로 GAN과 다른 generation paradigm을 만들었으며, 논문은 안정적인 training과 high-quality generation을 중요한 장점으로 강조한다.     1. A survey on deep learning fu…

결국 이 네 가지를 관통하는 흐름은

$$
\boxed{ \text{어떻게 더 좋은 representation을 만들 것인가} }
$$

와

$$
\boxed{ \text{어떻게 실제 data distribution을 이해하고 생성할 것인가} }
$$

라는 두 질문으로 압축할 수 있다.