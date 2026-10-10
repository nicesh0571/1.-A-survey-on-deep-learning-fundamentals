# 1. 딥러닝이란 무엇인가?

딥러닝을 이해하기 전에 먼저 \*\*신경망(Neural Network)\*\*이 무엇인지 생각해보자.

가장 기본적인 하나의 neuron은 여러 입력값을 받아 각 입력에 weight를 곱하고 bias를 더한 다음 activation function을 통과시킨다.

수식으로 표현하면

\\[ \boxed{ y=f(W^TX+b) } \\]

이다.

논문에서도 전통적인 neural network를 대부분의 deep learning technology의 기반으로 설명하고 있으며, neuron이 weight \\(W\\), bias \\(b\\), activation function \\(f\\)로 구성된다고 정리한다.     1. A survey on deep learning fu…

여기서

\\[ X \\]

는 입력,

\\[ W \\]

는 weight,

\\[ b \\]

는 bias,

\\[ f \\]

는 activation function이다.

쉽게 말하면 신경망은 기본적으로

\\[ \boxed{ \text{입력} \rightarrow \text{선형 변환} \rightarrow \text{비선형 변환} \rightarrow \text{출력} } \\]

의 반복이라고 볼 수 있다.

---

# 2. Weight와 Bias

가장 간단한 neuron부터 살펴보자.

입력이 세 개 있다고 가정하자.

\\[ x_1,\ x_2,\ x_3 \\]

각 입력마다 weight가 존재한다.

\\[ w_1,\ w_2,\ w_3 \\]

그러면 neuron에 들어가는 값은

\\[ z = w_1x_1+w_2x_2+w_3x_3+b \\]

가 된다.

그리고 activation function을 적용해서

\\[ y=f(z) \\]

를 얻는다.

즉 전체 구조는

\\[ x_1,x_2,x_3 \\]

↓

\\[ w_1x_1+w_2x_2+w_3x_3+b \\]

↓

\\[ f(\cdot) \\]

↓

\\[ y \\]

이다.

---

## Weight는 무엇인가?

Weight는 각 입력이 출력에 얼마나 영향을 주는지를 결정한다.

예를 들어 어떤 classification 문제에서

\\[ x_1=\text{색깔 정보} \\]\\[ x_2=\text{모양 정보} \\]\\[ x_3=\text{크기 정보} \\]

라고 생각해보자.

학습 결과

\\[ w_1=0.1,\qquad w_2=2.4,\qquad w_3=0.3 \\]

이 되었다면 해당 neuron에서는 \*\*모양 정보 \\(x_2\\)\*\*가 상대적으로 중요하게 사용되고 있다고 해석할 수 있다.

즉

\\[ \boxed{\text{Weight = feature의 영향도를 조절하는 학습 가능한 parameter}} \\]

라고 이해하면 된다.

---

# 3. Bias는 왜 필요한가?

bias는 입력과 독립적으로 neuron의 출력을 이동시키는 역할을 한다.

가장 간단한 선형식

\\[ y=wx+b \\]

를 생각해보면,

\\(w\\)는 기울기이고

\\[ b \\]

는 절편이다.

bias가 없다면

\\[ y=wx \\]

이므로 모든 함수가 반드시 원점

\\[ (0,0) \\]

을 지나야 한다.

bias를 사용하면

\\[ y=wx+b \\]

가 되어 함수 자체를 이동시킬 수 있다.

따라서 모델이 표현할 수 있는 함수의 범위가 넓어진다.

---

# 4. 하나의 neuron으로는 부족하다

하나의 neuron은 매우 제한된 표현만 가능하다.

그래서 여러 neuron을 하나의 layer에 배치한다.

예를 들어 입력

\\[ X= \begin{bmatrix} x_1\\\ x_2\\\ x_3 \end{bmatrix} \\]

가 있고 hidden neuron이 여러 개라면 weight를 matrix 형태로 표현할 수 있다.

\\[ Z=WX+b \\]

그리고 activation을 적용한다.

\\[ H=f(Z) \\]

여기서 \\(H\\)가 다음 layer의 입력이 된다.

---

# 5. MLP: 여러 Layer를 쌓기

이제 이러한 layer를 여러 개 연결하면 **Multilayer Perceptron, MLP**가 된다.

논문은 MLP를 기본 neural network를 여러 layer로 확장한 구조로 설명한다.     1. A survey on deep learning fu…

예를 들어

\\[ X \rightarrow H_1 \rightarrow H_2 \rightarrow Y \\]

와 같은 구조가 있을 수 있다.

수식으로 표현하면

\\[ H_1=f(W_1X+b_1) \\]\\[ H_2=f(W_2H_1+b_2) \\]\\[ Y=f(W_3H_2+b_3) \\]

가 된다.

한 번에 쓰면

\\[ \boxed{ f(X) = f(W_3 f(W_2 f(W_1X+b_1)+b_2) +b_3) } \\]

처럼 표현할 수 있다.

논문 역시 여러 activation과 linear transformation을 연속적으로 결합한 형태로 MLP를 나타낸다.     1. A survey on deep learning fu…

---

# 6. 왜 'Deep' Learning인가?

신경망의 layer가 여러 층으로 쌓이면 입력과 출력 사이에 여러 hidden layer가 생긴다.

논문에서는 입력과 출력 사이에 있는 층을 hidden layer라고 설명하며, 여러 층으로 구성된 network를 deep neural network의 형태로 설명한다.     1. A survey on deep learning fu…

직관적으로 보면

\\[ \text{Input} \\]

↓

\\[ \text{Hidden Layer 1} \\]

↓

\\[ \text{Hidden Layer 2} \\]

↓

\\[ \text{Hidden Layer 3} \\]

↓

\\[ \text{Output} \\]

처럼 깊어지기 때문에 **Deep Neural Network**라고 부른다.

그렇다면 굳이 여러 layer를 쌓는 이유는 무엇일까?

핵심은

\\[ \boxed{\text{Hierarchical Feature Representation}} \\]

이다.

예를 들어 이미지라면 초기 layer에서는 단순한 edge나 texture 같은 정보를 학습할 수 있고,

더 깊은 layer로 갈수록

\\[ \text{edge} \rightarrow \text{shape} \rightarrow \text{object part} \rightarrow \text{object} \\]

처럼 더 추상적인 representation을 만들 수 있다.

논문도 CNN에서 낮은 layer가 기본 edge와 같은 특징을 검출하고 깊은 layer가 보다 복잡한 object 수준의 정보를 표현한다고 설명한다.     1. A survey on deep learning fu…

---

# 7. 그런데 Linear Layer만 계속 쌓으면 될까?

여기서 중요한 문제가 하나 생긴다.

만약 모든 layer가 단순히 linear transformation만 한다면 어떨까?

예를 들어

\\[ H_1=W_1X \\]\\[ H_2=W_2H_1 \\]

이라고 하자.

두 식을 합치면

\\[ H_2=W_2W_1X \\]

이다.

새로운 matrix

\\[ W'=W_2W_1 \\]

라고 놓으면

\\[ H_2=W'X \\]

이다.

즉 linear layer를 아무리 많이 쌓아도 결국 하나의 linear transformation과 같다.

10개를 쌓더라도

\\[ W\_{10}W_9\cdots W_2W_1X \\]

는

\\[ W'X \\]

로 합칠 수 있다.

따라서

\\[ \boxed{ \text{Linear + Linear + Linear} = \text{Linear} } \\]

이다.

하지만 현실의 데이터는 대부분 단순한 linear relationship만으로 표현할 수 없다.

논문도 convolution과 fully connected operation만 사용하면 neural network가 linear mapping만 표현하게 되므로, 실제 nonlinear data를 다루기 위해 activation function이 필요하다고 설명한다.     1. A survey on deep learning fu…

그래서 필요한 것이 **Activation Function**이다.

---

# 8. Activation Function

Activation function의 핵심 역할은

\\[ \boxed{\text{Non-linearity 추가}} \\]

이다.

구조가

\\[ WX+b \\]

로 끝나는 것이 아니라

\\[ f(WX+b) \\]

가 된다.

이 \\(f\\)가 nonlinear function이기 때문에 network 전체도 nonlinear mapping을 표현할 수 있다.

논문에서는 대표 activation으로

\\[ \text{Sigmoid} \\]\\[ \text{Tanh} \\]\\[ \text{ReLU} \\]

를 설명한다.     1. A survey on deep learning fu…

---

# 9. Sigmoid

Sigmoid 함수는

\\[ \boxed{ \sigma(x) = \frac{1}{1+e^{-x}} } \\]

로 정의된다.

출력 범위는

\\[ 0<\sigma(x)<1 \\]

이다.

그래서 확률처럼 해석하기 좋다.

예를 들어

\\[ \sigma(x)=0.9 \\]

라면 어떤 binary classification 문제에서 positive class일 가능성을 90%처럼 표현할 수 있다.

그래프 형태는 대략 S자 모양이다.

큰 음수가 들어오면 0 근처,

\\[ x\rightarrow-\infty \Rightarrow \sigma(x)\rightarrow0 \\]

큰 양수가 들어오면 1 근처,

\\[ x\rightarrow+\infty \Rightarrow \sigma(x)\rightarrow1 \\]

가 된다.

---

# 10. Sigmoid의 문제: Vanishing Gradient

Sigmoid의 derivative는

\\[ \sigma'(x) = \sigma(x)(1-\sigma(x)) \\]

이다.

그런데 \\(x\\)가 매우 크거나 매우 작으면 sigmoid 값이 0 또는 1에 가까워진다.

예를 들어

\\[ \sigma(x)\approx1 \\]

이라면

\\[ \sigma'(x) \approx 1(1-1)=0 \\]

이다.

반대로

\\[ \sigma(x)\approx0 \\]

이면 역시

\\[ \sigma'(x)\approx0 \\]

이다.

즉 activation이 saturation 영역에 들어가면 gradient가 거의 사라진다.

논문도 sigmoid가 큰 입력이나 작은 입력에서 vanishing gradient 문제를 일으켜 deep network 학습을 어렵게 만들 수 있다고 설명한다. 또한 sigmoid 출력이 zero-centered가 아니라는 점도 gradient descent에 불리할 수 있다고 정리한다.     1. A survey on deep learning fu…

---

# 11. Tanh

Tanh는

\\[ \boxed{ \tanh(x) = \frac{e^x-e^{-x}} {e^x+e^{-x}} } \\]

이다.

출력 범위는

\\[ -1 < \tanh(x) < 1 \\]

이다.

Sigmoid와 달리

\\[ 0 \\]

을 중심으로 출력이 형성된다.

즉

\\[ \boxed{\text{zero-centered}} \\]

라는 장점이 있다.

논문도 Tanh가 sigmoid보다 zero-centered output을 제공한다는 점을 장점으로 설명한다. 하지만 큰 입력에서는 여전히 gradient가 작아져 vanishing gradient 문제가 발생할 수 있다고 지적한다.     1. A survey on deep learning fu…

---

# 12. ReLU

딥러닝에서 가장 대표적인 activation 중 하나가 \*\*ReLU(Rectified Linear Unit)\*\*이다.

수식은 매우 간단하다.

\\[ \boxed{ \text{ReLU}(x) = \max(0,x) } \\]

즉

\\[ \text{ReLU}(x)= \begin{cases} x & x>0\\\ 0 & x\leq0 \end{cases} \\]

이다.     1. A survey on deep learning fu…

예를 들어

\\[ x=-3 \Rightarrow 0 \\]\\[ x=2 \Rightarrow 2 \\]

이다.

양수 영역에서 derivative는

\\[ 1 \\]

이므로 sigmoid나 tanh보다 gradient가 잘 전달될 수 있다.

논문은 ReLU가 convergence를 빠르게 하고 sparse activation을 유도할 수 있어 feature extraction과 overfitting 감소에 도움이 될 수 있다고 설명한다.     1. A survey on deep learning fu…

---

# 13. Activation Function을 정리하면

딥러닝에서는 결국

\\[ z=Wx+b \\]

를 계산하고

\\[ a=f(z) \\]

를 적용하는 과정이 반복된다.

핵심은

\\[ \boxed{ \text{Linear Transformation} + \text{Nonlinear Activation} } \\]

이다.

이 조합을 반복하면서 복잡한 function을 학습하게 된다.

---

# 14. Convolution Layer

이미지 분야에서는 모든 pixel을 fully connected 방식으로 연결하면 parameter가 너무 많아진다.

그래서 CNN에서는 **Convolution Layer**를 사용한다.

논문은 convolution layer를 CNN의 핵심 구성요소로 설명하며, filter 또는 kernel을 입력의 local region, 즉 receptive field에 적용하여 feature map을 생성한다고 설명한다.     1. A survey on deep learning fu…

기본적인 형태는

\\[ \boxed{ z=a(W\*i+b) } \\]

로 표현된다.

여기서

\\[ i \\]

는 input,

\\[ W \\]

는 convolution filter,

\\[ b \\]

는 bias,

\\[ \* \\]

는 convolution,

\\[ a \\]

는 activation function이다.

---

# 15. Convolution의 직관

예를 들어 이미지가

\\[ 224\times224 \\]

이고 작은

\\[ 3\times3 \\]

filter가 있다고 하자.

filter를 이미지 전체에 이동시키면서 각 local region과 연산한다.

즉

\\[ \boxed{ \text{Local Region} \rightarrow \text{Filter} \rightarrow \text{Feature Response} } \\]

이다.

어떤 kernel은 horizontal edge에 민감할 수 있고,

또 다른 kernel은 vertical edge,

또 다른 kernel은 texture나 특정 pattern을 학습할 수 있다.

---

# 16. Receptive Field

Convolution에서 중요한 개념이 **Receptive Field**이다.

하나의 neuron이 입력 전체를 보는 것이 아니라 입력의 일부 local region만 본다.

예를 들어

\\[ 3\times3 \\]

kernel이면 한 번의 convolution에서는 주변

\\[ 3\times3 \\]

영역을 보는 것이다.

이를

\\[ \boxed{\text{Local Receptive Field}} \\]

라고 한다.

CNN이 깊어지면 여러 convolution을 거치면서 실제 receptive field가 점점 커진다.

따라서 초기 layer에서는 작은 edge를 보고, 깊은 layer에서는 더 큰 object structure를 볼 수 있게 된다.

---

# 17. Weight Sharing

CNN이 효율적인 또 하나의 중요한 이유가

\\[ \boxed{\text{Weight Sharing}} \\]

이다.

하나의 \\(3\times3\\) filter를 이미지 왼쪽 위에서 사용하고,

같은 filter를 가운데에서도 사용하고,

오른쪽 아래에서도 사용한다.

즉 위치마다 새로운 weight를 만들지 않는다.

논문도 CNN이 local region에서 학습한 convolution weight를 전체 이미지에 공유함으로써 parameter 수를 줄인다고 설명한다.     1. A survey on deep learning fu…

---

# 18. Convolution의 결과: Feature Map

한 개의 filter가 이미지 전체를 통과하면 하나의

\\[ \boxed{\text{Feature Map}} \\]

이 생성된다.

예를 들어 filter가 64개라면

\\[ 64 \\]

개의 feature map이 생성된다.

이를 tensor로 보면 대략

\\[ H\times W\times C \\]

형태가 된다.

여기서

\\[ C \\]

가 channel이다.

---

# 19. Normalization은 왜 필요한가?

Deep network에서는 layer가 많아지면서 각 layer로 들어오는 값의 scale이나 distribution이 달라질 수 있다.

이 때문에 training이 불안정하거나 느려질 수 있다.

논문은 이를 설명하면서 \*\*Batch Normalization(BN)\*\*을 중요한 구성요소로 다룬다. BN은 layer input을 정규화한 뒤 다시 학습 가능한 scale과 shift를 적용한다고 설명한다.     1. A survey on deep learning fu…

---

# 20. Batch Normalization

mini-batch 안에 값들이

\\[ x_1,x_2,\dots,x_m \\]

있다고 해보자.

먼저 평균을 구한다.

\\[ \mu_B = \frac{1}{m} \sum\_{i=1}^{m}x_i \\]

variance를 계산한다.

\\[ \sigma_B^2 = \frac{1}{m} \sum\_{i=1}^{m}(x_i-\mu_B)^2 \\]

그리고 normalize한다.

\\[ \hat{x}\_i = \frac{x_i-\mu_B} {\sqrt{\sigma_B^2+\epsilon}} \\]

그러면 평균이 대략 0,

variance가 대략 1인 distribution으로 변한다.

하지만 항상 평균 0, variance 1로 고정하면 network 표현력이 제한될 수 있기 때문에 학습 가능한 parameter

\\[ \gamma \\]

와

\\[ \beta \\]

를 추가한다.

\\[ \boxed{ y_i = \gamma\hat{x}\_i+\beta } \\]

즉

\\[ \gamma \\]

는 scale,

\\[ \beta \\]

는 shift 역할을 한다.

---

# 21. BN의 장점

논문에서는 BN이 gradient를 유지하고 training convergence를 빠르게 하는 역할을 한다고 설명한다.     1. A survey on deep learning fu…

직관적으로 보면 각 layer의 input distribution이 너무 크게 흔들리지 않도록 조절하는 것이다.

즉

\\[ \boxed{ \text{More Stable Training} } \\]

을 만드는 장치라고 이해하면 된다.

---

# 22. BN의 한계

하지만 Batch Normalization에는 중요한 문제가 있다.

이름 그대로 **Batch** statistics를 사용한다.

즉 mini-batch가 너무 작으면

\\[ \mu_B,\sigma_B^2 \\]

의 추정이 불안정해진다.

논문도 small batch에서는 mean과 variance 추정이 부정확해질 수 있으며, 반대로 큰 batch는 GPU memory를 많이 요구한다고 설명한다.     1. A survey on deep learning fu…

그래서 등장한 것이

\\[ \boxed{\text{Group Normalization}} \\]

과

\\[ \boxed{\text{Layer Normalization}} \\]

같은 방식이다.

---

# 23. BN / GN / LN의 차이

개념적으로만 이해하면 충분하다.

BN은

\\[ \boxed{\text{Batch 방향으로 normalize}} \\]

한다.

GN은 channel을 여러 group으로 나눈다.

\\[ \boxed{\text{Channel Group 단위 normalize}} \\]

한다.

LN은 하나의 sample 내부 feature를 기준으로 normalize한다.

\\[ \boxed{\text{Layer 단위 normalize}} \\]

한다.

논문은 GN과 LN이 batch size에 의존하지 않는 normalization 대안으로 사용될 수 있다고 설명한다.     1. A survey on deep learning fu…

특히 Transformer 계열에서는 LayerNorm이 매우 중요하지만, 이는 이후 Transformer를 정리할 때 다시 자세히 보자.

---

# 24. Pooling Layer

CNN에서 convolution으로 feature를 뽑았다면 feature map의 spatial resolution을 줄이는 과정이 필요할 수 있다.

이때 사용하는 것이

\\[ \boxed{\text{Pooling}} \\]

이다.

논문은 pooling layer가 convolution과 activation 뒤에 사용되어 feature map의 spatial dimension을 줄이고, computation을 줄이면서 overfitting 위험도 완화할 수 있다고 설명한다.     1. A survey on deep learning fu…

---

# 25. Max Pooling

예를 들어

\\[ 2\times2 \\]

영역이

\\[ \begin{bmatrix} 1&3\\\ 2&5 \end{bmatrix} \\]

라고 하자.

Max Pooling은 가장 큰 값만 선택한다.

\\[ \boxed{5} \\]

즉

\\[ \text{MaxPool}(1,3,2,5)=5 \\]

이다.

이렇게 하면 중요한 activation은 남기면서 feature map 크기를 줄일 수 있다.

---

# 26. Average Pooling

Average Pooling은 해당 영역의 평균을 사용한다.

같은 예제라면

\\[ \frac{1+3+2+5}{4} = 2.75 \\]

가 된다.

즉

\\[ \boxed{ \text{Max Pooling} = \text{가장 강한 특징 선택} } \\]

이라면,

\\[ \boxed{ \text{Average Pooling} = \text{영역 전체 평균 정보} } \\]

라고 이해할 수 있다.

---

# 27. Non-overlapping Pooling과 Overlapping Pooling

논문에서는 일반적인 pooling뿐만 아니라 **overlapping pooling**도 설명한다.     1. A survey on deep learning fu…

일반적으로 pooling window가

\\[ 2\times2 \\]

이고 stride도

\\[ 2 \\]

라면 각 window가 겹치지 않는다.

반면

\\[ \text{window size} > \text{stride} \\]

이면 pooling 영역이 서로 겹친다.

이를

\\[ \boxed{\text{Overlapping Pooling}} \\]

이라고 한다.

AlexNet에서도 이 방법이 사용되었다고 논문은 설명한다.     1. A survey on deep learning fu…

---

# 28. Spatial Pyramid Pooling

일반적인 Fully Connected Layer는 입력 크기가 고정되어야 한다.

그런데 실제 이미지 크기는 다양하다.

이를 해결하기 위한 방법 중 하나가

\\[ \boxed{\text{Spatial Pyramid Pooling, SPP}} \\]

이다.

논문에서는 SPP가 arbitrary-size convolution feature를 fixed-length vector로 변환해 FC layer에 전달할 수 있게 한다고 설명한다. 이 과정에서 단순 crop이나 image warping으로 인한 정보 손실을 줄일 수 있다고 정리한다.     1. A survey on deep learning fu…

---

# 29. Global Average Pooling

더 중요한 개념이

\\[ \boxed{\text{Global Average Pooling, GAP}} \\]

이다.

feature map 하나가

\\[ H\times W \\]

라면 해당 feature map의 모든 spatial value의 평균을 하나로 만든다.

예를 들어 channel \\(c\\)에 대해

\\[ GAP_c = \frac{1}{HW} \sum\_{i=1}^{H} \sum\_{j=1}^{W} x\_{ijc} \\]

이다.

만약 feature map이

\\[ 7\times7\times512 \\]

라면 GAP 이후에는

\\[ 1\times1\times512 \\]

가 된다.

즉 channel 하나당 숫자 하나만 남긴다.

---

# 30. 왜 GAP를 쓰는가?

Fully Connected Layer는 parameter가 매우 많아질 수 있다.

예를 들어

\\[ 7\times7\times512 \\]

feature map을 flatten하면

\\[ 25088 \\]

개의 feature가 된다.

이를 4096개의 neuron에 연결하면 weight 개수만

\\[ 25088\times4096 \\]

개다.

약

\\[ 102\text{ million} \\]

개의 parameter가 필요하다.

반면 GAP를 쓰면 각 channel 하나의 값으로 요약할 수 있다.

논문도 ResNet과 GoogLeNet 같은 network에서 FC 대신 GAP를 사용하여 feature를 통합하고 model size와 overfitting을 줄일 수 있다고 설명한다.     1. A survey on deep learning fu…

---

# 31. Fully Connected Layer

Fully Connected Layer는 이전 layer의 모든 neuron과 다음 layer의 neuron을 연결한다.

수식은

\\[ \boxed{ y=Wx+b } \\]

이다.

CNN에서는 convolution으로 feature를 뽑은 뒤 최종 classification 단계에서 FC layer를 사용하는 전통적인 구조가 많았다.

즉

\\[ \text{Image} \\]

↓

\\[ \text{Convolution} \\]

↓

\\[ \text{Feature Extraction} \\]

↓

\\[ \text{Pooling} \\]

↓

\\[ \text{FC} \\]

↓

\\[ \text{Classification} \\]

이다.

논문도 FC layer가 pooling 이후 학습한 distributed feature representation을 label space로 mapping한다고 설명한다.     1. A survey on deep learning fu…

---

# 32. Forward Propagation

지금까지 설명한 것들을 이용해서 모델이 prediction을 만드는 과정을

\\[ \boxed{\text{Forward Propagation}} \\]

이라고 한다.

예를 들어 MLP라면

\\[ X \\]

↓

\\[ Z_1=W_1X+b_1 \\]

↓

\\[ H_1=f(Z_1) \\]

↓

\\[ Z_2=W_2H_1+b_2 \\]

↓

\\[ \hat{Y}=f(Z_2) \\]

순서다.

여기서

\\[ \hat{Y} \\]

가 model prediction이다.

---

# 33. 그런데 Prediction이 틀렸다면?

예를 들어 정답이

\\[ Y=1 \\]

인데 모델 예측은

\\[ \hat{Y}=0.2 \\]

라고 하자.

모델이 틀렸다는 것을 숫자로 표현해야 한다.

이 역할을 하는 것이

\\[ \boxed{\text{Loss Function}} \\]

이다.

---

# 34. Loss Function

논문에서는 loss function을 prediction과 실제 target 사이의 차이를 계산하여 model parameter를 업데이트하기 위한 함수라고 설명한다.     1. A survey on deep learning fu…

즉

\\[ \boxed{ L(\hat{Y},Y) } \\]

이다.

prediction이 정답에 가까우면 loss가 작고,

틀리면 loss가 커지도록 설계한다.

학습 목표는 결국

\\[ \boxed{ \min\_\theta L(\theta) } \\]

이다.

여기서

\\[ \theta \\]

는 network의 모든 parameter를 의미한다.

\\[ \theta= \\{W_1,b_1,W_2,b_2,\dots\\} \\]

이다.

---

# 35. Mean Squared Error

Regression이나 image restoration에서 대표적으로 사용되는 것이

\\[ \boxed{\text{Mean Squared Error, MSE}} \\]

이다.

\\[ MSE = \frac{1}{N} \sum\_{i=1}^{N} (y_i-\hat{y}\_i)^2 \\]

이다.

prediction이 정답에서 멀수록 제곱해서 penalty가 커진다.

예를 들어

\\[ y=1 \\]

이고

\\[ \hat y=0.8 \\]

이면

\\[ (1-0.8)^2=0.04 \\]

이다.

논문에서도 mean square 계열 loss가 image denoising 및 resolution과 같은 reconstruction 문제에 사용될 수 있다고 설명한다.     1. A survey on deep learning fu…

---

# 36. Cross Entropy

Classification에서 가장 자주 사용되는 loss 중 하나가

\\[ \boxed{\text{Cross Entropy}} \\]

이다.

예를 들어 정답 class가 cat인데 model output이

\\[ P(\text{cat})=0.9 \\]

이면 loss는 작아진다.

반대로

\\[ P(\text{cat})=0.05 \\]

라면 loss는 크게 나온다.

binary classification에서는

\\[ L = -[y\log(p)+(1-y)\log(1-p)] \\]

와 같이 표현할 수 있다.

multi-class에서는

\\[ L = -\sum\_{c=1}^{C}y_c\log p_c \\]

이다.

---

# 37. Softmax

multi-class classification에서는 마지막 output을 probability distribution으로 만들기 위해 Softmax를 많이 사용한다.

\\[ \boxed{ P(y=i) = \frac{e^{z_i}} {\sum_j e^{z_j}} } \\]

예를 들어 raw output, 즉 logits가

\\[ [2.0,\ 1.0,\ 0.1] \\]

이라면 softmax를 거쳐

\\[ [0.66,\ 0.24,\ 0.10] \\]

같은 확률 distribution으로 바뀐다.

그리고

\\[ \sum_iP(y=i)=1 \\]

이 된다.

참고로 이 논문에서는 Softmax도 loss function 목록에 포함해 표현하는 부분이 있다.     1. A survey on deep learning fu…\
엄밀한 현대적 표현에서는 **Softmax 자체는 보통 output activation/probability normalization이고, classification loss로는 Softmax와 Cross Entropy를 결합한 Softmax Cross-Entropy를 사용하는 경우가 일반적**이라고 구분해 이해하는 편이 정확하다.

---

# 38. Gradient Descent

Loss를 계산했다고 끝이 아니다.

이제 loss가 작아지는 방향으로 weight를 바꿔야 한다.

이를 위해 gradient를 계산한다.

\\[ \frac{\partial L}{\partial W} \\]

이 값은

> weight \\(W\\)를 조금 바꾸면 loss가 얼마나 변하는가?

를 의미한다.

Gradient Descent에서는

\\[ \boxed{ W \leftarrow W-\eta \frac{\partial L}{\partial W} } \\]

로 parameter를 업데이트한다.

여기서

\\[ \eta \\]

는

\\[ \boxed{\text{Learning Rate}} \\]

이다.

---

# 39. Learning Rate

Learning rate는 한 번 업데이트할 때 얼마나 크게 움직일지를 결정한다.

너무 크면

\\[ \text{Minimum} \\]

을 지나쳐서 training이 불안정할 수 있다.

너무 작으면

\\[ \text{training} \\]

이 지나치게 느려질 수 있다.

즉

\\[ \boxed{ \eta \text{가 너무 큼} \Rightarrow \text{불안정} } \\]\\[ \boxed{ \eta \text{가 너무 작음} \Rightarrow \text{느린 학습} } \\]

이다.

---

# 40. Backpropagation

Deep Neural Network는 parameter가 수백만 개, 수십억 개가 될 수 있다.

그렇다면 모든 parameter가 loss에 어떤 영향을 주는지 어떻게 계산할까?

바로

\\[ \boxed{\text{Backpropagation}} \\]

이다.

논문 역시 neural network에서 prediction과 target을 비교한 뒤 backpropagation으로 parameter를 조정한다고 설명한다.     1. A survey on deep learning fu…

---

# 41. Backpropagation의 핵심은 Chain Rule

예를 들어

\\[ x \rightarrow z \rightarrow a \rightarrow L \\]

구조가 있다고 하자.

그러면

\\[ \frac{\partial L}{\partial x} \\]

를 직접 계산하는 대신 chain rule을 사용한다.

\\[ \boxed{ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial x} } \\]

즉 output 쪽에서 input 쪽으로 gradient를 전달한다.

그래서 이름이

\\[ \boxed{\text{Back-propagation}} \\]

이다.

---

# 42. Forward와 Backward

전체 training을 가장 간단하게 표현하면 다음과 같다.

\\[ \boxed{\text{1. Forward}} \\]

입력을 model에 넣는다.

\\[ X \rightarrow f\_\theta(X) \rightarrow \hat Y \\]

↓

\\[ \boxed{\text{2. Loss}} \\]

prediction과 target을 비교한다.

\\[ L(\hat Y,Y) \\]

↓

\\[ \boxed{\text{3. Backward}} \\]

gradient를 계산한다.

\\[ \nabla\_\theta L \\]

↓

\\[ \boxed{\text{4. Update}} \\]

parameter를 수정한다.

\\[ \theta \leftarrow \theta-\eta\nabla\_\theta L \\]

↓

다시 Forward.

이 과정을 수없이 반복한다.

---

# 43. Epoch / Batch / Iteration

여기서 training을 이해하기 위해 세 가지 용어도 알아두면 좋다.

전체 training dataset을 한 번 모두 사용하면

\\[ \boxed{\text{1 Epoch}} \\]

이다.

예를 들어 dataset이

\\[ 10,000 \\]

개이고 batch size가

\\[ 100 \\]

이라면 하나의 batch에는 100개 sample이 들어간다.

그러면 1 epoch에

\\[ \frac{10,000}{100}=100 \\]

번 parameter update가 일어난다.

즉

\\[ \boxed{\text{Batch Size}=100} \\]\\[ \boxed{\text{Iteration per Epoch}=100} \\]

이 된다.

---

# 44. Vanishing Gradient

Deep Learning 역사에서 매우 중요한 문제가 바로

\\[ \boxed{\text{Vanishing Gradient}} \\]

이다.

Backpropagation은 chain rule을 이용한다고 했다.

예를 들어 10개 layer를 거쳐야 한다면

\\[ \frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial H\_{10}} \frac{\partial H\_{10}}{\partial H_9} \cdots \frac{\partial H_2}{\partial H_1} \frac{\partial H_1}{\partial W_1} \\]

처럼 수많은 derivative가 곱해진다.

만약 각각이

\\[ 0.1 \\]

정도라면

\\[ 0.1^{10} = 10^{-10} \\]

이 된다.

즉 앞쪽 layer까지 gradient가 거의 전달되지 않는다.

\\[ \boxed{ \text{Gradient} \rightarrow 0 } \\]

이것이 vanishing gradient다.

논문 역시 sigmoid saturation으로 인한 gradient vanishing이 deep network training을 어렵게 만들었다고 설명한다.     1. A survey on deep learning fu…

---

# 45. 옛날 Deep Network는 어떻게 학습했을까?

초기의 깊은 network는 지금처럼 그냥 end-to-end로 쉽게 학습하기 어려웠다.

그래서 Hinton 등의 연구에서는

\\[ \boxed{\text{Unsupervised Pre-training}} \\]

을 먼저 하고,

그 뒤

\\[ \boxed{\text{Supervised Fine-tuning}} \\]

을 수행하는 방식이 사용되었다.

논문은 stacked autoencoder(SAE)나 deep belief network(DBN) 등을 이용한 pretraining이 gradient 문제를 완화하는 방법으로 사용되었지만, 과정이 복잡하고 시간이 많이 걸린다고 설명한다.     1. A survey on deep learning fu…

즉 과거에는

\\[ \text{Unsupervised Pre-training} \\]

↓

\\[ \text{Supervised Fine-tuning} \\]

처럼 여러 단계를 거쳤다.

---

# 46. 현대 Deep Learning: End-to-End

이후 activation, initialization, normalization, architecture, GPU 등의 발전으로 복잡한 pretraining 없이도 network 전체를 한 번에 학습하는 방식이 일반화됐다.

즉

\\[ \boxed{ X \rightarrow \text{Network} \rightarrow Y } \\]

전체를 하나의 loss를 이용해 동시에 optimize한다.

이를

\\[ \boxed{\text{End-to-End Learning}} \\]

이라고 한다.

논문도 modern deep learning에서는 CNN 계열, ResNet 및 RNN 등의 network가 end-to-end supervised training 방식으로 널리 사용됐다고 설명한다.     1. A survey on deep learning fu…

---

# 47. 딥러닝의 전체 학습 과정을 하나로 연결하면

지금까지 나온 모든 개념을 하나의 흐름으로 연결해보자.

예를 들어 이미지 classification을 한다고 하자.

입력:

\\[ X=\text{Image} \\]

---

### Step 1. Feature Extraction

Convolution을 이용한다.

\\[ F=W\*X+b \\]

---

### Step 2. Normalization

\\[ \hat F=BN(F) \\]

---

### Step 3. Non-linearity

\\[ A=ReLU(\hat F) \\]

---

### Step 4. Downsampling

\\[ P=Pooling(A) \\]

---

이 과정을 여러 번 반복한다.

\\[ \text{Conv} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Pooling} \\]

↓

\\[ \text{Conv} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Pooling} \\]

↓

feature representation 생성.

---

### Step 5. Prediction

마지막 feature를 FC 또는 GAP을 이용해 classification한다.

\\[ z=W_cF+b_c \\]

Softmax:

\\[ P(y=c) = \frac{e^{z_c}} {\sum_k e^{z_k}} \\]

---

### Step 6. Loss

정답이 cat인데

\\[ P(\text{cat})=0.3 \\]

밖에 안 나왔다고 해보자.

Cross Entropy를 계산한다.

\\[ L = -\log0.3 \\]

---

### Step 7. Backpropagation

\\[ \frac{\partial L}{\partial W} \\]

를 모든 layer에 대해 계산한다.

---

### Step 8. Parameter Update

\\[ W \leftarrow W-\eta \frac{\partial L}{\partial W} \\]

---

그리고 다시 새로운 image batch를 넣는다.

이 과정을 반복하면서

\\[ \boxed{ \text{Loss}\downarrow } \\]

하도록 network가 학습된다.

---

# 48. Training과 Inference의 차이

이것도 매우 중요하다.

## Training

training에서는

\\[ X,Y \\]

둘 다 있다.

즉 input과 정답을 같이 가지고 있다.

\\[ X \rightarrow Model \rightarrow \hat Y \\]

↓

\\[ \hat Y \leftrightarrow Y \\]

↓

\\[ Loss \\]

↓

\\[ Backpropagation \\]

↓

\\[ Parameter Update \\]

이다.

---

## Inference

학습이 끝난 뒤에는 정답이 없다.

새로운 input만 준다.

\\[ X\_{\text{new}} \rightarrow Model \rightarrow \hat Y \\]

끝이다.

Backpropagation도 하지 않고 parameter도 업데이트하지 않는다.

즉

\\[ \boxed{ \text{Training} = \text{Forward + Loss + Backward + Update} } \\]

이고,

\\[ \boxed{ \text{Inference} = \text{Forward only} } \\]

라고 기억하면 된다.

---

# 49. Feature란 무엇인가?

Deep Learning을 공부하면 계속 등장하는 단어가

\\[ \boxed{\text{Feature}} \\]

이다.

Feature는 model이 판단을 위해 사용하는 정보라고 생각하면 된다.

전통적인 computer vision에서는 사람이 직접 feature를 설계했다.

예를 들어

\\[ \text{edge} \\]\\[ \text{corner} \\]\\[ \text{color histogram} \\]\\[ \text{texture} \\]

등이다.

하지만 deep learning에서는

\\[ \boxed{\text{Feature 자체도 network가 학습}} \\]

한다.

그래서

\\[ \text{Raw Input} \rightarrow \text{Representation Learning} \rightarrow \text{Task Output} \\]

이라는 구조가 된다.

---

# 50. Representation Learning

딥러닝의 핵심을 한 단어로 표현하면

\\[ \boxed{\text{Representation Learning}} \\]

이라고 볼 수 있다.

예를 들어 raw image의 pixel은

\\[ [127,32,250,\dots] \\]

같은 숫자일 뿐이다.

하지만 neural network를 통과하면서

\\[ \text{Pixels} \\]

↓

\\[ \text{Edges} \\]

↓

\\[ \text{Textures} \\]

↓

\\[ \text{Shapes} \\]

↓

\\[ \text{Semantic Features} \\]

처럼 task에 필요한 representation으로 변환된다.

그리고 마지막 representation을 이용해

\\[ \text{Classification} \\]\\[ \text{Detection} \\]\\[ \text{Segmentation} \\]

등을 수행한다.

---

# 51. Deep Learning에서 중요한 것은 Architecture만이 아니다

처음 공부할 때는

> CNN은 어떤 구조고 Transformer는 어떤 구조인가?

에 집중하게 된다.

하지만 실제 deep learning은

\\[ \boxed{ \text{Architecture} + \text{Data} + \text{Loss} + \text{Optimization} + \text{Training Strategy} } \\]

가 함께 작동한다.

같은 architecture라도

- 어떤 data를 쓰는가
- 어떤 preprocessing을 하는가
- 어떤 loss를 쓰는가
- learning rate가 얼마인가
- optimizer가 무엇인가
- augmentation을 어떻게 하는가

에 따라 성능이 크게 달라질 수 있다.

이 점은 이후 AlexNet, ResNet, Transformer 등을 볼 때 계속 등장한다.

---

# 52. Deep Learning의 기본 Pipeline

이번 글 전체를 하나의 그림으로 요약하면 다음과 같다.

\\[ \boxed{\text{Input Data}} \\]

↓

\\[ \boxed{\text{Feature Extraction}} \\]

Convolution / Linear Layer

↓

\\[ \boxed{\text{Normalization}} \\]

BN / GN / LN

↓

\\[ \boxed{\text{Activation}} \\]

ReLU / Sigmoid / Tanh

↓

\\[ \boxed{\text{Feature Reduction}} \\]

Pooling / GAP

↓

\\[ \boxed{\text{Prediction}} \\]

FC / classifier

↓

\\[ \boxed{\text{Loss}} \\]

Cross Entropy / MSE

↓

\\[ \boxed{\text{Backpropagation}} \\]

Gradient 계산

↓

\\[ \boxed{\text{Optimization}} \\]

Parameter Update

이 과정을 반복하면서 model이 data로부터 representation을 학습한다.

---

# 53. 각 구성요소의 역할 한 번에 정리

| 구성 요소               | 역할                                          |
| ------------------- | ------------------------------------------- |
| Weight              | 입력 feature의 영향도를 결정하는 학습 가능한 parameter      |
| Bias                | activation 위치를 이동시켜 표현력 증가                  |
| Linear / FC         | feature를 새로운 공간으로 mapping                   |
| Convolution         | local pattern을 추출                           |
| Activation          | network에 non-linearity 추가                   |
| Batch Normalization | feature distribution을 안정화하여 학습 지원           |
| Pooling             | spatial dimension과 computation 감소           |
| GAP                 | feature map을 channel별 하나의 값으로 압축            |
| Softmax             | logits를 class probability distribution으로 변환 |
| Loss                | prediction과 target 사이의 error 측정             |
| Backpropagation     | 각 parameter가 loss에 미친 영향을 계산                |
| Gradient Descent    | loss를 줄이는 방향으로 parameter update             |

논문도 convolution, normalization, activation, pooling, fully connected layer, loss function을 deep network를 이해하기 위한 핵심 기본 요소로 정리한다.     1. A survey on deep learning fu…

---

# 54. 가장 중요한 흐름

Deep Learning을 처음 공부할 때 아래 흐름이 머릿속에 있으면 이후 대부분의 모델을 훨씬 쉽게 이해할 수 있다.

\\[ \boxed{ X \xrightarrow{\text{Network}} \hat Y } \\]

↓

\\[ \boxed{ L(\hat Y,Y) } \\]

↓

\\[ \boxed{ \nabla\_\theta L } \\]

↓

\\[ \boxed{ \theta \leftarrow \theta-\eta\nabla\_\theta L } \\]

즉,

**1. 입력을 넣는다.**

\\[ X \\]

**2. 모델이 예측한다.**

\\[ \hat Y \\]

**3. 정답과 비교한다.**

\\[ L(\hat Y,Y) \\]

**4. 잘못된 정도를 backward한다.**

\\[ \nabla\_\theta L \\]

**5. parameter를 수정한다.**

\\[ \theta\_{\text{new}} = \theta\_{\text{old}} - \eta\nabla\_\theta L \\]

**6. 반복한다.**

이것이 거의 모든 supervised deep learning의 기본 학습 구조다.

---

# 55. 결국 Deep Learning이 학습한다는 것은 무엇인가?

많이 헷갈리는 부분인데, model이 실제로 저장하는 것은 우리가 생각하는

> "고양이는 귀가 있고 수염이 있다."

같은 문장이 아니다.

실제로 학습되는 것은 수많은

\\[ W,b \\]

parameter 값이다.

즉 training은

\\[ W_1,W_2,\dots,W_n \\]

을 data에 맞게 계속 조정하는 과정이다.

예를 들어 초기에는 random weight에서 시작해서

\\[ W^{(0)} \\]

training을 반복하면서

\\[ W^{(1)} \\]\\[ W^{(2)} \\]\\[ W^{(3)} \\]\\[ \vdots \\]\\[ W^{(\*)} \\]

로 변화한다.

최종적으로 loss가 작은 parameter 조합을 찾는다.

따라서

\\[ \boxed{ \text{Deep Learning} = \text{Data로부터 적절한 parameter와 representation을 학습하는 과정} } \\]

이라고 이해할 수 있다.

---

# 56. 이 기본 구조에서 모든 모델이 갈라진다

그리고 여기서부터 각각의 deep learning architecture가 갈라진다.

이미지의 local pattern을 효율적으로 학습하려면

\\[ \boxed{\text{CNN}} \\]

을 사용한다.

새로운 data distribution을 생성하고 싶다면

\\[ \boxed{\text{GAN}} \\]

같은 generative architecture를 사용할 수 있다.

sequence 전체의 관계를 attention으로 보고 싶다면

\\[ \boxed{\text{Transformer}} \\]

를 사용한다.

noise에서 점진적으로 data를 복원하면서 생성하고 싶다면

\\[ \boxed{\text{Diffusion Model}} \\]

을 사용한다.

결국 이 모델들은 완전히 별개의 세계가 아니라,

\\[ \boxed{ \text{Deep Learning Fundamentals} } \\]

이라는 공통 토대 위에서 **정보를 처리하는 방식과 학습 목적을 다르게 설계한 architecture**라고 볼 수 있다.

---

# 마무리

딥러닝의 가장 기본적인 구조는 생각보다 단순하다.

\\[ \boxed{ \text{Input} \rightarrow \text{Feature Extraction} \rightarrow \text{Prediction} \rightarrow \text{Loss} \rightarrow \text{Backpropagation} \rightarrow \text{Parameter Update} } \\]

이 흐름을 수많은 layer와 parameter로 확장한 것이 modern deep learning이다.

그리고 network 내부에서는

\\[ \text{Linear / Convolution} \\]

을 통해 feature를 변환하고,

\\[ \text{Activation} \\]

을 통해 non-linearity를 추가하고,

\\[ \text{Normalization} \\]

을 이용해 training을 안정화하며,

\\[ \text{Pooling / GAP} \\]

으로 feature를 정리하고,

최종적으로

\\[ \text{Loss} \\]

를 이용하여 model이 얼마나 틀렸는지를 계산한다.

그 loss를 backpropagation하여

\\[ W,b \\]

를 반복적으로 수정하는 것이 딥러닝 학습의 본질이다. 논문 역시 neural network의 기본 구성요소와 MLP, backpropagation, 그리고 이후 현대적인 end-to-end network로 이어지는 흐름을 deep learning의 기초로 설명한다.&#x20;