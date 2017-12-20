# Caffe : Convolutional Architecture for Fast Feature Embedding

Caffe는 다양한 딥러닝 프레임워크들 중 하나로 버클리 인공지능 연구소 (Berkeley AI Research([BAIR](http://bair.berkeley.edu)))/ 버클리 비전 & 학습 센터 (The Berkeley Vision and Learning Center (BVLC)), 그리고 Caffe 커뮤니티의 수많은 기여자들에의해 개발된 것입니다.

좀 더 자세한 사항들을 확인하고자 한다면 [Caffe 공식 사이트](http://caffe.berkeleyvision.org)를 확인해주세요. 

[해당 공식 사이트(이전 버전)](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Documentation-:-Caffe-Tutorial-(Kor)) 번역 또한 같이 첨부하였습니다.

## [Caffe 튜토리얼](https://github.com/Hahnnz/Caffe_Tutorial/wiki)

Caffe는 딥러닝 프레임워크로 이 튜토리얼은 Caffe의 원리, 구성 그리고 사용법에 대해서 설명할 것입니다.

이 튜토리얼은 프레임워크 튜토리얼로써 작성한 것이며, 딥러닝의 전체 과정과 맥락 그리고 한계에 대해서는 다루지는 않습니다. 

이 튜토리얼이 여러분들에게 유용한 설명이 되기를 바라며 기계학습의 배경지식과 신경망에 대한 이해가 이 튜토리얼을 이해하는데에 도움이 될 것입니다.

**이 튜토리얼은 [Caffe 공식 튜토리얼](http://caffe.berkeleyvision.org/tutorial/)을 기반으로 하여 작성하였습니다.**

### 이론
여러분들은 해당 Caffe 튜토리얼을 진행하실 때마다 다음과 같은 점들을 이해하실 수 있을 것입니다.

1. 표현 : 모델과 최적화는 일반적 코딩 작업과는 달리 일반적문장으로 이루어진 평문을 작성하듯이 선언을 할 수 있습니다.

2. 속도 : 산업과 연구분야 쪽에서 수많은 데이터를 처리하고 이에 대한 최신의 모델에 있어 처리속도는 중요합니다.	

3. 모듈성 : 새로운 업무(계층의 새로운 역할)와 환경(구동환경)은 유연성과 확장성을 필요로 합니다.	

4. 공개성 : 과학적이고 응용된 진보기술들은 공동의 코드와 참조 모델들에 대한 재생산성이 좋아야합니다.	

5. 공동성 : 학문적 연구, 초기 프로토타입 그리고 산업분야의 응용품들 모두가 합동 연구회와 BSD-2프로젝트의 분야로 강하게 공유하고 있습니다.	

### Caffe 튜토리얼
그리고 다음과 같은 항목을 이 튜토리얼에서 알려주고자 합니다.

1.	[Nets, layers, Bolbs](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-1.Blobs,-Layers,-and-Nets-(Kor)) : ‘Caffe’ 모델의 분석
2.	[Forward / Backward](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-2.Forward-and-Backward-(kor)) : 계층화로 구성된 모델의 필수적인 연산
3.	[Loss](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-3.Loss-(Kor)) : 학습되어야할 업무(계층)를 손실로 정의합니다.
4.	[Solver] (https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-4.Solver-(Kor)) : Solver는 모델 최적화를 수행해줍니다.
5.	[Layer Catalogue](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-5.Layer-Catalogue-(Kor)) : 계층은 최신모델에 대한 계층을 포함하는 ‘Caffe’ 카탈로그인 모델링과 연산의 기본단위
6.	[Interfaces](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-6.Interface-(Kor)) : 커맨드 라인, Python, Matlab Caffe를 사용.
7.	[Data](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-7.Data-(Kor)) : 모델 입력에 대하여 어떻게 caffe화를 할 것인가
8.	[Caffeinated Convolution](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Tutorial-:-8.Caffeinated-Convolution-(Kor)) : 어떻게 Caffe가 컨볼루션을 계산할까 (심화내용)

### 심층학습 (Deeper Learning)
우리가 다루는 튜토리얼을 수행하는 심층학습에 대한 이해에 도움이 될 만한 참고사항이 온라인상에 많이 공개되어 있습니다. 
이는 입문사항과 발전된 요소, 배경지식과 역사 그리고 기술을 다룹니다.
CVPR’14에서 [시각에 대한 심층학습(Deep Learning)에 대한 튜토리얼](https://sites.google.com/site/deeplearningcvpr2014/)은 연구자들이 참고하기 좋은 튜토리얼입니다. 
여러분들이 Caffe 튜토리얼로부터 실제 기반과 작동, 프레임워크를 알게된다면, CVPR’14 tutorial에서 향상된 연구방향과 
핵심적인 아이디어를 탐구할 수 있을 것입니다.
이 입문서는 Michael Nielsen에 의한 [신경망과 심층학습](http://neuralnetworksanddeeplearning.com/index.html/)을 무료 온라인 드래프트로 제공합니다. 
실제로 신경망과 ‘backpropagation’이 작동하는 방법을 다루는 챕터들이 이 분야가 처음이라면 도움이 될 것입니다.
이러한 학문적 튜토리얼은 기계학습과 영상분야의 연구자들을 위한 심층학습에 대해 다룹니다.
코드와 회로상에서 신경망에 대한 해석으로, Andrej Karpathy (Stanford)가 작성한 
[프로그래머의 관점으로 이해하는 신경망들](http://karpathy.github.io/neuralnets/)를 참고해보세요!

## 커스텀 배포판
Caffe는 Intel에서 다중 노드를 지원해주고 CPU(특히 제논 프로세서 (HSW, BDW, SKX, Xeon Phi))에 좀 더 최적화된 Caffe를 만들어 배포하고 있으며 Window에서도 사용이 가능한 Caffe 등등을 배포하고 있습니다.


 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## 라이센스 및 인용

Caffe는 [BSD 2-Clause 라이센스](https://github.com/BVLC/caffe/blob/master/LICENSE)하에 배포됩니다.

BAIR/BVLC가 제공하는 신경망 모델들은 자유롭게 사용이 가능하도록 배포되었습니다.

Caffe가 여러분의 연구에 도움이 되었다면 여러분들의 출판물에 Caffe를 인용 자료로써 넣어주세요!

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
