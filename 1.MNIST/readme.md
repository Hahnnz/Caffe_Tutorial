---
제목: LeNet MNIST Tutorial
설명: 손으로 쓴 숫자 데이터셋을 활용하여 "Lenet"을 학습시켜보고 테스트를 해본다.
카테고리: Caffe_Tutorial
include_in_docs: true
priority: 1
---

# Caffe를 사용하여 MNIST 데이터 셋을 활용해 Lenet 학습시키기
해당 튜토리얼을 진행하기 앞서 Caffe를 성공적으로 컴파일 해야합니다. 아니라면 [설치 페이지](https://github.com/Hahnnz/Caffe_Tutorial/wiki/0.-Caffe-%EC%84%A4%EC%B9%98-%EB%B0%A9%EB%B2%95)를 참고하여 설치 및 컴파일을 완료해주시기를 바랍니다.
이 튜토리얼에서는 Caffe가 튜토리얼 레퍼지토리에 존재해야합니다.

## 데이터셋 준비하기

먼저 MNIST 웹사이트에 있는 데이터셋을 다운로드를 하고 데이터 형식을 변환해주어야 합니다. 아래의 코드를 실행하면 간단히 이를 수행할 수 있습니다.

    cd $1.MNIST
    ./get_mnist.sh
    ./create_mnist.sh

만약 `wget`이나 `gunzip`이 설치되어있지 않다는 문제가 발생하면, 이 둘 각각을 설치해주어야만 합니다. 그래서 최종적으로 다음 두 Shell 파일들을 실행시킨후에는 `mnist_train_lmdb`과 `mnist_test_lmdb`이라는 두 개의 데이터셋이 생성된 것을 확인하실 수 있습니다. 

## LeNet: MNIST 데이터 셋 분류 모델

실제로 우리가 이 학습 프로그램을 실행시켜보기전에, 무엇이 일어날지에 대해 설명드리겠습니다. 우리는 [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)이라는 신경망을 사용할 것인데, 이 신경망은 숫자 분류하는 기능을 아주 탁월하게 수행하기로 유명합니다. 우린 기존에 구현된 LeNet과는 조금은 다른 버전을 사용할 것입니다. 기존에는 활성화 함수를 Sigmoid 함수로 구현하였지면 여기서 사용하는 것은 이를 대체해 현재 아주 유명한 Rectified Linear Unit (ReLU)를 사용하여 구현한 버전을 사용할 것입니다.

LeNet의 디자인은 여전히 많이 사용되어지는 좀 더 큰 크기의 모델에 핵심이 되는 CNN을 포함하고 있습니다. 예를 들면 ImageNet에 속한 여러가지 모델들입니다. 일반적으로 convolution Layer 다음에 pooling layer를 붙이고, 또 그 뒤에 convolution layer, pooling layer를 붙이고, 그 뒤에는 전가산층(Fully Connected layers : 기존의 다층 퍼셉트론과 유사함)를 두 개 더 붙입니다. 이러한 구성은 다음의 파일에 구성되어 있습니다.


`$Tutorial_ROOT/1.MNIST/lenet_train_test.prototxt`.

## MNIST 신경망 정의

해당 섹션에서는 손으로 쓴 숫자 MNIST 데이터셋을 분류하는 LeNet 모델을 구현한 `lenet_train_test.prototxt`의 모델 정의를 살펴볼 것입니다.
[Google Protobuf](https://developers.google.com/protocol-buffers/docs/overview)에 어느정도 익숙하다고 가정을 하고 설명을 할 것이지만 모르더라도 충분히 따라올 수 있습니다. 그래도 가급적이면 `$CAFFE_ROOT/src/caffe/proto/caffe.proto`에 있는 Caffe에서 사용하는 protobuf를 한 번 읽어보시는 것을 추천합니다.

우린 `caffe::NetParameter` protobuf를(python에서는 `caffe.proto.caffe_pb2.NetParameter`) 작성할 것인데 먼저 신경망에게 이름을 지어주는 것부터 해서 시작한다.

    name: "LeNet"

### Data Layer 작성

현재 우리는 이전 위에서 데이터 셋을 다운로드 했는데, 신경망이 데이터 입력을 받을 수 있도록 다음과 같이 data layer를 작성해준다.

    layer {
      name: "mnist"
      type: "Data"
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "mnist_train_lmdb"
        backend: LMDB
        batch_size: 64
      }
      top: "data"
      top: "label"
    }

특별하게도, 이 계층의 이름은 `mnist`, 타입은 `data`로 선언되고, 이 계층은 주어진 lmdb 데이터셋으로 부터 데이터를 읽어들인다. 우린 일괄 처리량을 한번에 64장 씩 처리하도록 batch_size를 64로 정의하였으며, 그리고 입력 크기가 픽셀 단위이기에 \[0,1\)의 범위로 설정을 해주어야 한다. 하지만 왜 하필 0.00390625인가? 이는 1을 256으로 나눈 숫자이기 때문이다. 그래서 최종적으로 이 계층은 두개의 blob 데이터 덩어리를 생성해낸다. 하난 `data` 이고, 다른 하나는 `label`이다.

### Convolution Layer 작성하기

이제 첫번째 합성곱 계층 (first convolution layer)를 정의해보자.

    layer {
      name: "conv1"
      type: "Convolution"
      param { lr_mult: 1 }
      param { lr_mult: 2 }
      convolution_param {
        num_output: 20
        kernel_size: 5
        stride: 1
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "data"
      top: "conv1"
    }

이 계층은 데이터 계층에서 제공하는 `data`이라는 데이터 덩어리를 받아 `conv1`계층을 수행합니다. convolution 커널 크기는 5이며 stride 1로 convolution 연산을 수행하여, 출력은 20개의 채널로 나갑니다.

fillers는 weight와 bias 값을 임의로 초기화합니다. weight filler로 우리들은 `xavier` 알고리즘을 사용하는데, 이 알고리즘은 입력과 출력 뉴런들의 수에 기반하여서 초기화의 크기를 자동적으로 정해줍니다. bias filler로는 0의 값을 기본 값으로 하여 상수로 초기화 합니다.

`lr_mult`는 계층 내에서 학습을 하는 파라미터들의 학습률을 조정해주는 역할을 수행합니다. 여기서 우리는 실행중에 Solver가 주는 학습률과 동일하게 weight learning rate를 설정해줍니다. 그리고 bias learning rate는 보통 두배 크게 설정해줍니다. 이는 보통 더 나은 수렴률을 보입니다.

### Pooling Layer

후. Pooling Layer는 좀 더 쉽게 정의할 수 있습니다.

    layer {
      name: "pool1"
      type: "Pooling"
      pooling_param {
        kernel_size: 2
        stride: 2
        pool: MAX
      }
      bottom: "conv1"
      top: "pool1"
    }
    
이는 pool 커널 크기는 2로, stride 2로 (stride 2로 pooling region들이 서로 겹치지 않게 하기 위함이다.) 설정하여 Max pooling을 수행하는 것이다.
비슷하게, 두번째 Convolution과 Pooling layer를 작성할 수 있을 겁니다. 아직도 잘 모르시겠다면, `$TUTORIAL_ROOT/1.MNIST/lenet_train_test.prototxt`를 참고해주세요.


### Fully Connected Layer(전가산층) 작성하기

전가산층 작성하는 것도 간단합니다.

    layer {
      name: "ip1"
      type: "InnerProduct"
      param { lr_mult: 1 }
      param { lr_mult: 2 }
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "pool2"
      top: "ip1"
    }

(`InnerProduct` (내적) 계층으로써 Caffe에서 사용되는) 전가산 계층은 500개의 출력을 한다고 정의된다. 모든 다른 줄들은 친근해보이죠? 맞죠?

### ReLU Layer 작성하기
ReLU Layer도 작성하기 쉽습니다.

    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "ip1"
      top: "ip1"
    }

ReLU는 원소단위(신경망 뉴런단위) 연산이기에, 우린 어느정도 메모리를 아끼기 위해서 *준비된* 연산을 할 수 있다. 이는 하위 (bottom = 입력)와 상위 (top = 출력)blob들에게 같은 이름을 부여해 설정이 가능하다. 당연한 것이지만, 다른 타입의 계층들에서는 blob이름이 같아서는 안됩니다!

ReLU layer를 한 후에는, 다시 또 하나의 내적 계층 (innerproduct layer)을 작성합니다.

    layer {
      name: "ip2"
      type: "InnerProduct"
      param { lr_mult: 1 }
      param { lr_mult: 2 }
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "ip1"
      top: "ip2"
    }

### Loss Layer 작성하기

마지막으로, Loss 계층을 작성해볼 것입니다!

    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "ip2"
      bottom: "label"
    }

`softmax_loss` 계층은 다항식의 로지스틱 손실 함수와 소프트맥스를 모두 구현한 것입니다. 이렇게 하면 수치적 안정성을 향상시키고 시간을 절약할 수 있습니다. 이 계층은 두개의 blob 데이터 덩어리를 입력으로 받는데 하나는 예측된 것이고 다른 하나는 맨 앞의 데이터 계층에서 제공하는 `label`를 받아옵니다. 여기서는 더이상의 출력을 해주지는 않습니다. 이 계층에서 하는 것은 손실 함수 값을 연산하는 것으로, 역전파(backpropagation) 과정을 수행할 때 사용하고, `ip2`에 대하여 그래디언트를 초기화 합니다.


### 추가 노트 : 계층 규칙 작성

계층을 정의 할 때 include라는 인자를 사용해, (학습 혹은 테스트)어떤 과정에서 해당 계층을 사용할 지도 명시해 줄 수 있습니다.

    layer {
      // ...layer definition...
      include: { phase: TRAIN }
    }

이것이 계층 규칙으로, 현재 신경망 상태에 기반해서, 신경망에 포함시킬지 여부를 정할수 있습니다.
모델 스키마와 규칙에 대해 더 많은 정보가 필요하다면 `$CAFFE_ROOT/src/caffe/proto/caffe.proto` 참고를 해주세요.

위와 같은 예시로, 이 계층은 오직 `TRAIN` 학습 단계에서만 포함될 것입니다.
만약 `TRAIN`를 `TEST`로 바꾼다면, 해당 계층은 테스트 단계에서만 포함될 것입니다.
계층 규칙이 없다면, 기본값으로 해당 계층은 모든 단계에서 수행될 것입니다.
그래서 `lenet_train_test.prototxt`에 두개의 `DATA` 계층이 정의되어 있습니다. (심지어 `batch_size`값도 다릅니다) 하나는 학습 단계를 위한 것이고, 다른하나는 테스트 단계를 위한 것입니다.
또한 `Accuracy`는 `TEST` 단계에만 포함되기 때문에 오직 테스트 단계에서만 `lenet_solver.prototxt`에 정의되어 있는 사항들에 따라, 모델 정확도는 매 100회마다 출력됩니다.

## MNIST Solver 정의하기

`$TUTORIAL_ROOT/1.MNIST/lenet_solver.prototxt`에 있는 각 줄에 있는 설명 코멘트를 살펴봅시다.

    # 학습/테스트 신경망 모델 파일 정의
    net: "examples/mnist/lenet_train_test.prototxt"
    # test_iter는 테스트가 얼마나 순전파(forward) 과정을 진행할지 설정합니다.
    # MNIST의 경우, 우리는 100회 테스트 반복에, 일괄처리량을 100개로 설정하였다.
    # 모든 테스트 데이터 10,000 장 사용
    test_iter: 100
    # 매 학습 500회 진행후 테스트 수행
    test_interval: 500
    # 신경망의 기본 학습률, 모멘텀, weight decay
    base_lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
    # 학습정책
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    # 매 100회 반복 때마다 출력합니다.
    display: 100
    # 최대 반복 횟수
    max_iter: 10000
    # 매 5000번 반복마다 중간 Snapshot 저장 
    snapshot: 5000
    snapshot_prefix: "examples/mnist/lenet"
    # CPU 혹은 GPU 설정
    solver_mode: GPU


## 신경망 모델 학습 및 테스트 하기

신경망 모델 학습시키기는 여러분들이 신경망 정의 protobuf과 solver protobuf 파일들을 작성한 후에 간단하게 수행시켜 볼 수 있습니다. `train_lenet.sh`를 실행시키면 되므로 다음의 코드를 실행시켜봅시다.

    cd $TUTORIAL_ROOT
    ./1.MNIST/train_lenet.sh

`train_lenet.sh` 는 간단한 스크립트입니다. 하지만 학습을 시키기 위한 메인 툴은 `caffe`내의 툴인 `train`과 solver protobuf 파일을 `train_lenet.sh`가 인자로 사용을 합니다.

코드를 실행시켜보면 여러분들은 다음과 같이 마구마구 날라오는 상당량의 메세지를 볼 것입니다.

    I1203 net.cpp:66] Creating Layer conv1
    I1203 net.cpp:76] conv1 <- data
    I1203 net.cpp:101] conv1 -> conv1
    I1203 net.cpp:116] Top shape: 20 24 24
    I1203 net.cpp:127] conv1 needs backward computation.

이러한 메세지들은 여러분들에게 디버깅할 때 도움이 될만한 각 계층들, 그리고 그 계층의 출력 shape와 연결들에 대한 자세한 설명을 알려줍니다. 초기화를 한 뒤에 학습이 시작될 것입니다.

    I1203 net.cpp:142] Network initialization done.
    I1203 solver.cpp:36] Solver scaffolding done.
    I1203 solver.cpp:44] Solving LeNet

Solver 세팅에 기반해 우리는 각 100번 반복마다 학습 손실 함수를 출력할 것이고, 매 반복 500회마다 신경망을 테스트해볼 것입니다. 여러분들은 아마 다음과 같은 메세지를 확인할 수 있습니다.

    I1203 solver.cpp:204] Iteration 100, lr = 0.00992565
    I1203 solver.cpp:66] Iteration 100, loss = 0.26044
    ...
    I1203 solver.cpp:84] Testing net
    I1203 solver.cpp:111] Test score #0: 0.9785
    I1203 solver.cpp:111] Test score #1: 0.0606671

각각 반복마다, `lr`는 그 반복의 학습률을 의미하며, `loss`는 학습 함수중 하나입니다. 테스트 단계의 출력으로는 score 0은 정확도, score 1는 테스트 Loss 함수를 의미합니다.

그리고 어느 정도 기다려주면 학습이 완료될 것입니다!

    I1203 solver.cpp:84] Testing net
    I1203 solver.cpp:111] Test score #0: 0.9897
    I1203 solver.cpp:111] Test score #1: 0.0324599
    I1203 solver.cpp:126] Snapshotting to lenet_iter_10000
    I1203 solver.cpp:133] Snapshotting solver state to lenet_iter_10000.solverstate
    I1203 solver.cpp:78] Optimization Done.

최종 학습 모델은 binary protobuf file로 다음과 같이 저장될 것입니다.

    lenet_iter_10000

그리고 이를 활용해서, 여러분이 실제 어플리케이션의 데이터셋으로 학습시킨다면 여러분의 어플리케이션 속으로 여러분이 입력한 데이터로 학습된 모델로 저장될 것입니다.

### 음... GPU 학습은 어떻게 할 수 있죠?
대부분의 사람들이 Caffe를 컴파일 할 당시에 Makefile.config에서 CPU-ONLY로 컴파일을 수행해주었을 것입니다. 이를 Uncomment하여 다시 컴파일을 수행해준뒤에, Solver 파일에 들어가서 solver_mode를 CPU에서 GPU로 변경해주신 뒤에 학습을 시켜주면 됩니다. 이 튜토리얼 같은 경우에는 `lenet_solver.prototxt`의 다음과 같은 라인을 바꾸어 주시면 됩니다.

    # solver mode: CPU or GPU
    solver_mode: GPU

이렇게 아주 간단하게 바꿀수 있답니다.

MNIST는 규모가 작은 데이터셋입니다. 그래서 GPU로 학습시키는 것은 커뮤니케이션 오버헤드 덕택을 그렇게 많이 보지를 못합니다. 좀 더 규모가 크고 좀 더 복잡한 모델들, 예를 들면 ImageNet같은 모델들로 수행해본다면 연산 속도의 차이를 좀 더 확연하게 느끼실 수 있습니다. 

### 어떻게 정해진 단계수에서 학습률을 감소시킬 수 있는 것이죠?
lenet_multistep_solver.prototxt를 참고해주세요
