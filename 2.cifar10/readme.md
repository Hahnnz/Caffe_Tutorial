---
제목: CIFAR-10 튜토리얼
설명: CIFAR-10 data로 Caffe를 사용해 학습 및 테스트를 해본다.
카테고리: 튜토리얼
include_in_docs: true
priority: 5
---

Alex's CIFAR-10 tutorial, Caffe style
=====================================
Alex Krizhevsky의 [cuda-convnet](https://code.google.com/p/cuda-convnet/)에 신경망 모델 정의와 파라미터들 그리고 CIFAR-10 데이터셋을 활용해 훌륭한 결과를 도출한 학습 과정에 대해 상세히 나와있다. 이 예시는 Caffe에서 결과를 재구성한 것입니다.

해당 튜토리얼을 진행하기 앞서 Caffe를 성공적으로 컴파일 해야합니다. 아니라면 [설치 페이지](https://github.com/Hahnnz/Caffe_Tutorial/wiki/0.-Caffe-%EC%84%A4%EC%B9%98-%EB%B0%A9%EB%B2%95)를 참고하여 설치 및 컴파일을 완료해주시기를 바랍니다.
이 튜토리얼에서는 Caffe가 튜토리얼 레퍼지토리에 존재해야합니다.


데이터셋 준비하기
-------------------

여러분은 먼저 [CIFAR-10 웹사이트](http://www.cs.toronto.edu/~kriz/cifar.html)에서 데이터를 다운 받고 형식을 변환해주어야합니다. 이는 다음과 같은 명령어를 입력해주어 파일을 실행시켜주면 됩니다.

    cd $CAFFE_TUTORIAL/2.cifar10
    ./get_cifar10.sh
    ./create_cifar10.sh

만약 `wget`이나 `gunzip`이 설치되어있지 않다는 문제가 발생하면, 이 둘 각각을 설치해주어야만 합니다. 그래서 최종적으로 다음 두 Shell 파일들을 실행시킨후에는 `./cifar10-leveldb`라는 데이터셋 파일과 `./mean.binaryproto`라는 이미지 평균 값 파일을 확인하실 수 있습니다. 


신경망 모델
---------
CIFAR-10 model은 특징 추출을 위한 convolution, Subsampling을 위한pooling, 비선형성을 주기 위한rectified linear unit (ReLU) , 그리고 선형 분류기를 사용한 local contrast normalization으로 구성된 CNN(Convolution Neural Network)입니다. 이 신경망 모델은 `./2.cifar10` 디렉토리 안의 `cifar10_quick_train_test.prototxt` 파일에 모델이 정의되어 있습니다.


"빠른" 신경망 모델 학습 및 테스트하기
--------------------------------------

신경망 모델을 학습시키는 것은 이전에 튜토리얼에서 수행해보았던 것처럼 Solver protobuf파일과 신경망 정의 protobuf파일을 작성한 후에 학습을 진행할 수 있습니다. ([1.MNIST 튜토리얼](https://github.com/Hahnnz/Caffe_Tutorial/tree/master/1.MNIST)을 참고해 주세요!) `train_quick.sh`을 실행시키면 간단하게 학습을 수행시킬 수 있습니다.


    cd $CAFFE_TUTORIAL/2.cifar10
    ./train_quick.sh

`train_quick.sh`는 간단한 스크립트니, 한번 들어가서 둘러보세요. 하지만 학습을 시키기 위한 메인 툴은 `caffe`내의 툴인 `train`과 `solver` protobuf 파일을 train_quick.sh가 인자로 사용을 합니다.

코드를 실행시켜보면 여러분들은 다음과 같이 마구마구 날라오는 상당량의 메세지를 볼 것입니다.

    I0317 21:52:48.945710 2008298256 net.cpp:74] Creating Layer conv1
    I0317 21:52:48.945716 2008298256 net.cpp:84] conv1 <- data
    I0317 21:52:48.945725 2008298256 net.cpp:110] conv1 -> conv1
    I0317 21:52:49.298691 2008298256 net.cpp:125] Top shape: 100 32 32 32 (3276800)
    I0317 21:52:49.298719 2008298256 net.cpp:151] conv1 needs backward computation.
    
이러한 메세지들은 여러분들에게 디버깅할 때 도움이 될만한 각 계층들, 그리고 그 계층의 출력 shape와 연결들에 대한 자세한 설명을 알려줍니다. 초기화를 한 뒤에 학습이 시작될 것입니다.

    I0317 21:52:49.309370 2008298256 net.cpp:166] Network initialization done.
    I0317 21:52:49.309376 2008298256 net.cpp:167] Memory required for Data 23790808
    I0317 21:52:49.309422 2008298256 solver.cpp:36] Solver scaffolding done.
    I0317 21:52:49.309447 2008298256 solver.cpp:47] Solving CIFAR10_quick_train

Solver 세팅에 기반해 우리는 각 100번 반복마다 학습 손실 함수를 출력할 것이고, 매 반복 500회마다 신경망을 테스트해볼 것입니다. 여러분들은 아마 다음과 같은 메세지를 확인할 수 있습니다.

    I0317 21:53:12.179772 2008298256 solver.cpp:208] Iteration 100, lr = 0.001
    I0317 21:53:12.185698 2008298256 solver.cpp:65] Iteration 100, loss = 1.73643
    ...
    I0317 21:54:41.150030 2008298256 solver.cpp:87] Iteration 500, Testing net
    I0317 21:54:47.129461 2008298256 solver.cpp:114] Test score #0: 0.5504
    I0317 21:54:47.129500 2008298256 solver.cpp:114] Test score #1: 1.27805


각각 반복마다, `lr`는 그 반복의 학습률을 의미하며, `loss`는 학습 함수중 하나입니다. **테스트 단계의 출력으로는 score 0은 정확도, score 1는 테스트 Loss 함수를 의미**합니다.

그리고 어느 정도 기다려주면 학습이 완료될 것입니다!

    I0317 22:12:19.666914 2008298256 solver.cpp:87] Iteration 5000, Testing net
    I0317 22:12:25.580330 2008298256 solver.cpp:114] Test score #0: 0.7533
    I0317 22:12:25.580379 2008298256 solver.cpp:114] Test score #1: 0.739837
    I0317 22:12:25.587262 2008298256 solver.cpp:130] Snapshotting to cifar10_quick_iter_5000
    I0317 22:12:25.590215 2008298256 solver.cpp:137] Snapshotting solver state to cifar10_quick_iter_5000.solverstate
    I0317 22:12:25.592813 2008298256 solver.cpp:81] Optimization Done.

우리의 신경망 모델은 ~75%정도의 테스트 정확도를 달성했습니다. 신경망 모델 파라미터들은 binary protobuf형태로 CPU든 GPU든 언제든지 사용할 수 있는  

    cifar10_quick_iter_5000

라는 이름으로 저장되어 있습니다. 새로운 데이터를 불러와서 테스트할 수 있는 `./2.cifar10/cifar10_quick.prototxt`라는 파일 디플로이먼트 모델을 참고해보세요.

왜 GPU에서 학습을 하는건가요?
-------------------

CIFAR-10 데이터 세트는 여전히 그 세트의 크기가 작지만 GPU를 사용하여 학습을 하는데 매력을 느낄수 있을만큼의 충분한 데이터세트를 확보하고 있습니다.

CPU와 GPU의 학습 속도를 비교하기 위해서는 `cifar*solver.prototxt`에 있는 한줄만 바꾸어주면 됩니다.

    # solver mode: CPU or GPU
    solver_mode: CPU
    
    
이렇게 설정하면 CPU를 사용할 수 있습니다.

