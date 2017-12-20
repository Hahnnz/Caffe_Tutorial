---
제목: 이미지넷 튜토리얼
설명: Train and test "CaffeNet" on ImageNet data.
category: example
include_in_docs: true
priority: 1
---

# 이미지망 집중학습[(Brewing ImageNet)](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html)
이 가이드는 여러분들이 소유한 데이터에 대한 여러분들만의 모델을 학습할 준비가 되었다는하에 진행합니다. 만약 이미지넷으로 학습된 네트워크(ImageNet-trained network)를 원한다면, 학습에는 많은 에너지를 소비하기때문에 우리는 CaffeNet 모델을 [model zoo]이 가이드는 여러분들이 소유한 데이터에 대한 여러분들만의 모델을 학습할 준비가 되었다는하에 진행합니다. 만약 이미지넷으로 학습된 네트워크(ImageNet-trained network)를 원한다면, 학습에는 많은 에너지를 소비하기때문에 우리는 CaffeNet 모델을 [model zoo](https://github.com/Hahnnz/Caffe_Tutorial/wiki/Caffe-Documentation-:-Caffe-Model-Zoo-(Kor))에 있는 미리 학습된 데이터로 학습을 시키었습니다.
)에 있는 미리 학습된 데이터로 학습을 시키었습니다.
## 데이터준비(Data Preparation)
이 가이드는 모든 과정을 명시하며 모든 명령어가 Caffe Tutorial root directory에서 실행된다는 가정하에 진행합니다.
“ImageNet”에의해 우리가 여기 ILSVRC12 challenge를 말하지만, 약간의 디스크 공간과, 더 적은 학습시간으로 당신은 쉽게 ImageNet의 전체 또한 학습시킬수 있다.

여러분이 이미지넷 학습데이터와 유효데이터를 다운로드 해놓았다고 가정하며, 이들은 당신의 디스크상에 다음과 같이 저장됩니다:
```
/path/to/imagenet/train/n01440764/n01440764_10026.JPEG
/path/to/imagenet/val/ILSVRC2012_val_00000001.JPEG
```
우선 학습을 위해 몇개의 예비 데이터를 준비해놓을 필요가 있습니다. 이 데이터는 다음 파일을 실행시켜 다운로드할 수 있습니다.
```
./data/ilsvrc12/get_ilsvrc_aux.sh
```
학습입력과 validation입력은 모든 파일들과 그들의 라벨들을 리스트한 글로적힌 train.txt와 val.txt에 적혀있습니다. 우리는 ILSVRC devkit와는 달리 라벨에대해 인덱스를 매겨놓는 색다른 방법을 사용할 것임을 알아두어야합니다. 우리는 synset이름들을 그들의 아스키법칙에 따라 분류하고, 그러고나서 이를 0부터 999까지 라벨을 붙입니다. synset/name 맵핑에 대해서는 synset_words.txt을 참고해주세요.

여러분들은은 아마 미리 256x256로 이미지의 크기를 재설정하길 원하실겁니다. 디폴트값 덕분에, 우리는 딱히 이를 할 필요가 없는데, 이는 군집화 환경에서, mapreduce를 사용해서 병렬 방식에서 이미지의 크기를 조정할 수 있기 때문입니다. 예들들어 Yangqing은 그의 경량의 [mincepie](https://github.com/Yangqing/mincepie) 패키지를 사용했습니다. 만약 더 간단하게 하는 것을 좋아하신다면, 여러분들은 다음과 같은 shell 명령어를 사용하여 진행하실 수 있습니다.
```
for name in /path/to/imagenet/val/*.JPEG; do
    convert -resize 256x256\! $name $name
done
```
examples/imagenet/create_imagenet.sh을 실행시켜보세요! 필요하다면 학습디렉토리와 Validation 디렉토리 경로를 지정하고, 미리 이미지의 크기를 수정해두지 않았다면, 256x256로 모든 이미지의 크기를 재설정하기위해 “RESIZE=true”를 설정하세요. 지금, examples/imagenet/create_imagenet.sh로 leveldbs를 간단히 만들어보실 수 있을겁니다. examples/imagenet/ilsvrc12_train_leveldb와 examples/imagenet/ilsvrc12_val_leveldb를 실행해야 leveldb파일을 얻을 수 있습니다.

## 이미지 평균 연산하기(Compute Image Mean)
모델은 여러분들에게 각각 이미지로부터 이미지 평균을 달라고 요구할 것이고, 그래서 우리들은 이 평균을 계산해야만합니다. tools/compute_image_mean.cpp가 이를 수행해줄겁니다. 이는 또한 어떻게 다중요소들, 예를들면 프로토콜 버퍼(protocol buffer)나 leveldbs, 그리고 로깅(logging)와 같은 것들을 다루는 방법과 여러분들자신을 친숙하게 해줄 좋은 예입니다. 어쨌거나, 평균연산은 다음과 같이 수행할 수 있습니다:
```
./examples/imagenet/make_imagenet_mean.sh
```
이는 data/ilsvrc12/imagenet_mean.binaryproto를 만들어낼 것입니다.
## 모델정의(Model definition)
우리는 Krizhevsky, Sutskever, 그리고 Hinton가 처음 [NIPS 2012 paper](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)에 제시한 접근방법에 대하여 참조구현을 살펴볼 것입니다.

네트워크 정의(models/bvlc_reference_caffenet/train_val.prototxt)는 in Krizhevsky 와 그의 동료들이 만든 것을 따라 볼 것입니다. 만약 여러분들이 이 가이드에서 제시한 파일 경로와 다른 경로를 사용하신다면, .prototxt 파일들에서 적절한 경로를 조절할 필요가 있음을 주의하세요.

만약 여러분들이 유심히 models/bvlc_reference_caffenet/train_val.prototxt를 보았다면, phase: TRAIN이나 혹은 phase: TEST 둘 중 하나를 명시한 include 세션들 보았을 것입니다. 이러한 세션들은 한 파일에서 우리에게 둘이 가깝게 연관된 네트워크들(학습용으로 사용되는 네트워크와 실험용으로 사용되는 네트워크)을 정의할 수 있게 해줍니다. 이러한 두 네트워크들은 include { phase: TRAIN }이나 include { phase: TEST }로 적힌 것들을 제외하고 모든 계층들을 공유하여 동일하게 작동합니다.

* 입력계층 차이점(Input layer differences)

 학습 네트워크의 데이터 입력 계층은 examples/imagenet/ilsvrc12_train_leveldb로부터 해당하는 데이터를 서술하며 랜덤으로 입력이미지를 미러링합니다. 테스트 네트워크의 데이터 계층은 examples/imagenet/ilsvrc12_val_leveldb에서 데이터를 가져오지만 랜덤 미러링은 수행하지 않습니다.
 
* 출력계층 차이점(Output layer differences)

 두 네트워크 모두 softmax_loss 계층을 출력하며, 이는 학습에서 손실 함수를 연산하고 backpropagation을 초기화하는데 사용됩니다. 반면에 Validation에서는 이 손실은 간단히 보고만 합니다. 테스트 네트워크는 또한 두번째 출력 계층, 정확도를 지니고 있고, 이는 실험 세트에 대한 정확도를 보고하는데만 사용됩니다. 학습과정에서, 실험 네트워크는 때로는 예를들며 설명할 것이고 Test score #0: xxx 이나 Test score #1: xxx와 같은 줄을 생성하면서 테스트 세트에 대하여 실험할 것입니다. 이러한 경우, 스코어 0은 정확도이며(이 정확도는 학습되지 않은 네트워크에 대하여 1/1000 = 0.001 주변에서 시작합니다), 그리고 스코어 1은 손실입니다(이 손실은 학습되지않은 네트워크에 대하여 약 7 주변에서 시작합니다).

우리는 또한 Solver를 실행하기 위한 프로토콜 버퍼를 구현해야하죠. 계획을 세워봅시다:
* 256의 일회처리량으로 실행할 것이고, 전체 450,000만번의 반복 (약 90회의 에코)을 실행
* 매 1000번 반복마다 우리는 validation 데이터에 대하여 학습된 네트워크를 저장
* 초기 학습율을 0.01로 설정하고, 이를 매 100,000만번 (약 20회의 에코)에서 감소
* 매 20회 반복마다 정보들 출력
* 네트워크는 모멘텀 0.9 그리고 0.0005의 가중치 감소로 학습
* 매 10,000번 반복마다, 현재 상태의 스냅샷을 저장

좋아보이죠? 사실 이 계획은 models/bvlc_reference_caffenet/solver.prototxt.에 구현되어있습니다.
## 이미지망 학습하기(Training ImageNet)
준비되었나요? 시작해봅시다!
```
./build/tools/caffe train --solver=models/bvlc_reference_caffenet/solver.prototxt
```

학습을 즐겨봅시다!

K40 머신상에서는, 매 20회 반복마다 (K20머신상에서는 36초가 걸리는데 비해) 실행하는데 약 26.5초가 걸리며, 매우 효과적으로 정방향-역방향 과정 모두에 대하여 이미지당 약 5.2 ms가 걸립니다. 약 2 ms가 순방향연산에 사용되고, 나머지는 역방향연산에서 사용됩니다. 만약 이 연산시간을 해부하는데 흥미가있다면, 다음을 실행시켜보세요.
```
./build/tools/caffe time --model=models/bvlc_reference_caffenet/train_val.prototxt
```

## 학습 다시 시작하기?(Resume Training?)
우리는 모두 파워가 나갈 때를 경험해봤거나 배틀필드를 즐김으로써 거의 우리자신을 가치있게 보람있지않은 것 같이 느꼈다(누구 퀘이크를 아직도 기억하는 이가 있나요?). 우리가 학습중에 중간 결과를 스냅샷으로 저장하기 때문에, 우리는 스냅샷 횟수에서 부터 학습을 다시 시작할 수 있습니다. 이는 다음과 같이 간단히 수행할 수 있습니다.
```
./build/tools/caffe train --solver=models/bvlc_reference_caffenet/solver.prototxt --snapshot=models/bvlc_reference_caffenet/caffenet_train_iter_10000.solverstate
```
스크립트에서 caffenet_train_iter_10000.solverstate는 정확한 Solver 상태를 복구하는 모든 필수적인 정보를 저장하는 Solver 상태 스냅샷입니다.
## 작별인사(Parting Words)
여러분들이 이 튜토리얼을 좋아하기를 희망해요! 네트워크의 구조를 바꾸면서 그리고/혹은 새로운 데이터와 업무를 보내려는 네트워크내의 다양한 파라미터들을 잘 조율하면서, ILSVRC 2012 challenge이래로 많은 연구자들은 더욱 진전되었습니다. 간단히 적은 다양한 prototxt 파일들에의해 Caffe가 당신을 다양한 네트워크 선택을 좀 더 쉽게 탐색해보세요. - 흥미롭지 않은가요?

그리고 여러분들이 네트워크를 이제 막 훈련시켰기 때문에, [classifying ImageNet](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb)를 참고해서 Python 인터페이스로는 어떻게 사용할 수 있는지 확인해보세요!
