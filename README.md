# Caffe

Caffe는 다양한 딥러닝 프레임워크들 중 하나로 버클리 인공지능 연구소 (Berkeley AI Research([BAIR](http://bair.berkeley.edu)))/ 버클리 비전 & 학습 센터 (The Berkeley Vision and Learning Center (BVLC)), 그리고 Caffe 커뮤니티의 수많은 기여자들에의해 개발된 것입니다.

좀 더 자세한 사항들을 확인하고자 한다면 [Caffe 공식 사이트](http://caffe.berkeleyvision.org)를 확인해주세요. 

[해당 공식 사이트(이전 버전)](https://github.com/ys7yoo/BrainCaffe/wiki/Caffe-Documentation-:-Caffe-Tutorial-(Kor)) 번역 또한 같이 첨부하였습니다.

## [Caffe 튜토리얼](https://github.com/Hahnnz/Caffe_Tutorial/wiki)
Caffe는 딥러닝 프레임워크로 이 튜토리얼은 딥러닝의 원리, 구성 그리고 사용법에 대해서 설명할 것입니다.
해당 튜토리얼은 딥러닝의 전체적인 이로이나 맥락등 이론적인 내용은 다루지 않습니다.
기계학습의 배경지식과 신경망등, 딥러닝 이론에 대해 미리 숙지하고 참고하시면 도움이될 것입니다!
 - 0. [Caffe 설치 하기](https://github.com/Hahnnz/Caffe_Tutorial/wiki/0.-Caffe-%EC%84%A4%EC%B9%98-%EB%B0%A9%EB%B2%95)


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
