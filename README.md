# KBJ_ABRlab_practice_2019

## 2019년도 ABR lab 강의 실습 코드

> 2019.03 ~ 현재까지 경북대학교 ABR lab에서 deep learning 수업을 받고 있습니다(연구실 미지원 연구생)    
> 2019학년도 1학기+여름방학 기간동안 수강한 MLP , CNN 과제 실습 코드 입니다.

* dataset 은 cifar-10 dataset 을 사용하였습니다.
   링크 [cifar-10-dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
   - cifar10 data를 두가지 방법을 통해 사용하였습니다.
 
 1. google colab의 가상드라이브에 파일을 올린후, drive mount 하여 압축파일을 명령어를 통해 압축해제 한후,
    load , unpickle 등의 명령어를 정의하여 dataset 에대한 preprocessing 을 연습해 보았습니다(cifar10(실습).ipynb)                
          해당 파일의 함수들은 (https://github.com/snatch59/load-cifar-10/blob/master/load_cifar_10.py)
          의 코드를 참고하였습니다. 
    
 2. 두번째는 tensorflow 내부에 있는 cifar10 dataset 을 tensorflow 의 함수로 불러내어 전처리과정을 최소화 하여 모델학습을 위해
    사용하였습니다. (cifar10practic.ipynb, cnnpractice.ipynb)

------------------------------------------------------------------------

 ### cifar10(실습).ipynb
 > *cifar10 dataset 의 전처리를 연습하는 코드입니다. 작성된 합수들은 (https://github.com/snatch59/load-cifar-10/blob/master/load_cifar_10.py) 의 코드를 참고하였습니다.*
 > *  google colab을 사용하여 코드를 작성하였습니다. 파일을 업로드 하기위해서 colab drive mount 를 통해 tar file에 접근하였습니다
 > * `!tar -xvf cifar-10-python.tar.gz` 명령어를 통해 tar 파일 압축을 해제하였습니다
 > * 함수 unpickle에서 사용된 pickle 모듈은 텍스트가 아닌 리스트나 클래스 같은 자료형 데이터를 저장, 불러올때 사용하는 모듈로, 데이터를 바이트형식으로 읽거나 써진 데이터에 사용됩니다.
 > * 함수 load_cifar_10_data 의 Meta_data_dict 는 unpickle 함수로 데이터를 저장하고 Cifar_label_names = meta_data_dixt 에 있는 것들을 b형식으로 읽어와서 다시 array 담게됩니다. Train_data, filename, label 을 초기화 합니다
 > * For문 을 사용하여 파일 1에서6까지 접근합니다. 첫번째 파일인 data_batch_1은 빈곳에 담고 나머지 2에서6까지 파일은 np.vstack 을 통해 세로로 이어붙힙니다.
 > * Filename 과 label 은 빈리스트에 담습니다. 불러운 train_data를 reshape, inputdata의 개수(len(cifar_train_data)),RGB의 3채널, 32x32의 꼴로 변환합니다.
> * Test_data도 앞의 traindata 처리와 같이 실행합니다
> * 마지막으로 이미지를 확인합니다. Num_plot 을 설정하여 이미지를 출력합니다. np.random.shuffle()을 사용하여 실행될때마다 data 가 shuffle 되게 합니다.

------------------------------------------------------------------------

## cifar10practice.ipynb
> *cifar10 dataset 을 활용하여 MLP 모델을 구성하고 학습시키는것 까지를 연습한 코드입니다. 기본적인 모델을 구성하였을때의 성능과, layer 중간에 다양한 추가 기능들을 사용했을때의 차이를 분석해보았습니다.
>  https://eyeofneedle.tistory.com/14 와 https://colab.research.google.com/drive/1eHjcLuiWS2Yr42vnvzzQSBkZMOXaXh_x#scrollTo=kA-iZILBzhtm 의 코드를 참고하여 작성하였습니다*
>  * cifar10 dataset 은 tensorflow 안의 dataset을 import 하여 사용하였습니다. Load_data 함수로 train data 와 test data 를 정의 하였고 각 shape 을 확인해보았습니다.
> * 기초모델을 구성하였습니다. Tensorflow, numpy 라이브러리를 import 하여 X,Y 에 placeholder 함수로 feed 할 그릇을 생성하였습니다. tf.Variable로 weight 와 bias를 설정, 가중치를 계산후에는 relu 함수를 통과하도록 하였습니다.
> * 기초모델의 Layer는 input, hidden(3072,512), hidden(512,512), hidden(512,256), hidden(256,256), output 으로 구성하였고 softmax crossentropy를  사용하였습니다. Optimizer 는 adamoptimizer 에 learningrate 은 0.001 로 하였습니다

### 학습 및 결과 확인
> 위의 링크 코드에서 참고를 많이 하였는데, Batch_ys 부분에서 행렬크기가 안맞는 부분을 reshape 해주었습니다.
> * epoch = 100, batchsize = 2500 으로 하였습니다
> * 100번 학습하여 test 한 결과 정확도는 약 47.73%의 정확도를 가졌습니다.
> * 모델을 구성하던 중 4layer 에 50epcoh, 1000batch, bias 없음으로 했을때의 정확도는 50%를 넘었었습니다.
> * train set 에서의 cost 는 위 경우보다 많이 떨어졌지만 정화도가 더 낮게 나오는걸 봐서 오버피딩 되었다고 생각합니다.
> * 새로운 모델을 구성할때 이러한 오버피팅을 방지할 부분을 추가할 필요가 있다고 생각했습니다.

#### 모델 수정
> * Drop out 을 추가하였습니다. 이를 추가하여 오버피팅을 방지하고자 하였습니다.
> Accuracy 가 약 4%정도 증가하였습니다
> * 또 다른 방법으로 batch normalizationg 함수를 사용해 보았습니다 (tf.layers.batch_normalization, istraining = True)
> Accuracy 가 약 11% 정도 증가하였습니다
> * 위 함수들을 추가하자 traininset 에서 cost가 매우 감소함을 확인하였습니다

------------------------------------------------------------------------


------------------------------------------------------------------------

