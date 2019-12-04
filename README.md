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

 ### cifar10(실습).ipynb
 > *  google colab을 사용하여 코드를 작성하였습니다. 파일을 업로드 하기위해서 colab drive mount 를 통해 tar file에 접근하였습니다
 > * `!tar -xvf cifar-10-python.tar.gz` 명령어를 통해 tar 파일 압축을 해제하였습니다
 > * 함수 unpickle에서 사용된 pickle 모듈은 텍스트가 아닌 리스트나 클래스 같은 자료형 데이터를 저장, 불러올때 사용하는 모듈로, 데이터를 바이트형식으로 읽거나 써진 데이터에 사용됩니다.
 > * 함수 load_cifar_10_data 의 Meta_data_dict 는 unpickle 함수로 데이터를 저장하고 Cifar_label_names = meta_data_dixt 에 있는 것들을 b형식으로 읽어와서 다시 array 담게됩니다. Train_data, filename, label 을 초기화 합니다
 > * For문 을 사용하여 파일 1~6까지 접근합니다. 첫번째 파일인 data_batch_1은 빈곳에 담고 나머지 2~6파일은 np.vstack 을 통해 세로로 이어붙힙니다.
 > * Filename 과 label 은 빈리스트에 담습니다. 불러운 train_data를 reshape, inputdata의 개수(len(cifar_train_data)),RGB의 3채널, 32x32의 꼴로 변환합니다.
> * Test_data도 앞의 traindata 처리와 같이 실행합니다
> * 마지막으로 이미지를 확인합니다. Num_plot 을 설정하여 이미지를 출력합니다. np.random.shuffle()을 사용하여 실행될때마다 data 가 shuffle 되게 합니다.


