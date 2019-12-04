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
 > * 함수 load_cifar_10_data 



