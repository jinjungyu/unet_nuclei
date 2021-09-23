# ML_Study
## 이 페이지는 원래 스터디할 때 쓰려고 했는데, 복습 노트 + 오류 해결이 중점적으로 기록될 것 같다.
### docker 사용
docker run : 도커 이미지에서 컨테이너 생성  
docker exec : 이미 실행중인 컨테이너에 명령을 줄 때  
docker restart : 컨테이너 재실행  
docker start : 종료된 컨테이너 실행  
docker stop : 실행중인 컨테이너 종료  
서버에서 docker run으로 주피터 서버가 실행되는 이미지를 실행시켰기 때문에 docker exec으로 명령을 줄 일이 많다.  
- 자주 쓰는 명령
```bash
docker exec -it {container} bash - 컨테이너 bash CLI 터미널 
docker exec -it {container} nvidia-smi - GPU 사용 정보
docker exec -it {container} jupyter notebook list - 주피터 노트북 서버 리스트, 토큰 확인
```
### 오류 대응
#### 1. 머신 러닝 후 GPU 메모리가 해제되지 않아 out-of-memory 가 발생할 때
Tensorflow 에서는 아래처럼 말하고 있다.
```
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.
```
메모리 조각화를 막기 위해 GPU 메모리들에 모두 할당을 해놓는다고 한다.  
- 해결 방법 : ps -ef | grep python 이나 top 처럼 프로세스를 확인하고, 프로세스를 kill 한다.
- 예방 방법 : 특정 GPU만 사용하기.
```python
import os 
# GPU를 아예 못 보게 하려면: 
os.environ["CUDA_VISIBLE_DEVICES"]='' 
# GPU 0만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0' 
# GPU 1만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 0과 1을 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
```
```python
with tf.device('/device:GPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
```
