# onnx_tensorrt

onnx file link : https://drive.google.com/drive/folders/1YFbH1CzVI_aiPEOHBMrmLt0RtGypm_tZ


1. Docker(https://hub.docker.com/r/chenxyyy/tensorrt7) 를 다운받는다.
docker pull chenxyyy/tensorrt7
2. bash run_docker.sh
3. docker 밖 host 에서 xhost 명령어를 이용해 컨테이너 내 애플리케이션이 호스트의 디스플레이 서버에 접속할 수 있게 허가 해준다
xhost +local:host    
4. docker 안에서 소스코드로 접근한다.
cd /ws
5. build directory 를 생성한다. 
mkdir build 
cd build
6. cmake ..
build



