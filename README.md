# VMS
git clone https://github.com/DuongNo/VMS.git
rm -rf faceSystem/
git clone https://github.com/DuongNo/faceSystem.git
cd ..

mkdir traffic_process
cd traffic_process/

cp /home/vdc/project/computervision/python/VMS/faceprocess/setup.py .

to run vms
```
conda activate facesys
./scripts/run.sh
```


