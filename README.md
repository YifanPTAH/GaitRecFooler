# GaitRecFooler
Create Adversarial Examples which can attack GaitRecognitionCNN

Orgin Data of the input are from CASIA-B database:http://www.cbsr.ia.ac.cn/users/szheng/?page_id=71

GaitRecognizer are modified from GaitRecognitionCNN: https://github.com/nephashi/GaitRecognitionCNN

## Result of Experiment
### Source image-GEI
![Image 1](https://github.com/YifanPTAH/GaitRecFooler/blob/master/input/experiment-1/source/013-nm-04-090.png)
### Source image-silhouette
![Image 2](https://github.com/YifanPTAH/GaitRecFooler/blob/master/input/experiment-1/gif/source.gif)

GaitRecognizer output: 13
### target image
![Image 3](https://github.com/YifanPTAH/GaitRecFooler/blob/master/input/experiment-1/target/075-nm-04-090.png)

GaitRecognizer output: 75

### Created Adversarial Example
![image 4](https://github.com/YifanPTAH/GaitRecFooler/blob/master/output/experiment-1/gei/fake-gait-gei.png)
![image 5](https://github.com/YifanPTAH/GaitRecFooler/blob/master/output/experiment-1/gif/fake.gif)

GaitRecognizer output: 75




