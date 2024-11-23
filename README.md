## CMSC472_Final_Project

This project is about classifying UMD buildings (11 buildings). So far, we have 3300 labeled images for this project.

Update: 11/22/2024, it is possible we might need to change the structure of the project, or not.
I haven't have the time to run the code, please check it yourself. We did supervised learning, .....


We used GradCam and GradCam++ to visualize where the Network is focusing. Image Data is stored in Google Drive. 

Update: 11/23/2024, we tried restnet18 and restnet50. Model output is saved in the ./report/outputs folder. Based on the result, restnet18 seems to be better than resnet50 with 97% vs ~84% test acc. For the gradcam, resnet18 layer4 is showing promising results of where the building is, so the network is not looking at trees, skys, or grounds.
