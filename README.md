## CMSC472_Final_Project

Progress log:

This project is about classifying UMD buildings (20 buildings). So far, we have close to 4000 labeled images for this project.

Update: 11/22/2024, it is possible we might need to change the structure of the project, or not.
I haven't have the time to run the code, please check it yourself. We did supervised learning, .....


We used GradCam and GradCam++ to visualize where the Network is focusing. Image Data is stored in Google Drive. 

Update: 11/23/2024, we tried restnet18 and restnet50. Model output is saved in the ./report/outputs folder. Based on the result, restnet18 seems to be better than resnet50 with 97% vs ~84% test acc. For the gradcam, resnet18 layer4 is showing promising results of where the building is, so the network is not looking at trees, skys, or grounds.


Update: 11/26/2024, added 9 more classes of UMD building images on top of our original 11 classes of buildings. We figured 11 classes is still too little. Based on our previous test acc, it seems to be that our test images have overlappings with our train images. To solve this, we decided to do add some noise like rain effects to our test images to see how it works out. We will try the new test images later after the date 11/29/2024 since there are much more stuff we need to test out.


Update: 11/27/2024, have some progress on FixMatch implementation but needs improvement. The time it takes to train is way too long.


Update: 11/28/2024, added vision transformer as our model to train on classification. Trained on 9 classes, with test acc 91% with the pretrained weights in mind; if trained from scratch, test acc is about 60%, kin  of in the expected range between 50 and 70 percent.


Update: 11/29/2024, added GradCam and GradCam++ methods to visualize vision transformer layers. FixMatch is still not doing good. Need to either fix or move on and try other methods.
