## CMSC472_Final_Project

Progress log:

This project is about classifying UMD buildings (20 buildings). So far, we have close to 4000 labeled images for this project.

Update: 11/22/2024, it is possible we might need to change the structure of the project, or not.
I haven't have the time to run the code, please check it yourself. We did supervised learning, .....


We used GradCam and GradCam++ to visualize where the Network is focusing. Image Data is stored in Google Drive. 

Update: 11/23/2024, we tried restnet18 and restnet50. Model output is saved in the ./report/outputs folder. Based on the result, restnet18 seems to be better than resnet50 with 97% vs ~84% test acc. For the gradcam, resnet18 layer4 is showing promising results of where the building is, so the network is not looking at trees, skys, or grounds.


Update: 11/26/2024, added 9 more classes of UMD building images on top of our original 11 classes of buildings. We figured 11 classes is still too little. Based on our previous test acc, it seems to be that our test images have overlappings with our train images. To solve this, we decided to add some noise like rain effects to our test images to see how it works out. We will try the new test images later after the date 11/29/2024 since there are much more stuff we need to test out.


Update: 11/27/2024, have some progress on FixMatch implementation but needs improvement. The time it takes to train is way too long.


Update: 11/28/2024, added vision transformer as our model to train on classification. Trained on 9 classes, with test acc 91% with the pretrained weights in mind; if trained from scratch, test acc is about 60%, kin  of in the expected range between 50 and 70 percent.


Update: 11/29/2024, added GradCam and GradCam++ methods to visualize vision transformer layers. FixMatch is still not doing good. Need to either fix or move on and try other methods.


Update: 11/30/2024, implemented MeanTeacher, got some gradcam results. The results are quite interesting that the gradcam shows us the neural network is looking at the right places and building parts, but we have a test acc of around 54% for student and teacher models. Well, the data we used for this quick 20 classification experiment is from the fixmatch photos, where the number of labeled images are very little like 10-20 per building, and the rest of the thousand photos are regarded as unlabeled. We decided to increase the labeled images like 2 times, so 40-ish labeled images per building, and see how it goes. It is possible we might do some data augmentation to some unlabeled data and see how it goes out in MeanTeacher.


Update: 12/2/2024, just training different pretrained models like resnet18, vgg11, vit16_224, etc to get some results for the report. FixMatch, after a few days of constant training, is giving us top5 with value 78 something, which is a good sign overall compared to a few days ago with only 48 something. Might add learning rate scheduler in MeanTeacher. Realized if we need to do VQA models, it requires use to have a lot of questions and answers for each image, which is not practical to do given limited time and energy. So what other ways can we try to figure out to make the classification better? Perhaps Human interaction during the training? Currently trying to code a quick prove of concept idea of having a feedback loop where the user can see the gradcam image and select which class the image belong to. Never tried it, so we will see. 
