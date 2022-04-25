# Master-Thesis



Around 30% of Earth's surface is covered by Forests. Food, fuel, and medicine are provided for billions of people by these forests. Otherwise, millions of people work in the forest sector.
Over the years, tree removal increased significantly in various ways, some of these operations were by humans for construction or even to make the trees fuel for winter, or without human intervention. The permanent removal of trees leads to the term "Deforestation"[1,2]. The impact of deforestation reached to climate change and emissions of greenhouse gas. 

In terms of forest cover, such deforested areas are usually investigated by forestry experts. But sometimes the field studies are hard in remote areas, therefore, remote sensing is essential to make study in these areas affected by deforestation. One of the methods used in remote sensing is satellite images[3]. Satellites able to capture thousands of hectares on a single shoot with resolution arrived at 30 meters per pixel[4], and it is used significantly for capturing the forested areas so that many websites have been created for this purpose such as global forest watch[5] and  Planet[6]. 

Lately, Deep Learning has come to solve various problems in various areas. It has shown great effectiveness in studying and analysing satellite images, So that a few researchers are keen to apply deep learning to investigate and detect deforestation around the world. Mhatre et al.[7], proposed a CNN-based approach to monitor the vegetarian distribution using satellite images. The CNN model was trained by using supervised learning to get an accurate determination of forest cover. The object features identified by CNN were very complex, this returned to the number of filters applied to identify different patterns. The dataset that was used was a combination of Kaggle's dataset and a collected set from the Planet website[5]. 
Also, three CNN architectures have been used in [8], ResUnet, SharpMask and U-Net. ResUnet model performance was the best in overall measures and the least amount of errors. Rakshit et al. [9] used a classification VGG16 model to track the changing land pattern in the Amazon rainforests. It achieved high accuracy after one hour of training. The experiment was done on a dataset collected from Planets  Flock 2 satellites between January 1, 2016, and February 1, 2017.
Similarly, Ortega et al. [10] worked on deforestation monitoring using CNN-based models and the PRODES database. Siamese Convolutional Neural Network (S-CNN), Early Fusion (EF),  and the Support Vector Machine (SVM) as a baseline were tested. S-CNN was superior in terms of F1-score and Overall Accuracy to its counterparts, it identified deforested areas. Whereas EF was better than SVM that gave a weak performance for deforestation detection. Another study using FCN-based was in[3], where SegNet and U-Net algorithms showed low performance for land use types with a small number of pixels, and within non-forest lands, a misclassification occurred when dividing the U-Net results. 
Chantharaj et al.  [11] presented a modified SegNet framework in addition to a method for adding more feature bands into satellite images which helps in improving the F1-score. This framework applied to medium resolution satellite images. The performance of SegNet after modification was better on the RGB band and the feature band improved the overall accuracy.

Semantic segmentation is known as a hard task in a system of computer vision and plays an essential role in analysis tasks and images understanding. It has various applications in artificial intelligence and computer vision such as robot navigation, medical sciences, remote sensing and so on. Modern work in deep learning has been improved significantly to deal with semantic segmentation through using a neural network, allowing faster and more precise segmentation. The studies that employ semantic segmentation in satellite imagery isn't that much, so in our work, we will attempt to employ deep learning in order to do an investigation around the deforestation issue. So we are going to use satellite images in this study and apply semantic segmentation [12,13] as a method to identify the deforestation areas.


III. Aims of the study:
Nowadays,  object detection has become one of the most important problems that researchers are concerned to solve, not to mention image segmentation which is the most interesting and difficult problem. It is a primary task in remote sensing imagery and computer vision, it involves finding the exact boundary of the object in a particular image. 
In our work, we will go through this concept in all of its dimensions, so we aim to:
Acquiring satellite imagery for Jordan regions. 


Creating Masks for Forests.


Monitoring deforestation using AI's techniques.


Reduce deforestation through monitoring.


Testing the ability of semantic segmentation to identify regions that are regressing.


Training the model on remote sensing to be able to diagnose potentially affected areas.



IV. Materials and Methods:

   

In this work we are going to do the next: 

Dataset collection and preparation: We aim at collecting a set of a new collection of satellite images for the forests and green landscapes in Jordan. This dataset will represent many regions over many years to monitor any changes in the forest coverage areas, which enable us to monitor them automatically. Annotation of images will be done by a specific annotation tool.


Dataset Pre-Processing: The aim of the pre-processing step is to improve the image data that suppresses distortions and enhances the important features such as noise reduction, contrast, image scaling, colour space conversion for more processing.






Image data augmentation: The aim is to artificially increase the size of a training dataset by generating modified versions of images in the dataset to prevent overfitting and get accurate results.


Semantic segmentation: We aim to segment the forest cover which appears on satellite images of any region of interest. This process helps in identifying deforestation that has been monitored over several years.


Deep learning model: We aim to learn the model to execute a classification task. This will be done by training a model using a large set of labelled data and neural network architectures that include various layers.


Satellite image analysis: We aim at this process to extract the information  and make decisions on the final detected and segmented images that have been monitored over several years.




V. References
[1] “Deforestation: Facts, Causes & Effects | Live Science.” https://www.livescience.com/27692-deforestation.html (accessed Mar. 06, 2021).
[2] “Deforestation and Forest Degradation | Threats | WWF.” https://www.worldwildlife.org/threats/deforestation-and-forest-degradation (accessed Mar. 06, 2021).
[3] S. H. Lee, K. J. Han, K. Lee, K. J. Lee, K. Y. Oh, and M. J. Lee, “Classification of landscape affected by deforestation using high‐resolution remote sensing data and deep‐learning techniques,” Remote Sens., vol. 12, no. 20, pp. 1–16, 2020, doi: 10.3390/rs12203372.

[4] “Remote Sensing for Forest Landscapes. | by Alexander Watson | openforests | Medium.” https://medium.com/openforests/remote-sensing-for-forest-landscapes-83e246261c21 (accessed Mar. 06, 2021).
[5] “Interactive World Forest Map & Tree Cover Change Data | GFW.” https://www.globalforestwatch.org/map/ (accessed Mar. 07, 2021).
[6] “Planet | Homepage.” https://www.planet.com/ (accessed Mar. 07, 2021).
[7] A. Mhatre, N. K. Mudaliar, M. Narayanan, A. Gurav, A. Nair, and A. Nair, “Using deep learning on satellite images to identify deforestation/afforestation,” in Advances in Intelligent Systems and Computing, Sep. 2020, vol. 1108 AISC, pp. 1078–1084, doi: 10.1007/978-3-030-37218-7_113.
[8] P. P. de Bem, O. A. de Carvalho, R. F. Guimarães, and R. A. T. Gomes, “Change detection of deforestation in the brazilian amazon using landsat data and convolutional neural networks,” Remote Sens., vol. 12, no. 6, 2020, doi: 10.3390/rs12060901.
[9] S. Rakshit, S. Debnath, and D. Mondal, “Identifying Land Patterns from Satellite Imagery in Amazon Rainforest using Deep Learning,” arXiv, Sep. 2018, Accessed: Mar. 16, 2021. [Online]. Available: http://arxiv.org/abs/1809.00340.
[10] M. X. Ortega, J. D. Bermudez, P. N. Happ, A. Gomes, and R. Q. Feitosa, “EVALUATION of DEEP LEARNING TECHNIQUES for DEFORESTATION DETECTION in the AMAZON FOREST,” ISPRS Ann. Photogramm. Remote Sens. Spat. Inf. Sci., vol. 4, no. 2/W7, pp. 121–128, 2019, doi: 10.5194/isprs-annals-IV-2-W7-121-2019.
[11] S. Chantharaj et al., “Semantic Segmentation on Medium-Resolution Satellite Images Using Deep Convolutional Networks with Remote Sensing Derived Indices,” Sep. 2018, doi: 10.1109/JCSSE.2018.8457378.
[12] J. Long, E. Shelhamer, and T. Darrell, “Fully Convolutional Networks for Semantic Segmentation.”
[13] F. Lateef and Y. Ruichek, “Survey on semantic segmentation using deep learning techniques,” Neurocomputing, vol. 338, pp. 321–348, Apr. 2019, doi: 10.1016/j.neucom.2019.02.003.







