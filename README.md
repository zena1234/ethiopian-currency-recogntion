# ethiopian-currency-recogntion
Ethiopian currency recognition for visually impaired people
The billing system for ethiopian visualy impired people is very difficult.untill now it follows a traditional way. There are so many cases for this system to be difficult.When the visual impired man wants to deal with his currency he asks another person, this consumes human labour,time and there is phychological effect , in addition to this cheating the money from the victim,pisychological disblife are occurred.,time consumeing for the person who try to help the blind person. 
Such problems could be solved by ethiopian currency recongnition system .the visualy impaired people can use the system at any pMatch input image with datasets 
In order to confirm image similarity, we check whether the key points in the test image are in spatial consistency with the retrieved images. We use the popular method of geometric verification (GV) by fitting fundamental matrix (adopted from [16]) to find out the number of key points of the test image that are spatially consistent with those of the retrieved images. 5) Classification: In the voting mechanism, each retrieved image adds votes to its image class (type of bill) by the number of spatially consistent key points it has (computed in the previous step). The class with the highest vote is declared as the result. 
lace and time they want. 

The overall objective of currency recognition system is for the provision of improving the billing system for visually impaired people through fast, timely  convenient billing system by insuring the visually impaired  person the birr you see it is the one you bill it .

                                Methodology
Image retrieval 
The first stage of any vision system is the image acquisition stage. After the image has been obtained, various methods of processing can be applied to the image to perform the many different tasks. Performing image acquisition in image processing is always the first step in the workflow sequence because, without an image, no processing is possible. There are various ways to obtain image such as with the help of camera or scanner. Acquired image should keep all the features.

Pre-processing 
The main goal of the pre-processing to increase the visual appearance of images and improve the impact of datasets. Pre-processing of image are those operations that are normally required earlier to the main data analysis and extraction of information. Image pre-processing, also called image restoration, and involves the correction of distortion, degradation, and noise introduced during the imaging process. Image pre-processing can notably increase the accuracy of an optical inspection. Image Adjusting is done with the help of image interpolation. Interpolation is the technique mostly used for tasks such as zooming, rotating, shrinking etc. Removing the noise is an important step when image processing is being performed. However noise can affect segmentation and pattern matching. When performing smoothing process on a pixel, and neighbour of the pixel is used to do some conversion. After that a new value of the pixel is created. 
 Remove background
 As illustrated in architecture, the images are captured in a wide variety of environment, in association to lighting condition and background while the currency in the image itself could be damaged. Image segmentation is important for reducing the data to process and remove unwanted features (background region) that would involve the decision-making. We start with a fixed rectangular region of interest (ROI) which is forty pixels smaller from all four sides than the image itself. We assume that a major part of the currency will be present inside this region. Once this region is obtained, it must be extended to a segmentation of the entire image. For removing the unwanted background here we are use Grab cut algorithm.
 Feature Extraction
 Feature extraction is a special type of dimensional reduction. When the input of an algorithm is too large to be processed and it is not needed then the input data will be converted into a reduced representation set of features. Transforming the input data into the set of features is called feature extraction. If the features extracted are carefully selected it is supposed to the features set will extract the related information from the input data to perform the required task using this reduced representation instead of the full size input. 


      Match input image with datasets 
In order to confirm image similarity, we check whether the key points in the test image are in spatial consistency with the retrieved images. We use the popular method of geometric verification (GV) by fitting fundamental matrix (adopted from [16]) to find out the number of key points of the test image that are spatially consistent with those of the retrieved images. 5) Classification: In the voting mechanism, each retrieved image adds votes to its image class (type of bill) by the number of spatially consistent key points it has (computed in the previous step). The class with the highest vote is declared as the result. 

        Audio output generation 
The recognized text codes are recorded in script files. Then we employ the text to speech converter to load these files and display the audio output of text information. Blind users can adjust speech rate, volume and language according to their preferences.

installations:
1) Install Anaconda(https://docs.anaconda.com/anaconda/install/)
2) install numpy( using conda prompt)
conda install numpy
3) install Opencv(using conda prompt)
conda install -c  conda -forge opencv
4) install time
conda install time
5) install thread
conda install thread
6) install system
 conda install -c conda​-forge r-sys
 conda install -c bioconda gnu-getopt 
 7) install wave
conda install -c contango wave
8) install pyaudio
conda install -c anaconda pyaudio
N.B common is a package found in folder by the format of common.py , you can access it through importing to billdetect.py
