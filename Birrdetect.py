import numpy as np
import cv2
import common
import time
import thread
import sys, getopt
import wave
import pyaudio
from common import anorm, getsize

#--------------------------------------------------------------------------------------------------------
MIN_POINT = 30
chunk = 1024
FLANN_INDEX_KDTREE = 5

def init_feature():
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1    
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = MIN_POINT)
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


    
def play_Sound(cash):
    if cash == '1':
        stream = p.open(format = p.get_format_from_width(sound1.getsampwidth()),  
                        channels = sound1.getnchannels(),  
                        rate = sound1.getframerate(),  
                        output = True)
        playnow = sound1
    elif cash == '5':
        stream = p.open(format = p.get_format_from_width(sound5.getsampwidth()),  
                        channels = sound5.getnchannels(),  
                        rate = sound5.getframerate(),  
                        output = True)
        playnow = sound5
    elif cash == '10':
        stream = p.open(format = p.get_format_from_width(sound10.getsampwidth()),  
                        channels = sound10.getnchannels(),  
                        rate = sound10.getframerate(),  
                        output = True)
        playnow = sound10
    elif cash == '50':
        stream = p.open(format = p.get_format_from_width(sound50.getsampwidth()),  
                        channels = sound50.getnchannels(),  
                        rate = sound50.getframerate(),  
                        output = True)
        playnow = sound50
    elif cash == '100':
        stream = p.open(format = p.get_format_from_width(sound100.getsampwidth()),  
                        channels = sound100.getnchannels(),  
                        rate = sound100.getframerate(),  
                        output = True)
        playnow = sound100



    data = playnow.readframes(chunk)
    #paly stream  
    while data != '':
        stream.write(data)  
        data = playnow.readframes(chunk)


    playnow.rewind()
    #stop stream  
    stream.stop_stream()  
    stream.close()

    #close PyAudio  
    p.terminate()
   
    

def filter_matches(kp1, kp2, matches, ratio = 0.75): #ratio = 0.75
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if H is not None :
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, showText, (corners[0][0],corners[0][1]), font, 1,(0,0,255),2)
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
        
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = (0, 255, 0)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
    return vis

def match_and_draw(win, checksound, found , count):
    if(len(kp2) > 0):
	#matching feature
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= MIN_POINT:
            if not found:
                checksound = True
            found = True
            count = count + 1
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            vis = explore_match(win, img1, img2, kp_pairs, status, H)
	    #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
            
        else:
            found = False
            checksound = False
            H, status = None, None
            count = 0
            #print '%d matches found, not enough for homography estimation' % len(p1)
            vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1+w2] = img2
        
        cv2.imshow('find_obj', vis)
        if(count > 3):
            if checksound :
                checksound = False
                play_Sound(showText)
                count = 0
        
    return found, checksound ,count 
        
#--------------------------------------------------------------------------------------------------------
cv2.useOptimized()

cap = cv2.VideoCapture(0)
detector, matcher = init_feature()

checksound = True
found = False
searchIndex = 1
count = 0

#preLoad resource
birr1 = cv2.imread( 'birr/1f.jpg' , 0 )
temp_kp1, temp_desc1 = detector.detectAndCompute(birr1, None)

birr2 = cv2.imread( 'birr/1b.jpg' , 0 )
temp_kp2, temp_desc2 = detector.detectAndCompute(birr2, None)

birr3 = cv2.imread( 'birr/5f.jpg' , 0 )
temp_kp3, temp_desc3 = detector.detectAndCompute(birr3, None)

birr4 = cv2.imread( 'birr/5b.jpg' , 0 )
temp_kp4, temp_desc4 = detector.detectAndCompute(birr4, None)

birr5 = cv2.imread( 'birr/10f.jpg' , 0 )
temp_kp5, temp_desc5 = detector.detectAndCompute(birr5, None)

birr6 = cv2.imread( 'birr/10b.jpg' , 0 )
temp_kp6, temp_desc6 = detector.detectAndCompute(birr6, None)

birr7 = cv2.imread( 'birr/50f.jpg' , 0 )
temp_kp7, temp_desc7 = detector.detectAndCompute(birr7, None)

birr8 = cv2.imread( 'birr/50b.jpg' , 0 )
temp_kp8, temp_desc8 = detector.detectAndCompute(birr8, None)

birr9 = cv2.imread( 'birr/100f.jpg' , 0 )
temp_kp9, temp_desc9 = detector.detectAndCompute(birr9, None)

birr10 = cv2.imread( 'birr/100b.jpg' , 0 )
temp_kp10, temp_desc10 = detector.detectAndCompute(birr10, None)

#preLoad sound
auido1 = wave.open(r"auido/1.wav","rb")
auido5 = wave.open(r"auido/5.wav","rb")
auido10 = wave.open(r"auido/10.wav","rb")
auido50 = wave.open(r"auido/50.wav","rb")
auido100 = wave.open(r"auido/100.wav","rb")

while(True):
    t1 = cv2.getTickCount()
    p = pyaudio.PyAudio() 
    #switch template
    if not found:
        if searchIndex <= 10:
            if searchIndex == 1:
                img1 = birr1
                kp1 = temp_kp1
                desc1 = temp_desc1
                showText = '1'
            elif searchIndex == 2:
                img1 = birr2
                kp1 = temp_kp2
                desc1 = temp_desc2
                showText = '1'
            elif searchIndex == 3:
                img1 = birr3
                kp1 = temp_kp3
                desc1 = temp_desc3
                showText = '5'
            elif searchIndex == 4:
                img1 = birr4
                kp1 = temp_kp4
                desc1 = temp_desc4
                showText = '5'
            elif searchIndex == 5:
                img1 = birr5
                kp1 = temp_kp5
                desc1 = temp_desc5
                showText = '10'
            elif searchIndex == 6:
                img1 = birr6
                kp1 = temp_kp6
                desc1 = temp_desc6
                showText = '10'
            elif searchIndex == 7:
                img1 = birr7
                kp1 = temp_kp7
                desc1 = temp_desc7
                showText = '50'
            elif searchIndex == 8:
                img1 = birr8
                kp1 = temp_kp8
                desc1 = temp_desc8
                showText = '50'
            elif searchIndex == 9:
                img1 = birr9
                kp1 = temp_kp9
                desc1 = temp_desc9
                showText = '100'
            elif searchIndex == 10:
                img1 = birr10
                kp1 = temp_kp10
                desc1 = temp_desc10
                showText = '100'
                
            searchIndex = searchIndex+1
        else:
            searchIndex = 1
            img1 = birr1
            
                    
    #Capture frame-by-frame
    ret, frame = cap.read()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    #calculate features
    kp2, desc2 = detector.detectAndCompute(img2, None)
    #print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    found, checksound, count = match_and_draw('find_obj',checksound,found,count)

    t2 = cv2.getTickCount()
    #calculate fps
    time = (t2 - t1)/ cv2.getTickFrequency()
    print 'FPS = ',1/time
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
# When everything done, release the capture
 
#close PyAudio  
p.terminate()
 
cap.release()
cv2.destroyAllWindows()
