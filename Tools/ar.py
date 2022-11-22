"""
Credit: this code was adapted from an open-source project, here is the original project : https://github.com/cedishppy/Markerless-AR-Overlay-Python-OpenCV 
"""

import cv2, numpy as np, glob

marker = 'koi.png'


img1 = cv2.imread(marker) # This is the image identified for overlaying
win_name = 'Camera Matching'
MIN_MATCH = 10
images = glob.glob('*.JPG') #all the jpg images in the folder could be displayed
currentImage=0  #the first image is selected
replaceImg=cv2.imread(images[currentImage])
rows,cols,ch = replaceImg.shape
pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
zoomLevel = 0   #when zoomLevel is positive it zooms in, when its negative it zooms out
processing = True   #boolean variable using for disabling the image processing
maskThreshold=10
# Detector used runs ORB (Oriented FAST and Rotated BRIEF)
detector = cv2.ORB_create(1000)
# Flann for approximating the nearest neighbour
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Start of video capture and setting the frame size
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if img1 is None:
        res = frame
    else:
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)
        ratio = 0.75
        good_matches = [m[0] for m in matches \
                            if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        matchesMask = np.zeros(len(good_matches)).tolist()
        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            #mask = cv2.erode(mask, (3, 3))
            #mask = cv2.dilate(mask, (3, 3))
            if mask.sum() > MIN_MATCH:
                matchesMask = mask.ravel().tolist()
                h,w, = img1.shape[:2]
                pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                dst = cv2.perspectiveTransform(pts,mtrx)
                dst = cv2.getPerspectiveTransform(pts1,dst)
                rows, cols, ch = frame.shape
                distance = cv2.warpPerspective(replaceImg,dst,(cols,rows))
                rt, mk = cv2.threshold(cv2.cvtColor(distance, cv2.COLOR_BGR2GRAY), maskThreshold, 1,cv2.THRESH_BINARY_INV)
                mk = cv2.erode(mk, (3, 3))
                mk = cv2.dilate(mk, (3, 3))
                for c in range(0, 3):
                    frame[:, :, c] = distance[:,:,c]*(1-mk[:,:]) + frame[:,:,c]*mk[:,:]
        cv2.imshow('img', frame)
        #Wait for the key
        key = cv2.waitKey(1)
        #decide the action based on the key value (quit, zoom, change image)
        if key == ord('q'): # quit
            print ('Quit')
            break
        if key == ord('i'): # + zoom in
            zoomLevel=zoomLevel+0.05
            rows,cols,ch = replaceImg.shape
            A=[-zoomLevel*cols,-zoomLevel*rows]
            B=[-zoomLevel*cols,zoomLevel*rows]
            C=[zoomLevel*cols,zoomLevel*rows]
            D=[zoomLevel*cols,-zoomLevel*rows]
            pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
            pts1 = pts1 + np.float32([A,B,C,D])
            print ('Zoom in')

        if key == ord('o'): # - zoom out
            zoomLevel=zoomLevel-0.05
            rows,cols,ch = replaceImg.shape
            pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
            pts1 = pts1 + np.float32([[-zoomLevel*cols,-zoomLevel*rows],
                                      [-zoomLevel*cols,zoomLevel*rows],
                                      [zoomLevel*cols,zoomLevel*rows],
                                      [zoomLevel*cols,-zoomLevel*rows]])
            print ('Zoom out')
        if key == ord('n'): # -> next image
            if currentImage<len(images)-1:
                currentImage=currentImage+1
                replaceImg=cv2.imread(images[currentImage])
                rows, cols, ch = replaceImg.shape
                pts1 = np.float32([[0, 0], [0, rows], [(cols), (rows)], [cols, 0]])
                pts1 = pts1 + np.float32([[-zoomLevel * cols, -zoomLevel * rows],
                                          [-zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, -zoomLevel * rows]])
                print ('Next image')
            else:
                print ('No more images on the right')
        if key == ord('m'): # <- previous image
            if currentImage>0:
                currentImage=currentImage-1
                replaceImg=cv2.imread(images[currentImage])
                rows, cols, ch = replaceImg.shape
                pts1 = np.float32([[0, 0], [0, rows], [(cols), (rows)], [cols, 0]])
                pts1 = pts1 + np.float32([[-zoomLevel * cols, -zoomLevel * rows],
                                          [-zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, zoomLevel * rows],
                                          [zoomLevel * cols, -zoomLevel * rows]])
                print ('Previous image')
            else:
                print ('No more images on the left')
        else:
            print("We are doomed")
cap.release()
cv2.destroyAllWindows()
# KILL ME

#Using the open cv library, we first set  our chosen image and detect unique points on the image. Next the program
# Matches these points to whatever is captured on your computer's video frame. If the unique points are detected,
# the program will overlay an image of your choice at the exact coordinates