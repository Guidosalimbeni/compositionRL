# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:37:29 2018
Class for implementing the scores for the composition UI and also the display image
with all the scores@author: Guido Salimbeni
"""

import cv2
import numpy as np
import itertools
from scipy.spatial import distance as dist
from skimage.measure import compare_ssim as ssim
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import pandas as pd
from SaliencyMap import Saliency

class AutoScoreML ():
    
    def __init__(self, extractedFeatures ):
        self.df = pd.DataFrame(np.array(extractedFeatures))
      
    def autoScoreML(self):
        filepath_01 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoringApril30.csv'
        filepath_02 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoringApril30_B.csv'
# =============================================================================
#         filepath_03 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring20apr2018_c.csv'
#         filepath_04 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring20apr2018_d.csv'
#         filepath_05 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring20apr2018_e.csv'
#         filepath_06 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring21apr2018_a.csv'
#         filepath_07 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring21apr2018_b.csv'
#         filepath_08 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring21apr2018_c.csv'
#         filepath_09 = 'D:\\google drive\A PhD Project at Godlsmiths\ArtistSupervisionMaya\csv_file\scoring22apr2018_a.csv'
#          
# =============================================================================
        df_01 = pd.read_csv(filepath_01)
        df_02 = pd.read_csv(filepath_02)
# =============================================================================
#         df_03 = pd.read_csv(filepath_03)
#         df_04 = pd.read_csv(filepath_04)
#         df_05 = pd.read_csv(filepath_05)
#         df_06 = pd.read_csv(filepath_06)
#         df_07 = pd.read_csv(filepath_07)
#         df_08 = pd.read_csv(filepath_08)
#         df_09 = pd.read_csv(filepath_09)
# =============================================================================
        
        frames= [df_01, df_02 
                 #,df_03, df_04, df_05, df_06, df_07, df_08, df_09
                 ]
        
        df = pd.concat(frames)
        df.reset_index(drop = True, inplace = True)
        
        # drop the Null Value
        df.dropna(inplace=True)
        # select the features to use:
        df.drop(['file', 'CompositionUserChoice'], axis=1, inplace=True)
        
        X_train = df.drop('judge', axis = 1)
        #y = df['judge']
        
        
        X_test = self.df
        
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        
        # construct the ANN
        # import the Keras Library and the required packages
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.models import model_from_json
        import os
        # load json and create model
        json_file = open("D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\code\\classifier.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\code\\classifier.h5")
        print("Loaded model from disk")
         
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# =============================================================================
#         score = loaded_model.evaluate(X_test, y_test, verbose=0)
#         print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
# =============================================================================
        
        # predict the test set results
        
        # =============================================================================
        y_pred = loaded_model.predict(X_test)
        for y in y_pred:
            res = np.argmax(y)
            
            return res

        
        
class CompositionAnalysis ():
    
    def __init__ (self, image = None, imagepath = None, mask = None):
        
        if imagepath:
            self.image = cv2.imread(imagepath)
            self.imagepath = imagepath
        else:
            
            self.image = image
            
        
        
        self.totalPixels = self.image.size
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
# =============================================================================
#     
#     def _borderCut(self, borderCutted):
#         
#         
#         borderCutted[0:2, :] = 0
#         borderCutted[-2:self.image.shape[0], :] = 0
#         borderCutted[:, 0:2] = 0
#         borderCutted[:, -2:self.image.shape[1]] = 0
#         
#         return borderCutted
# =============================================================================
       
        
        
    
    def synthesisScores (self):
        
        # return the display image for the UI
        rows, cols, depth = self.image.shape
        scoreSynthesisImg = np.zeros(self.image.shape, dtype="uint8")
        # make solid color for the background
        scoreSynthesisImg[:] = (218,218,218)
        cv2.line(scoreSynthesisImg, ( int(self.image.shape[1] * 0.6), 20), ( int(self.image.shape[1] * 0.6),self.image.shape[0]), (50,50,140), 1)
        cv2.line(scoreSynthesisImg, ( int(self.image.shape[1] * 0.75), 20), ( int(self.image.shape[1] * 0.75),self.image.shape[0]), (60,140,90), 1)
        
        # collect the balance scores:
        VisualBalanceScore = (  self.scoreVisualBalance + self.scoreHullBalance ) / 2
        # corner balance and line
        lineandcornerBalance = (self.cornersBalance  +  self.verticalandHorizBalanceMean ) / 2
        # collect the rythm scores:
        #asymmetry = (self.scoreFourier + self.verticalandHorizBalanceMean + self.ssimAsymmetry) / 3
        asymmetry = (self.ssimAsymmetry +self.diagonalAsymmetry) / 2
        scoreFourier = self.scoreFourier
        # collect the gold proportion scores:
        goldScore = self.scoreProportionAreaVsGoldenRatio
        
        #score composition
        scoreCompMax = max(self.diagonalasymmetryBalance, self.ScoreFourTriangleAdapted,self.ScoreBigTriangle)
        ruleOfThird = self.ScoreRuleOfThird
        # diagonal balance commposition
        #diagonalasymmetryBalance = self.diagonalasymmetryBalance
        # spiral
        spiralScore = self.scoreSpiralGoldenRatio
        # fractal
        fractalScoreFromTarget = self.fractalScoreFromTarget
        
        cv2.putText(scoreSynthesisImg, "Balance", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[20:24, 10:int(VisualBalanceScore*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Rule of Third", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[35:39, 10:int(ruleOfThird*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Composition Max", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[50:54, 10:int(scoreCompMax*cols*0.9)] = (120,60,120)
        #cv2.putText(scoreSynthesisImg, "Diagonal Comp", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        #scoreSynthesisImg[65:70, 10:int(diagonalasymmetryBalance*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Spiral ", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[80:84, 10:int(spiralScore*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Asymmetry ", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[95:99, 10:int(asymmetry*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Fourier ", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[110:114, 10:int(scoreFourier*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "CornerLinesBalance ", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[125:129, 10:int(lineandcornerBalance*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Proportion ", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[140:144, 10:int(goldScore*cols*0.9)] = (120,60,120)
        cv2.putText(scoreSynthesisImg, "Fractal ", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        scoreSynthesisImg[155:159, 10:int(fractalScoreFromTarget*cols*0.9)] = (120,60,120)
        
        #cv2.putText(scoreSynthesisImg, "Balance, asymmetry, Proportion, corner, spiral ", (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        #cv2.putText(scoreSynthesisImg, "Possible Comp: {} ".format(selectedComp), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
        
        return scoreSynthesisImg
    
    
    def fourierOnEdgesDisplay (self):
        
        ImgImpRegionA, contours, keypoints = self._orbSegmentation ( maxKeypoints = 10000, edged = False, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
        
        cropped_img_lf = ImgImpRegionA[0:int(ImgImpRegionA.shape[0]), 0: int(ImgImpRegionA.shape[1] / 2) ]
        cropped_img_rt = ImgImpRegionA[0:int(ImgImpRegionA.shape[0]), int(ImgImpRegionA.shape[1] / 2): ImgImpRegionA.shape[1] ]
        
        
        #imgDftGray = self._returnDFT(ImgImpRegionA)
        imgDftGraylf = self._returnDFT(cropped_img_lf)
        imgDftGrayRt = self._returnDFT(cropped_img_rt)
        
        # number of pixels in left and number of pixels in right
        numberOfWhite_lf = (imgDftGraylf>0).sum()
        numberOfWhite_Rt = (imgDftGrayRt > 0).sum()

        # create the stiched picture
        stichedDft = self.image.copy()
        
        stichedDft = np.concatenate((imgDftGraylf,imgDftGrayRt ), axis = 1)
        score = (abs(numberOfWhite_lf - numberOfWhite_Rt)) / (numberOfWhite_lf + numberOfWhite_Rt)
        # to penalise the change in rithm
        scoreFourier = np.exp(-score * self.image.shape[0]/2)
 
        #cv2.putText(stichedDft, "diff: {:.3f}".format(score), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        self.scoreFourier = scoreFourier
        
        return stichedDft, scoreFourier
    
    def _returnDFT (self, imageForDft):
        ImgImpRegionA = imageForDft
        ImgImpRegionA = cv2.cvtColor(ImgImpRegionA, cv2.COLOR_BGR2GRAY)
        
        #dft = cv2.dft(np.float32(self.gray),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft = cv2.dft(np.float32(ImgImpRegionA),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
      
        cv2.normalize( magnitude_spectrum, magnitude_spectrum, alpha = 0 , beta = 1 , norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        imgDftGray = np.array(magnitude_spectrum * 255, dtype = np.uint8)
        
        meanThres = np.mean(imgDftGray)
        
        _, imgDftGray = cv2.threshold(imgDftGray,meanThres, 255, cv2.THRESH_BINARY)
        
        imgDftGray = cv2.cvtColor(imgDftGray, cv2.COLOR_GRAY2BGR)
        
        return imgDftGray
    
    def HOGcompute (self):
        
        gray = self.gray.copy()
        # h x w in pixels
        cell_size = (8, 8) 
        
         # h x w in cells
        block_size = (2, 2) 
        
        # number of orientation bins
        nbins = 9
        
        # Using OpenCV's HOG Descriptor
        # winSize is the size of the image cropped to a multiple of the cell size
        hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1] * cell_size[1],
                                          gray.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        
        # Create numpy array shape which we use to create hog_feats
        n_cells = (gray.shape[0] // cell_size[0], gray.shape[1] // cell_size[1])
        
        # We index blocks by rows first.
        # hog_feats now contains the gradient amplitudes for each direction,
        # for each cell of its group for each group. Indexing is by rows then columns.
        hog_feats = hog.compute(gray).reshape(n_cells[1] - block_size[1] + 1,
                                n_cells[0] - block_size[0] + 1,
                                block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))  
        
        # Create our gradients array with nbin dimensions to store gradient orientations 
        gradients = np.zeros((n_cells[0], n_cells[1], nbins))
        
        # Create array of dimensions 
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
        
        # Block Normalization
        for off_y in range(block_size[0]):
            for off_x in range(block_size[1]):
                gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                          off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                           off_x:n_cells[1] - block_size[1] + off_x + 1] += 1
        
        # Average gradients
        gradients /= cell_count
        
# =============================================================================
#         # Plot HOGs using Matplotlib
#         # angle is 360 / nbins * direction
#         print (gradients.shape)
#         
#         color_bins = 5
#         plt.pcolor(gradients[:, :, color_bins])
#         plt.gca().invert_yaxis()
#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.colorbar()
#         plt.show()
#         cv2.destroyAllWindows()
# =============================================================================
        
        return
    
    def goldenProportionOnCnts(self,  numberOfCnts = 25, method = cv2.RETR_CCOMP, minArea = 2):
        
        edgedForProp = self._edgeDetection( scalarFactor = 1, meanShift = 0, edgesdilateOpen = True, kernel = 3)
        
        goldenPropImg = self.image.copy()
        
        # create the contours from the segmented image
        ing2, contours, hierarchy = cv2.findContours(edgedForProp, method,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        
        innerCnts = []
        for  cnt, h in zip (contours, hierarchy[0]):
            if h[2] == -1 :
                innerCnts.append(cnt)
        
        sortedContours = sorted(innerCnts, key = cv2.contourArea, reverse = True)
        
        selectedContours = [cnt for cnt in sortedContours if cv2.contourArea(cnt) > minArea]
        
        for cnt in selectedContours[0: numberOfCnts]:
            cv2.drawContours(goldenPropImg, [cnt], -1, (255, 0, 255), 1)
            
        # get all the ratio to check
        ratioAreas = []
        for index, cnt in enumerate(selectedContours[0: numberOfCnts]):
            if index < len(selectedContours[0: numberOfCnts]) -1:
                areaGoldenToCheck_previous = cv2.contourArea(selectedContours[index])
                areaGoldenToCheck_next = cv2.contourArea(selectedContours[index + 1])
                ratioArea = areaGoldenToCheck_previous / areaGoldenToCheck_next
                ratioAreas.append(ratioArea)
        
        meanAreaRatio = (np.mean(ratioAreas))
        diffFromGoldenRatio = abs(1.618 - meanAreaRatio)
        scoreProportionAreaVsGoldenRatio = np.exp(-diffFromGoldenRatio)
        
        cv2.putText(goldenPropImg, "GoldPr: {:.3f}".format(scoreProportionAreaVsGoldenRatio), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        
        self.scoreProportionAreaVsGoldenRatio = scoreProportionAreaVsGoldenRatio
        
        return goldenPropImg, scoreProportionAreaVsGoldenRatio
        

    def cornerDetectionVisualBalance (self, maxCorners = 40 , minDistance = 6, midlineOnCornersCnt =  True):
        
        # based on the idea that there is a balance in balanced distribution of corner
        # the mid axis is the mid of the extremes corners detected
        corners = cv2.goodFeaturesToTrack(self.gray, maxCorners, 0.01, minDistance )
        
        cornerimg = self.image.copy()
        
        cornersOntheLeft = 0
        cornersOntheRight = 0
        cornersOnTop = 0
        cornersOnBottom = 0
        # find the limit x and y of the detected corners
        listX = [corner[0][0] for corner in corners]
        listY = [corner[0][1] for corner in corners]
        minX = min(listX)
        maxX = max (listX)
        minY = min(listY)
        maxY = max (listY)

        for corner in corners:
            x, y = corner[0]
            x = int(x)
            y = int(y)
            if midlineOnCornersCnt:
                # find the middle x and middle y
                midx = minX + int((maxX - minX)/2)
                midy = minY + int((maxY - minY)/2)
                pass
            else:
                midx = int(self.image.shape[1] / 2)
                midy = int(self.image.shape[0] / 2)
                
            cv2.rectangle(cornerimg,(x-2,y-2),(x+2,y+2),(0,255,0), 1)
            if x < midx:
                cornersOntheLeft += 1
            if x > midx:
                cornersOntheRight += 1
            if y < midy:
                cornersOnTop += 1
            if y > midy:
                cornersOnBottom += 1
        scoreHorizzontalCorners = np.exp(-(abs(cornersOntheLeft - cornersOntheRight )/(maxCorners/3.14)))
        scoreVerticalCorners = np.exp(-(abs(cornersOnTop - cornersOnBottom )/(maxCorners/3.14)))
        
        cv2.putText(cornerimg, "Corn H: {:.3f} V: {:.3f}".format(scoreHorizzontalCorners, scoreVerticalCorners), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.cornersBalance = (scoreHorizzontalCorners + scoreVerticalCorners) / 2
        
        return cornerimg, scoreHorizzontalCorners, scoreVerticalCorners


    def goldenSpiralAdaptedDetection (self, displayall = False , displayKeypoints = True, maxKeypoints = 100, edged = True):
        
        goldenImgDisplay = self.image.copy()
        
        # segmentation with orb and edges
        ImgImpRegion, contours, keypoints = self._orbSegmentation ( maxKeypoints = maxKeypoints, edged = edged, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
        # find the center zig zag orb silhoutte
        copyZigZag, ratioGoldenRectangleZigZagOrb , sorted_contoursZigZag, zigzagPerimeterScore= self._zigzagCntsArea()

        #draw the bounding box
        c = max(sorted_contoursZigZag, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        if x==0 or x+w == self.image.shape[1] or y==0 or y+w == self.image.shape[0]:
            cv2.rectangle(goldenImgDisplay, (0,0), (self.image.shape[1], self.image.shape[0]), (0,255,0), 1)

        else:
            cv2.rectangle(goldenImgDisplay,(x,y),(x+w,y+h),(0,255,0),1)
        
        # create the guidelines
        im, im2,im3, im4 = self._drawGoldenSpiral(drawRectangle=False, drawEllipses = True,  x = w, y = h)
        transX = x
        transY = y
        T = np.float32([[1,0,transX], [0,1, transY]])
        imTranslated = cv2.warpAffine(im, T, (self.image.shape[1], self.image.shape[0]))
        T2 = np.float32([[1,0, -self.image.shape[1] + transX + w], [0,1, -self.image.shape[0] + transY + h]])
        imTranslated2 = cv2.warpAffine(im2, T2, (self.image.shape[1], self.image.shape[0]))
        T3 = np.float32([[1,0, transX], [0,1, -self.image.shape[0] + transY + h]])
        imTranslated3 = cv2.warpAffine(im3, T3, (self.image.shape[1], self.image.shape[0]))
        T4 = np.float32([[1,0, -self.image.shape[1] + transX + w], [0,1,  transY ]])
        imTranslated4 = cv2.warpAffine(im4, T4, (self.image.shape[1], self.image.shape[0]))
        
        # bitwise the guidlines for one display img
        goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated)
        goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated2)
        if displayall:
            goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated3)
            goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated4)
        
        if displayKeypoints:
            goldenImgDisplay = cv2.drawKeypoints(goldenImgDisplay, keypoints,goldenImgDisplay, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        
        # dilate the spirals
        kernel = np.ones((5,5),np.uint8)
        imTranslated = cv2.dilate(imTranslated,kernel,iterations = 3)
        imTranslated2 = cv2.dilate(imTranslated2,kernel,iterations = 3)
        imTranslated3 = cv2.dilate(imTranslated3,kernel,iterations = 3)
        imTranslated4 = cv2.dilate(imTranslated4,kernel,iterations = 3)
        # loop to collect the intersection
        intersection = cv2.bitwise_and(ImgImpRegion,imTranslated)
        intersection2 = cv2.bitwise_and(ImgImpRegion,imTranslated2)
        intersection3 = cv2.bitwise_and(ImgImpRegion,imTranslated3)
        intersection4 = cv2.bitwise_and(ImgImpRegion,imTranslated4)
        # sum of imgImpRegion
        sumOfAllPixelInImgImpRegion = (ImgImpRegion>0).sum()
        # sum of all intersections
        sum1 = (intersection>0).sum()
        sum2 = (intersection2>0).sum()
        sum3 = (intersection3>0).sum()
        sum4 = (intersection4>0).sum()
        maxSumIntersection = max(sum1, sum2, sum3, sum4)
        # calculate the ratio of the max vs whole
        scoreSpiralGoldenRatio = maxSumIntersection / sumOfAllPixelInImgImpRegion
        
        cv2.putText(goldenImgDisplay, "Gold: {:.3f}".format(scoreSpiralGoldenRatio), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.scoreSpiralGoldenRatio = scoreSpiralGoldenRatio
        
# =============================================================================
#         cv2.imshow('ImgImpRegion', ImgImpRegion)
#         cv2.imshow('imTranslated', imTranslated)
#         cv2.imshow('inter', intersection)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        

        
        return goldenImgDisplay, scoreSpiralGoldenRatio
    
    def goldenSpiralFixDetection (self, displayall = False , displayKeypoints = True, maxKeypoints = 100, edged = True, numberOfCnts = 40, scaleFactor = 0.5, bonus = 10):
        
        #goldenImgDisplay = self.image.copy()
        
        # segmentation with orb and edges
        ImgImpRegion, contours, keypoints = self._orbSegmentation ( maxKeypoints = maxKeypoints, edged = edged, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
        # implement the segmentation including the edges
        edgedImg = self._edgeDetection(scalarFactor = 1, meanShift = 0, edgesdilateOpen = False, kernel = 5)
        edgedImg = cv2.cvtColor(edgedImg, cv2.COLOR_GRAY2BGR)
        # give a weight to the edges detection smaller than the orb
        #edgedImg[np.where((edgedImg ==[255,255,255]).all(axis=2))] = [255,255,255]
        # implement with inner shape
        segmentationOnInnerCnts, contours = self._innerCntsSegmentation(numberOfCnts = numberOfCnts, method = cv2.RETR_CCOMP, minArea = 5)
        segmentationOnInnerCnts[np.where((segmentationOnInnerCnts ==[255,255,255]).all(axis=2))] = [40,40,40]
        
        
        # merge the masks
        ImgImpRegion = cv2.bitwise_or(ImgImpRegion,edgedImg)
        ImgImpRegion = cv2.bitwise_or(ImgImpRegion,segmentationOnInnerCnts)
        
        goldenImgDisplay = ImgImpRegion.copy()
# =============================================================================
#         # find the center zig zag orb silhoutte
#         copyZigZag, ratioGoldenRectangleZigZagOrb , sorted_contoursZigZag, zigzagPerimeterScore= self._zigzagCntsArea()
# 
#         #draw the bounding box
#         c = max(sorted_contoursZigZag, key=cv2.contourArea)
#         x,y,w,h = cv2.boundingRect(c)
# =============================================================================
        
        
        # set this way to make the boundig box the size of the frame.. for adaptive unmask above and adjust
        x=0
        y=0
        w = self.image.shape[1]
        h = self.image.shape[0]

        if x==0 or x+w == self.image.shape[1] or y==0 or y+h == self.image.shape[0]:
            cv2.rectangle(goldenImgDisplay, (0,0), (self.image.shape[1], self.image.shape[0]), (0,255,0), 1)

        else:
            cv2.rectangle(goldenImgDisplay,(x,y),(x+w,y+h),(0,255,0),1)
        
        # create the guidelines
        im, im2,im3, im4 = self._drawGoldenSpiral(drawRectangle=False, drawEllipses = True,  x = w, y = h)
        transX = x
        transY = y
        T = np.float32([[1,0,transX], [0,1, transY]])
        imTranslated = cv2.warpAffine(im, T, (self.image.shape[1], self.image.shape[0]))
        T2 = np.float32([[1,0, -self.image.shape[1] + transX + w], [0,1, -self.image.shape[0] + transY + h]])
        imTranslated2 = cv2.warpAffine(im2, T2, (self.image.shape[1], self.image.shape[0]))
        T3 = np.float32([[1,0, transX], [0,1, -self.image.shape[0] + transY + h]])
        imTranslated3 = cv2.warpAffine(im3, T3, (self.image.shape[1], self.image.shape[0]))
        T4 = np.float32([[1,0, -self.image.shape[1] + transX + w], [0,1,  transY ]])
        imTranslated4 = cv2.warpAffine(im4, T4, (self.image.shape[1], self.image.shape[0]))
        
        # dilate the spirals
        kernel = np.ones((5,5),np.uint8)
        AimTranslated = cv2.dilate(imTranslated,kernel,iterations = 3)
        AimTranslated2 = cv2.dilate(imTranslated2,kernel,iterations = 3)
        AimTranslated3 = cv2.dilate(imTranslated3,kernel,iterations = 3)
        AimTranslated4 = cv2.dilate(imTranslated4,kernel,iterations = 3)
        # loop to collect the intersection
        intersection = cv2.bitwise_and(ImgImpRegion,AimTranslated)
        intersection2 = cv2.bitwise_and(ImgImpRegion,AimTranslated2)
        intersection3 = cv2.bitwise_and(ImgImpRegion,AimTranslated3)
        intersection4 = cv2.bitwise_and(ImgImpRegion,AimTranslated4)
        # sum of imgImpRegion
        sumOfAllPixelInSilhoutte = (ImgImpRegion > 0).sum()
        
        sumofAlledgedandorb = (ImgImpRegion==255).sum()
        sumofAllInnerCnts = (ImgImpRegion==40).sum()
        sumOfAllPixelInImgImpRegion = sumofAlledgedandorb + (scaleFactor* sumofAllInnerCnts)
        
        # sum of all intersections
        sum1_orb = (intersection==255).sum()
        sum2_orb = (intersection2==255).sum()
        sum3_orb = (intersection3==255).sum()
        sum4_orb = (intersection4==255).sum()
        # for the inner shape
        sum1_inn = (intersection==40).sum()
        sum2_inn = (intersection2==40).sum()
        sum3_inn = (intersection3==40).sum()
        sum4_inn = (intersection4==40).sum()
        # weight
        
        sum1 = sum1_orb * bonus + (scaleFactor * sum1_inn)
        sum2 = sum2_orb * bonus + (scaleFactor * sum2_inn)
        sum3 = sum3_orb * bonus + (scaleFactor * sum3_inn)
        sum4 = sum4_orb * bonus + (scaleFactor * sum4_inn)

        maxSumIntersection = max(sum1, sum2, sum3, sum4)

        # calculate the ratio of the max vs whole and weighted with the overall area of te silhoutte compare to the size of the frame
        scoreSpiralGoldenRatio = maxSumIntersection / sumOfAllPixelInImgImpRegion * (sumOfAllPixelInSilhoutte / self.gray.size)
        
        cv2.putText(goldenImgDisplay, "Gold: {:.3f}".format(scoreSpiralGoldenRatio), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.scoreSpiralGoldenRatio = scoreSpiralGoldenRatio
        # bitwise the guidlines for one display img
        if displayall == False:
            if sum1 == max(sum1, sum2, sum3, sum4):
                goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated)
            if sum2 == max(sum1, sum2, sum3, sum4):
                goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated2)
            if sum3 == max(sum1, sum2, sum3, sum4):
                goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated3)
            if sum4 == max(sum1, sum2, sum3, sum4):
                goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated4)
        if displayall:
            goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated)
            goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated2)
            goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated3)
            goldenImgDisplay = cv2.bitwise_or(goldenImgDisplay, imTranslated4)
        
        if displayKeypoints:
            goldenImgDisplay = cv2.drawKeypoints(goldenImgDisplay, keypoints,goldenImgDisplay, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        return goldenImgDisplay, scoreSpiralGoldenRatio
    
    
    def displayandScoreExtremePoints(self, numberOfCnts = 20):
        

        
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        mean = np.mean(blur)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(blur, mean, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # cut the border for more precision
        thresh[0:2, :] = 0
        thresh[-2:self.image.shape[0], :] = 0
        thresh[:, 0:2] = 0
        thresh[:, -2:self.image.shape[1]] = 0
         
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = cnts[1]
        sorted_contours = sorted(cnts, key=cv2.contourArea, reverse = True)
        

        
# =============================================================================
#         c = max(cnts, key=cv2.contourArea)
# 
#         # determine the most extreme points along the contour
#         extLeft = tuple(c[c[:, :, 0].argmin()][0])
#         extRight = tuple(c[c[:, :, 0].argmax()][0])
#         extTop = tuple(c[c[:, :, 1].argmin()][0])
#         extBot = tuple(c[c[:, :, 1].argmax()][0])
# =============================================================================
        extLeftList=[]
        extRightList = []
        extTopList = []
        extBotList =[]
        for c in sorted_contours[0:numberOfCnts]:
           
            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            extLeftList.append(extLeft)
            extRightList.append(extRight)
            extTopList.append(extTop)
            extBotList.append(extBot)
            
        # sort the list of tuple by x
        extLeftListSorted = sorted(extLeftList, key=lambda x: x[0])
        extRightListSorted = sorted(extRightList, key=lambda x: x[0], reverse = True)
        extTopListSorted = sorted(extTopList, key=lambda x: x[1])
        extBotListSorted = sorted(extBotList, key=lambda x: x[1], reverse = True)
        
        extLeft = extLeftListSorted[0]
        extRight = extRightListSorted[0]
        extTop = extTopListSorted[0]
        extBot = extBotListSorted[0]
        
        
        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        image = self.image.copy()
        
        for c in sorted_contours[0:numberOfCnts]:
            cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
        
        
        cv2.circle(image, extLeft, 5, (0, 0, 255), -1)
        cv2.circle(image, extRight, 5, (0, 255, 0), -1)
        cv2.circle(image, extTop, 5, (255, 0, 0), -1)
        cv2.circle(image, extBot, 5, (255, 255, 0), -1)
        
        #calculate Manhattan distance from half side point
        leftHalfSidePoint = (0, int(self.image.shape[0]/2))
        rightHalfSidePoint =(self.image.shape[1], int(self.image.shape[0]/2))
        topHalfSidePoint = (int(self.image.shape[1] /2), 0) 
        bottomHalfSidePoint = (int(self.image.shape[1] /2), self.image.shape[0])
        
        cv2.circle(image, leftHalfSidePoint, 3, (0, 0, 255), -1)
        cv2.circle(image, rightHalfSidePoint, 3, (0, 0, 255), -1)
        cv2.circle(image, topHalfSidePoint, 3, (0, 0, 255), -1)
        cv2.circle(image, bottomHalfSidePoint, 3, (0, 0, 255), -1)
        
        #halfHight = int(self.image.shape[0]/2)
        dist01 = dist.euclidean(extLeft, leftHalfSidePoint)
        dist02 = dist.euclidean(extRight, rightHalfSidePoint)
        #meanDistA = int((dist01 + dist02)/2 )
        #scoreA = meanDistA / halfHight
        
        #halfwidth = int(self.image.shape[1]/2)
        dist03 = dist.euclidean(extTop, topHalfSidePoint)
        dist04 = dist.euclidean(extBot, bottomHalfSidePoint)
        #meanDistB = int((dist03 + dist04)/2 )
        #scoreB = meanDistB / halfwidth
        
        #meanScore = (scoreA + scoreB)/2
        #scoreMeanDistanceOppositeToHalfSide =  np.exp(-meanScore*1.9) # used with mean on negative 

        if extLeft[1] < (self.image.shape[0] / 2):
            DistExtLeft = - dist01
        else:
            DistExtLeft = dist01
        if extRight[1] < (self.image.shape[0] / 2):
            DistExtRight = - dist02
        else:
            DistExtRight = dist02
        if extTop[0] < (self.image.shape[1] / 2):
            DistExtTop = - dist03
        else:
            DistExtTop = dist03
        if extBot[0] < (self.image.shape[1] / 2):
            DistExtBot = - dist04
        else:
            DistExtBot = dist04
        
        # make the script indipendent from the size of the image
        if self.image.shape[1]> self.image.shape[0]:
            ratio = self.image.shape[1]
        else:
            ratio = self.image.shape[0]
        
        DistExtLeftToHalf = DistExtLeft / ratio
        DistExtRightToHalf = DistExtRight / ratio
        DistExtTopToHalf = DistExtTop / ratio
        DistExtBotToHalf = DistExtBot / ratio
        

# =============================================================================
#         if self.image.shape[1]> self.image.shape[0]:
#             ratio = self.image.shape[1] / self.image.shape[0]
#         else:
#             ratio = self.image.shape[0] / self.image.shape[1]
#         
#         meanNeg = (DistExtLeft + DistExtRight + DistExtTop + DistExtBot) / 4
#         scoreMeanDistanceOppositeToHalfSideadapted =  np.exp(-meanNeg / (ratio * 10))
# =============================================================================
        
        #cv2.putText(image, "Epts: {:.3f}".format(scoreMeanDistanceOppositeToHalfSideadapted), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
# =============================================================================
#         
#         self.scoreMeanDistanceOppositeToHalfSide = scoreMeanDistanceOppositeToHalfSideadapted
# =============================================================================
        
        # distances from borders:
        
        DistLeftBorder = abs(extLeft[0] - 0)
        DistRightBorder = abs(extRight[0] - self.image.shape[1])
        DistTopBorder = abs(extTop[1] - 0)
        DistBotBorder = abs(extBot[1] - self.image.shape[0])
        # make it indipendent from the size of the image by normalised by the lenght of the related side frame
        DistLeftBorder = DistLeftBorder / (self.image.shape[1])
        DistRightBorder = DistRightBorder / (self.image.shape[1])
        DistTopBorder = DistTopBorder / (self.image.shape[0])
        DistBotBorder = DistBotBorder / (self.image.shape[0])

        return image, DistExtLeftToHalf,DistExtRightToHalf,DistExtTopToHalf, DistExtBotToHalf, DistLeftBorder, DistRightBorder, DistTopBorder, DistBotBorder
    
    def vertAndHorizLinesBalance (self):
        
        edges = self._edgeDetection(scalarFactor = 1, meanShift = 0, edgesdilateOpen = False)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, 100, 1)
        
        copyImg = self.image.copy()
        
        verticalLines = 0
        horizontalLines = 0
        verticalLinesLeft = 0
        verticalLinesRight = 0
        horizontalLinesLeft = 0
        horizontalLinesRight = 0
        allX = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            allX.append(x1)
            allX.append(x2)
        # only horizzontal counts along the middle detection relevance
        midX = int((max(allX) - min(allX))/2) + min(allX)  
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2-x1) *180 / np.pi
            # horizzontal lines
            if angle == 0:
                cv2.line(copyImg, (x1, y1), (x2, y2), (0,0,255), 1)
                horizontalLines += 1
                if x1 < midX and x2 < midX:
                    horizontalLinesLeft += 1
                if x1 > midX and x2 > midX:
                    horizontalLinesRight += 1
            # vertical lines
            if angle == 90 or angle == -90 :
                cv2.line(copyImg, (x1, y1), (x2, y2), (0,255,0), 1)
                verticalLines += 1
                if x1 < midX and x2 < midX:
                    verticalLinesLeft += 1
                if x1 > midX and x2 > midX:
                    verticalLinesRight += 1
        diffVerticals = abs(verticalLinesLeft - verticalLinesRight)
        diffHorizontal = abs(horizontalLinesLeft -horizontalLinesRight )
        
        if verticalLines == 0 or horizontalLines == 0:
            verticalLinesBalanceScore = 0
            horizontalLinesBalanceScore = 0
            cv2.putText(copyImg, "Lines V: {:.3f} H: {:.3f}".format(verticalLinesBalanceScore,horizontalLinesBalanceScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            self.verticalandHorizBalanceMean = (verticalLinesBalanceScore + horizontalLinesBalanceScore) / 2
            
            return copyImg, verticalLinesBalanceScore, horizontalLinesBalanceScore
        else:
            verticalLinesBalanceScore = (1 - (diffVerticals/verticalLines))
            horizontalLinesBalanceScore = (1 - (diffHorizontal / horizontalLines))
            cv2.putText(copyImg, "Lines V: {:.3f} H: {:.3f}".format(verticalLinesBalanceScore,horizontalLinesBalanceScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            self.verticalandHorizBalanceMean = (verticalLinesBalanceScore + horizontalLinesBalanceScore) / 2
            
            return copyImg, verticalLinesBalanceScore, horizontalLinesBalanceScore
    
    def numOfTangentandBalance (self):
        
        edges = self._edgeDetection(scalarFactor = 1, meanShift = 0, edgesdilateOpen = False)
        
        # first template
        template = np.zeros((16,16), np.uint8)
        template[7:9,0:16] = 255
        template[0:16, 7:9] = 255
        # w and h to also use later to draw the rectagles
        w, h = template.shape[::-1]
        # rotated template
        M = cv2.getRotationMatrix2D((w/2,h/2),15,1)
        template15 = cv2.warpAffine(template,M,(w,h))
        template30 = cv2.warpAffine(template15,M,(w,h))
        template45 = cv2.warpAffine(template30,M,(w,h))
        template60 = cv2.warpAffine(template45,M,(w,h))
        template75 = cv2.warpAffine(template60,M,(w,h))
        # run the matchtemplate
        result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF)
        result15 = cv2.matchTemplate(edges, template15, cv2.TM_CCOEFF)
        result30 = cv2.matchTemplate(edges, template30, cv2.TM_CCOEFF)
        result45 = cv2.matchTemplate(edges, template45, cv2.TM_CCOEFF)
        result60 = cv2.matchTemplate(edges, template60, cv2.TM_CCOEFF)
        result75 = cv2.matchTemplate(edges, template75, cv2.TM_CCOEFF)
        #find the points of match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = max_val * 0.96
        loc = np.where(result >= threshold)
        loc90 = np.where(result15 >= threshold)
        loc180 = np.where(result30 >= threshold)
        loc270 = np.where(result45 >= threshold)
        loc180a = np.where(result60 >= threshold)
        loc270a = np.where(result75 >= threshold)
        #convert edges for display
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        points = []
        for pt  in zip (*loc[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc90[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc180[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc270[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc180a[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        for pt  in zip (*loc270a[::-1]):
            cv2.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            points.append(pt)
        
        score = np.exp(- len(points) )
        
# =============================================================================
#         leftCount = 0
#         rightCount = 0
#         for p in points:
#             if p[0] < self.image.shape[0]:
#                 leftCount += 1
#             else:
#                 rightCount += 1
# =============================================================================
        cv2.putText(edges, "Cross: {:.3f}".format(score), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        #self.crossDetectionbalance = score
        
        return edges , score
    
    def numberEdgesConvexCnt (self, minArea = True, 
                             numberOfCnts = 8, areascalefactor = 1000 ):
        
        HullimgCopy = self.image.copy()
        gray = cv2.cvtColor(HullimgCopy,cv2.COLOR_BGR2GRAY)
        meanThresh = np.mean(gray)
        ret,thresh = cv2.threshold(gray, meanThresh, 255, cv2.THRESH_BINARY)
        ing2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)
        # cut the border for more precision
        thresh[0:2, :] = 0
        thresh[-2:self.image.shape[0], :] = 0
        thresh[:, 0:2] = 0
        thresh[:, -2:self.image.shape[1]] = 0
        
      
        # sort contours by area
        if minArea == False:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            minArea = self.gray.size / areascalefactor
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > minArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the bigger contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        # mask creation
        blankHull = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        
        listofHullsPoints = []
        for cnt in contoursSelection:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(HullimgCopy, [hull], -1, (0,255,0),1)
            cv2.drawContours(blankHull, [hull], -1, (255,255,255),-1)
            for coord in hull:
                listofHullsPoints.append(coord[0])
        # sort the points on the hull by the x coordinates
        listofHullsPointsSorted = sorted(listofHullsPoints, key=lambda x: x[0])
        #extRightListSorted = sorted(extRightList, key=lambda x: x[0], reverse = True)
        
        firstPointLeft = (listofHullsPointsSorted[0][0], listofHullsPointsSorted[0][1])
        secondtPointLeft = (listofHullsPointsSorted[1][0], listofHullsPointsSorted[1][1])
        thirdPointLeft = (listofHullsPointsSorted[2][0], listofHullsPointsSorted[2][1])
        firstPointRight = (listofHullsPointsSorted[-1][0], listofHullsPointsSorted[-1][1])
        secondtPointRight = (listofHullsPointsSorted[-2][0], listofHullsPointsSorted[-2][1])
        thirdPointRight = (listofHullsPointsSorted[-3][0], listofHullsPointsSorted[-3][1])
        # draw the point on the image for visualisaton purpose
        cv2.circle(HullimgCopy, firstPointLeft, 5, (0, 0, 255), -1)
        cv2.circle(HullimgCopy, secondtPointLeft, 5, (0, 255, 0), -1)
        cv2.circle(HullimgCopy, thirdPointLeft, 5, (0, 255, 0), -1)
        cv2.circle(HullimgCopy, firstPointRight, 5, (0, 0, 255), -1)
        cv2.circle(HullimgCopy, secondtPointRight, 5, (0, 255, 0), -1)
        cv2.circle(HullimgCopy, thirdPointRight, 5, (0, 255, 0), -1)
        # we only need the y coordinate since the column will tell the one is come first second and third (we focus on the slope here)
        # and normalised to the height to make it indipendent from the size of the image
        firstPointLeftY = firstPointLeft[1] / self.image.shape[0]
        secondtPointLeftY = secondtPointLeft[1] / self.image.shape[0]
        thirdPointLeftY = thirdPointLeft[1] / self.image.shape[0]
        firstPointRightY = firstPointRight[1] / self.image.shape[0]
        secondtPointRightY = secondtPointRight[1] / self.image.shape[0]
        thirdPointRightY = thirdPointRight[1] / self.image.shape[0]
        
        #left mask and right mask
        x = self.gray.shape[1]
        blankHullLeft = blankHull.copy()
        blankHullRight = blankHull.copy()
        blankHullLeft[ : , 0: int(x/2)] = 0
        blankHullRight[ : , int(x/2): x ] = 0
        
        totalWhiteinHull = (blankHull > 0).sum()
        totalWhiteinLeft = (blankHullLeft > 0).sum()
        totalWhiteInRight = (blankHullRight > 0).sum()
        
        #calculate the score for negative space balance
        alpha = 3.14
        scoreHullBalance = np.exp(-(abs(totalWhiteinLeft - totalWhiteInRight ) / totalWhiteinHull) * 1.618 * alpha )

        
        self.scoreHullBalance = scoreHullBalance

        cv2.putText(HullimgCopy, "NegBal: {:.3f}".format(scoreHullBalance), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return HullimgCopy, scoreHullBalance , firstPointLeftY,secondtPointLeftY,thirdPointLeftY,firstPointRightY,secondtPointRightY,thirdPointRightY

    def rectangularComposition (self, segmentation = 'ORB'):

        # calculate the area of the template triangle based on the golden ratio
        h, w, s = self.image.shape
        upLeftX = int((w - (w*0.618)) / 2)
        upLeftY = int((h - (h*0.618)) / 2)
        baseRightX = upLeftX + int(w* 0.618)
        baseRightY = upLeftY + int(h * 0.618)
        # draw the rect mask
        blankForMasking = np.zeros(self.image.shape, dtype = "uint8")
        rectangularMask = cv2.rectangle(blankForMasking,(upLeftX,upLeftY),(baseRightX,baseRightY),   (255, 255, 255), 2)
        
        # segmentation using 
        if segmentation == 'ORB' :
            ImgImpRegion, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, ImgImpRegion = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegion = cv2.cvtColor(ImgImpRegion, cv2.COLOR_GRAY2BGR) 
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, ImgImpRegion = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'both':
            ImgImpRegionA, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
            contours, ImgImpRegionB = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegionB = cv2.cvtColor(ImgImpRegionB, cv2.COLOR_GRAY2BGR)
            # create the both mask
            ImgImpRegion = cv2.bitwise_or(ImgImpRegionA,ImgImpRegionB)
        
        # dilate the mask to capture more relevant pixels
        kernel = np.ones((5,5),np.uint8)
        rectangularMask = cv2.dilate(rectangularMask,kernel,iterations = 3)
        
        # count the total number of segmentation pixel bigger than 0
        maskedImage = cv2.bitwise_and(ImgImpRegion, rectangularMask)
        sumOfrelevantPixels = (maskedImage > 0).sum()
        totalRelevantPixels = (ImgImpRegion > 0).sum()

        # ratio of the number counted in and out of the triangle
        rectCompScore =  sumOfrelevantPixels/totalRelevantPixels
         
        # draw the image for display
        cv2.putText(ImgImpRegion, "RectComp: {:.3f}".format(rectCompScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        rectangularMask = cv2.rectangle(ImgImpRegion,(upLeftX,upLeftY),(baseRightX,baseRightY),   (255, 255, 255), 2)
        
        self.rectCompScore = rectCompScore
        
        return ImgImpRegion, rectCompScore
        

    def circleComposition (self, segmentation = 'ORB'):

        # calculate the area of the template triangle based on the golden ratio
        h, w, s = self.image.shape
        
        # draw the ellipse mask
        blankForMasking = np.zeros(self.image.shape, dtype = "uint8")
        ellipseMask = cv2.ellipse(blankForMasking, (int(w/2),int(h/2)), (int(w*0.618/2), int(h*0.618/2)),0,0,360,(255, 255, 255), 2)
        
        # segmentation using 
        if segmentation == 'ORB' :
            ImgImpRegion, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, ImgImpRegion = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegion = cv2.cvtColor(ImgImpRegion, cv2.COLOR_GRAY2BGR) 
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, ImgImpRegion = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'both':
            ImgImpRegionA, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
            contours, ImgImpRegionB = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegionB = cv2.cvtColor(ImgImpRegionB, cv2.COLOR_GRAY2BGR)
            # create the both mask
            ImgImpRegion = cv2.bitwise_or(ImgImpRegionA,ImgImpRegionB)
        
        # dilate the mask to capture more relevant pixels
        kernel = np.ones((5,5),np.uint8)
        ellipseMask = cv2.dilate(ellipseMask,kernel,iterations = 2)
        
        # count the total number of segmentation pixel bigger than 0
        maskedImage = cv2.bitwise_and(ImgImpRegion, ellipseMask)
        sumOfrelevantPixels = (maskedImage > 0).sum()
        totalRelevantPixels = (ImgImpRegion > 0).sum()

        # ratio of the number counted in and out of the triangle
        circleCompScore =  sumOfrelevantPixels/totalRelevantPixels
         
        # draw the image for display

        cv2.putText(ImgImpRegion, "circleComp: {:.3f}".format(circleCompScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.ellipse(ImgImpRegion, (int(w/2),int(h/2)), (int(w*0.618/2), int(h*0.618/2)),0,0,360,(255, 255, 255), 2)
        
        self.circleCompScore = circleCompScore
        
        return ImgImpRegion, circleCompScore

    def fourTriangleCompositionAdapted(self, segmentation = 'inner', minArea = True, 
                             numberOfCnts = 100, areascalefactor = 2000, distanceMethod = 'segment'):
        
        FourTriangleImg = self.image.copy()
        
        # draw the lines of the diagonal
        topLeft = (0,0)
        lowerRight = (self.image.shape[1], self.image.shape[0])
        lowerLeft = (0, self.image.shape[0])
        topright = (self.image.shape[1],0)
        
        #blankFourTriangle = np.array(blankFourTriangle)
        cv2.line(FourTriangleImg, topright , lowerLeft, (255,255,255), 1) # topright - lowerleft
        cv2.line(FourTriangleImg, topLeft , lowerRight, (255,0,255), 1) # topleft - lowerright

        # draw the two perpendicular lines
        leftIntersectionX, leftIntersectionY = self._find_perpendicular_through_point_to_line(lowerLeft[0], lowerLeft[1],topright[0],topright[1], topLeft[0], topLeft[1] )
        cv2.line(FourTriangleImg, topLeft , (int(leftIntersectionX), int(leftIntersectionY) ), (255,255,255), 1)
        rightIntersectionX, righttIntersectionY = self._find_perpendicular_through_point_to_line(lowerLeft[0], lowerLeft[1],topright[0],topright[1], lowerRight[0], lowerRight[1] )
        cv2.line(FourTriangleImg, lowerRight , (int(rightIntersectionX), int(righttIntersectionY) ), (255,255,255), 1)
        # second
        leftIntersectionXB, leftIntersectionYB = self._find_perpendicular_through_point_to_line(topLeft[0], topLeft[1],lowerRight[0],lowerRight[1], lowerLeft[0], lowerLeft[1] )
        cv2.line(FourTriangleImg, lowerLeft , (int(leftIntersectionXB), int(leftIntersectionYB) ), (255,0,255), 1)
        rightIntersectionXB, righttIntersectionYB = self._find_perpendicular_through_point_to_line(topLeft[0], topLeft[1],lowerRight[0],lowerRight[1], topright[0], topright[1] )
        cv2.line(FourTriangleImg, topright , (int(rightIntersectionXB), int(righttIntersectionYB) ), (255,0,255), 1)
        
        # calculate the segmentation
        if segmentation == 'ORB' :
            blank, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = False, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, SaliencyMask = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, threshImg = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'inner':
            segmentationOnInnerCnts, contours = self._innerCntsSegmentation(numberOfCnts = numberOfCnts, method = cv2.RETR_CCOMP, minArea = 2)

        # sort contours        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            minArea = self.gray.size / areascalefactor
            for cnt in sorted_contours[0:numberOfCnts]:
                area = cv2.contourArea(cnt)
                if area > minArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the bigger contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        
        # find the center of each contours and draw cnts, not using approx contours
        imageDisplay, listOfCenterPoints = self._findCentreOfMass(image = FourTriangleImg, contours = contoursSelection, approxCnt = False)
        
        # calculate the distance from the center points and the rule of third point(as in the paper)
        # min distance of each center to the 4 points
        distances_option_1 = []
        distances_option_2 = []
                
        if distanceMethod == 'segment':
            for point in listOfCenterPoints:
                centerPoint = np.asarray(point)
                topLeftA = np.asarray((topLeft[0], topLeft[1]))
                lowerRightA = np.asarray((lowerRight[0], lowerRight[1]))
                lowerLeftA = np.asarray((lowerLeft[0], lowerLeft[1]))
                toprightA = np.asarray((topright[0], topright[1]))
                
                leftIntersectionPointA = np.asarray((leftIntersectionX, leftIntersectionY))
                rightIntersectionPointA = np.asarray((rightIntersectionX,righttIntersectionY))
                leftIntersectionPointB = np.asarray((leftIntersectionXB, leftIntersectionYB))
                rightIntersectionPointB = np.asarray((rightIntersectionXB,righttIntersectionYB))

                
                dist_01 = self._point_to_line_dist(centerPoint, [topLeftA,lowerRightA])
                dist_02 = self._point_to_line_dist(centerPoint, [lowerLeftA,leftIntersectionPointA])
                dist_03 = self._point_to_line_dist(centerPoint, [toprightA,rightIntersectionPointA])
                
                dist_04 = self._point_to_line_dist(centerPoint, [lowerLeftA,toprightA])
                dist_05 = self._point_to_line_dist(centerPoint, [topLeftA,leftIntersectionPointB])
                dist_06 = self._point_to_line_dist(centerPoint, [lowerRightA,rightIntersectionPointB])
                
                
                minDistance_option_1 = min(dist_01, dist_02, dist_03)
                minDistance_option_2 = min(dist_04, dist_05, dist_06)
                
                distances_option_1.append(minDistance_option_1)
                distances_option_2.append(minDistance_option_2)
        
        
        # initialise the result and set a paramenter that is linked to the size of the image
        res_option_1 = 0
        res_option_2 = 0
        parameter = self.gray.size / ((self.gray.shape[0]+self.gray.shape[1]) * 1.618)

        for distance in distances_option_1:
            res_option_1 += distance * (np.exp((-distance/parameter)))
        for distance in distances_option_2:
            res_option_2 += distance * (np.exp((-distance/parameter)))
                    
        if len(distances_option_1) == 0:
            cv2.putText(imageDisplay, "Nan: 0 ", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            self.ScoreFourTriangleAdapted = 0
            
            return imageDisplay, 0
        
        elif len(distances_option_2) == 0:
            cv2.putText(imageDisplay, "Nan: 0 ", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            self.ScoreFourTriangleAdapted = 0
            
            return imageDisplay, 0
        
        else:
            
            score_option_1 = res_option_1 / sum(distances_option_1)
            score_option_2 = res_option_2 / sum(distances_option_2)
            
            ScoreFourTriangleAdapted = max(score_option_1,score_option_2)
            
            
            if distanceMethod == 'segment':
                cv2.putText(imageDisplay, "TriangSeg: {:.3f}".format(ScoreFourTriangleAdapted), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if segmentation == 'saliency':
                cv2.putText(imageDisplay, "RTS: {:.3f}".format(ScoreFourTriangleAdapted), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if segmentation == 'ORB':
                cv2.putText(imageDisplay, "RTorb: {:.3f}".format(ScoreFourTriangleAdapted), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
            self.ScoreFourTriangleAdapted = ScoreFourTriangleAdapted
            
            return imageDisplay, ScoreFourTriangleAdapted
    
    def bigTriangleCompositionAdapted(self, segmentation = 'inner', minArea = True, 
                             numberOfCnts = 100, areascalefactor = 2000, distanceMethod = 'segment'):
        # calculate the area of the template triangle based on the golden ratio
        h, w, s = self.image.shape
        baseLeftX = int((w - (w*0.618)) / 2)
        baseLeftY = int(h - ((h*0.618) / 2))
        baseRightX = baseLeftX + int(w* 0.618)
        baseRightY = baseLeftY
        vertexX = int( w / 2)
        vertexY =  h - baseLeftY 
        centerImgX = int(self.image.shape[1]/2)
        centerImgY = int(self.image.shape[0]/2)
        
        listOfPotins = [[baseLeftX, baseLeftY], [vertexX, vertexY], [baseRightX, baseRightY]]
        ctr = np.array(listOfPotins).reshape((-1,1,2)).astype(np.int32)
        
        bigTriangeImg = self.image.copy()
        
        # fill the triangle for the mask
        blankForMasking = np.zeros(self.image.shape, dtype = "uint8")
        cv2.drawContours(blankForMasking, [ctr], -1, (255, 255, 255), 2)        
        
        if segmentation == 'ORB' :
            blank, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = False, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, SaliencyMask = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, threshImg = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'inner':
            segmentationOnInnerCnts, contours = self._innerCntsSegmentation(numberOfCnts = numberOfCnts, method = cv2.RETR_CCOMP, minArea = 2)

        # sort contours        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            minArea = self.gray.size / areascalefactor
            for cnt in sorted_contours[0:numberOfCnts]:
                area = cv2.contourArea(cnt)
                if area > minArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the bigger contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        
        # find the center of each contours and draw cnts, not using approx contours
        imageDisplay, listOfCenterPoints = self._findCentreOfMass(image = bigTriangeImg, contours = contoursSelection, approxCnt = False)

        # calculate the distance from the center points and the rule of third point(as in the paper)
        # min distance of each center to the 4 points
        distancePoints = []
        
        if distanceMethod == 'point':
            for point in listOfCenterPoints:
                cX = point[0]
                cY = point[1]
                ManhattanDistanceNormalised_01 = abs(baseLeftX - cX) / self.image.shape[1] + abs(baseLeftY - cY) / self.image.shape[0]
                ManhattanDistanceNormalised_02 = abs(baseRightX - cX) / self.image.shape[1] + abs(baseRightY - cY) / self.image.shape[0]
                ManhattanDistanceNormalised_03 = abs(vertexX - cX) / self.image.shape[1] + abs(vertexY- cY) / self.image.shape[0]
                ManhattanDistanceNormalised_04 = abs(centerImgX - cX) / self.image.shape[1] + abs(centerImgY - cY) / self.image.shape[0]
                
                minDistance = min(ManhattanDistanceNormalised_01,ManhattanDistanceNormalised_02,ManhattanDistanceNormalised_03,ManhattanDistanceNormalised_04)
    
                distancePoints.append(minDistance)
                
        if distanceMethod == 'segment':
            for point in listOfCenterPoints:
                centerPoint = np.asarray(point)
                baseLeftPoint = np.asarray((baseLeftX, baseLeftY))
                baseRigthPoint = np.asarray((baseRightX, baseRightY))
                vertexPoint = np.asarray((vertexX, vertexY))
                centerOfImgPoint = np.asarray((centerImgX,centerImgY))

                
                dist_01 = self._point_to_line_dist(centerPoint, [baseLeftPoint,baseRigthPoint])
                dist_02 = self._point_to_line_dist(centerPoint, [baseLeftPoint,vertexPoint])
                dist_03 = self._point_to_line_dist(centerPoint, [baseRigthPoint,vertexPoint])
                dist_04 = self._point_to_line_dist(centerPoint, [centerOfImgPoint,vertexPoint])
                
                minDistance = min(dist_01, dist_02, dist_03, dist_04)
                
                distancePoints.append(minDistance)
        
        # initialise the result and set a paramenter that is linked to the size of the image
        res = 0
        parameter = self.gray.size / ((self.gray.shape[0]+self.gray.shape[1]) * 1.618)
        
        if len(distancePoints) == 0:
            ScoreBigTriangle = 0
            self.ScoreBigTriangle = ScoreBigTriangle
            return imageDisplay, ScoreBigTriangle
        else:
            for distance in distancePoints:
                res += distance * (np.exp((-distance/parameter)))
            
            ScoreBigTriangle = res / sum(distancePoints)
    
            
            # draw the guides rules and saliency on panel
            cv2.line(imageDisplay,(baseLeftX,baseLeftY), (baseRightX,baseRightY), (255,0,255), 1)
            cv2.line(imageDisplay,(baseLeftX,baseLeftY), (vertexX, vertexY), (255,0,255), 1)
            cv2.line(imageDisplay,(baseRightX,baseRightY), (vertexX, vertexY), (255,0,255), 1)
            cv2.line(imageDisplay,(centerImgX,centerImgY), (vertexX, vertexY), (255,0,255), 1)
            
            
            if distanceMethod == 'segment':
                cv2.putText(imageDisplay, "TriangSeg: {:.3f}".format(ScoreBigTriangle), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if segmentation == 'saliency':
                cv2.putText(imageDisplay, "RTS: {:.3f}".format(ScoreBigTriangle), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if segmentation == 'ORB':
                cv2.putText(imageDisplay, "RTorb: {:.3f}".format(ScoreBigTriangle), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
            self.ScoreBigTriangle = ScoreBigTriangle
            
            return imageDisplay, ScoreBigTriangle
        
        
    def bigTriangleComposition (self, segmentation = 'ORB'):

        # calculate the area of the template triangle based on the golden ratio
        h, w, s = self.image.shape
        baseLeftX = int((w - (w*0.618)) / 2)
        baseLeftY = int(h - ((h*0.618) / 2))
        baseRightX = baseLeftX + int(w* 0.618)
        baseRightY = baseLeftY
        vertexX = int( w / 2)
        vertexY =  h - baseLeftY 
        
        listOfPotins = [[baseLeftX, baseLeftY], [vertexX, vertexY], [baseRightX, baseRightY]]
        ctr = np.array(listOfPotins).reshape((-1,1,2)).astype(np.int32)
        
        # fill the triangle for the mask
        blankForMasking = np.zeros(self.image.shape, dtype = "uint8")
        cv2.drawContours(blankForMasking, [ctr], -1, (255, 255, 255), 2)
        # dilate the mask to capture more relevant pixels
        kernel = np.ones((5,5),np.uint8)
        blankForMasking = cv2.dilate(blankForMasking,kernel,iterations = 2)
        # flip to make the triangle with the base to the lower base
        #blankForMasking = cv2.flip(blankForMasking, -1)
        
        # segmentation using 
        if segmentation == 'ORB' :
            ImgImpRegion, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, ImgImpRegion = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegion = cv2.cvtColor(ImgImpRegion, cv2.COLOR_GRAY2BGR) 
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, ImgImpRegion = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'both':
            ImgImpRegionA, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
            contours, ImgImpRegionB = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegionB = cv2.cvtColor(ImgImpRegionB, cv2.COLOR_GRAY2BGR)
            # create the both mask
            ImgImpRegion = cv2.bitwise_or(ImgImpRegionA,ImgImpRegionB)
        
        # count the total number of segmentation pixel bigger than 0
        maskedImage = cv2.bitwise_and(ImgImpRegion, blankForMasking)
        sumOfrelevantPixels = (maskedImage > 0).sum()
        totalRelevantPixels = (ImgImpRegion > 0).sum()
        
        # ratio of the number counted in and out of the triangle
        bigTriangleCompScore =  sumOfrelevantPixels/totalRelevantPixels
         
        # draw the image for display
        cv2.drawContours(ImgImpRegion, [ctr], -1, (255, 0, 0), 2)
        cv2.putText(ImgImpRegion, "TriComp: {:.3f}".format(bigTriangleCompScore), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.bigTriangleCompScore = bigTriangleCompScore
        
        return ImgImpRegion, bigTriangleCompScore
        
    
    def fourTriangleDistance (self, segmentation = 'ORB', edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL):
        
        # draw the guidelines of the images
        guideLinesA = self._fourTriangleGuidelines(flip = False)
        guideLinesB = self._fourTriangleGuidelines(flip = True)
        
        # dilate theguidelines 
        kernel = np.ones((5,5),np.uint8)
        maskA = cv2.dilate(guideLinesA,kernel,iterations = 3)
        maskB = cv2.dilate(guideLinesB,kernel,iterations = 3)
        # segmentation using ORB or Saliency or Thresh or Edges expanded
        if segmentation == 'ORB' :
            ImgImpRegion, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = edged, edgesdilateOpen = edgesdilateOpen, method = method)
        if segmentation == 'saliency':
            contours, ImgImpRegion = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegion = cv2.cvtColor(ImgImpRegion, cv2.COLOR_GRAY2BGR) 
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, ImgImpRegion = self._thresholdSegmentation(imageToTresh = None, method = cv2.RETR_LIST )
        if segmentation == 'both':
            ImgImpRegionA, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = edged, edgesdilateOpen = edgesdilateOpen, method = method)
            contours, ImgImpRegionB = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
            ImgImpRegionB = cv2.cvtColor(ImgImpRegionB, cv2.COLOR_GRAY2BGR)
            # create the both mask
            ImgImpRegion = cv2.bitwise_or(ImgImpRegionA,ImgImpRegionB)
        
        intersectionA = cv2.bitwise_and(maskA, ImgImpRegion)
        intersectionB = cv2.bitwise_and(maskB, ImgImpRegion)
        
        # count the number of white pixels remained
        validPointInA = ((intersectionA > 0).sum())
        validPointInB = ((intersectionB > 0).sum())
        
# =============================================================================
#         cv2.imshow('imp', ImgImpRegion)
#         cv2.imshow ('maskA',intersectionA)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        
        validPoints = max(validPointInA, validPointInB)
        totalPossiblePoints =  ((ImgImpRegion > 0).sum())
        
        scoreFourTriangleDistance = validPoints / totalPossiblePoints

        displayImg = cv2.bitwise_or(guideLinesA, ImgImpRegion)
        cv2.putText(displayImg, "4TC: {:.3f}".format(scoreFourTriangleDistance), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.scoreFourTriangleDistance = scoreFourTriangleDistance
        
        return displayImg, scoreFourTriangleDistance
        
    
    
    def triangleAreaGoldenRatio (self, segmentation = 'ORB', minArea = False, 
                             numberOfCnts = 10, areascalefactor = 3000):
        
        triangleDisplay = self.image.copy()
        # from paper 41 
        # calculate the area of the template triangle
        h, w, s = self.image.shape
        areaGoldenFittedTriangle = ((w * 0.618) * (h * 0.618)) / 2
        # segmentation of the image 
        #TODO put other possible segmentation modes
        #TODO also add the bitwise operation of saliency + orb
        if segmentation == 'ORB' :
            blank, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = True, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, SaliencyMask = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, threshImg = self._thresholdSegmentation(method = cv2.RETR_LIST )
        # narrow the contorus to the most relevant    
        # sort contours        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            MinArea = self.gray.size / areascalefactor
            for cnt in sorted_contours[0:numberOfCnts]:
                area = cv2.contourArea(cnt)
                if area > MinArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the biggest contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        
        # find the center points of the contours
        triangleDisplay, listOfCenterPoints = self._findCentreOfMass(image = triangleDisplay, contours = contoursSelection, approxCnt = False)
        # calculate the area of the detected traingle using iteration over 6 points
        # Return r length subsequences of elements from the input iterable.
        listOfCenterPoints = [list(elem) for elem in listOfCenterPoints]
        candidates = list(itertools.combinations(listOfCenterPoints, 3))
        candidates = [list(elem) for elem in candidates]
        
        areaCntDict = {}

        for cnt in candidates:
            cnt = np.array(cnt, dtype = np.int32)

            cv2.drawContours(triangleDisplay, [cnt], -1, (0, 255, 0), 1)
            areaCnt = cv2.contourArea(cnt)
            areaCntDict[areaCnt] = cnt
        
        
        # use the dictionary to find the bigger triangle
        sortedKeys = sorted(areaCntDict, reverse = True)
        
        if sortedKeys == []:
            
            scoreGoldenTriangle = 0
            
            return triangleDisplay, blank, scoreGoldenTriangle
        
        # draw the bigger triangle in red in most cases is the same of the golden triangle
        #biggerTriangle =  (areaCntDict[sortedKeys[0]])
        #cv2.drawContours(triangleDisplay, [biggerTriangle], -1, (0, 0, 255), 2)
            
        # select the element value closest to the areaGolden
        
        closestAreaTriangleToGoldeRation = min(enumerate(sortedKeys), key=lambda x: abs(x[1]-areaGoldenFittedTriangle))
        #print (closestAreaTriangleToGoldeRation, areaGoldenFittedTriangle )

        ClosestTriangle =  (areaCntDict[closestAreaTriangleToGoldeRation[1]])
        cv2.drawContours(triangleDisplay, [ClosestTriangle], -1, (255, 0, 0), 2)
        # calculate the score as the minimum distance of the area
        
        scoreGoldenTriangle = (closestAreaTriangleToGoldeRation[1] / areaGoldenFittedTriangle)
        
        #TODO return also the mask in case is ORB or create separate PANEL in UI for the mask calling a dedicated funtion
        
        cv2.putText(triangleDisplay, "Torb: {:.3f}".format(scoreGoldenTriangle), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.scoreGoldenTriangle = scoreGoldenTriangle
        
        return triangleDisplay, blank, scoreGoldenTriangle
    
    def diagonalDistanceBalance(self, segmentation = 'inner', minArea = True, 
                             numberOfCnts = 100, areascalefactor = 3000, distanceMethod = 'lines'):
        # https://stackoverflow.com/questions/45766534/finding-cross-product-to-find-points-above-below-a-line-in-matplotlib
        
        diagonalImg = self.image.copy()
        
        # draw the lines of the diagonal
        topLeft = (0,0)
        lowerRight = (self.image.shape[1], self.image.shape[0])
        lowerLeft = (0, self.image.shape[0])
        topright = (self.image.shape[1],0)
        
        #blankFourTriangle = np.array(blankFourTriangle)
        cv2.line(diagonalImg, topright , lowerLeft, (255,255,255), 1) # topright - lowerleft
        cv2.line(diagonalImg, topLeft , lowerRight, (255,0,255), 1) # topleft - lowerright

        # calculate the segmentation
        if segmentation == 'ORB' :
            blank, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = False, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, SaliencyMask = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, threshImg = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'inner':
            segmentationOnInnerCnts, contours = self._innerCntsSegmentation(numberOfCnts = numberOfCnts, method = cv2.RETR_CCOMP, minArea = 2)

        # sort contours        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            minArea = self.gray.size / areascalefactor
            for cnt in sorted_contours[0:numberOfCnts]:
                area = cv2.contourArea(cnt)
                if area > minArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the bigger contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        
        # find the center of each contours and draw cnts, not using approx contours
        imageDisplay, listOfCenterPoints = self._findCentreOfMass(image = diagonalImg, contours = contoursSelection, approxCnt = False)
        
        # calculate how many points are below and above each diagonals and take the best result
        aboveA = 0
        belowA = 0
        aboveB = 0
        belowB = 0
        
        if distanceMethod == 'segment':
            for point in listOfCenterPoints:
                centerPoint = np.asarray(point)
                topLeftA = np.asarray((topLeft[0], topLeft[1]))
                lowerRightA = np.asarray((lowerRight[0], lowerRight[1]))
                lowerLeftA = np.asarray((lowerLeft[0], lowerLeft[1]))
                toprightA = np.asarray((topright[0], topright[1]))
                
                isabove = lambda p, a, b : np.cross (p-a, b-a) > 0
                
                if isabove(centerPoint,topLeftA,lowerRightA):
                    aboveA += 1
                else:
                    belowA += 1
                if isabove(centerPoint,lowerLeftA,toprightA):
                    aboveB += 1
                else:
                    belowB += 1
       
        if len(contoursSelection) == 0:
            diagonalasymmetryBalance = 0
            self.diagonalasymmetryBalance = diagonalasymmetryBalance
            return imageDisplay, diagonalasymmetryBalance
            
        else:
            
            ratio_A = max (aboveA, belowA)
            resultA = ratio_A / len(contoursSelection)
            ratio_B = max (aboveB, belowB)
            resultB = ratio_B / len(contoursSelection)
            
# =============================================================================
#             
#             resultA = abs(aboveA-belowA) / len(contoursSelection)
#             resultB = abs(aboveB - belowB) / len(contoursSelection)
# =============================================================================
            
            diagonalasymmetryBalance = max(resultA,resultB)
            
            
            # draw label
            cv2.putText(imageDisplay, "Dcomp: {:.3f}".format(diagonalasymmetryBalance), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            self.diagonalasymmetryBalance = diagonalasymmetryBalance
            
            return imageDisplay, diagonalasymmetryBalance
            
    
    def diagonalsDistance (self, saliencySegmentation = True, minArea = True, 
                             numberOfCnts = 4, areascalefactor = 3500):
        
        # using adapted version of the paper 'Autonomous Viepoint Selection of Robots..'
        diagonal = self.image.copy()
        
        
        # take the 4 points of the two diagonal lines 
        topLeft = (0,0)
        lowerRight = (self.image.shape[1], self.image.shape[0])
        topright = (0, self.image.shape[0])
        lowerLeft = (self.image.shape[1],0)
        
        # extract the center points using saliency segmentation or threshold 
        # create the segmentation of the image using Saliency Segmentation
        if saliencySegmentation == True:
            contours, SaliencyMask = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
        # create the segmentations of the images using threshold
        if saliencySegmentation == False:
            contours, threshImg = self._thresholdSegmentation(method = cv2.RETR_LIST )
            
        # narrow the contorus to the most relevant    
        # sort contours        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            MinArea = self.gray.size / areascalefactor
            for cnt in sorted_contours[0:numberOfCnts]:
                area = cv2.contourArea(cnt)
                if area > MinArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the biggest contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        
        # find the center of cnts
        diagonal, listOfCenterPoints = self._findCentreOfMass(image = diagonal, contours = contoursSelection, approxCnt = False)
        
        # create the function to calculate the distance
        # descending diagonal
        distanceFromDescendingDiagonalList = []
        for point in listOfCenterPoints:
            x,y = point
            p1 =  topLeft
            p2 =  lowerRight
            distanceFromDescendingDiagonal = self._distance_to_line( p1, p2, x, y)
            distanceFromDescendingDiagonalList.append(distanceFromDescendingDiagonal)
        meandistanceFromDescendingDiagonalList = np.mean(distanceFromDescendingDiagonalList)
        # ascending diagonal
        distanceFromAscendingDiagonalList = []
        for point in listOfCenterPoints:
            x,y = point
            p1 = lowerLeft
            p2 = topright
            distanceFromAscendingDiagonal = self._distance_to_line( p1, p2, x, y)
            distanceFromAscendingDiagonalList.append(distanceFromAscendingDiagonal)
        meanddistanceFromAscendingDiagonalList = np.mean(distanceFromAscendingDiagonalList)
        
        # extract the min distances 
        minDistancefromDiagonals = min(meandistanceFromDescendingDiagonalList,meanddistanceFromAscendingDiagonalList)

        # use exponential formula to score the results
        pointA = (int(self.image.shape[0]/2), 0)
        maxDistanceA = self._distance_to_line( topLeft, lowerRight, pointA[0], pointA[1])
        pointB = (0, int(self.image.shape[1]/2))
        maxDistanceB = self._distance_to_line( topLeft, lowerRight, pointB[0], pointB[1])
        maxdistance = max(maxDistanceA,maxDistanceB)
        ScoreMinDistancefromDiagonals = 1 - (minDistancefromDiagonals / maxdistance)
        
        # draw the 2 diagonals
        cv2.line(diagonal, topLeft,lowerRight , (255,255,255), 1) # topleft - lowerright
        cv2.line(diagonal, topright , lowerLeft, (255,255,255), 1) # topright - lowerleft
        # draw label
        cv2.putText(diagonal, "Ddist: {:.3f}".format(ScoreMinDistancefromDiagonals), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.ScoreMinDistancefromDiagonals = ScoreMinDistancefromDiagonals
        
        return diagonal, ScoreMinDistancefromDiagonals
    
    def ruleOfThirdOnThreshPixelsCount(self):
        
        meanThresh = np.mean(self.gray)
   
        copyImg = self.gray.copy()
        # decided not to use Saliency Map and only thresh
        ret,masked = cv2.threshold(copyImg, meanThresh, 255, cv2.THRESH_BINARY)
      
        totalWhiteInthresh = (masked > 0).sum()
        
        rows = masked.shape[0]
        cols = masked.shape[1]
        masked[0: rows//12*2 , :] = 0
        masked[rows//12*10: rows , :] = 0
        masked[: , 0 : cols//12*2] = 0
        masked[: , cols//12*10 : cols ] = 0
        masked[rows//12*4 : rows//12*8, cols//12*4 : cols//12*8] = 0
        
        totalWhiteInthreshRuleofThird = (masked > 0).sum()
        
        
        ratioRuleOfThird = totalWhiteInthreshRuleofThird / totalWhiteInthresh
        self.ratioRuleOfThird = ratioRuleOfThird
        
        return ratioRuleOfThird


    def ruleOfThirdDistance (self, segmentation = 'ORB', minArea = True, 
                             numberOfCnts = 100, areascalefactor = 2000, distanceMethod = 'lines'):
        # using paper 'Autonomous Viewpoint Selection of Robots...'
        
        ruleOfThirdImg = self.image.copy()
        
        # calculate the coordinates of the 4 points
        w = self.image.shape[1]
        h = self.image.shape[0]
        
        LowLeftThirdPoint = (int(w/3), int(h*(2/3)))
        UppperLeftThirdPoint = (int(w/3), int(h/3))
        LowRightThirdPoint = (int(w*(2/3)), int(h*(2/3)))
        UpppeRightThirdPoint = (int(w*(2/3)), int(h/3))
        midUpperThirdPoint = (int(w/2), int(h/6))
        midLowerThirdPoint = (int(w/2), int(h*(5/6)))
        
        
        if segmentation == 'ORB' :
            blank, contours, keypoints = self._orbSegmentation ( maxKeypoints = 1000, edged = False, edgesdilateOpen = False, method = cv2.RETR_EXTERNAL)
        if segmentation == 'saliency':
            contours, SaliencyMask = self._saliencySegmentation(method = cv2.RETR_EXTERNAL )
        # create the segmentations of the images using threshold
        if segmentation == 'thresh':
            contours, threshImg = self._thresholdSegmentation(method = cv2.RETR_LIST )
        if segmentation == 'inner':
            segmentationOnInnerCnts, contours = self._innerCntsSegmentation(numberOfCnts = numberOfCnts, method = cv2.RETR_CCOMP, minArea = 2)

        # sort contours        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        
        # sorted and selected list of areas in contours if minArea is True
        if minArea:
            selected_contours = []
            minArea = self.gray.size / areascalefactor
            for cnt in sorted_contours[0:numberOfCnts]:
                area = cv2.contourArea(cnt)
                if area > minArea:
                    selected_contours.append(cnt)
            sorted_contours = sorted(selected_contours, key = cv2.contourArea, reverse = True)
        
        # select only the bigger contours
        contoursSelection = sorted_contours[0:numberOfCnts]
        
        # find the center of each contours and draw cnts, not using approx contours
        imageDisplay, listOfCenterPoints = self._findCentreOfMass(image = ruleOfThirdImg, contours = contoursSelection, approxCnt = False)

        # calculate the distance from the center points and the rule of third point(as in the paper)
        # min distance of each center to the 4 points
        distancePoints = []
        
        if distanceMethod == 'point':
            for point in listOfCenterPoints:
                cX = point[0]
                cY = point[1]
                ManhattanDistanceNormalised_01 = abs(LowLeftThirdPoint[0] - cX) / self.image.shape[1] + abs(LowLeftThirdPoint[1] - cY) / self.image.shape[0]
                ManhattanDistanceNormalised_02 = abs(UppperLeftThirdPoint[0] - cX) / self.image.shape[1] + abs(UppperLeftThirdPoint[1] - cY) / self.image.shape[0]
                ManhattanDistanceNormalised_03 = abs(LowRightThirdPoint[0] - cX) / self.image.shape[1] + abs(LowRightThirdPoint[1] - cY) / self.image.shape[0]
                ManhattanDistanceNormalised_04 = abs(UpppeRightThirdPoint[0] - cX) / self.image.shape[1] + abs(UpppeRightThirdPoint[1] - cY) / self.image.shape[0]
                
                minDistance = min(ManhattanDistanceNormalised_01,ManhattanDistanceNormalised_02,ManhattanDistanceNormalised_03,ManhattanDistanceNormalised_04)
    
                distancePoints.append(minDistance)
        # use exponential formula to score the results
        if distanceMethod == 'lines':
            for point in listOfCenterPoints:
                
                cX = point[0]
                cY = point[1]
                dist_01 = self._distance_to_line( UppperLeftThirdPoint, UpppeRightThirdPoint, cX, cY)
                dist_02 = self._distance_to_line( UppperLeftThirdPoint, LowLeftThirdPoint, cX, cY)
                dist_03 = self._distance_to_line( UpppeRightThirdPoint, LowRightThirdPoint, cX, cY)
                dist_04 = self._distance_to_line( LowLeftThirdPoint, LowRightThirdPoint, cX, cY)
                
                minDistance = min(dist_01, dist_02, dist_03, dist_04)
                
                distancePoints.append(minDistance)
                
        if distanceMethod == 'segment':
            for point in listOfCenterPoints:
                centerPoint = np.asarray(point)
                UppperLeftThirdPointA = np.asarray(UppperLeftThirdPoint)
                UpppeRightThirdPointA = np.asarray(UpppeRightThirdPoint)
                LowLeftThirdPointA = np.asarray(LowLeftThirdPoint)
                LowRightThirdPointA = np.asarray(LowRightThirdPoint)
                midUpperThirdPointA = np.asarray(midUpperThirdPoint)
                midLowerThirdPointA = np.asarray(midLowerThirdPoint)
                
                dist_01 = self._point_to_line_dist(centerPoint, [UppperLeftThirdPointA,UpppeRightThirdPointA])
                dist_02 = self._point_to_line_dist(centerPoint, [UppperLeftThirdPointA,LowLeftThirdPointA])
                dist_03 = self._point_to_line_dist(centerPoint, [UpppeRightThirdPointA,LowRightThirdPointA])
                dist_04 = self._point_to_line_dist(centerPoint, [LowLeftThirdPointA,LowRightThirdPointA])
                
                dist_05 = self._point_to_line_dist(centerPoint, [midUpperThirdPointA,midLowerThirdPointA])
                
                minDistance = min(dist_01, dist_02, dist_03, dist_04, dist_05)
                
                distancePoints.append(minDistance)

        # initialise the result and set a paramenter that is linked to the size of the image
        res = 0
        parameter = self.gray.size / ((self.gray.shape[0]+self.gray.shape[1]) * 1.618)
        
        if len(distancePoints) == 0:
            ScoreRuleOfThird = 0
            self.ScoreRuleOfThird = ScoreRuleOfThird
            return imageDisplay, ScoreRuleOfThird
        else:
        
            for distance in distancePoints:
                res += distance * (np.exp((-distance/parameter)))
            
            ScoreRuleOfThird = res / sum(distancePoints)
    
            
            # draw the guides rules and saliency on panel
            cv2.line(imageDisplay,LowLeftThirdPoint, UppperLeftThirdPoint, (255,0,255), 1)
            cv2.line(imageDisplay,LowRightThirdPoint, UpppeRightThirdPoint, (255,0,255), 1)
            cv2.line(imageDisplay,LowLeftThirdPoint, LowRightThirdPoint, (255,0,255), 1)
            cv2.line(imageDisplay,UppperLeftThirdPoint, UpppeRightThirdPoint, (255,0,255), 1)
            cv2.line(imageDisplay,midUpperThirdPoint, midLowerThirdPoint, (255,0,255), 1)
            
            
            if distanceMethod == 'segment':
                cv2.putText(imageDisplay, "RTSeg: {:.3f}".format(ScoreRuleOfThird), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if segmentation == 'saliency':
                cv2.putText(imageDisplay, "RTS: {:.3f}".format(ScoreRuleOfThird), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if segmentation == 'ORB':
                cv2.putText(imageDisplay, "RTorb: {:.3f}".format(ScoreRuleOfThird), (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            self.ScoreRuleOfThird = ScoreRuleOfThird
            
            return imageDisplay, ScoreRuleOfThird
        
    def VisualBalanceForeground(self, numberOfCnts = 50, kernel = 3,areascalefactor = 3000, segmentation = 'inner'):
    
    # display center of mass and calculate the disstance from the frame center
    # using paper 'Autonomous Viewpoint Selection of Robots...'
    # adapted from book under loomis but not loomis the other one
        if segmentation == 'thresh':
            meanThresh = np.mean(self.gray)
       
            copyThresh = self.gray.copy()
            # decided not to use Saliency Map and only thresh - but then opted for inner contours
            ret,mask = cv2.threshold(copyThresh, meanThresh, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(mask, (3, 3), 0)
            ret,thresh = cv2.threshold(blurred, meanThresh, 255, cv2.THRESH_BINARY)
            kernelM = np.ones((kernel,kernel),np.uint8)
            #thresh = cv2.dilate(thresh,kernelM,iterations = 1)
            mask = cv2.erode(mask,kernelM,iterations = 1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelM)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelM)
            
            ing2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #convert back to BGR for display purpose
            maskForeground = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        if segmentation == 'inner':
            maskForeground, contours = self._innerCntsSegmentation(numberOfCnts = numberOfCnts, method = cv2.RETR_CCOMP, minArea = 1)
# =============================================================================
#             kernelM = np.ones((kernel,kernel),np.uint8)
#             #thresh = cv2.dilate(thresh,kernelM,iterations = 1)
#             maskForeground = cv2.erode(maskForeground,kernelM,iterations = 1)
#             maskForeground = cv2.morphologyEx(maskForeground, cv2.MORPH_OPEN, kernelM)
#             maskForeground = cv2.morphologyEx(maskForeground, cv2.MORPH_OPEN, kernelM)
# =============================================================================
        
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

        BigContours = sorted_contours [0:numberOfCnts]
        
        minArea = self.gray.size / areascalefactor
        
        selectedContours = [cnt for cnt in BigContours if cv2.contourArea(cnt) > minArea]
        
        listOfxs = []
        listOfys = []
        listOfCenters = []
        listofAreas = []
        
        for c in selectedContours:
            
        	# compute the center of the contour
            M = cv2.moments(c)
            cX = int((M["m10"]) / (M["m00"]))
            cY = int((M["m01"]) / (M["m00"]))
            listOfxs.append(cX)
            listOfys.append(cY)
            listOfCenters.append([cX, cY])
            listofAreas.append(cv2.contourArea(c))
         
        	# draw the contour and center of the shape on the image
            cv2.drawContours(maskForeground, [c], -1, (0, 255, 0), 1)
            cv2.circle(maskForeground, (cX, cY), 7, (255, 0, 0), -1)
        
      
          # how far is the centroid to the image centre
        ImageCentreX = int(self.image.shape[1] / 2)
        ImageCentreY = int(self.image.shape[0] / 2)
        
        extremeLeftx = 0
        extremeLefty = 0
        extremeRightx = 0
        extremeRighty = 0
        
        if len(selectedContours)<3:
            # draw the extreme line
            for i in [i for i, x in enumerate(listOfxs) if x == min(listOfxs)]:
                extremeLeftx = listOfxs[i]
                extremeLefty = listOfys[i]
            for i in [i for i, x in enumerate(listOfxs) if x == max(listOfxs)]:
                extremeRightx = listOfxs[i]
                extremeRighty = listOfys[i]
        
            cv2.line(maskForeground, (extremeLeftx,extremeLefty), (extremeRightx,extremeRighty), (0,0,255), 2)
              
        if len (selectedContours)>= 3:

            listofIndexLeft = []
            for i, x in enumerate(listOfxs):
                if x < int(self.image.shape[1]/2):
                   listofIndexLeft.append(i) 
            
            if len(listofIndexLeft)== 1:
                for i in [i for i, x in enumerate(listOfxs) if x == min(listOfxs)]:
                    extremeLeftx = listOfxs[i]
                    extremeLefty = listOfys[i]
                    cv2.circle(maskForeground, (extremeLeftx,extremeLefty), 7, (0, 255, 0), -1)
                    
            if len(listofIndexLeft) >1:
                cntsLeft = []
                for i in listofIndexLeft:
                    cntsLeft.append(listOfCenters[i])
                # calculate the coordinates of the centroid
                x = [p[0] for p in cntsLeft]
                y = [p[1] for p in cntsLeft]
                centroid = (sum(x) / len(cntsLeft), sum(y) / len(cntsLeft))
                # draw centroid left
                extremeLeftx = int(centroid[0])
                extremeLefty = int(centroid[1])
                cv2.circle(maskForeground, (extremeLeftx,extremeLefty ), 7, (0, 255, 0), -1)
                    
            listofIndexRight = []
            for i, x in enumerate(listOfxs):
                if x >= int(self.image.shape[1]/2):
                   listofIndexRight.append(i) 
            
            if len(listofIndexRight)== 1:
                for i in [i for i, x in enumerate(listOfxs) if x == max(listOfxs)]:
                    extremeRightx = listOfxs[i]
                    extremeRighty = listOfys[i]
                    cv2.circle(maskForeground, (extremeRightx,extremeRighty), 7, (0, 255, 0), -1)
                
            if len(listofIndexRight) >1:
                cntsRight = []
                for i in listofIndexRight:
                    cntsRight.append(listOfCenters[i])
                # calculate the coordinates of the centroid
                x = [p[0] for p in cntsRight]
                y = [p[1] for p in cntsRight]
                centroid = (sum(x) / len(cntsRight), sum(y) / len(cntsRight))
                # draw centroid left
                extremeRightx = int(centroid[0])
                extremeRighty = int(centroid[1])
                cv2.circle(maskForeground, (extremeRightx,extremeRighty ), 7, (0, 255, 0), -1)
        # draw connected lines if there are 2 green spot 
        scoreVisualBalance = 0
        if len(selectedContours) == 1:
            # Manhattan Distance
            ManhattanDistanceNormalised = abs(ImageCentreX - cX) / self.image.shape[1] + abs(ImageCentreY - cY) / self.image.shape[0]
            scoreVisualBalance = (1 - ManhattanDistanceNormalised)
            slope = 0 # which is GOOD
            
        if len(selectedContours) == 2:
            
            # get the areas  of cnts left and right
            leftArea = 0
            rightArea = 0
            for x, a in zip(listOfxs, listofAreas):
                if x < int(self.image.shape[1]/2):
                    leftArea += a
                else:
                    rightArea += a
            # return also the angle of the line
            slope = ((extremeLefty - extremeRighty ) / (extremeLeftx -extremeRightx ))
            intercept = extremeLefty - (slope * extremeLeftx)
            # find the point where the leverage meets the middle line of the frame
            ycenterIntercept = (slope * self.image.shape[1]/2) + intercept
            xcenterIntercept = int(self.image.shape[1]/2)
            cv2.circle(maskForeground, (int(self.image.shape[1]/2),int(ycenterIntercept) ), 4, (255, 0, 255), -1)
            
            # handle exceptions
            if leftArea == 0 or rightArea == 0:
            
                scoreVisualBalance = 0
                
            else:
                # weight the distance and find the fulcrum
                if leftArea > rightArea:
                    # how long should be the input level to be in balance
                    # and how long it is
                    LenghtOfTheResistantLevel = dist.euclidean((extremeLeftx,extremeLefty), (xcenterIntercept,ycenterIntercept))
                    lenghtOfActualInput = dist.euclidean((extremeRightx,extremeRighty), (xcenterIntercept,ycenterIntercept))
                    # calculate the lenght of the input to have balance
                    lenghtofInputToHaveBalance = (leftArea/rightArea) * LenghtOfTheResistantLevel                     
                    scoreVisualBalance = np.exp(-abs(lenghtOfActualInput - lenghtofInputToHaveBalance) / self.image.shape[0]*1.618)
                    
                if leftArea <= rightArea:
                    # how long should be the input level to be in balance
                    # and how long it is
                    LenghtOfTheResistantLevel = dist.euclidean((extremeRightx,extremeRighty), (xcenterIntercept,ycenterIntercept))
                    lenghtOfActualInput = dist.euclidean((extremeLeftx,extremeLefty), (xcenterIntercept,ycenterIntercept))
                    # calculate the lenght of the input to have balance
                    
                    lenghtofInputToHaveBalance = (rightArea/leftArea) * LenghtOfTheResistantLevel                     
                    scoreVisualBalance = np.exp(-abs(lenghtOfActualInput - lenghtofInputToHaveBalance) / self.image.shape[0]*1.618)
            
            slope = ((extremeLefty - extremeRighty ) / (extremeLeftx -extremeRightx ))
            
        if len(selectedContours) > 2:
            if extremeLeftx == 0 or extremeLefty == 0 or extremeRightx == 0 or extremeRighty == 0:
                if extremeLeftx == 0:
                    ManhattanDistanceNormalised = abs(extremeRightx - cX) / self.image.shape[1] + abs(extremeRighty - cY) / self.image.shape[0]
                    scoreVisualBalance = (1 - ManhattanDistanceNormalised) / 4 # harcoded value
                    slope = 0.5 # which is bad
                else:
                    ManhattanDistanceNormalised = abs(extremeLeftx - cX) / self.image.shape[1] + abs(extremeLefty - cY) / self.image.shape[0]
                    scoreVisualBalance = (1 - ManhattanDistanceNormalised) / 4
                    slope = 0.5 # which is bad
            else:  
                cv2.line(maskForeground, (extremeLeftx,extremeLefty), (extremeRightx,extremeRighty), (0,0,255), 2)
                # get the areas  of cnts left and right
                leftArea = 0
                rightArea = 0
                for x, a in zip(listOfxs, listofAreas):
                    if x < int(self.image.shape[1]/2):
                        leftArea += a
                    else:
                        rightArea += a
                # return also the angle of the line
                slope = ((extremeLefty - extremeRighty ) / (extremeLeftx -extremeRightx ))
                intercept = extremeLefty - (slope * extremeLeftx)
                # find the point where the leverage meets the middle line of the frame
                ycenterIntercept = (slope * self.image.shape[1]/2) + intercept
                xcenterIntercept = int(self.image.shape[1]/2)
                cv2.circle(maskForeground, (int(self.image.shape[1]/2),int(ycenterIntercept) ), 4, (255, 0, 255), -1)
                # weight the distance and find the fulcrum
                            # handle exceptions
                if leftArea == 0 or rightArea == 0:
                
                    scoreVisualBalance = 0
                    
                else:
                    if leftArea > rightArea:
                        # how long should be the input level to be in balance
                        # and how long it is
                        LenghtOfTheResistantLevel = dist.euclidean((extremeLeftx,extremeLefty), (xcenterIntercept,ycenterIntercept))
                        lenghtOfActualInput = dist.euclidean((extremeRightx,extremeRighty), (xcenterIntercept,ycenterIntercept))
                        # calculate the lenght of the input to have balance
                        lenghtofInputToHaveBalance = (leftArea/rightArea) * LenghtOfTheResistantLevel                     
                        scoreVisualBalance = np.exp(-abs(lenghtOfActualInput - lenghtofInputToHaveBalance) / self.image.shape[0]*1.618)
                    if leftArea <= rightArea:
                        # how long should be the input level to be in balance
                        # and how long it is
                        LenghtOfTheResistantLevel = dist.euclidean((extremeRightx,extremeRighty), (xcenterIntercept,ycenterIntercept))
                        lenghtOfActualInput = dist.euclidean((extremeLeftx,extremeLefty), (xcenterIntercept,ycenterIntercept))
                        # calculate the lenght of the input to have balance
                        lenghtofInputToHaveBalance = (rightArea/leftArea) * LenghtOfTheResistantLevel                     
                        scoreVisualBalance = np.exp(-abs(lenghtOfActualInput - lenghtofInputToHaveBalance) / self.image.shape[0]*1.618)
       
        SlopeVisualBalance = 1 - abs(slope)
        # draw center cross for visualiation
        cv2.line(maskForeground,(ImageCentreX, 0), (ImageCentreX, self.image.shape[0]), (255,255,255), 1)
        cv2.line(maskForeground,(0, ImageCentreY), (self.image.shape[1], ImageCentreY), (255,255,255), 1)
        # draw the label
        cv2.putText(maskForeground, "VB: {:.3f} Slop: {:.3f}".format(scoreVisualBalance, SlopeVisualBalance), (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.scoreVisualBalance = scoreVisualBalance

        return maskForeground, scoreVisualBalance, SlopeVisualBalance
    
    
    
    def superPixelSegmentation(self, num_segments = 2):
        # https://www.pyimagesearch.com/2014/12/29/accessing-individual-superpixel-segmentations-python/
        
        #load the image and convert it to a floating point data type
        copy = self.image.copy()
        image = img_as_float(copy)
        
        
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image, n_segments = num_segments, sigma = 5)
        
        segImg = mark_boundaries(copy, segments)
        # show the output of SLIC
        
        cv2.imshow('segm', segImg)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        return segments
    
        
    def maskFromSuperpixelSegmentation(self, num_segments = 2):
        
        
        # TODO apply color differents color and plot the resulting figure
        # to see how the perceived images are seen once segmented in 4, 5 etc. or just 2
        segments = self.superPixelSegmentation(num_segments = num_segments)
        for (i, segVal) in enumerate(np.unique(segments)):
            # construct a mask for the segment
            mask = np.zeros(self.image.shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255
             
            # show the masked region
# =============================================================================
#             supImg = self.image.copy()             
#             cv2.imshow("Mask", mask)
#             cv2.imshow("Applied", cv2.bitwise_and(supImg, supImg, mask = mask))
#             cv2.waitKey()
#             cv2.destroyAllWindows()
# =============================================================================
            

    
    def _findCentreOfMass(self, image = None, contours = None, approxCnt = False):
        
        if image is None:
            image = self.image
            
        imageCopy = image.copy()
        
        gray = cv2.cvtColor(imageCopy,cv2.COLOR_BGR2GRAY)
        
        #a get the outer silhouette and max one shape only
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
        ret,thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY)
        
        if contours is None:
            ing2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
            contours = sorted_contours[0:7]
        
        listOfCenterPoints = []

        for c in contours:   
        	# compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            point = (cX, cY)
            listOfCenterPoints.append(point)
         
        	# draw the contour and center of the shape on the image
            # use approx cnts drawing
            if approxCnt:
                imageCopy = self._drawApproxContours(contours = [c], image = imageCopy)
            if approxCnt == False:
                cv2.drawContours(imageCopy, [c], -1, (0, 255, 0), 1)
            
            # draw circle and write text for the center of cnt
            cv2.circle(imageCopy, (cX, cY), 5, (255, 0, 0), -1)
            #cv2.putText(imageCopy, "C", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
        return imageCopy, listOfCenterPoints

    def _drawApproxContours(self, contours, image = None):
        
        if image is None:
            image = self.image.copy()
        # iterate trough each contour and compute the approx contour
        approxContours = []
        for cnt in contours:
            #calculate accuracy as a percent of contour perimeter
            accuracy = 0.003 * cv2.arcLength(cnt,True)
            approxCnt = cv2.approxPolyDP(cnt, accuracy, True)
            approxContours.append(approxCnt)
            
        cv2.drawContours(image, approxContours, -1, (0,255,0), 1)
            
        return image
   
    def _saliencySegmentation(self, method = cv2.RETR_EXTERNAL,  factor = 3):
        
        saliency = Saliency(self.imagepath)
        SaliencyMask = saliency.get_proto_objects_map( factor = factor)
        SaliencyMask = cv2.cvtColor(SaliencyMask, cv2.COLOR_BGR2GRAY)
        # create the contours from the segmented image
        ing2, contours, hierarchy = cv2.findContours(SaliencyMask, method,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        
        return contours, SaliencyMask
    
    def _thresholdSegmentation(self, imageToTresh = None, method = cv2.RETR_LIST, factor = 1 ):

            
        if imageToTresh is None:
            copied = self.image.copy()
            copied = cv2.cvtColor(copied,cv2.COLOR_BGR2GRAY)
        else:
            copied = imageToTresh

        mean = int(np.mean(copied) * factor)
        ret,threshImg = cv2.threshold(copied, mean, 255, cv2.THRESH_BINARY)
    
        # create the contours from the segmented image
        ing2, contours, hierarchy = cv2.findContours(threshImg, method,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        
# =============================================================================
#         cv2.imshow('thresh', threshImg)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        
        return contours, threshImg
    
    def _orbSegmentation (self, maxKeypoints = 500, edged = False, edgesdilateOpen = True, method = cv2.RETR_EXTERNAL):
        
        grayCopy = self.image.copy()
        ImgForOrb = grayCopy
        # in case use edge to detect the ORB
        if edged:
            #ImgForOrb = self._edgeDetectionAuto(sigma = 0.33)
            ImgForOrb = self._edgeDetection( scalarFactor = 1, meanShift = 0, edgesdilateOpen = edgesdilateOpen)
        # create ORB object and setup number of keypoints we desire
        orb = cv2.ORB_create(maxKeypoints)
        # determine Keypoints
        keypoints = orb.detect(ImgForOrb, None)
        #obtain the descriptors
        keypoints, descriptors = orb.compute(self.gray, keypoints)
        # draw rich keypoints on input image
        imageout = self.image.copy()
        imageout = cv2.drawKeypoints(imageout, keypoints,imageout, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # mask creation
        blank = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        
        coordinates = []
        for i, keypoint, in enumerate(keypoints):
            X, Y =  (keypoints[i].pt)
            x = int(X)
            y = int(Y)
            coordinates.append([y,x])
            
        for coord in coordinates:
            #blank[coord[0],coord[1] ] = 255
            cv2.circle(blank,( coord[1],coord[0]), 3, (255,255,255), -1)
        
        if edgesdilateOpen:
            kernel = np.ones((5,5),np.uint8)
            #blank = cv2.dilate(blank,kernel,iterations = 1)
            blank = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel)
        
        blankforcnts = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)

        # create the contours from the segmented image
        ing2, contours, hierarchy = cv2.findContours(blankforcnts, method,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        
# =============================================================================
#         cv2.imshow('orb', ImgForOrb )
#         cv2.imshow('orb2', imageout )
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        
        return blank, contours, keypoints
    
    def _innerCntsSegmentation (self,numberOfCnts = 50, method = cv2.RETR_CCOMP, minArea = 2):
        
        edgeForSegm_first = self._edgeDetection( scalarFactor = 1, meanShift = 0, edgesdilateOpen = True, 
                                                kernel = 3, superEdgesdilateOpen = False)
        contours, thresholdMask = self._thresholdSegmentation()
        thresholdMaskEdges = cv2.Canny(thresholdMask,125,255,apertureSize = 3)
        # merge the two edged
        edgeForSegm = cv2.bitwise_or(edgeForSegm_first,thresholdMaskEdges)
        segmentationOnInnerCnts = np.zeros(self.image.shape, dtype = "uint8")

        
        # create the contours from the segmented image
        ing2, contours, hierarchy = cv2.findContours(edgeForSegm, method,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        
        innerCnts = []
        for  cnt, h in zip (contours, hierarchy[0]):
            if h[2] == -1 :
                innerCnts.append(cnt)
        
        sortedContours = sorted(innerCnts, key = cv2.contourArea, reverse = True)
        
        selectedContours = [cnt for cnt in sortedContours if cv2.contourArea(cnt) > minArea]
        
        for cnt in selectedContours[0: numberOfCnts]:
            cv2.drawContours(segmentationOnInnerCnts, [cnt], -1, (255, 255, 255), -1)
        
        copyInner = segmentationOnInnerCnts.copy()
        copyGray = cv2.cvtColor(copyInner, cv2.COLOR_BGR2GRAY)
        # create the contours from the segmented image
        ing2, contours, hierarchy = cv2.findContours(copyGray, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

# =============================================================================
#         cv2.drawContours(segmentationOnInnerCnts, contours, -1, (255, 0, 255), 1)
#         
#         cv2.imshow('innercnts', segmentationOnInnerCnts)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        
        return segmentationOnInnerCnts, contours
        
        
    
    def _orbSegmentationConnectedLines (self, maxKeypoints = 1000):
        
        
        imgBlank = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        
        # create ORB object and setup number of keypoints we desire
        orb = cv2.ORB_create(maxKeypoints)
        # determine Keypoints
        keypoints = orb.detect(self.gray, None)
        #obtain the descriptors
        keypoints, descriptors = orb.compute(self.gray, keypoints)
        # draw rich keypoints on input image
        imageout = self.image.copy()
        imageout = cv2.drawKeypoints(imageout, keypoints,imageout, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        pts = [p.pt for p in keypoints]
        listOfKeyPoints = [list(elem) for elem in pts]
        cnt = np.array(listOfKeyPoints, dtype = np.int32)
        cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), 1)
        
        originalCopy = self.image.copy()
        cv2.drawContours(originalCopy, [cnt], -1, (255, 0, 0), 1)
        
        
        return imgBlank, originalCopy
    
    def _zigzagCntsArea(self, drawLabel = False):
        
        copyZigZag = self.image.copy()
        #get the mask by connected keypoints
        imageBlank, imageOriginal = self._orbSegmentationConnectedLines ( maxKeypoints = 1000)
        
        imageBlank = cv2.cvtColor(imageBlank,cv2.COLOR_BGR2GRAY)
        meanThresh = np.mean(imageBlank)
        ret,thresh = cv2.threshold(imageBlank, meanThresh, 255, cv2.THRESH_BINARY)

        im2, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse = True)
        
        cv2.drawContours(copyZigZag, sorted_contours, -1, (0, 255, 255), 1)
        # area golden rectangle
        
        goldenRectangleArea = (self.image.shape[0] * 0.618 ) * (self.image.shape[1]* 0.618)
        areaCnt = cv2.contourArea(sorted_contours[0])
        
        ratioGoldenRectangleZigZagOrb = areaCnt/goldenRectangleArea
        
        # the longer the perimeter the more the eye travel the pictures
        perimeter = cv2.arcLength(sorted_contours[0],True)
        pixelCount = self.image.shape[0] * self.image.shape[1]
        
        zigzagPerimeterScore = (perimeter/pixelCount * 10 * 1.618)
        
        
        if drawLabel:
            cv2.putText(copyZigZag, "ZigGolden: {:.3f}".format(ratioGoldenRectangleZigZagOrb), (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        self.ratioGoldenRectangleZigZagOrb = ratioGoldenRectangleZigZagOrb
        
        self.zigzagPerimeterScore = zigzagPerimeterScore
        
        return copyZigZag, ratioGoldenRectangleZigZagOrb, sorted_contours, zigzagPerimeterScore
    
    def _edgeDetectionAuto(self, sigma = 0.33):
        
        # compute the median of the single channel pixel intensities
        v = np.median(self.image)
        
        copyforedge = self.image.copy()
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(copyforedge, lower, upper)
        
        # return the edged image
        return edged
    
    def _edgeDetection(self, scalarFactor = 1, meanShift = 0, edgesdilateOpen = True, kernel = 5, superEdgesdilateOpen = False):
        
        # edges history: edges can be defined as sudden changes(discontinuities)in an image
        # and they can encode just as much information as pixels
        # there are 3 main types of Edge Detection: Sobel, to emphasis vertical,horizontal edges
        # laplacian gets all orientations
        # canny (John F.Canny in 1986) Optimal due to low error rate,well defined edges
        # and accurate detection (1.Applies gaussian bluring, finds intesity gradient,
        # 3.. and 4... see Udemy course)
        copyImg = self.image.copy()
        gray = cv2.cvtColor(copyImg,cv2.COLOR_BGR2GRAY)
        
        mean = np.mean(gray)
        mean += meanShift
        std = np.std(gray)

        minThres = int((mean - std) * scalarFactor)
        maxThresh = int((mean+std) * scalarFactor)
        edges = cv2.Canny(gray,minThres,maxThresh,apertureSize = 3)
        
        if edgesdilateOpen:
            kernel = np.ones((kernel,kernel),np.uint8)
            edges = cv2.dilate(edges,kernel,iterations = 1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            
        if superEdgesdilateOpen:
            kernel = np.ones((3,3),np.uint8)
            edges = cv2.dilate(edges,kernel,iterations = 1)
            edges = cv2.dilate(edges,kernel,iterations = 1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        return edges
    
    def _fourTriangleGuidelines (self, flip = True):
        
        # draw the lines of the diagonal
        topLeft = (0,0)
        lowerRight = (self.image.shape[1], self.image.shape[0])
        lowerLeft = (self.image.shape[1],0)
        topright = (0, self.image.shape[0])
        blankFourTriangle = np.zeros(self.image.shape, dtype = np.uint8)
        #blankFourTriangle = np.array(blankFourTriangle)
        cv2.line(blankFourTriangle, topright , lowerLeft, (255,255,255), 1) # topright - lowerleft

        # draw the two perpendicular lines
        leftIntersectionX, leftIntersectionY = self._find_perpendicular_through_point_to_line(lowerLeft[0], lowerLeft[1],topright[0],topright[1], topLeft[0], topLeft[1] )
        cv2.line(blankFourTriangle, topLeft , (int(leftIntersectionX), int(leftIntersectionY) ), (255,255,255), 1)
        rightIntersectionX, righttIntersectionY = self._find_perpendicular_through_point_to_line(lowerLeft[0], lowerLeft[1],topright[0],topright[1], lowerRight[0], lowerRight[1] )
        cv2.line(blankFourTriangle, lowerRight , (int(rightIntersectionX), int(righttIntersectionY) ), (255,255,255), 1)
        # swithch for reverese
        
        if flip:
            blankFourTriangle = cv2.flip( blankFourTriangle, 0)

        # return one or reversed as bgr
        
        return blankFourTriangle

    
    def _find_perpendicular_through_point_to_line(self, x1, y1, x2, y2, x3, y3):
        
        k = ((y2-y1) * (x3-x1) - (x2-x1) * (y3-y1)) / ((y2-y1)**2 + (x2-x1)**2)
        x4 = x3 - k * (y2-y1)
        y4 = y3 + k * (x2-x1)
        
        return x4, y4
    
    def _distance_to_line(self, p1, p2, x, y):
        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        num = abs(y_diff*x - x_diff*y + p2[0]*p1[1] - p2[1]*p1[0])
        den = np.sqrt(y_diff**2 + x_diff**2)
        return num / den
    
    def grabcut(self, image, x,y,w,h, pad):
        
        img = image
        mask = np.zeros(img.shape[:2],np.uint8)
        
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        rect = (x-pad,y-pad,w+pad,h+pad)
        rect = (2,2,self.image.shape[1]-10, self.image.shape[0]-10)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        
# =============================================================================
#         cv2.imshow('rabcut', img)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        return img
        
    def grabcutOnOrb (self):
        
        imgCut = self.image.copy()
        blank, contours, keypoints = self._orbSegmentation(maxKeypoints = 500)
# =============================================================================
#         
#         
#         coordinates = []
#         for i, keypoint, in enumerate(keypoints):
#             X, Y =  (keypoints[i].pt)
#             x = int(X)
#             y = int(Y)
#             coordinates.append([y,x])
#         
#         topLeft = min(coordinates)
#         botRight = max (coordinates)
# =============================================================================
        
        
        
        copyZigZag, ratioGoldenRectangleZigZagOrb , sorted_contoursZigZag, zigzagPerimeterScore= self._zigzagCntsArea()

        #draw the bounding box
        c = max(sorted_contoursZigZag, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        
        imgCut = self.grabcut(imgCut, x,y,w,h, pad = 30)
        
        cv2.imshow('imgcut', imgCut)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def _point_to_line_dist(self, point, line):
        """Calculate the distance between a point and a line segment.
    
        To calculate the closest distance to a line segment, we first need to check
        if the point projects onto the line segment.  If it does, then we calculate
        the orthogonal distance from the point to the line.
        If the point does not project to the line segment, we calculate the 
        distance to both endpoints and take the shortest distance.
    
        :param point: Numpy array of form [x,y], describing the point.
        :type point: numpy.core.multiarray.ndarray
        :param line: list of endpoint arrays of form [P1, P2]
        :type line: list of numpy.core.multiarray.ndarray
        :return: The minimum distance to a point.
        :rtype: float
        """
        # unit vector
        unit_line = line[1] - line[0]
        norm_unit_line = unit_line / np.linalg.norm(unit_line)
    
        # compute the perpendicular distance to the theoretical infinite line
        segment_dist = (
            np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
            np.linalg.norm(unit_line)
        )
    
        diff = (
            (norm_unit_line[0] * (point[0] - line[0][0])) + 
            (norm_unit_line[1] * (point[1] - line[0][1]))
        )
    
        x_seg = (norm_unit_line[0] * diff) + line[0][0]
        y_seg = (norm_unit_line[1] * diff) + line[0][1]
    
        endpoint_dist = min(
            np.linalg.norm(line[0] - point),
            np.linalg.norm(line[1] - point)
        )
    
        # decide if the intersection point falls on the line segment
        lp1_x = line[0][0]  # line point 1 x
        lp1_y = line[0][1]  # line point 1 y
        lp2_x = line[1][0]  # line point 2 x
        lp2_y = line[1][1]  # line point 2 y
        is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
        is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
        if is_betw_x and is_betw_y:
            return segment_dist
        else:
            # if not, then return the minimum distance to the segment endpoints
            return endpoint_dist
        
    
    def _drawGoldenSpiral(self, drawRectangle = False, drawEllipses = True, x0 = None, y0 = None, x = None, y = None):

        im = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        
        # Golden Rectangle
        # Draws a golden reactangle, its decomposition into squares
        # and the spiral connecting the corners
        # Jerry L. Martin, December 2014
        
        phi = (1 + 5 ** 0.5) / 2 # The golden ratio. phi = 1.61803...
        
        # x = 0 and y= 0 are the top left corner , x=x and y=y are the bottom right corner
        if x0 == None:
            x0 = 0
        if y0 == None:
            y0 = 0
        if y == None:
            y = im.shape[0]
        if x == None:
            x = im.shape[1]
            
        
        lenght1rect = int(x * (1/phi))
        hight1rect = y
        # second rectangle
        lenght2rect = x - lenght1rect
        height2rect = int(y*(1/phi))
        # third rectangle
        lenght3rect = lenght2rect # confusion here but it works fxed later TODO clean the code
        hight3rect = y - height2rect
        leftCorn3rectX = int(int(x*(1 /phi)) + (lenght3rect * ( 1 - (1/phi))))
        leftCorn3rectY = int(y*(1/phi))
        # fourth rectangle
        leftCorn4rectX = lenght1rect
        leftCorn4rectY = (x + int(y*(1/phi)))
        rigthLowerCorn4rectX = lenght1rect + int((1-(1/phi)) * int(lenght2rect))
        rigthLowerCorn4rectY = height2rect + int((y - height2rect) * (1 -(1/phi)))
        #be careful I drew the little rect here the 4 rect is the bigger
        #correct hight and width of the 4 rect
        lenght4rect = int((1 -(1/phi)) * lenght3rect)
        hight4rect = int(((1/phi)) * hight3rect)
        # fifth rectangle
        leftCorn5rectX = lenght1rect
        leftCorn5rectY = height2rect
        rigthLowerCorn5rectX = lenght1rect + int(lenght4rect * (1/phi))
        rigthLowerCorn5rectY = height2rect + int((y - height2rect) * (1 -(1/phi)))
        lenght5rect = int((1/phi) * lenght4rect)
        hight5rect = y - height2rect - hight4rect
        # sixth rectangle
        lenght6rect = int((1 - (1/phi)) * lenght4rect)
        hight6rect = int((1/phi) * hight5rect)
        leftCorn6rectX = lenght1rect + lenght5rect
        leftCorn6rectY = height2rect
        rigthLowerCorn6rectX = lenght1rect + lenght4rect
        rigthLowerCorn6rectY = height2rect + hight6rect
        # seventh rectangle
        lenght7rect = int((1/phi) * lenght6rect)
        hight7rect = int((1-(1/phi)) * hight5rect)
        leftCorn7rectX = lenght1rect + lenght5rect + int((1 -(1/phi))*  lenght6rect )
        leftCorn7rectY = height2rect + hight6rect
        rigthLowerCorn7rectX = lenght1rect + lenght4rect
        rigthLowerCorn7rectY = height2rect + hight5rect
        # eighth rectangle
        lenght8rect = int((1 - (1/phi)) * lenght6rect)
        hight8rect = int((1/phi) * hight7rect)
        leftCorn8rectX = lenght1rect + lenght5rect
        leftCorn8rectY = height2rect + hight6rect + int((1 -(1/phi))*  hight7rect )
        rigthLowerCorn8rectX = lenght1rect + lenght5rect + lenght8rect
        rigthLowerCorn8rectY = height2rect + hight6rect + hight7rect
        
        if drawRectangle:
            # first rectangle on the left
            cv2.rectangle(im,(0,0), (int(x*(1 /phi)), y), (255, 255, 255), 1)
            cv2.rectangle(im,( int(x*(1 /phi)), 0  ), (x, int(y*(1/phi))), (255, 255, 255), 1 )
            cv2.rectangle(im,( leftCorn3rectX , leftCorn3rectY  ), (x, y), (255, 255, 255), 1 )
            cv2.rectangle(im,( leftCorn4rectX , leftCorn4rectY  ), (rigthLowerCorn4rectX, rigthLowerCorn4rectY), (255, 255, 255), 1)
            cv2.rectangle(im,( leftCorn5rectX , leftCorn5rectY  ), (rigthLowerCorn5rectX, rigthLowerCorn5rectY), (255, 255, 255), 1)
            cv2.rectangle(im,( leftCorn6rectX , leftCorn6rectY  ), (rigthLowerCorn6rectX, rigthLowerCorn6rectY), (255, 255, 255), 1)
            cv2.rectangle(im,( leftCorn7rectX , leftCorn7rectY  ), (rigthLowerCorn7rectX, rigthLowerCorn7rectY), (255, 255, 255), 1)
            cv2.rectangle(im,( leftCorn8rectX , leftCorn8rectY  ), (rigthLowerCorn8rectX, rigthLowerCorn8rectY), (255, 255, 255), 1)
            # main rectangle
            cv2.rectangle(im,(0,0), (x, y), (255, 255, 255), 2)
        
        if drawEllipses:
            #draw ellipse
            #first ellipse
            cv2.ellipse(im,  (int(x *(1/phi)), y) , (int(lenght1rect), int(hight1rect))  , 0,  180, 270, (255, 255, 255), 1)
            # 2 ellipse
            cv2.ellipse(im,  (lenght1rect, height2rect) , (lenght2rect, height2rect)  , 0,  270, 360, (255, 255, 255), 1)
            # 3 ellipse
            cv2.ellipse(im,  ((lenght1rect+lenght4rect), height2rect) , (int(lenght2rect*(1/phi)), hight3rect)  , 0,  0, 90, (255, 255, 255), 1)
            # 4 ellipse
            cv2.ellipse(im,  ((lenght1rect+lenght4rect), (height2rect + hight5rect) ), (lenght4rect, hight4rect)  , 0,  90, 180, (255, 255, 255), 1)
            # 5 ellipse
            cv2.ellipse(im,  ((lenght1rect+lenght5rect), (height2rect + hight5rect) ), (lenght5rect, hight5rect)  , 0,  180, 270, (255, 255, 255), 1)
            # 6 ellipse
            cv2.ellipse(im,  ((lenght1rect+lenght5rect), (height2rect + hight6rect) ), (lenght6rect, hight6rect)  , 0,  270, 360, (255, 255, 255), 1)
            # 7 ellipse
            cv2.ellipse(im,  ((lenght1rect+lenght5rect+lenght8rect), (height2rect + hight6rect) ), (lenght7rect, hight7rect)  , 0,  0, 90, (255, 255, 255), 1)
            # 8 ellipse
            cv2.ellipse(im,  ((lenght1rect+lenght5rect+lenght8rect), (height2rect + hight6rect + int((1-(1/phi)) * hight7rect)  ) ), (lenght8rect, hight8rect)  , 0,  90, 180, (255, 255, 255), 1)
        
        im2 = cv2.flip(im, -1 )
        im3 = cv2.flip(im2, 1 )
        im4 = cv2.flip(im, 1)
        
        
        return im, im2, im3, im4
            
    def collectScoresImage (self):
        
        # collect all the scores that are not part of the display label images
        
        ssimAsymmetry = self.calculateAsimmetry()
        fractalScoreFromTarget = self.fractalDimMinkowskiBoxCount()
        ratioForeVsBackground = self.areaForegroundVsbackground()
        diagonalAsymmetry = self.calculateDiagonalAsymmetry()
        histHueCorrelationBalance = self.calculateHistBalance()
        warmColorBalance = self.calculateWarmOrColdBalance()
        
# =============================================================================
#         # return the display image for the UI
#         rows, cols, depth = self.image.shape
#         blackboard = np.zeros(self.image.shape, dtype="uint8")
#         # make solid color for the background
#         blackboard[:] = (218,218,218)
#         
#         
#         scoringListtoDisplay= []
#         scoringListtoDisplay.append(ssimAsymmetry)
#         scoringListtoDisplay.append(fractalScoreFromTarget)
#         scoringListtoDisplay.append(ratioForeVsBackground)
#         
#         scalar = .9
#         padding = 10
#         switch = True
#         colors = [(120,60,120), (60,120,120)]
#         for score in scoringListtoDisplay:
#             if switch:
#                 blackboard[padding:padding+10, 10:int(score*scalar*rows)] = colors[0]
#                 padding += padding
#             else:
#                 blackboard[padding:padding+10, 10:int(score*scalar*rows)] = colors[1]
#                 padding += padding
#             switch = not switch
#         #scores = np.array([[10, int(- 10 + rows -(percentageWarm * rows * .6))], [20, int(- 10 + rows - (percentageCold * rows * .6))]])
#         #cv2.polylines(blackboard, np.int32([scores]), 1, (255,0,255))
#         cv2.putText(blackboard, "Asym , fract, ratioFore", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
#         # send to UI json writing (as a return)
# =============================================================================
        
        #return blackboard, ssimAsymmetry, fractalScoreFromTarget, ratioForeVsBackground
        
        self.ssimAsymmetry = ssimAsymmetry
        self.fractalScoreFromTarget = fractalScoreFromTarget
        self.diagonalAsymmetry = diagonalAsymmetry
        
        return  ssimAsymmetry, fractalScoreFromTarget, ratioForeVsBackground, diagonalAsymmetry,histHueCorrelationBalance, warmColorBalance
    
    def areaForegroundVsbackground (self):
        
        imageGray = self.gray.copy()
        meanGray = np.mean(imageGray)
        # blur to increase extraction
        imageBlurred = cv2.GaussianBlur(imageGray, (3,3), 0)

        ret, threshImageBinary = cv2.threshold(imageBlurred, meanGray, 255, cv2.THRESH_BINARY)

        # count the number of pixel greater than 1 so in the area foreground
        totalPixelInForeground = (threshImageBinary > 1).sum()
        
        ratioForeVsBackground = totalPixelInForeground / threshImageBinary.size
        

        return ratioForeVsBackground
    
    def calculateAsimmetry(self, image = None):
        # calculate the the hist correlation of flipped image
        # if image is not provided uses self.image
        if image is None:
            image = self.image
        
        imageCopy = image.copy()
        imageFlipLfRt = cv2.flip(imageCopy, 1)
        imageFlipLf = imageFlipLfRt.copy()
        #imageFlipLfHalfRows = int(imageFlipLf.shape[0] /2)
        imageFlipLfHalfCols = int(imageFlipLf.shape [1] / 2)
        imageHalfCols = int(imageCopy.shape [1] / 2)
        
        imageFlipLf [:,0 : imageFlipLfHalfCols ] = 0
        imageCopy [:, 0 : imageHalfCols] = 0
        
        imageFlipLf = imageFlipLf [0 : image.shape[0], imageFlipLfHalfCols : image.shape[1]]
        imageCopy = imageCopy [0 : image.shape[0], imageHalfCols : image.shape[1]]

        similaritySSIM = self.compare_images (imageFlipLf, imageCopy)

        ssimAsymmetry = 1 - similaritySSIM
        
        return ssimAsymmetry
    
    def calculateDiagonalAsymmetry(self):
        
        diagonalImg = self.image.copy()

        imageFlippedDiagonal = cv2.flip(diagonalImg, -1)
        
        pts = np.array([[0,0],[int(self.image.shape[1]),0],[int(self.image.shape[1]),int(self.image.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        
        cv2.drawContours(diagonalImg, [pts], 0, (0,0,0), -1)
        cv2.drawContours(imageFlippedDiagonal, [pts], 0, (0,0,0), -1)
        
        similaritySSIMDiagonal = self.compare_images (diagonalImg, imageFlippedDiagonal)

        diagonalAsymmetry = 1 - similaritySSIMDiagonal
        
        return diagonalAsymmetry
    
    def calculateHistBalance(self, image = None):
        
        if image is None:
            image = self.image
        
        imageCopy = image.copy()
        # convert to HSV
        imageCopy = cv2.cvtColor(imageCopy,cv2.COLOR_BGR2HSV)
        
        imageFlipLfRt = cv2.flip(imageCopy, 1)
        imageFlipLf = imageFlipLfRt.copy()
        #imageFlipLfHalfRows = int(imageFlipLf.shape[0] /2)
        imageFlipLfHalfCols = int(imageFlipLf.shape [1] / 2)
        imageHalfCols = int(imageCopy.shape [1] / 2)
        
        imageFlipLf [:,0 : imageFlipLfHalfCols ] = 0
        imageCopy [:, 0 : imageHalfCols] = 0
        
        imageFlipLf = imageFlipLf [0 : image.shape[0], imageFlipLfHalfCols : image.shape[1]]
        imageCopy = imageCopy [0 : image.shape[0], imageHalfCols : image.shape[1]]
        
        histA = cv2.calcHist([imageFlipLf], [0, 1], None, [180, 256], [0, 180, 0, 256])
        histB = cv2.calcHist([imageCopy], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
# =============================================================================
#         histA = cv2.calcHist([imageFlipLf], [0], None, [180], [0, 180])
#         histB = cv2.calcHist([imageCopy], [0], None, [180], [0, 180])
# =============================================================================
        
        histHueCorrelationBalance = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
        
        return histHueCorrelationBalance

    def calculateWarmOrColdBalance(self):
        
        
        imageCopy = self.image.copy()
        # convert to HSV
        imageCopy = cv2.cvtColor(imageCopy,cv2.COLOR_BGR2HSV)
        
        imageFlipLfRt = cv2.flip(imageCopy, 1)
        imageFlipLf = imageFlipLfRt.copy()
        #imageFlipLfHalfRows = int(imageFlipLf.shape[0] /2)
        imageFlipLfHalfCols = int(imageFlipLf.shape [1] / 2)
        imageHalfCols = int(imageCopy.shape [1] / 2)
        
        imageFlipLf [:,0 : imageFlipLfHalfCols ] = 0
        imageCopy [:, 0 : imageHalfCols] = 0
        
        imageFlipLf = imageFlipLf [0 : self.image.shape[0], imageFlipLfHalfCols : self.image.shape[1]]
        imageCopy = imageCopy [0 : self.image.shape[0], imageHalfCols : self.image.shape[1]]

        
        LftotalWarmPixels30 = (imageFlipLf[:,:,0] < 30).sum()
        LftotalWarmPixels150 = (imageFlipLf[:,:,0] > 150).sum()
        LftotalWarmPixels = LftotalWarmPixels30 + LftotalWarmPixels150
        
        RttotalWarmPixels30 = (imageCopy[:,:,0] < 30).sum()
        RttotalWarmPixels150 = (imageCopy[:,:,0] > 150).sum()
        RttotalWarmPixels = RttotalWarmPixels30 + RttotalWarmPixels150
        
        diff = abs(LftotalWarmPixels - RttotalWarmPixels)
        mean = (LftotalWarmPixels + RttotalWarmPixels) / 2
        
        warmColorBalance = diff / mean 
        
        return warmColorBalance

            

    
    def compare_images(self, imageA, imageB):
    
        s = ssim(imageA,imageB, win_size=None, gradient=False, data_range=None, 
                 multichannel = True, gaussian_weights=False, full=False, dynamic_range = None)
        
        return s
        
        
    def calculateWarmOrCold(self):

        imageHSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #calculate the percentage, assuming warm from 150 to 0 and 0 to 30
        # and cold from 31 to 149
        warm = 0
        cold = 0
        rows, cols, depth = imageHSV.shape
        for row in range (0, rows):
            for col in range( 0, cols):
                pixel = imageHSV[row, col, 0]
                if pixel < 30:
                    warm += 1 
                elif pixel > 150:
                    warm += 1
                else:
                    cold += 1
        percentageWarm = warm / self.image[:,:,0].size
        percentageCold = cold / self.image[:,:,0].size
        
        return percentageWarm, percentageCold
    
    def fractalDimMinkowskiBoxCount(self, target = 1.5):
        
        Z = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # calculate the threshold on the mean
        mean, standardDeviation = cv2.meanStdDev(Z)
        threshold = mean[0][0]
        
        # Only for 2d image
        assert(len(Z.shape) == 2)
    
        # From https://github.com/rougier/numpy-100 (#87)
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
    
            # We count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k*k))[0])
    
        # Transform Z into a binary array
        Z = (Z < threshold)
    
        # Minimal dimension of image
        p = min(Z.shape)
    
        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p)/np.log(2))
    
        # Extract the exponent
        n = int(np.log(n)/np.log(2))
    
        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)
    
        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
    
        # Fit the successive log(sizes) with log (counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        # print ("this is the fractal {}".format(-coeffs[0]))
        fractal = -coeffs[0]
        
        distFromTarget = abs(target - fractal) 
        # avoiding nan results
        if str(distFromTarget) == 'nan':
            distFromTarget = 1
        
        fractalScoreFromTarget = 1 - distFromTarget
            
        return fractalScoreFromTarget
    
