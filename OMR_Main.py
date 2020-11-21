
#spark intership Computer
# Vision &
#
# Internet of
#
# Things
#GRIPNOV20
# Implement a Fault Detection System which, detects and eliminates the faulty
# products based on the shape/colour.

# omr scnner

import cv2
import numpy as np
import utlis

path = "MCQPaper_LI.jpg"
widthImg = 700
heightImg = 700
question = 5
choices = 5
ans = [0, 1, 0, 2, 3]
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, countours, -1, (0, 255, 0), 10)


def rectCountours(contours):
    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    # print(len(rectCon))
    return rectCon


rectCon = rectCountours(countours)


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)  # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)  # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx


biggestContour = getCornerPoints(rectCon[0])
print(biggestContour.shape)
gradePoints = getCornerPoints(rectCon[1])


# print(biggestContour)

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))  # REMOVE EXTRA BRACKET
    print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32)  # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    print(add)
    print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  # [0,0]
    myPointsNew[3] = myPoints[np.argmax(add)]  # [w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [h,0]

    return myPointsNew


def splitBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes


if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -3, (0, 255, 0), 30)
    cv2.drawContours(imgBiggestContours, gradePoints, -3, (255, 0, 0), 50)
    biggestContour = reorder(biggestContour)
    gradePoints = reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    pt1G = np.float32(gradePoints)
    pt2G = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(pt1G, pt2G)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    # cv2.imshow("Grade",imgGradeDisplay)

    # apply therhold
    imgwarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgwarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("thresh", imgThresh)

    boxes = splitBoxes(imgThresh)

    # non zero pixel value
    myPixelVal = np.zeros((question, choices))
    countc = 0
    countr = 0

    for image in boxes:
        totalPixel = cv2.countNonZero(image)
        myPixelVal[countr][countc] = totalPixel
        countc += 1
        if (countc == choices): countr += 1; countc = 0
        print(myPixelVal)
    # finding index of value
    myIndex = []
    for x in range(0, question):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)

    # grading
    grading = []
    for x in range(0, question):
        if ans[x] == myIndex[x]:
            grading.append((1))
        else:
            grading.append(0)
    # print(grading)

    score = sum(grading) / question * 100  # final graing
    print(score)

# dispaly answer
imgResult = imgWarpColored.copy()


def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2

        if grading[x] == 1:
            myColor = (0, 255, 0)
            # cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            # cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
        else:
            myColor = (0, 0, 255)

            # cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            # cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 0, 255)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2), 20, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
    return img


imgResult = showAnswers(imgResult, myIndex, grading, ans, question, choices)
imgRawDrawing = np.zeros_like(imgWarpColored)
imgRawDrawing = showAnswers(imgRawDrawing, myIndex, grading, ans, question, choices)
invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
imgInvwrap = cv2.warpPerspective(imgRawDrawing, invmatrix, (widthImg, heightImg))

# # imgRawGrade = np.zeros_like(imggrade)
# # cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
# # cv2.imshow("grade", imgRawGrade)  # print scrore
# imgRawGrade = np.zeros_like(imgGradeDisplay) # NEW BLANK IMAGE WITH GRADE AREA SIZE
# cv2.putText(imgRawGrade,str(int(score))+"%",(80,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
# cv2.imshow("grade",imgRawGrade)
#
# # invmatrixG = cv2.getPerspectiveTransform(pt2G, pt1G)
# # imginvgrade = cv2.warpPerspective(imgRawGrade, invmatrixG, (widthImg, heightImg))
# #
# # imgFinal = cv2.addWeighted(imgFinal, 1, imgInvwrap, 1, 0)
# # imgFinal = cv2.addWeighted(imgFinal, 1, imginvgrade, 1, 0)
# invMatrixG = cv2.getPerspectiveTransform(pt2G, pt1G) # INVERSE TRANSFORMATION MATRIX
# imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # INV IMAGE WARP
#
#             # SHOW ANSWERS AND GRADE ON FINAL IMAGE
# imgFinal = cv2.addWeighted(imgFinal, 1, imgInvwrap, 1,0)
# imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)
# cv2.imshow("inver", imgFinal)
# cv2.waitKey(1)


# DISPLAY GRADE
imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)  # NEW BLANK IMAGE WITH GRADE AREA SIZE
cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)  # ADD THE GRADE TO NEW IMAGE
cv2.imshow("grade",imgRawGrade)
invMatrixG = cv2.getPerspectiveTransform(pt2G, pt1G)  # INVERSE TRANSFORMATION MATRIX
imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))  # INV IMAGE WARP

# SHOW ANSWERS AND GRADE ON FINAL IMAGE
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvwrap, 1, 0)
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

# IMAGE ARRAY FOR DISPLAY
imageArray = ([img, imgGray, imgCanny, imgContours],
              [imgBiggestContours, imgThresh, imgWarpColored, imgFinal])
cv2.imshow("Final Result", imgFinal)

imgBlank = np.zeros_like(img)


def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)
            # cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            # cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
        else:
            myColor = (0, 0, 255)
            # cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2), 20, myColor, cv2.FILLED)
    return img


def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver


imageArray = (
[img, imgContours, imgThresh, imgResult],[ imgRawDrawing, imgInvwrap, imgFinal,imgBlank])
# lables =[["original","Contours","threshold","REsult"],[" rawdrawing ","invrwrap","final","-"]]
imggStacked = stackImages(imageArray, 0.3)
cv2.imshow("original", imggStacked)
cv2.waitKey(0)

# import cv2
#
#
# path = "MCQPaper.jpg"
# img = cv2.imread(path)
#
# cv2.imshow("original", img)
# cv2.waitKey(0)
