import numpy as np
import cv2 as cv
'''
Goal: create a program that
- segments all objects in the given image
- draw them on a blank image
- print the perimeter and area of each object
- only draw large objects with an area of greater than 1,000 px2 --> changed to 2k
- Each object should have its own color but that color does not need to match the original

Additional learning goal: apply different filters even if not actually used later to see how they each affect the image
'''

img = cv.imread("fuzzy.png",1)
cv.imshow("original", img)

#region experiment_with_filters
# #Apply thresh to image
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh1 = cv.threshold(gray, 127,255, cv.THRESH_BINARY)
# cv.imshow("thresh1", thresh1)

# # Apply Gaussian to image
# blur = cv.GaussianBlur(img, (5,5),0)
# cv.imshow("blur", blur)

# #Apply dilate and erode to image
# kernel = np.ones((5,5), 'uint8')
# dilate = cv.dilate(img, kernel, iterations=5)
# erode = cv.erode(img, kernel, iterations=1)
# cv.imshow("dilated", dilate)
# cv.imshow("eroded", erode)
#endregion

# Apply adaptive threshold to image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3,3),0)
#cv.imshow("blur", blur)
# Had to refer to lesson files for the settings on both blur and thresh - very picky
thresh = cv.adaptiveThreshold(blur,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 205,1)
cv.imshow("Thresh", thresh)

#Find contours
thickness = 4
color = (255,255,0)

contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))

#Filter out unwanted contours
# (of an area less than 1k originally, then increased to 2k to get rid of additional contour)
filtered = []
for i in contours:
    if cv.contourArea(i)< 2000: continue
    filtered.append(i)

# print(len(filtered))
# print(np.shape(thresh))

#Remember to add color channel in this line
original_image_shape = np.shape(thresh)
out_image1 = np.zeros([original_image_shape[0], original_image_shape[1], 3])
out_image2 = np.zeros([original_image_shape[0], original_image_shape[1], 3])

color_list = [(100,100,100), (0,200,0), (0,100,100),(200,0,0)]

for i, contour in enumerate(filtered):
    color = color_list[i%len(color_list)]
    print(i%len(color_list))
    print(color)
    cv.drawContours(out_image1, [contour], -1, color, -1)

    #Grab moments from given contour, print
    moments = cv.moments(contour) #Typically called M
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    print(f'for contour {i}, the perim is {round(perimeter,2)} and the area is {round(area,2)}')


    #Try to get an approximate of the contour (more like a rectangle)
    epsilon = 0.1*cv.arcLength(contour, True) #sets epsilon to 10% of the contour
    approx = cv.approxPolyDP(contour, epsilon, True)
    cv.drawContours(out_image2, [approx], -1, color, -1)


cv.imshow("contours", out_image1)
cv.imshow("contours with approximation applied", out_image2)

cv.waitKey(0)
cv.destroyAllWindows()
