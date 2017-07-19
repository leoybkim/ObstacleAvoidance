import cv2
import numpy as np

# dictionary of all contours
contours = {}

# Camera
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while (cap.isOpened()):
    # Capture
    ret, frame = cap.read()
    if ret == True:
        # Canny edge detection
        canny = cv2.Canny(frame,80,240,3)
        # Contours
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Fill from bottom side
        # For each column, find the maximum row index:
        # 1) Reverse the array vertically
        # 2) Use np.argmax to return the first instance of the maximum in an array
        h, w = canny2.shape[:2]
        max_row_index = h - np.argmax(canny2[::-1], axis=0)

        # Create boolean array where every index greater than or equal to the max index is True
        row_index = np.indices((h, w))[0]
        index_after_edges = row_index >= max_row_index

        # Fill with white pixels
        filled_from_bottom = np.zeros((h, w))
        filled_from_bottom[index_after_edges] = 255

        # erosion attempt1
        vertical = filled_from_bottom.copy();
        (rows, cols) = filled_from_bottom.shape
        # #horizontalSize = cols/30
        verticalSize = int(rows/30)
        # #horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(int(horizontalSize), 1))
        # verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalSize))
        # erosion = cv2.erode(vertical, verticalStructure)


        # Erosion on thin vertical lines
        kernel = np.ones((1,verticalSize),np.uint8)
        erosion = cv2.erode(filled_from_bottom,kernel,iterations=10)

        # pixels = np.argwhere(erosion == 255)
        #
        # if (pixels[0][0]):
        #     # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) → img
        #     cv2.circle(erosion,(pixels[0][0],pixels[0][1]), 3, (255,0,255), -1)


        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)
        cv2.imshow('filled', erosion)
        if cv2.waitKey(1) == 1048689:
            break

cap.release()
cv2.destroyAllWindows()
