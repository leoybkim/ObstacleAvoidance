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
        # h, w = frame.shape[:2]
        # row_inds = np.indices((h, w))[0] # gives row indices in shape of img
        # row_inds_at_edges = row_inds.copy()
        # row_inds_at_edges[contours==0] = 0 # only get indices at edges, 0 elsewhere
        # max_row_inds = np.amax(row_inds_at_edges, axis=0) # find the max row ind over each col
        #
        # inds_after_edges = row_inds >= max_row_inds
        #
        # filled_from_bottom = np.zeros((h, w))
        # filled_from_bottom[inds_after_edges] = 255

        h, w = canny2.shape[:2]
        filled_from_bottom = np.zeros((h, w))
        for col in range(w):
            for row in reversed(range(h)):
                if canny2[row][col] < 255:
                    filled_from_bottom[row][col] = 255
                else:
                    break

        pixels = np.argwhere(filled_from_bottom == 255)

        if (pixels[0][0]):
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) → img
            cv2.circle(filled_from_bottom,(pixels[0][0],pixels[0][1]), 3, (255,0,255), -1)


        out.write(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('canny',canny)
        cv2.imshow('filled', filled_from_bottom)
        if cv2.waitKey(1) == 1048689:
            break

cap.release()
cv2.destroyAllWindows()
