import cv2
import numpy as np
import os

# def find_best_rotation(contour_imgs, rot) : 

#     # for each contour, for every rotation angle, find its bounding box, find the difference in area
  
#     for contour in contours : 
#         for angle in range(-rot, rot) : 

def rotate_contour(image, contour, angle, center=None):
    # Convert angle from degrees to radians

    padding = 500
    
    x, y, w, h = cv2.boundingRect(contour)
    channels = image.shape[2] if len(image.shape) == 3 else 1
    blank_image = np.zeros((h+padding, w+padding, channels), dtype=image.dtype)
    offset_contour = contour - [x, y] + [int(padding/2), int(padding/2)]
    cv2.drawContours(blank_image, [offset_contour], -1, (255, 255, 0), thickness=10)
    cv2.imshow("original contour", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    angle_rad = np.deg2rad(angle)

    # If no center is specified, calculate the center of the contour
    if center is None:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            center = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        else:
            # If the contour is a line (zero area), fall back to the first point as the center
            center = contour[0][0]

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation matrix to each point in the contour
    rotated_contour = []
    for point in contour:
        rotated_point = np.dot(rotation_matrix, np.array([point[0][0], point[0][1], 1]))
        rotated_contour.append(rotated_point)

    # Convert to an array with the same shape as the input contour
    rotated_contour = np.array(rotated_contour, dtype=np.int32).reshape(-1, 1, 2)

    x, y, w, h = cv2.boundingRect(rotated_contour)
    channels = image.shape[2] if len(image.shape) == 3 else 1
    blank_image = np.zeros((h+padding, w+padding, channels), dtype=image.dtype)
    rot_offset_contour = rotated_contour - [x, y] + [int(padding/2), int(padding/2)]
    cv2.drawContours(blank_image, [rot_offset_contour], -1, (255, 255, 0), thickness=10)
    cv2.imshow("rotated contour", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rotated_contour