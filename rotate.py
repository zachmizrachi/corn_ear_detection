import cv2
import numpy as np
import os

# def find_best_rotation(contour_imgs, rot) : 

#     # for each contour, for every rotation angle, find its bounding box, find the difference in area
  
#     for contour in contours : 
#         for angle in range(-rot, rot) : 

def rotate_contour(contour, angle) : 

    angle_rad = np.deg2rad(angle)

    M = cv2.moments(contour)
    if M['m00'] != 0:
        center = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation matrix to each point in the contour
    rotated_contour = []
    for point in contour:
        rotated_point = np.dot(rotation_matrix, np.array([point[0][0], point[0][1], 1]))
        rotated_contour.append(rotated_point)

    # Convert to an array with the same shape as the input contour
    rotated_contour = np.array(rotated_contour, dtype=np.int32).reshape(-1, 1, 2)

    return rotated_contour

def rotate_main(image, contour, sweep, center=None):
    # Convert angle from degrees to radians

    padding = 500 
    x, y, w, h = cv2.boundingRect(contour)
    channels = image.shape[2] if len(image.shape) == 3 else 1
    blank_image = np.zeros((h+padding, w+padding, 3), dtype=image.dtype)
    offset_contour = contour - [x, y] + [int(padding/2), int(padding/2)]
    # cv2.drawContours(blank_image, [offset_contour], -1, (255, 255, 255), thickness=10)
    offset = [int(padding/2), int(padding/2)]
    # cv2.rectangle(blank_image, tuple(offset), (offset[0] + w, offset[1] + h), (255, 255, 255), thickness=10)
    
    blank_image, area = measure_area(offset, offset_contour, blank_image, h, w, True)

    # cv2.imshow("Original: " + str(area), blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    angle_area_dict = {}

    for angle in range(-sweep, sweep) : 

        rotated_contour = rotate_contour(contour, angle)
        x, y, w, h = cv2.boundingRect(rotated_contour)
        # channels = image.shape[2] if len(image.shape) == 3 else 1
        # blank_image = np.zeros((h+padding, w+padding, 3), dtype=image.dtype)
        rot_offset_contour = rotated_contour - [x, y] + [int(padding/2), int(padding/2)]
        # cv2.drawContours(blank_image, [rot_offset_contour], -1, (255,255,255), thickness=10)
        offset = [int(padding/2), int(padding/2)]
        # cv2.rectangle(blank_image, tuple(offset), (offset[0] + w, offset[1] + h), (255, 255, 255), thickness=10)
        # cv2.rectangle(blank_image, (x, y), (x+w, y+h), (255, 255, 0), thickness=10)

        blank_image, area = measure_area(offset, rot_offset_contour, blank_image, h, w, False)
        # print(angle, area)
        angle_area_dict[angle] = area

        # cv2.imshow("Rotated: " + str(area), blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    # min_area = min(angle_area_dict, key=angle_area_dict.get)
    # optimal_angle = min(angle_area_dict)

    # print(angle_area_dict)
        
    optimal_angle = min(angle_area_dict, key=angle_area_dict.get)
    min_area = angle_area_dict[optimal_angle]

    # Print the result
    print(f"The minimum area is {min_area} at angle {optimal_angle} degrees.")

    optimal_rotated_contour = rotate_contour(contour, optimal_angle)
    
    x, y, w, h = cv2.boundingRect(optimal_rotated_contour)
    channels = image.shape[2] if len(image.shape) == 3 else 1
    blank_image = np.zeros((h+padding, w+padding, 3), dtype=image.dtype)
    opt_rot_offset_contour = optimal_rotated_contour - [x, y] + [int(padding/2), int(padding/2)]
    # cv2.drawContours(blank_image, [opt_rot_offset_contour], -1, (255,255,255), thickness=10)
    offset = [int(padding/2), int(padding/2)]
    # cv2.rectangle(blank_image, tuple(offset), (offset[0] + w, offset[1] + h), (255, 255, 255), thickness=10)
    # cv2.rectangle(blank_image, (x, y), (x+w, y+h), (255, 255, 0), thickness=10)
    blank_image, area = measure_area(offset, rot_offset_contour, blank_image, h, w, False)
    # cv2.imshow("Optimal Rotation: " + str(min_area), blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return optimal_rotated_contour


def measure_area(offset, contour, image, h, w, draw) : 

    horizontal_distances = []
    area = 0
    for i in range(offset[1], offset[1] + h + 1, int(h / 10)):
        point1 = (offset[0], i)
        point2 = (offset[0] + w, i)
        # cv2.line(blank_image, point1, point2, (255, 255, 0), thickness=2)

        # Find intersection points of the line with the contour
        intersections = []
        for pt in contour :
            if pt[0][1] == i:
                if offset[1] <= pt[0][0] <= offset[1] + w:
                    intersections.append((pt[0][0], pt[0][1]))
 
        # Draw circles at intersection points
        for intersection in intersections:
            if draw : cv2.circle(image, intersection, 5, (255, 255, 255), 30)

        # Calculate horizontal distances
        if len(intersections) > 1: 
            horizontal_distances.append(abs(intersections[0][0] - intersections[-1][0]))
            point1 = (offset[0], i)
            point2 = intersections[0]
            if draw: cv2.line(image, point1, point2, (0 , 255, 0), thickness=5)
            point3 = intersections[0]
            point4 = intersections[-1]
            if draw: cv2.line(image, point3, point4, (255 , 0, 0), thickness=5)
            point5 = intersections[-1]
            point6 = (offset[0] + w, i)
            if draw: cv2.line(image, point5, point6, (0, 0, 255), thickness=5)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            font_thickness = 5 

            # Line from point1 to point2
            dist_left = abs(point2[0] - point1[0])
            text1 = f"{dist_left}"
            if draw: cv2.putText(image, text1, (int((point1[0] + point2[0]) / 2), i - 10), font, font_scale, (255, 255, 255), font_thickness)
            # Line from point3 to point4
            dist_middle = abs(point4[0] - point3[0])
            text2 = f"{dist_middle}"
            if draw: cv2.putText(image, text2, (int((point3[0] + point4[0]) / 2), i - 10), font, font_scale, (255, 255, 255), font_thickness)

            # Line from point5 to point6
            dist_right = abs(point6[0] - point5[0])
            text3 = f"{dist_right}" 
            if draw: cv2.putText(image, text3, (int((point5[0] + point6[0]) / 2), i - 10), font, font_scale, (255, 255, 255), font_thickness)

            area += (dist_left + dist_right)


    return image, area