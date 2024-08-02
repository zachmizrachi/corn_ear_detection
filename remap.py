import cv2
import numpy as np
import os


def filter_black_areas(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 130])  # Adjust the upper bound as needed
    # Create a mask for black areas
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    return black_mask

def find_sheet(image):

    # filter for green (paper) in hsv space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    lower_green = np.array([10, 50, 100])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_parts = cv2.bitwise_and(image, image, mask=mask)

    # filter for green (paper) in rgb space
    lower_green = np.array([0, 100, 50])
    upper_green = np.array([100, 255, 200])
    mask = cv2.inRange(image, lower_green, upper_green)
    green_parts = cv2.bitwise_and(image, image, mask=mask)
    
    # combine the two filters
    both_mask = cv2.bitwise_or(yellow_parts, green_parts)
    gray_image = cv2.cvtColor(both_mask, cv2.COLOR_BGR2GRAY)

    # find the contours
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    if contours:
        largest_contour = max(contours, key=cv2.contourArea) # the page will be the largest contour in the filtered image
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        filtered_image = cv2.bitwise_and(image, image, mask=mask)
        return filtered_image
    else:
        print("No contours found.")
        return None

def find_corner_coords(img) : 

    black_mask = filter_black_areas(img)
    kernel_size=(2, 2)
    kernel = np.ones(kernel_size, np.uint8)
    blurImg = cv2.blur(black_mask, (10,10))  # helps to smooth contour detections (not over detect)
   
    contours, hierarchy = cv2.findContours(image=blurImg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Define circle parameters
    circle_radius = 20
    text_color = (255, 0, 255) 

    # list to store contours that meet corner requirements
    contours_with_area = []

    min_area = 1000
    w, h, d = img.shape
    area_sheet = w*h

    # Loop through contours and filter by vertices
    for contour in contours:
        epsilon = 0.01  * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Approximate the contour to a polygon
        num_vertices = len(approx)
        
        if num_vertices == 6: # the page corner markers have 6 vertices 
            area = cv2.contourArea(contour)
            if  area > min_area and area < (area_sheet*0.2): # approximation for size of corner
                contours_with_area.append((approx, area))
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0

                    # text = f"Vertices: {len(approx)}"
                    # cv2.putText(img, text, (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 1, cv2.LINE_AA)
                
                text = f" Area: {area}"
                cv2.putText(img, text, (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 5 , cv2.LINE_AA)
                cv2.drawContours(image=img, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

    # cv2.imshow("blur", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
                
    w, h, d = img.shape
    area_sheet = w*h
   
    # Sort the contours by size
    sorted_contours = sorted(contours_with_area, key=lambda x: x[1])

    # for i in sorted_contours :  print(i[1])
    # print("# of detections: " +  str(len(sorted_contours)))

    min_diff = float('inf')
    best_subset = []

    # sliding window approach to find the closest group of 4 (the four corners are closest in area)
    for i in range(len(sorted_contours) - 3): 
        current_subset = sorted_contours[i:i + 4] 
        # print(str(i) + " - " + str(i+4))
        current_areas = [area for _, area in current_subset]
        current_diff = current_areas[-1] - current_areas[0]
        # print(current_areas)
        # print(current_diff) 
        if abs(current_diff) < abs(min_diff) :
            min_diff = current_diff
            best_subset = current_subset

    # print('\n ')


    sheet_coordinates = np.empty((0, 2), dtype="float32")  # Initialize as an empty 2D array

    for contour, area in best_subset:

        cv2.drawContours(image=img, contours=[contour], contourIdx=-1, color=(255, 255, 0), thickness=4, lineType=cv2.LINE_AA)

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # Draw a circle at the centroid of the contour
        cv2.circle(img, (cx, cy), circle_radius, (255, 0, 0), -1)
        sheet_coordinates = np.vstack([sheet_coordinates, [cx, cy]])
 
        
    # Show the images
    # cv2.imshow("Detections", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return sheet_coordinates


def transform_sheet(image, corners) : 
    # Read the image
    height, width = image.shape[:2]

    sorted_corners = np.empty((0, 2), dtype="float32")
    # Define the four destination points (corners of the rectangle)

    pageDim = 2000

    pts_dst = np.array([[0, 0], [pageDim, 0], [0, pageDim], [pageDim, pageDim]], dtype="float32")
    # top left, top right, bottom left, bottom right

    # need to match the original corner marker to its respective corner (TL TR BL BR) before warp
    for point_src in pts_dst:
        distances = np.linalg.norm(corners - point_src, axis=1)
        closest_index = np.argmin(distances)
        # print("Closest: " + str(corners[closest_index]))
        sorted_corners = np.vstack([sorted_corners, corners[closest_index]])
        # sorted_corners = np.append(sorted_corners,corners[closest_index])
        # print("Appending: " + str(corners[closest_index]))
        corners = np.delete(corners, closest_index, 0)

    offset = 50
    sorted_corners = sorted_corners + np.array([[-offset, -offset], [offset, -offset], [-offset, offset], [offset, offset]])
 
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(sorted_corners), np.float32(pts_dst))
    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, M, (pageDim, pageDim))

    sorted_corners = sorted_corners.reshape(-1, 1, 2)  # Reshape for perspectiveTransform
    transformed_corners = cv2.perspectiveTransform(np.float32(sorted_corners), M)
    # Convert transformed corners back to a simple list of points
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Draw the transformed corners on the warped image
    for corner in transformed_corners:
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(warped_image, (x, y), 5, (255, 255, 0), 30)  # Draw green circles at each corner

    # Display the warped image with the corners
    # cv2.imshow('Warped Image with Corners', warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return warped_image

def scale_img(img) : 
    sheet_filtered = find_sheet(img)
    corners = find_corner_coords(sheet_filtered)
    warped = transform_sheet(img, corners)

    # top left, top right, bottom left, bottom right
    # 0         1          2            3
    # Calculate the distances between consecutive points
    # dTop = np.linalg.norm(corners[0] - corners[1])
    # dLeft = np.linalg.norm(corners[0] - corners[2])
    # dBot = np.linalg.norm(corners[2] - corners[3])
    # dRight = np.linalg.norm(corners[3] - corners[0])

    # print("base sheet width and height: " + str(warped.shape[0]) + ", " + str(warped.shape[1]))
    # print("Side lengths: " + str(dLeft) + " | " + str(dRight))
    # print("Top Bot lengths: " + str(dTop) + " | " + str(dBot))

    # Calculate the average lengths of the parallel sides
    topImage = 2000
    sideImage = 2000

    topbot_Real = 7.67 #195
    sides_Real = 9.64 #245

    ratio1 = topbot_Real / topImage
    ratio2 = sides_Real / sideImage

    finalRatio = (ratio1 + ratio2)/2

    # print("Ratio 1 and 2: " + str(ratio1) + " " + str(ratio2))
    # print("Final Ratio: " + str(finalRatio))
    # print("Top to bottom pixels: " + str(sideImage))

    # cv2.imshow("Warped", warped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return warped, finalRatio


if __name__ == "__main__":

    # Define the folder containing the images
    folder_path = '/Users/zachmizrachi/Documents/Documents - Zach’s MacBook Pro (2) - 1/Corn Ear Labeling/Popcorn Images/Ears'

    # Initialize an empty list to store image information

    # Loop through the images in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):
    
        image_path = os.path.join(folder_path, filename)
        # image_path =   "Popcorn Images/Ears/IMG_9737.jpeg"
        img = cv2.imread(image_path)
        # find_coords(img)

        sheet_filtered = find_sheet(img)

        # cv2.imshow("sheet", sheet_filtered)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        corners = find_corner_coords(sheet_filtered)
        warped = transform_sheet(img, corners)

        # cv2.imshow("warped", warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if idx == 10 :  
        #     cv2.destroyAllWindows()
        #     break
    
    # img = cv2.imread("Popcorn Images/Ears/IMG_9721.jpeg")
    # sheet_filtered = find_sheet(img)
    # find_corner_coords(sheet_filtered)


