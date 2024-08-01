import cv2
import numpy as np
import os

# def filter_yellow(image):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # t = 50
    
#     lower_green = np.array([10, 50, 100])
#     upper_green = np.array([100, 255, 255])
#     mask = cv2.inRange(hsv_image, lower_green, upper_green)
#     yellow_parts = cv2.bitwise_and(image, image, mask=mask)
#     # cv2.imshow('Original Image', image)
#     # cv2.imshow('Yellow Mask', mask)
#     # # cv2.imshow('Filtered Yellow Parts', yellow_parts)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     lower_green = np.array([0, 100, 50])
#     upper_green = np.array([100, 255, 200])
#     mask = cv2.inRange(image, lower_green, upper_green)
#     # cv2.imshow('Green Mask', mask)
#     green_parts = cv2.bitwise_and(image, image, mask=mask)

#     both_mask = cv2.bitwise_or(yellow_parts, green_parts)
    
#     # cv2.imshow('Original Image', image)
#     # cv2.imshow('Both Mask', both_mask)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     return both_mask

def filter_black_areas(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 130])  # Adjust the upper bound as needed
    # Create a mask for black areas
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    # blurred_mask = cv2.GaussianBlur(black_mask, (5, 5), 0)  # Adjust kernel size as needed
 
    
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Black Mask', cv2.bitwise_not(black_mask))
    # cv2.imshow('Filtered Black Parts', black_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(  )
    return black_mask

def find_sheet(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    lower_green = np.array([10, 50, 100])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_parts = cv2.bitwise_and(image, image, mask=mask)

    lower_green = np.array([0, 100, 50])
    upper_green = np.array([100, 255, 200])
    mask = cv2.inRange(image, lower_green, upper_green)
    green_parts = cv2.bitwise_and(image, image, mask=mask)
    both_mask = cv2.bitwise_or(yellow_parts, green_parts)

    gray_image = cv2.cvtColor(both_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                
        # Apply the mask to the original image
        filtered_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Display the results
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Mask', mask)
        # cv2.imshow('Filtered Image', filtered_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return filtered_image
    else:
        print("No contours found.")
        return None

def find_corner_coords(img) : 

    # print("here")
    # Load the image
    # img = cv2.imread(img_path)
    black_mask = filter_black_areas(img)

    kernel_size=(2, 2)
    kernel = np.ones(kernel_size, np.uint8)
    # dilated_image = cv2.dilate(black_mask, kernel, iterations=3)
    blurImg = cv2.blur(black_mask, (10,10))  
    # cv2.imshow("blur", blurImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("black_mask", black_mask)
    # cv2.waitKey(0)

    # Detect contours
    contours, hierarchy = cv2.findContours(image=blurImg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Define circle parameters
    circle_radius = 20
    circle_color = (0, 0, 255)  # Red color in BGR
    text_color = (255, 0, 255)  # White color for text

    # List to store contours with 6 points and their areas
    contours_with_area = []

    min_area = 1000
    w, h, d = img.shape
    area_sheet = w*h


    # Loop through contours and filter by vertices
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01  * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices == 6:

            area = cv2.contourArea(contour)

            if  area > min_area and area < (area_sheet*0.2):
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

    # Sort contours by area in descending order and get the top 4 largest contours
    # contours_with_area = sorted(contours_with_area, key=lambda x: x[1], reverse=True)

    contour_count = 0
    w, h, d = img.shape
    area_sheet = w*h
   
    mean_area = np.mean([area for contour, area in contours_with_area])
    # Sort the contours by their deviation from the mean area
    # sorted_contours = sorted(contours_with_area, key=lambda x: abs(x[1] - mean_area))
    sorted_contours = sorted(contours_with_area, key=lambda x: x[1])


    # for i in sorted_contours : 
    #     print(i[1])
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

    pts_dst = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype="float32")
    # top left, top right, bottom left, bottom right

    for point_src in pts_dst:
        # Compute the Euclidean distances
        distances = np.linalg.norm(corners - point_src, axis=1)
        closest_index = np.argmin(distances)
        # print("Closest: " + str(corners[closest_index]))
        sorted_corners = np.vstack([sorted_corners, corners[closest_index]])
        # sorted_corners = np.append(sorted_corners,corners[closest_index])
        # print("Appending: " + str(corners[closest_index]))
        corners = np.delete(corners, closest_index, 0)

    offset = 50
    sorted_corners = sorted_corners + np.array([[-offset, -offset], [offset, -offset], [-offset, offset], [offset, offset]])

    # print("Src: " + str(corners))
    # print("Corners: " + str(corners))
    # print("Sorted Corners: ")
    # for i in sorted_corners : 
    #     print(str(i) + ": " + str(i[0] + (width*i[1])))
    # print("Dest: " +   str(pts_dst))
 
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(sorted_corners), np.float32(pts_dst))

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, M, (width, height))

    return warped_image

def scale_img(img) : 
    sheet_filtered = find_sheet(img)
    corners = find_corner_coords(sheet_filtered)
    warped = transform_sheet(img, corners)
    return warped


if __name__ == "__main__":

    # Define the folder containing the images
    folder_path = '/Users/zachmizrachi/Documents/Documents - Zach’s MacBook Pro (2) - 1/Corn Ear Labeling/Popcorn Images/Ears'

    # Initialize an empty list to store image information
    # image_data = []

    image_names = [
        "IMG_9785.jpeg",
        "IMG_9655.jpeg",
        "IMG_9655.jpeg",
        "IMG_9602.jpeg",
        "IMG_9602.jpeg",
        "IMG_9841.jpeg",
        "IMG_9841.jpeg",
        "IMG_9529.jpeg",
        "IMG_9943.jpeg",
        "IMG_9943.jpeg",
        "IMG_9934.jpeg"
    ]   

    # Loop through the images in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):

        if filename not in image_names : continue
    
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


