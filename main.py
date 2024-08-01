import os
import cv2
import json
import numpy as np

from remap import *

def mask_yellow(image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([10, 40, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def detect_contours(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def detect_lines(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=100, maxLineGap=10)

    return lines

def draw_lines(image, lines):
    # Draw detected lines on the original image
    image_with_lines = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_with_lines

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def convert_and_dilate(image, kernel_size=(5, 5)):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply dilation to the grayscale image
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(gray_image, kernel, iterations=5)
    
    return gray_image, dilated_image


def contours(image, filename):
    # Find Canny edges
    edged = cv2.Canny(image, 100, 200)

    cv2.imwrite("edged.png", edged)
    # cv2.waitKey(0)

    # Finding Contours
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(edged, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("edged w contours", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]  # Select the four largest contours

    # for contour in contours : 
    #     print(cv2.contourArea(contour))

    # Sort contours by top-to-bottom difference (top two)
    contours = sorted(contours, key=lambda x: x[:, :, 1].max() - x[:, :, 1].min(), reverse=True)[:2]

    ret = np.array(np.zeros((2,4)))

    # Draw selected contours and mark highest and lowest points
    for idx, contour in enumerate(contours):
        # Calculate highest and lowest points of contour
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

        # Draw contour
        cv2.drawContours(image, [contour], -1, (255, 255, 0), thickness=10)

        # cv2.imshow("im", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Draw dots on highest and lowest points
        cv2.circle(image, ext_top, 5, (0, 0, 255), -1)
        cv2.circle(image, ext_bot, 5, (0, 0, 255), -1)

        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), thickness=10)

        # Draw horizontal lines connecting sides of bounding box with contour
        horizontal_distances = []
        for i in range(y, y + h + 1, int(h / 10)):
            point1 = (x, i)
            point2 = (x + w, i)
            cv2.line(image, point1, point2, (0, 255, 0), thickness=2)

            # Find intersection points of the line with the contour
            intersections = []
            for pt in contour:
                if pt[0][1] == i:
                    if x <= pt[0][0] <= x + w:
                        intersections.append((pt[0][0], pt[0][1]))

            # Draw circles at intersection points
            for intersection in intersections:
                cv2.circle(image, intersection, 5, (255, 0, 0), 30)

            # Calculate horizontal distances
            if len(intersections) > 1:
                horizontal_distances.append(abs(intersections[0][0] - intersections[-1][0]))

        # cv2.imwrite('Horizontal Intersections.png', image)
        # cv2.waitKey(0)

        # Draw vertical lines connecting sides of bounding box with contour
        vertical_distances = []
        for i in range(x, x + w + 1, int(w / 10)):
            point1 = (i, y)
            point2 = (i, y + h)
            cv2.line(image, point1, point2, (0, 255, 0), thickness=2)

            # Find intersection points of the line with the contour
            intersections = []
            for pt in contour:
                if pt[0][0] == i:
                    if y <= pt[0][1] <= y + h:
                        intersections.append((pt[0][0], pt[0][1]))

            # Draw circles at intersection points
            for intersection in intersections:
                cv2.circle(image, intersection, 5, (255, 0, 0), 20)

            # Calculate vertical distances
            if len(intersections) > 1:
                vertical_distances.append(abs(intersections[0][1] - intersections[-1][1]))

        # cv2.imshow("intersections", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite('detections/detect_' + str(filename), image)
        # cv2.waitKey(0)

        # Calculate average width and height based on horizontal and vertical distances
        avg_width = np.mean(horizontal_distances) if horizontal_distances else 0
        avg_height = np.mean(vertical_distances) if vertical_distances else 0

        ret[idx, 0] = avg_width
        ret[idx, 1] = w
        ret[idx, 2] = avg_height
        ret[idx, 3] = h

    # cv2.destroyAllWindows()

    return ret

# image_names = [
#     "IMG_9785.jpeg",
#     "IMG_9655.jpeg",
#     "IMG_9655.jpeg",
#     "IMG_9602.jpeg",
#     "IMG_9602.jpeg",
#     "IMG_9841.jpeg",
#     "IMG_9841.jpeg",
#     "IMG_9529.jpeg",
#     "IMG_9943.jpeg",
#     "IMG_9943.jpeg",
#     "IMG_9934.jpeg"
# ]

def main():

    # Define the folder containing the images
    folder_path = '/Users/zachmizrachi/Documents/Documents - Zach’s MacBook Pro (2) - 1/Corn Ear Labeling/Popcorn Images/Ears'

    # Initialize an empty list to store image information
    image_data = []

    # Loop through the images in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):

        # filename = "IMG_9853.jpeg"
       
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        # if filename not in image_names : continue

        height, width = img.shape[:2]

        if width > height:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 

        scaled_img = scale_img(img)

        # cv2.imshow("scaled_img", scaled_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        yellow_img = mask_yellow(scaled_img )
        cv2.imwrite("yellow.png", yellow_img)
    #     # cv2.waitKey(0)
        gray_image, dilated_image = convert_and_dilate(yellow_img)
        cv2.imwrite("dilated_image.png", dilated_image)
    #     # cv2.waitKey(0)
        blurred =  cv2.medianBlur(dilated_image, 51)
        cv2.imwrite("blurred.png", blurred)
    #     # cv2.waitKey(0)
        
        data = contours(blurred, filename)

        # Create a dictionary with image information
        for i in range(data.shape[0]): 
            image_info = {
                "name": str(filename) + "_" + str(i),
                "avg_width": data[i][0],
                "max_width": data[i][1],
                "avg_height": data[i][2],
                "max_height": data[i][3],
            }
            image_data.append(image_info)


        print("Finished image: " + str(filename))

    # Write the image data to a JSON file
    output_file = 'image_data.json'
    with open(output_file, 'w') as json_file:
        json.dump(image_data, json_file, indent=4)

    print(f"Image data has been written to {output_file} successfully.")

if __name__ == "__main__":
    main()
