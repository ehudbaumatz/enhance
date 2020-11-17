import cv2


def split_image_horizontally(path):

    img = cv2.imread(path)

    height, width = img.shape[:2]

    # Let's get the starting pixel coordiantes (top left of cropped top)
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(height), int(width * .5)
    cropped_left = img[start_row:end_row, start_col:end_col]


    # Let's get the starting pixel coordiantes (top left of cropped bottom)
    start_row, start_col = int(0), int(width * .5)
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height), int(width)
    cropped_right = img[start_row:end_row, start_col:end_col]

    return cropped_left, cropped_right