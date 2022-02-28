from HALIntelT265 import HALIntelT265
import cv2
import numpy as np

def main(args=None):
    
    hal_intel = HALIntelT265(mode="separated_rect_with_disparity")

    # Set up an OpenCV window to visualize the results
    WINDOW_TITLE1 = 'Realsense - Raw Image Left'
    cv2.namedWindow(WINDOW_TITLE1, cv2.WINDOW_NORMAL)

    WINDOW_TITLE2 = 'Realsense - Raw Image Right'
    cv2.namedWindow(WINDOW_TITLE2, cv2.WINDOW_NORMAL)

    WINDOW_TITLE3 = 'Realsense - Disparity Map'
    cv2.namedWindow(WINDOW_TITLE3, cv2.WINDOW_NORMAL)
    while True:
        image = hal_intel.get_full_stack()

        if image is not None:
            cv2.imshow(WINDOW_TITLE1, image["left"])

            cv2.imshow(WINDOW_TITLE2, image["right"])

            cv2.imshow(WINDOW_TITLE3, image["disp"])
            key = cv2.waitKey(1)


if __name__ == '__main__':
    main()
