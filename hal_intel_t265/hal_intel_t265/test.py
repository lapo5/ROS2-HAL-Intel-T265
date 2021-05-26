from HALIntelT265 import HALIntelT265
import cv2
import numpy as np

def main(args=None):
    
    hal_intel = HALIntelT265()

    # Set up an OpenCV window to visualize the results
    WINDOW_TITLE1 = 'Realsense - Raw Images'
    cv2.namedWindow(WINDOW_TITLE1, cv2.WINDOW_NORMAL)

    WINDOW_TITLE2 = 'Realsense - Rectified Images'
    cv2.namedWindow(WINDOW_TITLE2, cv2.WINDOW_NORMAL)

    WINDOW_TITLE3 = 'Realsense - Disparity Map'
    cv2.namedWindow(WINDOW_TITLE3, cv2.WINDOW_NORMAL)
    while True:
        image = hal_intel.get_full_stack()

        if image is not None:
            displayer = np.hstack((hal_intel.frame_data["left"], hal_intel.frame_data["right"]))
            cv2.imshow(WINDOW_TITLE1, displayer)

            cv2.imshow(WINDOW_TITLE2, image)

            cv2.imshow(WINDOW_TITLE3, hal_intel.disparity_map)
            key = cv2.waitKey(1)
            if key == ord('s'): hal_intel.mode = "stack"
            if key == ord('o'): hal_intel.mode = "overlay"
            if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE1, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE2, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE3, cv2.WND_PROP_VISIBLE) < 1:
                break



if __name__ == '__main__':
    main()
