import cv2
import numpy as np
import os

dataDir = 'images'


# A reference table with some useful functions:
# 
# | Function | Description |
# | ------------- | ------------- |
# | **Low-level NumPy** | |
# | `x = np.array(list)` | Converts Python list to `ndarray` |
# | `x = np.zeros((80, 80, 3))` | Create an array 80x80x3 of zeros (i.e., a black image). |
# | `x = np.ones((80, 80, 3))` | Same with ones. |
# | `x = np.random.rand((80, 80, 3))` | Same but each value is an uniform sample from [0,1]. |
# | `x = np.random.randn((80, 80, 3))` | Same but each value is a Gaussian sample from N(0,1). |
# | `print(x.shape)` | Print the shape of the `ndarray`. |
# | **Arithmetic** | |
# | `x[:, :, 0]` | Access the first slice of the third-axis (i.e., if `x` is an image with format BGR, this would be the blue channel. |
# | `x += 50` | Adds 50 to all pixels. |
# | `x[:, :, 1] *= 0.5` | Divides the green channel by 2. |
# | **OpenCV2 basic functions** | |
# | `img = cv2.imread(filename)` | Opens the image from the disk given by filename as a `ndarray`. |
# | `cv2.imwrite(filename, img)` | Save the given image in the disk with the given filename. |
# | `cv2.imshow(window_name, img)` | Open the given image in a window. |
# | `cv2.destroyWindow(window_name)` | Destroys the window. |
# | **OpenCV2 color conversion** | |
# | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | Converts the color format. |
# | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | |
# | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | |
# | **OpenCV2 user interaction** | |
# | `cv2.setMouseCallback(window_name, callback)` | Calls the given callback function whenver user interacts with the window. |
# | `cv2.selectROI(window_name, img)` | Asks the user to select a part of the image and returns that. |
# | `key = cv2.waitKey(0)` | Waits until the user presses a key. |
# | `key = cv2.waitKey(delay)` | Wait until the user presses a key or a certain delay passes (in seconds). |

# ### 1. Images – read, write and display; ROIs

# In[3]:


# Opening an image
img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))

# Showing the image
#cv2.imshow("ml.jpg", img)

#cv2.waitKey(0)

# Close the window after user pressed a key
#cv2.destroyWindow("ml.jpg")


# In[4]:


# Check image size
h, w, c = img.shape
print(f'height: {h}')
print(f'width: {w}')
print(f'channels: {c}')


# In[5]:


# Saving image in bmp format
cv2.imwrite('ml_new.bmp', img)


# Exercise 1.1 - When the user moves the mouse, print the coordinate and RGB component of the pixel under the cursor. When the user clicks on a pixel, modify that pixel to blue.

# In[ ]:


# TODO


# Exercise 1.2 - Allow the user to select a region of interest (ROI) in the image, by clicking on two points that identify two opposite corners of the selected ROI, and save the ROI into another file.

# In[ ]:


# TODO


# ### 2. Images – representation, grayscale and color, color spaces

# In[ ]:


# Create a white image
m = np.ones((100,200,1), np.uint8)

# Change the intensity to 100
m = m * 100

# Display the image
cv2.imshow('Grayscale image', m)
cv2.waitKey(0)
cv2.destroyWindow('Grayscale image')


# In[ ]:


# Draw a line with thickness of 5 px
cv2.line(m, (0,0), (100,200), 255, 5)
cv2.line(m, (200, 0), (0, 100), 255, 5)
cv2.imshow('Grayscale image with diagonals', m)
cv2.waitKey(0)
cv2.destroyWindow('Grayscale image with diagonals')


# Exercise 2.1 - Create a color image with 100(lines)x200(columns) pixels with yellow color. Then draw two diagonal lines across the image, one in red color, the other in blue color. Display the image.

# In[ ]:


# TODO


# Exercise 2.2 - Read any color image, in RGB format, display it in one window, convert it to grayscale, display the grayscale image in another window and save the grayscale image to a different file

# In[ ]:


# TODO


# Exercise 2.3 - Split the 3 RGB channels and show each channel in a separate window. Add a constant value to one of the channels, merge the channels into a new color image and show the resulting image.

# In[ ]:


# TODO


# Exercise 2.4 - Convert the image to HSV, split the 3 HSV channels and show each channel in a separate window. Add a constant value to the saturation channel, merge the channels into a new color image and show the resulting image.

# In[ ]:


# TODO


# ### 3. Video – acquisition and simple processing

# In[ ]:


# Define a VideoCapture Object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_nr = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('webcam', frame)

    # Wait for user to press s to save frame
    if cv2.waitKey(1) == ord('s'):
        frame_name = 'frame' + str(frame_nr) + '.png'
        cv2.imwrite(frame_name, frame)
        cv2.imshow("Saved frame: " + frame_name, frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Saved frame: " + frame_name)

    # Wait for user to press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


# Exercise 3.1 - Using the previous example as the baseline, implement a script that acquires the video from the webcam, converts it to grayscale, and shows the frames in binary format (i.e. the intensity of each pixel is 0 or 255); use a threshold value of 128.

# In[ ]:


# TODO


# Exercise 3.2 - Implement a simple detection/tracking algorithm for colored objects, using the following steps:
# 1) take each frame of the video;
# 2) convert from BGR to HSV color-space;
# 3) threshold the HSV image for a range of color values (creating a binary mask);
# 4) erase everything in the original image except the mask.

# In[ ]:


# TODO

