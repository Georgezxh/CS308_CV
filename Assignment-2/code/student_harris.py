from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    m = image.shape[0]
    n = image.shape[1]
    # 1. Image derivatives
    Ix = cv2.Sobel(image,cv2.CV_64F,1,0)
    Iy = cv2.Sobel(image,cv2.CV_64F,0,1)
    # 2. Square of derivatives
    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)
    kernel = cv2.getGaussianKernel(3,1)
    # 3. Gaussian filter
    Ixx = cv2.filter2D(Ixx, ddepth=-1, kernel=kernel)
    Ixy = cv2.filter2D(Ixy, ddepth=-1, kernel=kernel)
    Iyy = cv2.filter2D(Iyy, ddepth=-1, kernel=kernel)
    # 4. Cornerness function
    # A numpy array of shape (m,n,1)
    r_harris = np.multiply(Ixx, Iyy) - np.square(Ixy) - 0.05*np.square(Ixx + Iyy)
    # Remove interest points that are too close to a border
    larger = np.where(r_harris > np.abs(3*np.mean(r_harris)))
    larger = np.transpose(larger)
    half = feature_width/2
    no_border = []
    for i in range(len(larger)):
        if larger[i][0] > half and larger[i][0] < m - half:
             if larger[i][1] > half and larger[i][1] < n - half:
                     no_border.append(larger[i])
             

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    num_return = 2000
    tri = np.zeros((len(no_border),3))
    y = np.zeros(len(no_border))
    x = np.zeros(len(no_border))
    strength = np.zeros(len(no_border))
    for i in range(len(no_border)):
        y_idx = no_border[i][0]
        x_idx = no_border[i][1]
        val = r_harris[y_idx][x_idx]
        #print(val)
        tri[i][0] = val
        tri[i][1] = y_idx
        tri[i][2] = x_idx
    
    tri = sorted(tri,key=lambda x:x[0],reverse=True)
    for i in range(len(tri)):
        strength[i] = tri[i][0]
        y[i] = tri[i][1]
        x[i] = tri[i][2]
    four = np.zeros((len(no_border),4))

    four[0] = np.array([tri[0][0],tri[0][1],tri[0][2],m*m + n*n + 1])

    for i in range(len(tri) - 1):
        stronger = np.where(0.9*strength[i+1] < strength)
        #stronger = np.transpose(stronger)
        stronger = stronger[0]
        stronger = np.array(stronger)
        r = np.square(x[i+1] - x[stronger]) + np.square(y[i+1] - y[stronger])
        #print(x[a])
        r.sort()
        #第一个元素是0，与本身距离
        radius = r[1]
        four[i] = np.array([radius,tri[i+1][0],tri[i+1][1],tri[i+1][2]])
    four = sorted(four,key=lambda x:x[0],reverse=True)
    y = np.zeros(num_return)
    x = np.zeros(num_return)
    for i in range(num_return):
        x[i] = four[i][3]
        y[i] = four[i][2]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations


