import cv2
import numpy as np
from math import pi
from math import atan2

#with help from : https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking/blob/master/Part%20I%20-%20Simple%20Lane%20Detection/P1.ipynb

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)  
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 0

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(img, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img,img)
    return masked_image

#used below
def get_slope(x1,y1,x2,y2):
    if not (x2-x1) == 0:
        return (y2-y1)/(x2-x1)
    else:
        return 0

def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept 
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    first_frame = 0
    cache = 0
    
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    alpha =0.2 
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function
    if not lines is None:
        for line in lines:
            #1
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                if slope > det_slope:
                    r_slope.append(slope)
                    r_lane.append(line)
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)
    else:
        return 0
    
        #2
        y_global_min = min(y1,y2,y_global_min)
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        #print ('no lane detected')
        return 1
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1  
    
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")
   
    """if first_frame == 1:
        next_frame = current_frame
        first_frame = 0        
    else :
        prev_frame = cache
        next_frame = (1-alpha)*prev_frame+alpha*current_frame
        """
    next_frame = current_frame        
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)
    med_top = (int(next_frame[0]) + int(next_frame[4]))/2
    med_bot = (int(next_frame[2]) + int(next_frame[6]))/2
    med_htop = (int(next_frame[1]) + int(next_frame[5]))/2
    med_hbot = (int(next_frame[3]) + int(next_frame[7]))/2

    if not (med_hbot-med_htop) == 0:
        angle = atan2((med_top - med_bot),(med_hbot-med_htop))
    else:
        angle = 0
    
    cv2.arrowedLine(img, (med_bot, med_hbot), (med_top,med_htop),color,thickness)
    cache = next_frame

    return angle

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(image=img,rho=rho,theta=theta,
        minLineLength=min_line_len,threshold=threshold ,maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    angle = draw_lines(line_img,lines)
    return line_img,angle

def weighted_img(img, initial_img, alpha=0.8, beta=1., lam=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img *
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lam)

def process_image(image):

    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b
    
    med_filter = cv2.medianBlur(gray_image,15)
    
    #cv2.imshow('median',med_filter)

    #same as quiz values
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(med_filter,low_threshold,high_threshold)

    #cv2.imshow('canny',canny_edges)
    
    imshape = image.shape
    lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)    
    #cv2.imshow('canndy',roi_image)
    #ar = np.zeros_like(canny_edges)    
        
    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 4
    theta = np.pi*1/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 30
    max_line_gap = 1000
    #my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video
    hough = hough_lines(canny_edges, rho, theta, threshold, min_line_len, max_line_gap)
    line_image = hough[0]
    angle = hough[1]    
    result = weighted_img(line_image, image, alpha=0.8, beta=1., lam=0.)
    return result,angle

def get_angle(img):
    processed = process_image(img)
    angle = processed[1]
    if angle < pi/4:
        return -1
    elif angle > pi/4:
        return 1
    else:
        return 0

    """
image = cv2.imread('sidewalk-raleigh.jpg')
#image = cv2.imread('test.jpg')
processed = process_image(image)
cv2.imshow('processed',processed[0])
angle = processed[1]
print 'angle: ' + str(angle) + ' radians'
"""
