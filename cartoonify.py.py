import cv2

# Load input image
img = cv2.imread('test.jpg')
cv2.imshow("Original RGB Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Downsample the image to reduce processing load
img_small = cv2.pyrDown(img)
cv2.imshow("Downsampled Image", img_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply bilateral filter 7 times to smooth colors while preserving edges
for i in range(7):
    img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)

# Upsample to original size
img2 = cv2.pyrUp(img_small)
cv2.imshow("Bilateral Filter Applied", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale for edge detection
img_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
cv2.imshow("Grayscale Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply median blur to reduce noise before edge detection
img_blur = cv2.medianBlur(img_gray, 7)
cv2.imshow("Median Blurred Image", img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect edges using adaptive thresholding
img_edge = cv2.adaptiveThreshold(
    img_blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    7, 2
)
cv2.imshow("Edge Mask", img_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Combine color image and edge mask
cartoon_img = cv2.bitwise_and(img_edge, img2)
cv2.imshow("Cartoonified Image (RGB)", cartoon_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save cartoonified image
cv2.imwrite("cartoon_img.jpg", cartoon_img)
