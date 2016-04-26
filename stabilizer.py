import numpy as np
import imutils
import cv2

class Stabilizer:
  def __init__(self, videoPath, ratio, reprojThresh):
    self.videoPath = videoPath
    self.vidcap = cv2.VideoCapture(videoPath)
    initialFrame = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, initialFrame / 2)
    self.ratio = ratio
    self.reprojThresh = reprojThresh
    self.isv3 = imutils.is_cv3()

  def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
      # ensure the distance is within a certain ratio of each
      # other (i.e. Lowe's ratio test)
      if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
      # construct the two sets of points
      ptsA = np.float32([kpsA[i] for (_, i) in matches])
      ptsB = np.float32([kpsB[i] for (i, _) in matches])

      # compute the homography between the two sets of points
      (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
        self.reprojThresh)

      # return the matches along with the homograpy matrix
      # and status of each matched point
      return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None


  def detectAndDescribe(self, image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we are using OpenCV 3.X
    if self.isv3:
      # detect and extract features from the image
      descriptor = cv2.xfeatures2d.SIFT_create()
      (kps, features) = descriptor.detectAndCompute(image, None)

    # otherwise, we are using OpenCV 2.4.X
    else:
      # detect keypoints in the image
      detector = cv2.FeatureDetector_create("SIFT")
      kps = detector.detect(gray)

      # extract features from the image
      extractor = cv2.DescriptorExtractor_create("SIFT")
      (kps, features) = extractor.compute(gray, kps)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)

  def run(self):
    # Frames count
    count = 0
    success = True

    # infos of actual frame
    actualFrame = None
    actualKeyPoints = None
    actualFeatures = None

    while success:
      # Update last frame information
      lastFrame, lastKeyPoints, lastFeatures = actualFrame, actualKeyPoints, actualFeatures

      # Extract frame and get informations
      success, actualFrame = self.vidcap.read()
      actualKeyPoints, actualFeatures = self.detectAndDescribe(actualFrame)

      print 'Read a new frame: ', success
      if lastFrame is not None and actualFrame is not None:
        M = self.matchKeypoints(lastKeyPoints, actualKeyPoints, lastFeatures, actualFeatures)
        print(M)
      cv2.imwrite("frame%d.jpg" % count, actualFrame)     # save frame as JPEG file
      count += 1

if __name__ == '__main__':
  stab = Stabilizer('./Curtis-Biotech_RAW-output_360.mp4', 0.75, 4.0)
  stab.run()
