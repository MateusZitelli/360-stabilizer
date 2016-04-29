import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2

from kmeans.dataset import Dataset
from kmeans.bisectingKmeans import BisectingKmeans

class Stabilizer:
  def __init__(self, videoPath, ratio, reprojThresh):
    self.videoPath = videoPath
    self.vidcap = cv2.VideoCapture(videoPath)
    initialFrame = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.videoSize = (int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, initialFrame / 6)
    self.ratio = ratio
    self.reprojThresh = reprojThresh
    self.isv3 = imutils.is_cv3()

  def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("FlannBased")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
      # ensure the distance is within a certain ratio of each
      # other (i.e. Lowe's ratio test)
      matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
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

  def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
      # only process the match if the keypoint was successfully
      # matched
      if s == 1:
        # draw the match
        ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
        ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis

  def drawOffsets(self, image, kpsA, kpsB, matches, status):
    (hI, wI) = image.shape[:2]
    vis = np.zeros((hI, wI, 3), dtype="uint8")
    vis[:, :] = image
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
      # only process the match if the keypoint was successfully
      # matched
      if s == 1:
        ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
        ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
        cv2.line(vis, ptA, ptB, (0, 0, 255), 1)
    return vis

  def drawGroups(self, groups, image):
    (hI, wI) = image.shape[:2]
    vis = np.zeros((hI, wI, 3), dtype="uint8")
    vis[:, :] = image
    limits = (self.offsetsDataset.maximuns, self.offsetsDataset.minimuns)
    # loop over the matches
    for i, group in enumerate(groups):
      color = [0, 0, 0]
      color[i] = 255
      for offset in group.getCoveredDataset(limits=limits):
        p0 = self.getMercatorCoords(offset[:2])
        p1 = self.getMercatorCoords(offset[:2]) + self.getMercatorCoords(offset[2:])
        ptA = (int(p0[0]), int(p0[1]))
        ptB = (int(p1[0]), int(p1[1]))
        cv2.line(vis, ptA, ptB, color, 1)
    return vis

  def getOffsets(self, kpsA, kpsB, matches, status):
    return [np.concatenate((
      self.getSphericalCoords(kpsB[trainIdx]),
      self.getSphericalCoords(kpsB[trainIdx]) - \
      self.getSphericalCoords(kpsA[queryIdx])))
      for ((trainIdx, queryIdx), s) in zip(matches, status) if s == 1]

  def getOffsetsGrouped(self, groups, image, kpsA, kpsB, matches, status):
    offsets = self.getOffsets(kpsA, kpsB, matches, status)
    self.offsetsDataset = Dataset(data=offsets)
    k = BisectingKmeans(dataset=self.offsetsDataset, k=groups, trials=5, maxRounds=10, key=lambda x: [x[0], x[2], x[3]])
    k.run()
    return k.means

  def detectAndDescribe(self, image):
    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)

  def fixOffset(self, offset, img):
    size = img.shape
    finalImg = np.ndarray(size)
    indices = np.indices((self.videoSize[0],self.videoSize[1])).swapaxes(0,2).swapaxes(0,1)
    indices = np.around(indices, decimals=1)
    indices.shape = (self.videoSize[1] * self.videoSize[0], 2)
    phi = 2 * np.arctan(np.exp(indices[:, 1] / self.videoSize[1])) - 1/2 * np.pi - offset[0]
    lamb = indices[:, 0] - offset[1]
    x = lamb
    y = np.log(np.tan(np.pi / 4 + 1/2 * phi)) * self.videoSize[1]
    finalIdx = np.ndarray((self.videoSize[1] * self.videoSize[0], 2))
    finalIdx = np.around(finalIdx, decimals=1).astype(int)
    finalIdx[:, 1] = y % self.videoSize[1]
    finalIdx[:, 0] = x % self.videoSize[0]
    finalImg[indices[:,1], indices[:,0]] = img[finalIdx[:,1], finalIdx[:,0]]
    return finalImg

  def getSphericalCoords(self, position):
    if isinstance(position, np.matrix):
      p0 = position[:, 0]
      p1 = position[:, 1]
    else:
      p0 = position[1]
      p1 = position[0]
    phi = 2 * np.arctan(np.exp(p0 / self.videoSize[1])) - 1/2 * np.pi
    lamb = p1
    return np.array((phi, lamb))

  def getMercatorCoords(self, position):
    x = position[1]
    y = np.log(np.tan(np.pi / 4 + 1/2 * position[0])) * self.videoSize[1]
    return np.array([x, y])

  def run(self):
    def unnormalize(value, limits):
      maximuns, minimuns = limits

      def unnormalizeValue(ceil, floor, value):
        return value * (ceil - floor) + floor

      unnormalizedData =  [
        unnormalizeValue(maximuns[i], minimuns[i], d)
        for i, d in enumerate(value)]
      return unnormalizedData

    # Frames count
    count = 0
    success = True

    # infos of actual frame
    actualFrame = None
    actualKeyPoints = None
    actualFeatures = None

    totalOffset = np.zeros(2)
    while success:
      # Update last frame information
      lastFrame, lastKeyPoints, lastFeatures = actualFrame, actualKeyPoints, actualFeatures

      # Extract frame and get informations
      success, actualFrame = self.vidcap.read()
      actualKeyPoints, actualFeatures = self.detectAndDescribe(actualFrame)

      print('Read a new frame: ', success)
      if lastFrame is not None and actualFrame is not None:
        M = self.matchKeypoints(lastKeyPoints, actualKeyPoints, lastFeatures, actualFeatures)
        (matches, H, status) = M
        groups = self.getOffsetsGrouped(2, lastFrame, lastKeyPoints, actualKeyPoints, matches, status)
        print([g.position for g in groups])
        groupInTop = max(groups, key=lambda group: group.position[0])
        limits = (self.offsetsDataset.maximuns, self.offsetsDataset.minimuns)
        offset = unnormalize(groupInTop.position, limits)
        totalOffset = totalOffset - offset[2:]
        vis2 = self.drawGroups(groups, actualFrame)
        vis = self.fixOffset(totalOffset, actualFrame)
        cv2.imwrite("results/offset%d.png" % count, vis2)     # save frame as JPEG file
        cv2.imwrite("results/frame%d.png" % count, vis)     # save frame as JPEG file
      count += 1

if __name__ == '__main__':
  stab = Stabilizer('./Curtis-Biotech_RAW-output_360.mp4', 1.0, 20.0)
  stab.run()
