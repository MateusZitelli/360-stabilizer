import numpy as np
import cv2

class ObliqueEquirectangular:
  def __init__(self, image):
    def generateIndices():
      indices = np.indices(image.shape[:2]).swapaxes(0,2).swapaxes(0,1)
      indices.shape = (image.shape[1] * image.shape[0], 2)
      return indices

    def generateSphereIndices(indices):
      return self.XYtoSpherical(indices, 0)

    self.image = image
    self.imageSize = [x - 1 for x in self.image.shape[:2]]
    self.R = (image.shape[1])
    self.indices = generateIndices()
    self.sphereIndices = generateSphereIndices(self.indices)

  def sphericalToXY(self, points, h0, lamb0):
    (phi, lamb) = points[:, 0], points[:, 1]
    x = self.R * (lamb - lamb0)
    y = self.R * np.sin(phi)
    projection = np.ndarray(points.shape)
    projection[:, 1] = x
    projection[:, 0] = y
    return projection

  def getObliquePole(self, center0, center1):
    (phi0, lamb0) = center0
    (phi1, lamb1) = center1
    A = np.cos(phi0) * np.sin(phi1) * np.cos(lamb0) - np.sin(phi0) * np.cos(phi1) * np.cos(lamb1)
    B = np.sin(phi0) * np.cos(phi1) * np.sin(lamb1) - np.cos(phi0) * np.sin(phi1) * np.sin(lamb0)
    lambP = np.arctan2(A, B)
    phiP = np.arctan2(- np.cos(lambP - lamb0), np.tan(phi0))
    return (phiP, lambP)

  def sphericalToXYOblique(self, points, h0, center0, center1):
    (phi, lamb) = points[:, 0], points[:, 1]
    (phiP, lambP) = self.getObliquePole(center0, center1)
    (phiO, lambO) = (0, lambP + np.pi / 2)
    C = np.tan(phi) * np.cos(phiP) + np.sin(phiP) * np.sin(lamb - lambO)
    D = np.cos(lamb - lambO)
    x = self.R * h0 * np.arctan2(C, D)
    y = self.R / h0 * (np.sin(phiP) * np.sin(phi) - np.cos(phiP) * np.cos(phi) * np.sin(lamb - lambO))
    projection = np.ndarray(points.shape)
    projection[:, 0] = y
    projection[:, 1] = x
    return projection

  def XYtoSpherical(self, points, lamb0):
    (y, x) = points[:, 0] - self.imageSize[0] / 2, points[:, 1] - self.imageSize[1] / 2
    phi = np.arcsin(- y * 2 / self.imageSize[0])
    lamb = - x * 2 * np.pi / self.imageSize[1] + lamb0
    projection = np.ndarray(points.shape)
    projection[:, 0] = phi
    projection[:, 1] = lamb
    return projection

  def XYtoSphericalOblique(self, points, h0, center0, center1):
    (y, x) = points[:, 0], points[:, 1]
    (phiP, lambP) = self.getObliquePole(center0, center1)
    ratioY = (y * h0 / self.R)
    ratioX = x / (self.R * h0)
    phi = np.arcsin(ratioY * np.sin(phiP) + np.sqrt(1 - ratioY ** 2) * np.cos(phiP) * sin(ratioX))
    A = np.sqrt(1 - ratioY ** 2) * np.sin(phiP) * np.sin(ratioX) - ratioY * np.cos(phiP)
    B = np.sqrt(1 - ratioY ** 2) * np.cos(ratioX)
    lamb = lamb0 + self.arctan2(A, B)
    projection[:, 0] = phi
    projection[:, 1] = lamb
    return projection

  def reproject(self, center0, center1):
    h0 = 2
    newCoords = self.sphericalToXYOblique(self.sphereIndices, h0, center0, center1)
    minX, maxX = min(newCoords[:, 1]), max(newCoords[:, 1])
    minY, maxY = min(newCoords[:, 0]), max(newCoords[:, 0])
    finalImage = np.ndarray(self.image.shape)
    newCoords[:, 1] = ((newCoords[:, 1]) - minX) / (maxX - minX) * self.imageSize[1]
    newCoords[:, 0] = ((newCoords[:, 0]) - minY) / (maxY - minY) * self.imageSize[0]
    newCoords = np.around(newCoords, decimals=1).astype(int)
    finalImage[self.indices[:, 0], self.indices[:, 1]] = self.image[newCoords[:, 0], newCoords[:, 1]]
    return finalImage

if __name__ == '__main__':
  vidcap = cv2.VideoCapture('./Curtis-Biotech_RAW-output_360.mp4')
  totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  initialFrame = totalFrames / 5 * 2
  vidcap.set(cv2.CAP_PROP_POS_FRAMES, initialFrame)
  success, actualFrame = vidcap.read()
  proj = ObliqueEquirectangular(actualFrame)
  for i in range(0, 100):
    c1 = (np.pi / 8, np.pi * i / 10 * 2)
    c2 = (-np.pi / 8, 0)
    reprojection = proj.reproject(c1, c2)
    cv2.imwrite('projection%i.jpg' % (i), reprojection)
