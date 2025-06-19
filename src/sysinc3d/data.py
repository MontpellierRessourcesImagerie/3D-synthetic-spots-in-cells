"""
Create synthetic nuclei ghosts (ellipsoids) and point sets within a nucleus with different
distributions (CSR, clustered, dispersed or on structures such as tubes or spheres).
"""


import random
import numpy as np
from pyellipsoid import drawing
from skimage.measure import regionprops
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter


class NucleiGhostGenerator(object):
    """
        Create the ghost of a nucleus as an ellipsoid.
    """

    def __init__(self, image_shape=(512, 512, 512), pos=None, radii=None, angles=None):
        super(NucleiGhostGenerator, self).__init__()
        self.imageShape = image_shape
        self.position = pos
        if not self.position:
            self.position = (self.imageShape[0] // 2, self.imageShape[1] // 2, self.imageShape[2] // 2)
        self.radii = radii
        if not self.radii:
            self.radii = (self.imageShape[0] // 3, self.imageShape[1] // 4, self.imageShape[2] // 4)
        self.angles = angles
        if not self.angles:
            self.angles = (0, 0, 0)
        self.image = None


    def getImage(self):
        if not self.image:
            self.calculateImage()
        return self.image


    def calculateImage(self):
        angles = np.deg2rad(self.angles)
        self.image = drawing.make_ellipsoid_image(self.imageShape,
                                                  (self.position[2], self.position[1], self.position[0]),
                                                  (self.radii[2], self.radii[1], self.radii[0]),
                                                  (angles[2], angles[1], angles[0])
                                                  )



class PointPatternGenerator(object):
    """
        Create different types of point patterns within a mask.
    """

    def __init__(self, mask, nr_of_points=120):
        super(PointPatternGenerator, self).__init__()
        self.mask = mask
        self.numberOfPoints = nr_of_points


    def getCSRPoints(self):
        nz = np.nonzero(self.mask)
        coords = np.transpose(nz)
        maxSample = len(coords) - 1
        indices = random.sample(range(maxSample + 1), self.numberOfPoints)
        randPoints = coords[indices]
        return PointPattern(randPoints, self.mask)


    def getClusteredPoints(self, nrOfClusters=4, maxDist=5):
        nz = np.nonzero(self.mask)
        coords = np.transpose(nz)
        maxSample = len(coords) - 1
        indices = random.sample(range(maxSample + 1), nrOfClusters)
        centers = coords[indices]
        randPoints = None
        for counter in range(nrOfClusters):
            centerIndex = random.randrange(nrOfClusters)
            center = centers[centerIndex]
            image = drawing.make_ellipsoid_image(self.mask.shape,
                                                      (center[2], center[1], center[0]),
                                                      (maxDist, maxDist, maxDist),
                                                      (0,0,0)
                                                      )
            nz = np.nonzero(image)
            coords = np.transpose(nz)
            maxSample = len(coords) - 1
            numberOfSamples = self.numberOfPoints // nrOfClusters
            indices = random.sample(range(maxSample + 1), numberOfSamples)
            if randPoints is None:
                randPoints = coords[indices]
            else:
                randPoints = np.concatenate((randPoints, coords[indices]))
        return PointPattern(randPoints, self.mask)



class PointPattern(object):


    def __init__(self, points, mask):
        super(PointPattern, self).__init__()
        self.points = points
        self.mask = mask
        self.croppedMask = None
        self.density = None
        self.volume = None
        self.scale = (1, 1, 1)
        self.unit = 'voxel'
        self.localDensityImage = None
        self.pointImage = None
        self.croppedPointImage = None
        self.props = regionprops(self.mask)[0]


    def setScale(self, scale, unit):
        self.scale = scale
        self.unit = unit
        self.density = None


    def getGlobalDensity(self):
        if not self.density:
            self.calculateGlobalDensity()
        return self.density


    def calculateGlobalDensity(self):
        self.volume = self.props.area * self.scale[0] * self.scale[1] * self.scale[2]
        self.density = (len(self.points) / self.volume).item()


    def getPointImage(self):
        if self.pointImage is None:
            self.pointImage = np.zeros(shape=self.mask.shape, dtype=np.uint8)
            for z, y, x in self.points:
                self.pointImage[z][y][x] = 1
        return self.pointImage


    def getCroppedMask(self):
        if self.croppedMask is None:
            region = self.props.bbox
            self.croppedMask = self.mask[region[0]:region[3], region[1]:region[4], region[2]:region[5]]
        return self.croppedMask


    def getCroppedPoints(self):
        pointImage = self.getPointImage()
        if self.croppedPointImage is None:
            region = self.props.bbox
            self.croppedPointImage = pointImage[region[0]:region[3], region[1]:region[4], region[2]:region[5]]
        return self.croppedPointImage


    def getLocalDensityHeatMap(self, radius=None, mode='gaussian'):
        pointImage = self.getCroppedPoints()
        if not radius:
            diameter = min(pointImage.shape) // 2
            radius = (diameter - 1) // 2
        if mode == 'uniform':
            self.localDensityImage = uniform_filter(pointImage.astype(np.float64), 2*radius+1)
        if mode == 'gaussian':
            sigma = radius / 2.0
            print("sigma=", sigma)
            self.localDensityImage = gaussian_filter(pointImage.astype(np.float64), sigma)
        self.localDensityImage = self.localDensityImage * self.getCroppedMask()
        return self.localDensityImage





