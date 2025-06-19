from sysinc3d.data import NucleiGhostGenerator
from sysinc3d.data import PointPatternGenerator


class Examples(object):


    def __init__(self):
        super(Examples, self).__init__()


    def getSTEDNucleusAndPoints(self):
        ngh = NucleiGhostGenerator(image_shape=(62, 320, 320), radii=(23, 31, 35), angles=(4, 17, 150))
        image = ngh.getImage()
        ppg = PointPatternGenerator(image, nr_of_points=40)
        points = ppg.getCSRPoints()
        return image, points


