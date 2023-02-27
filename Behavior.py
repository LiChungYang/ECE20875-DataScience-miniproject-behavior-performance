import math
import numpy as np

class Behavior:
    
    def __init__(self, coords):

        self.coords = coords
        self.currCluster = None

    @property
    def dim(self):
        return len(self.coords)

    def distFrom(self, other):
        """Calculates distance between two Points.
        """
        # Error checking, keep this here.
        if self.dim != other.dim:
            raise ValueError(
                "dimension mismatch: self has dim {} and other has dim {}".format(
                    self.dim, other.dim
                )
            )

        # fill in

        return math.sqrt(sum([(self.coords[i] - other.coords[i]) ** 2 for i in range(self.dim)]))

    def moveToCluster(self, dest):
        """Reassigns this Point to a new Cluster.
        """
        if self.currCluster is dest:
            return False
        else:
            if self.currCluster:
                self.currCluster.removePoint(self)
            dest.addPoint(self)
            self.currCluster = dest
            return True

    def closest(self, objects):
        """Return the object that is closest to this point.
        """
        minDist = self.distFrom(objects[0])
        minPt = objects[0]
        for p in objects:
            if self.distFrom(p) < minDist:
                minDist = self.distFrom(p)
                minPt = p
        return minPt

    def __getitem__(self, i):
        """p[i] will get the ith coordinate of the Point p."""
        return self.coords[i]

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return f"Point({self.__str__()})"


def makePointList(data):
    """Creates a list of points from initialization data.
    """
    
    return [Behavior(row) for row in data]
        