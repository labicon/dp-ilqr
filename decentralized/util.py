#!/usr/bin/env python

"""Various utilities used in other areas of the code."""


class Point(object):
    """Point in 2D"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        return Point(self.x * other.x, self.y * other.y)
    
    def __repr__(self):
        return str((self.x, self.y))
    
