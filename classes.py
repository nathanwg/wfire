class Test:
    """ A class that creates a Test object which contains all of data
    parameters from ignition tests
    """
    def __init__(self,filename,date,testnumber,set_type,fmc,ignition_time,spatial_calibration,eof,flame_height):
        self.filename = filename
        self.date = date
        self.testnumber = testnumber
        self.set_type = set_type # set_type should be tuple that includes (set,orientation,height,temperature)
        self.fmc = fmc
        self.ignition_time = ignition_time # ignition time will be tuple that includes both time in secs and frame #
        self.spatial_calibration = spatial_calibration
        self.eof = eof
        self.flame_height = flame_height

class Line:
    def __init__(self,p1,p2,line_type,xpoints,ypoints,m,b):
        self.p1 = p1
        self.p2 = p2
        self.line_type = line_type
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.m = m
        self.b = b

