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

class AvgTest:
    """ A class that creates a AvgTest object which contains all of data
    parameters from an averaged group of ignition tests (i.e., a set of tests) 
    """
    def __init__(self,filenames,set_type,height,temperature,fmc,ignition_time,ignition_frame):
        self.filenames = filenames # tuple containing all filenames that this particular averaged group is made up of
        self.set_type = set_type
        self.fmc = fmc
        self.ignition_time = ignition_time
