from dis import dis
from email import message
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import os
import mplcursors as mpl

##################################
#
###################################

def readfile(filename,is_multi):
    """ The user is prompted to choose a file.
    The input should be .tif file
    """
    if filename is None:
        mydir = os.getcwd()
        mydir = mydir.replace('\wfire','')
##        tk.Tk().withdraw()
        filename = askopenfilename(filetypes=[('tif files','*.tif')],initialdir=mydir)
    if is_multi is True:
        img = cv.imreadmulti(filename)
    elif is_multi is False:
        img = cv.imread(filename)
##    if img[0] is False:                 
##        raise Exception('File was not read correctly')
##    else:
##        print('File was read succesfully') # If it correctly reads a stacked .tif file, it will a tuple
##        print(filename)
    return img,filename

def get_image_properties(img):
    """ Returns information about the dimensions and 
    number of frames in a stacked .tif file
    """
    frames = img[1] # First tuple element is skipped because it is a boolean value.... 
    num_frames = len(frames)  # ....second element is another tuple, each element of this one is an ndarray (i.e., frame)
    print('The number of frames in this file is:',num_frames)
    num_rows = frames[0].shape[0] # number of rows or length along the y-axis
    num_cols = frames[0].shape[1] # number of columns or length along the x-axis
    return frames,num_frames,num_rows,num_cols

def calc_numpixels_frame(ref_frame,intensity_threshold):
    """ Loops through an individual frame and counts the number of pixels
    above the intensity threshold
    """
    if ref_frame.max() < intensity_threshold:
        return 0
    ref_frame = ref_frame.astype(float) # Change to float to handle subtraction correctly
    x = ((ref_frame-intensity_threshold)>0)
    numpixels_frame = x.sum()
    return numpixels_frame

def calc_rate_numpixels(numpixels,num_frames,istest):
    """ Returns the rate of change (over the course of an entire
    test) of numpixels
    """
    rate_numpixels = np.zeros(num_frames)
    for i in range(0,num_frames-1):
        num_1 = numpixels[i,0]
        num_2 = numpixels[i+1,0]
        rate_numpixels[i]=np.abs(num_2-num_1)
    if istest != True:
        rate_numpixels[num_frames-2]=0
    return rate_numpixels

def calc_avgrate_numpixels(rate_numpixels,round_val,block,num_blocks):
    """ The number of frames is broken up into blocks of time (e.g., 50 frames at a time)
    and the average rate of change of numpixels is calculated over these blocks. 
    Returns both a rounded and unrounded array
    """
    avgrate_numpixels = np.zeros((1,num_blocks))
    avgrate_numpixels_r = np.zeros((1,num_blocks))
    for i in range(0,num_blocks):
        avgrate_numpixels[0,i] = np.mean(rate_numpixels[i*block:(i+1)*block])
        temp_value = avgrate_numpixels[0,i]
        avgrate_numpixels_r[0,i] = round(temp_value/round_val)*round_val
    return avgrate_numpixels,avgrate_numpixels_r

def detect_events(rate_numpixels,block,num_blocks,fps,round_val):
    """ Returns an array called events. This array holds a start frame, end frame,
    and duration for all detected events that show significant changes in light intensity.
    (These are meant to be very general frame/time markers since this is based on 
    larger time blocks)
    """
    avgrate_numpixels_r = calc_avgrate_numpixels(rate_numpixels,round_val,block,num_blocks)[1]
    events = np.zeros((1,3))
    event_num = 0
    event_start = False
    for i in range(0,num_blocks):
        if avgrate_numpixels_r[0,i] > 0 and event_start is False:
            if event_num == 0:
                events[event_num,0] = block*(i-1)
                if events[event_num,0] < 0:
                    events[event_num,0] = 0
                event_start = True
            else:
                events = np.append(events,[[block*(i-1),0,0]],0)
                if events[event_num,0] < 0:
                    events[event_num,0] = 0
                event_start = True
        elif avgrate_numpixels_r[0,i] == 0 and event_start is True:
            events[event_num,1] = block*i
            events[event_num,2] = (events[event_num,1]-events[event_num,0])/fps
            event_start = False
            event_num += 1
        else:
            continue
    if event_start is True:
        events[event_num,1] = block*i
        events[event_num,2] = (events[event_num,1]-events[event_num,0])/fps
        event_start = False
    return events

def check_ignition(start_frame,pause_frame,frame,fps,event_start,event_pause,ignition):
    """ Based on frame numbers passed from 'detect_ignition' function, this will
    check if ignition has occurred (i.e., if event is significantly long). Also checks
    if time of 'pause' is long enough to assume significant activity is not happening
    """
    if event_pause is True:
        duration_pause = (frame-pause_frame)/fps
        if duration_pause >= 0.25:
            event_start,event_pause = False,False
    duration_event = (frame-start_frame)/fps
    if duration_event >= 0.5:
        ignition = True
    return ignition,event_start,event_pause

def detect_ignition(rate_numpixels,rate_threshold,block,num_blocks,fps,round_val):
    """ Detects ignition event by looking at significant 'events' where light intensity
    is rapidly changing. If duration of significant event is long enough, ignition is
    assumed to have occurred
    """
    events = detect_events(rate_numpixels,block,num_blocks,fps,round_val)
    for i in events:
        if i[2] >= 0.2:
            FOI = i[0]-block # FOI is 'frame of interest'
            if FOI < 0:
                FOI = 0
            break
    ignition_frame,ignition = 0,False
    event_start,event_pause,pause_frame = False,False,0
    for i in range(0,len(rate_numpixels)):
        frame = FOI+i
        rate = rate_numpixels[int(frame)]
        if rate >= rate_threshold and event_start is False:
            start_frame,event_start = frame,True
        elif rate < rate_threshold and event_start is True and event_pause is False:
            pause_frame,event_pause = frame,True
        elif rate >= rate_threshold and event_pause is True:
            event_pause = False
        if event_start is True:
            ignition,event_start,event_pause = check_ignition(start_frame,pause_frame,frame,fps,event_start,event_pause,ignition)
        if ignition is True:
            ignition_frame = start_frame
            break
    return ignition_frame

def load_numpixels(filename):
    """ Loads a pre-computed array of numpixels
    """
    if filename is None:
        mydir = os.getcwd()
        tk.Tk().withdraw()
        filename = askopenfilename(filetypes=[('npy files','*.npy')],initialdir=mydir)
    numpixels = np.load(filename)
    numpixels = numpixels.T
    return numpixels

def calc_numpixels(intensity_threshold,filename):
    """ Calculates numpixels by asking user for file and process raw image data
    """
    img,filename = readfile(filename,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    numpixels = np.zeros((1,num_frames))
    for i in range(0,num_frames):
        ref_frame = frames[i]
        numpixels[0,i] = calc_numpixels_frame(ref_frame,intensity_threshold)
    numpixels = numpixels.T
    numpixels[-1,0] = numpixels[-2,0]
    return numpixels,num_frames,frames
    
def get_num_frames(numpixels):
    """ Returns the number of frames
    """
    num_frames = numpixels.shape[0]
    return num_frames

def get_num_blocks(num_frames,block):
    """ Returns the number of time blocks
    """
    num_blocks = int(np.floor(num_frames/block))
    return num_blocks

def display_plot(num_frames,ignition_frame,numpixels):
    """ Plots numpixels over time with a vertical line representing ignition
    """
    x = np.linspace(1,num_frames,num_frames)  # Plot used for testing code output
    x_ig = [ignition_frame,ignition_frame]
    pixel_length = 0.4 #mm/pixel
    pixel_area = pixel_length**2
    areapixels = pixel_area*numpixels
    y_ig = [0,np.amax(areapixels)]
    plt.plot(x,areapixels,linewidth=0.5)
    # plt.plot(x_ig,y_ig,linewidth=0.5)
    plt.xlabel('Frames (500 fps)')
    plt.ylabel('Area with higher light intensity ($mm^{2}$)')
    plt.title('Light intensity over time during ignition test')
    mpl.cursor()
    plt.show()

def show_ignition(ignition_frame,frames):
    """ Displays images of an ignition event.
    """
    ignition_frame = int(ignition_frame)
    img01 = frames[ignition_frame]
    img02 = frames[ignition_frame+25]
    img03 = frames[ignition_frame+50]
    img04 = frames[ignition_frame+75]
    img05 = frames[ignition_frame+100]
    img06 = frames[ignition_frame+125]
    img07 = frames[ignition_frame+150]
    img08 = frames[ignition_frame+175]
    font = cv.FONT_HERSHEY_COMPLEX
    bottomLeft = (10,245)
    fontScale = 0.5
    fontColor = (255,255,255)
    thickness = 1
    lineType = 2
    cv.putText(img08,'Hello there',bottomLeft,font,fontScale,fontColor,thickness,lineType)

    imgs01 = np.concatenate((img01,img02,img03,img04),axis=1)
    imgs02 = np.concatenate((img05,img06,img07,img08),axis=1)
    img = np.concatenate((imgs01,imgs02),axis=0)
    cv.imshow("window",img)
    while cv.getWindowProperty('window', cv.WND_PROP_VISIBLE) > 0:
        k = cv.waitKey(50)
        if k != -1:
            cv.destroyAllWindows()
            break
        
def disp_events(events):
    """ Prints start/stop and duration of 'flaming events'
    """
    for i in range(0,events.shape[0]):
        if events[i,2] <= 0.1:
            continue
        else:
            print('Flaming event: start - ',events[i,0],' stop - ',events[i,1],' Duration - ', events[i,2])


def base_values():
    """ Returns basic values needed for every test.
    """
    intensity_threshold = 35  # These are values that could be edited by user
    rate_threshold = 10       # Should be updated so these are default and user can specify 
    fps = 500                 # from command line as well 
    block = int(0.1*fps)
    round_val = 10           # round_val is the number that is used when averaging the 'rate_numpixels'
    return intensity_threshold,rate_threshold,fps,block,round_val
