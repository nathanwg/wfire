from dis import dis
from email import message
import tkinter as tk
import matplotlib.pyplot as plt
import os
import mplcursors as mpl
from func_wfipa import *

def run_loadfile():
    """ Runs the part of the program that loads a raw .tif file, calculates a 
    'numpixels array', and calculates and ignition time
    """
    global numpixels
    global frames
    global ignition_frame
    intensity_threshold,rate_threshold,fps,block,round_val = base_values()
    numpixels,num_frames,frames = calc_numpixels(intensity_threshold,None)
    num_blocks = get_num_blocks(num_frames,block)
    rate_numpixels = calc_rate_numpixels(numpixels,num_frames,False)
    ignition_frame = detect_ignition(rate_numpixels,rate_threshold,block,num_blocks,fps,round_val)
    print(ignition_frame)
    return

def run_loadarray():
    """ Runs the part of the program that loads a pre-computed 'numpixels' array
    and calculates an ignition time.
    """
    global numpixels
    global frames
    global ignition_frame
    frames = None
    intensity_threshold,rate_threshold,fps,block,round_val = base_values()
    numpixels = load_numpixels(None)
    num_frames = get_num_frames(numpixels)
    num_blocks = get_num_blocks(num_frames,block)
    rate_numpixels = calc_rate_numpixels(numpixels,num_frames,False)
    ignition_frame = detect_ignition(rate_numpixels,rate_threshold,block,num_blocks,fps,round_val)
    print(ignition_frame)
    return

def run_flamingevents():
    """ Runs the part of the program that detects flaming events and prints them to the screen.
    Note this will not run if a numpixels array hasn't been loaded or calculated.
    """
    if numpixels is None:
        print('You need to load data first')
        tk.messagebox.showwarning(title=' ',message='You need to load data first')
        return
    rate_numpixels = calc_rate_numpixels(numpixels=numpixels,num_frames=get_num_frames(numpixels),istest=False)
    intensity_threshold,rate_threshold,fps,block,round_val = base_values()
    num_blocks = get_num_blocks(rate_numpixels.shape[0]+1,block)
    flaming_events =  detect_events(rate_numpixels,block,num_blocks,fps,20)
    disp_events(flaming_events)
    return

def show_ignitiontime():
    """...
    """
    if ignition_frame is None:
        tk.messagebox.showwarning(message='No data has been loaded yet')
        return
    fps = base_values()[2]
    displaytext = 'The predicted ignition time is: '+str(round(ignition_frame/fps,2))+' s\n Frame: '+str(int(ignition_frame))
    print(displaytext)
    tk.messagebox.showinfo(message=displaytext)
    return

def run_display_plot():
    """...
    """
    if ignition_frame is None:
        tk.messagebox.showwarning(message='No data has been loaded yet')
        return
    num_frames = get_num_frames(numpixels)
    display_plot(num_frames,ignition_frame,numpixels)
    return

def run_show_ignition():
    """....
    """
##    #####
##
##    num_set = input('Please enter the number of sets you would like to look at: ')
##    sets = np.zeros((num_set,2))
##    for i in range(0,num_set):
##        sets[i,0] = input('Enter beginning of set: ')
##        sets[i,1] = input('Enter end of set: ')
##
##    ####
    if frames is None:
        tk.messagebox.showwarning(message='No image data has been loaded yet. Please select \'Load file\' to load a tif image file')
        return
    show_ignition(ignition_frame,frames)
    return

def quit_program():
    """ Exits the program
    """
    top.after(1,top.destroy)
    return

################################################
## Main
################################################
def main():
    """ Main
    """
    global top
    top = tk.Tk()
    top.title('Main window')
    top.geometry('450x400')
    top.eval('tk::PlaceWindow . center')
    labl = tk.Label(top,text='Load raw image file or array?',font=('Courier 15')).pack()
    B1 = tk.Button(top,text='Load file',command=run_loadfile,font=('Courier')).pack(pady=10)
    B2 = tk.Button(top,text='Load array',command=run_loadarray,font=('Courier')).pack(pady=10)
    B3 = tk.Button(top,text='Find flaming events',command=run_flamingevents,font=('Courier')).pack(pady=10)
    B4 = tk.Button(top,text='Show ignition time',command=show_ignitiontime,font=('Courier')).pack(pady=10)
    B5 = tk.Button(top,text='Display light intensity plot',command=run_display_plot,font=('Courier')).pack(pady=10)
    B6 = tk.Button(top,text='Display ignition images',command=run_show_ignition,font=('Courier')).pack(pady=10)
##    B7 = tk.Button(top,text='Quit',command=quit_program,font=('Courier')).pack(pady=10)
    B7 = tk.Button(top,text='Quit',command=quit_program,font=('Courier')).pack(pady=10)
##    top.protocol('WM_DELETE_WINDOW',quit_program)
    top.protocol('WM_DELETE_WINDOW',top.destroy)
    top.mainloop()
    return
 
if __name__ == "__main__":
    numpixels,frames,ignition_frame = None,None,None
    main()
    quit()
