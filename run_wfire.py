import os
import run_wffpa as wffpa
import run_wfipa as wfipa

running,error = True, False
while running is True:
    os.system('cls')
    # print('\n\n\n\n\n','--------------------------------------------------','\n')
    # print('Type option from the following list and hit enter:')
    # print(' Run Wildland Fuel Ignition Process Analysis (wfipa) ----------- i')
    # print(' Run Wildland Fuel Flaming Process Analysis (wffpa) ------------ f')
    # print(' Quit program -------------------------------------------------- q','\n')
    if error is True:
        print('Error -- selected invalid option')
        error = False
        print()
    # usr = input('Selected option: ')
    usr,running = 'f',False
    if usr == 'i':
        wfipa.main()
    elif usr == 'f':
        wffpa.main()
    elif usr == 'q':
        running = False
    else:
        error = True
