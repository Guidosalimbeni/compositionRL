import os, time, cv2
import numpy as np
from screenGrab import grab_screen
from getkeys import key_check





def main():


    paused = False
    print('STARTING!!!')
    while(True):
        if not paused:
            x = 420
            y = 120
            screen = grab_screen(region = (x, y, 930 + x, 615 + y))
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (465, 307))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


##            print('loop took {} seconds'.format(time.time()-last_time))

            cv2.imshow('window',screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

if __name__ == "__main__":
    main()