import numpy as np
import numpy
import cv2
import cv2.cv as cv
import sys


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


def sm2d(bl, ks=11):
    bls0 = smooth(bl[:, 0], window_len = ks)
    bls1 = smooth(bl[:, 1], window_len = ks)
    smbl = np.empty([bls0.shape[0], 2])
    smbl[:, 0] = bls0
    smbl[:, 1] = bls1
    return smbl


if len(sys.argv) < 3:
    print "Not enough aguments"
    print "Usage: bgSubstract.py <path_to_video_file> <path_to_background_file [seconds_start, default -> first frame]"
    print "[seconds_start] - the offset from the beginning of the video at which the background is projected"
    sys.exit(1)

bl = np.load('./data/coords_video1/1st_coords.npy')
br = np.load('./data/coords_video1/2nd_coords.npy')
pl = np.load('./data/coords_video1/3rd_coords.npy')
pr = np.load('./data/coords_video1/4th_coords.npy')

bl = bl[:-20, :]
br = br[:-20, :]
pl = pl[:-20, :]
pr = pr[:-20, :]
l = bl.shape[0]

ks = 19
bl = sm2d(bl, ks)
br = sm2d(br, ks)
pl = sm2d(pl, ks)
pr = sm2d(pr, ks)
bl = bl[ks / 2: - ks / 2, :]
br = br[ks / 2: - ks / 2, :]
pl = pl[ks / 2: - ks / 2, :]
pr = pr[ks / 2: - ks / 2, :]


filename = sys.argv[1]
background_file = sys.argv[2]
seconds_start = 0
if len(sys.argv) > 3:
    seconds_start = float(sys.argv[3])

cap = cv2.VideoCapture(filename)
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
codec =  cv.CV_FOURCC('F', 'L', 'V', '1')
bg = cv2.imread(background_file)
frame_start = int(fps * seconds_start)

noExt = filename[:filename.rfind('.')]
out_filename = noExt + '_bg_substract_bg_at_' + str(seconds_start)+".avi"
print "Output will be written to ", out_filename
out = cv2.VideoWriter(out_filename, codec, 24, (frame_height*3, frame_width))


paused = False
bgPts = [bl[frame_start], br[frame_start], pl[frame_start], pr[frame_start]]
prevPts = [bl[0], br[0], pl[0], pr[0]]
ret, firstFrame = cap.read()
frame_index = 1
frame_count = 1

print frame_height, frame_width
while(ret and frame_index < bl.shape[0]):
    if paused == False:
        # print frame_index, cap.get(cv.CV_CAP_PROP_POS_FRAMES)
        res, frame = cap.read()
        # Get the anchor points for the current frame
        pts = [bl[frame_index], br[frame_index], pl[frame_index], pr[frame_index]]
        # Compute the homography induced by current frame and the  background
        H = cv2.findHomography(np.array(bgPts, dtype='float'), np.array(pts, dtype='float'))[0]
        # Warp the current frame with the homoraphy
        warpedFrame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

        # Warp the background, substract it from the framse
        bgWarped = cv2.warpPerspective(bg, H, (frame.shape[1], frame.shape[0]))
        substracted = np.float32(frame) - bgWarped

        frame = np.concatenate((frame, substracted, bgWarped), axis = 0)
        cv2.imshow('img', cv2.convertScaleAbs(frame))
        resized_image = cv2.resize(cv2.convertScaleAbs(substracted), (frame_height*3, frame_width))
        out.write(resized_image)


        prevPts = pts
        frame_index += 1
        frame_count += 1
    k = cv2.waitKey(1)
    if k & 0xFF == ord('p'):
        paused = not paused
    elif k & 0xFF == ord('q'):
        break

print "Done"
out.release()
cap.release()
