import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



def smoothPositions(playerPositions):
	points = playerPositions.shape[1]

	smoothingFilter = gaussianFilter(2,5)
	for pt in range(points):
		fig1 = plt.figure(1)
		plt.plot(range(playerPositions.shape[0]), playerPositions[:,pt,:], 'r-')
		fig1.savefig('smoothed'+str(pt+points)+'.png')
		playerPositions[:,pt,0] = smoothArray(playerPositions[:,pt,0], smoothingFilter)
		playerPositions[:,pt,1] = smoothArray(playerPositions[:,pt,1], smoothingFilter)
		fig2 = plt.figure(1)
		plt.plot(range(playerPositions.shape[0]), playerPositions[:,pt,:], 'r-')
		fig2.savefig('smoothed'+str(pt)+'.png')

	return playerPositions


def smoothArray(array, smoothingFilter):

	filterSize = len(smoothingFilter)
	appendSize = len(smoothingFilter)/2

	# don't pad with zeros!! That introduces sharp derivatives
	appendedArray = array
	appendedArray = np.append(appendedArray, np.full([appendSize], array[len(array)-1]))
	appendedArray = np.append(np.full([appendSize], array[0]), appendedArray)

	smoothedArray = appendedArray

	for i in range(appendSize, len(smoothedArray)-appendSize):
		extract = appendedArray[i-appendSize: i+appendSize+1]
		smoothedArray[i] = sum(extract*smoothingFilter)

	smoothedArray = smoothedArray[appendSize:len(smoothedArray)-appendSize]

	return smoothedArray

def gaussianFilter(stdev, extent):
	stdev += 0.0
	filter = np.zeros(extent)
	centre = extent/2
	for i in range(extent):
		filter[i] = (1/(np.sqrt(2*np.pi)*stdev))*np.exp(-(i-centre)*(i-centre)*0.5*(1/stdev)*(1/stdev))

	total = sum(filter)

	for i in range(len(filter)):
		filter[i] = filter[i]/total

	return filter




def smoothPointsArray(points, MISSING=-9999, ks=11):
    '''
        points -> F x P x 2 array, where F is the number of frames, P the number of points
                and points[i, j] hold the x,y coords of point j in frame i
        MISSING -> The value which replaces coordinates for unknown points in the array
        ks -> window size for the smoothing
        returns - an array with the same length, with all continuos patches smoothened
    '''
    F, P, d = points.shape
    result = np.zeros([F-23, P, d])

    currentSeqStarts = - np.ones(P)

    for i, pos in enumerate(points[0, :, :]):
        if not (pos[0] == pos[1] == MISSING):
            currentSeqStarts[i] = 0

    # Add a dummy MISSING row in the end so as not to worry about edge cases
    dummy = MISSING * np.ones([1, P, d])
    points = np.concatenate((points, dummy), axis=0)

    for i in range(F+1):
        row = points[i, :, :]
        for j in range(P):
            pos = row[j, :]
            if pos[0] <= -1000: # We don't see this point in this frame
                if currentSeqStarts[j] >= 0: # We just lost this point, we want to smooth the patch
                    seqStart = currentSeqStarts[j]
                    seqEnd = i
                    if ks <= seqEnd - seqStart: # Make sure the patch is long enough to convolve with our kernel
                        seqEnd = 1172
                        print j, seqStart, seqEnd
                        for k in range(d):
                            smoothened = smooth(points[seqStart: seqEnd, j, k], window_len=ks)
                            result[seqStart:seqEnd, j, k] = smoothened #[ks/2 : -ks/2 + 1]
                    currentSeqStarts[j] = -1 #We're not seeing this point anymores
            else: # We see this point in this frame
                if points[i - 1, j, 0] == points[i - 1, j, 1] == MISSING: # We did not have it in the last frame
                    currentSeqStarts[j] = i
        #print i, currentSeqStarts
    # Remove dummy row, return result
    points = points[:-1, :, :]
    return result

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


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        #w=eval('np.'+window+'(window_len)')
        y=eval('signal.savgol_filter(x, 91, 3)')

    #y=np.convolve(w/w.sum(),s,mode='valid')
    return y