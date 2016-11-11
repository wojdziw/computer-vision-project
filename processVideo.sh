

echo $@
cd ./homography;
echo "Extracting background";
python2 extractBackground.py $@;
echo "Stitching video";
python2 stitchVideo.py $@;

cd ../ballDetection;
echo "Tracking ball"
python2 generateParaboles.py;
python2 ball_detection.py $@;
python2 drawBall.py $@ 1;

cd ../topDownView;
echo "Generating top-down view"
python2 topDownView.py $@

cd ../visualization;
echo "Visualization"
python2 main.py $@




