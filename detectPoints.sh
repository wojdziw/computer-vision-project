echo $@

cd ./pointsDetection;
echo "Finding the points";
python2 -W ignore pointsDetection.py $@;
echo "Sorting and Guessing the points";
python2 -W ignore pointsSortingAndGuess.py $@;
cd ../;
