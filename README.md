# TGS-Salt-Identification-Challenge
This is for TGS Salt Identification Challenge. All codes are in src folder and are tested in python (before you run it, make sure to run requirements.txt.)

1 - unet :0.640
The first result is a baseline without any further comment.
2 - clean unet:0.664
The second result is a test to prove code base cleanup success.
3 - 4-8-16-23-64-128 unet:0.637
The third try is to see if a much finer CNN cell could help or not. The result is negative obviously.
4 - the same model of 2 + remove black pictures:0.676
5 - time to seek new structure:0.688
change activation function from sigmoid to tanh.
6 - revision based on model 5: no submission.
change activation function to elu but the result is very bad, after finishing 30 epoch ti didn't get best result.
Then I tried to enlarged the training dataset by rotate image 90 degree and the effect is just so so.
7 - add K-means based on aggregation + model 5:
use K-mean to generate masks for test pictures.
8 - double the depth based on model 5.
and the result is not good.



