# vaibhav-rj
Handwritten devnagari(Hindi) numerals(0-9) classifier

The handwritten devnagari(Hindi) numerals(0-9) classifier which I've created consists of 2 file:-

(1)The 1st file:-'image_recog_train2.py' contains the code for training the model for classifying the devnagari numerals(0-9) handwritten
images.For the purpose I have trained the model using the train set of extracted version of zip file 'DevanagariHandwrittenCharacterDataset'
that contains handwritten devnagari numerals(0-9 in hindi) images along with the images of other characters and is uploaded on github by me.

(2)The 2nd file:-'image_recog_test2.py' contains the code where the random handwritten images of the devnagari numerals(0-9),
either taken from the test set of 'DevanagariHandwrittenCharacterDataset' extracted file or is drawn in MS-Paint on:-
  (i)32*32 white canvas with thickest black pencil stroke.
  (ii)32*32 black canvas with thickest white pencil stroke.
  Note:The other specifications related with the recognition of the images(i) and(ii) are mentioned in the comment section of the file:-
  'image_recog_test2.py'.
  
 Also to mention that my classification model is based on convoluted neural networks(CNNs) about which one can find details
 regarding the various steps/layers for creating a CNN on the
 link:-(i)http://cs231n.github.io/convolutional-networks/
     
 All the other explanations have been mentioned in the comment sections of the file:-'image_recog_train2.py' and 'image_recog_test2.py'.
 So,please go through the codes of the files:-'image_recog_train2.py' and 'image_recog_test2.py' along with referring to the link above.
  
