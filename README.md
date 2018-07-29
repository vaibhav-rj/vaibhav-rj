# vaibhav-rj
# Handwritten devnagari(Hindi) numerals(0-9) classifier

The handwritten devnagari(Hindi) numerals(0-9) classifier which I've created consists of 2 file:-

(1)The 1st file:-'image_recog_train2.py' contains the code for training the model for classifying the devnagari numerals(0-9) handwritten
images.For the purpose I have trained the model using the train set of extracted version of zip file 'DevanagariHandwrittenCharacterDataset'
that contains handwritten devnagari numerals(0-9 in hindi) images along with the images of other characters and can be accessed from the 
given link:-'https://archive.ics.uci.edu/ml/machine-learning-databases/00389/'.Also,the accessed file:-'DevanagariHandwrittenCharacterDataset.zip' and its extracted version needs to be saved along with 'image_recog_train2.py' and 'image_recog_test2.py' at the same location.

(2)The 2nd file:-'image_recog_test2.py' contains the code where the random handwritten images of the devnagari numerals(0-9),
either taken from the test set of 'DevanagariHandwrittenCharacterDataset' extracted file or is drawn in MS-Paint on:-
  (i)32 by 32 white canvas with thickest black pencil stroke and saved with(.png) extension.
  (ii)32 by 32 black canvas with thickest white pencil stroke and saved with(.png) extension.
  Note:The other specifications related with the recognition of the images(i) and(ii) are mentioned in the comment section of the file:-
  'image_recog_test2.py'.
  
 Also to mention that my classification model is based on convoluted neural networks(CNNs) about which one can find details
 regarding the various steps/layers for creating a CNN on the
 link:-(i)http://cs231n.github.io/convolutional-networks/
     
 All the other explanations have been mentioned in the comment sections of the file:-'image_recog_train2.py' and 'image_recog_test2.py'.
 So,please go through the codes of the files:-'image_recog_train2.py' and 'image_recog_test2.py' along with referring to the link above.
  
Following are the steps to train the classifier and then predict the value/label of any random handwritten devnagari numeral(0-9):-

(1)First go to the command line(command-prompt) and reach to the location where 'DevanagariHandwrittenCharacterDataset.zip' and its extracted version along with 'image_recog_train2.py' and 'image_recog_test2.py' are saved.

(2)Now for training the model type:-"python image_recog_train2.py"
It would take 20-30 minutes to train the model depending on your processor.

(3)Now testing a random image on the trained model type:-"python image_recog_test2.py 'location of the file to be tested\file_name.png'.
For eg:-If name of my image file is 5422 and its saved in (.png extension) my desktop,
I would write:-"python image_recog_test2.py C:\Users\NPC\Desktop\5422.png".

Please avoid including the double inverted commas while writing in the command line.They have been used here only to distinguish
the syntax of the command from normal text.
