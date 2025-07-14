# DATA-LEARNING-PROJECT
COMPANY : CODTECH IT SOLUTION
NAME : DEVANK
INTERN ID: CT04DH1688
DOMAIN : DATA SCIENCE 
DURATION : 4 WEEK
MENTOR : NEELA SANTOSH
YOU HAVE TO ENTER DESCRIPTION OF YOUR TASK :
Generating images
To generate images, pass the path to a trained checkpoint to the generate.py file.

python generate.py --checkpoint <path to checkpoint>
Scripts
Some scripts are provided in the scripts folder. They attempt to emulate some of the experiments conducted in the original DCGAN paper; in particular:

arithmetic.py: given a trained generator, computes vector arithmetic on three chosen vectors
classify.py: trains and tests a classifier (which can be modified in the models.py file) on the CIFAR-10 datasets, by using a given trained discriminator as feature extractor
interpolation.py: performs linear interpolation with a given trained generator
All three scripts take a trained model via the --checkpoint argument.

Additional notes for all scripts
The number of channels of input images needs to be manually set for all scripts if the input images are grayscale (with the --nc 1 argument)
Use --help to get all available arguments for a script
References
Alec Radford, Luke Metz and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial nets
Official implementation (in Torch)
OUTPUT :
