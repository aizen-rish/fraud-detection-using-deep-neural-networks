For all the dependencies requirment, install using 'pip install' 

The latest version of the code can be downloaded from Github using the command: 
  $ git clone https://github.com/aizen-rish/fraud-detection-using-deep-neural-networks.git

Steps for basic run: 
1. Run the python file 'simpleANN.py' 
2. Give the appropriate dataset in the 'data' folder
3. Accuracy is printed on the screen.

The encoder and auto-encoder trained network are already pre-saved as models in the 'saved_models' folder.
If you wish to edit/re-train the encoder network, modify the encoded_all.py and correspondingly the use_encoder.py ( the latter just uses the models trained in from the previous one).