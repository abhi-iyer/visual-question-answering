# Visual Question Answering with Stacked Attention Networks (CNN, LSTM, and Attention Nets)
#### Abhiram Iyer, Aravind Mahadevan, Neil Sengupta
##### UC San Diego

### Description

Based off of the paper on VQA here: https://arxiv.org/pdf/1511.02274.pdf

Link to Google Drive folder, containing logs and checkpoints for various experiments run on VQA model:
https://drive.google.com/drive/folders/1HFTv1vd2nEoso_5tEEbKR1YlrdylhKp0?usp=sharing

Please note:

logs_batch200 and exp_batch200 are the logs and checkpoint files to load and run the best trained model (includes proper parameter initializations; trained with RMSprop). Download the checkpoint file and run demo.ipynb to see results. 

logs_batch150 and logs_batch250 describe 2 different experiments run with Adam. exp_batch250 is the checkpoint file for the experiment corresponding to logs_batch250. Both experiments are highly prone to overfitting, and do not contain proper parameter weight initializations.

Code written in PyTorch.


### Code organization
##### demo.ipynb        --  Run a demo of our code
##### train.py          --  Train the model. Can change hyperparameters here, as well as save model parameters in checkpoints while training

##### models.py         --  Contains CNN (VGG16), LSTM, and Attention Net modules used in training
##### preprocess.py     --  Pre-process the dataset (created a torch Dataset)
##### nntools.py        --  Sample experiment/checkpoint saving code that train.py was built from
