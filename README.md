# ESFNet: Efficient Networks for Building Extraction from High-Resolution Images
The implementation of novel efficient neural network ESFNet

### Clone the Repository
git clone https://github.com/mrluin/ESFNet-Pytorch.git
cd ./ESFNet-Pytorch


### Installation using Conda
conda env create -f environment.yml
conda activate esfnet

### Directory Structure
```
Directory:
            #root | -- train 
                  | -- valid
                  | -- test
                  | -- save | -- {model.name} | -- datetime | -- ckpt-epoch{}.pth.format(epoch)
                            |                               | -- best_model.pth
                            |
                            | -- log | -- {model.name} | -- datetime | -- history.txt
                            | -- test| -- log | -- {model.name} | --datetime | -- history.txt
                                     | -- predict | -- {model.name} | --datetime | -- *.png
```
### Training
1. set `root_dir` in `./configs/config.cfg`, change the root_path like mentioned above.
2. set `divice_id` to choose which GPU will be used.
3. set `epochs` to control the length of the training phase.
4. setup the `train.py` script as follows:
```
python -m visdom.server -env_path='./visdom_log/' -port=8097 # start visdom server
python train.py
```
`-env_path` is where the visdom logfile store in, and `-port` is the port for `visdom`. You could also change the `-port` in `train.py`.



**If my work give you some insights and hints, star me please! Thank you~**
