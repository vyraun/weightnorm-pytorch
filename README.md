# Weight normalization

This repository implements weight normalization in pytorch, following [openai/weightnorm](https://github.com/openai/weightnorm).

I implemented weight norm for Linear and Conv2d and ConvTranspose2d in wn_nn.py which I call WN_Linear, WN_Conv2d, WN_ConvTranspose2d.

You can run a simple MNIST classification by `python test.py`. Most of the code is from [yunjey/pytorch-tutorial 04 - Convolutional Neural Network](https://github.com/yunjey/pytorch-tutorial).

I only used WN_Linear and WN_Conv2d for now in `test.py`, and it seems to work prpoerly. Don't know if WN_ConvTranspose2d would work.

It's quite possible that there are bugs in the code. Feel free to open an issue or create a pull request.

