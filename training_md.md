$ python train.py --dataroot ./datasets/selfie2anime --name selfie2anime_FastCUT --CUT_mode FastCUT --num_threads 0 --continue_train --epoch_count 91 --epoch 90

$ python -m visdom.server


$ python train.py --dataroot ./datasets/selfie2anime --name selfie2anime_CUT --CUT_mode CUT --num_threads 0

<!-- Apply hinge loss -->

$ python train.py --dataroot ./datasets/selfie2anime --name selfie2anime_Hinge_CUT --CUT_mode CUT --hinge_loss --num_threads 0 

## <!-- Net D options ['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2', 'basic_sn', 'self_attn'] -->
<!--  In my previous GAN research, Always apply spectral normalization and Hinge loss-->

<!-- Apply spectral normalization and hinge loss -->
$ python train.py --dataroot ./datasets/selfie2anime --name selfie2anime_SP_Hinge_CUT --CUT_mode CUT --hinge_loss --netD basic_sn --num_threads 0 

<!-- Apply spectral normalization, hinge loss and self-attn(default sp norm) -->
$ python train.py --dataroot ./datasets/selfie2anime --name selfie2anime_SP_Hinge_Selfattn_CUT --CUT_mode CUT --hinge_loss --netD self_attn --num_threads 0 

# TODO: TTUR, Two times scale ....
