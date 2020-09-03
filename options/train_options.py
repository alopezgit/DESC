from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--src_train_datafile', type=str, default='train.txt', help='stores data list, in src_root')
        parser.add_argument('--tgt_train_datafile', type=str, default='train.txt', help='stores data list, in tgt_root')
        
        parser.add_argument('--use_semantic_const', action='store_true', help='use semantic consistency in the target domain')
        parser.add_argument('--use_stereo', action='store_true', help='use stereo supervision in target domain')
        parser.add_argument('--pretrain_semantic_module', action='store_true', help='pretrain semantic data to depth network')
        parser.add_argument('--train_image_generator', action='store_true', help='train image network')
        parser.add_argument('--load_pretrained', action='store_true', help='load pretrained networks from 1st training step')

        parser.add_argument('--print_freq', type=int, default=32, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--n_train_epochs', type=int, default=20, help='number of training epochs (an epoch is defined using the target domain)')
        parser.add_argument('--n_train_iterations', type=int, default=1000000, help='number of training iterations, the script will use the min(n_train_epochs, n_train_iterations)')

        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr_task', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_trans', type=float, default=5e-5, help='initial learning rate for adam')
        parser.add_argument('--pool_size', type=int, default=20, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')

        self.isTrain = True
        return parser
