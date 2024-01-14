from .base_options_feature2face import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--dataset_names', type=str, default='name', help='chooses test datasets.')
        self.parser.add_argument('--test_dataset_names', type=str, default='name', help='chooses validation datasets.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--save_input', type=int, default=1)
        self.parser.add_argument('--size', type=str, default='normal')
        self.parser.add_argument('--load_epoch', default='latest')
        self.parser.add_argument('--frame_jump', type=int, default=1, help='jump frame for training, 1 for not jump')

  
        self.isTrain = False
