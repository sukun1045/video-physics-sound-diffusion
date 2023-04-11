from yacs.config import CfgNode as CN


cfg = CN()

cfg.device = 'cuda'

cfg.dist_backend = 'nccl'

cfg.log_dir = 'logs/'
cfg.output_dir = 'outputs/'
cfg.result_dir = 'results/'

cfg.seed = 42

cfg.workers = 4

cfg.model = 'base_model'


# dataset
cfg.dataset = CN()

cfg.dataset.img_num_per_gpu = 1

cfg.dataset.H = 224
cfg.dataset.W = 224
cfg.dataset.ratio = 0.5
cfg.dataset.name = ''
cfg.dataset.data_root = 'data/'
cfg.dataset.file = ''

cfg.dataset.train = CN()

cfg.dataset.train.sampler = ''
cfg.dataset.train.drop_last = True
cfg.dataset.train.shuffle = True
cfg.dataset.train.chunk = 400


cfg.dataset.test = CN()
cfg.dataset.test.sampler = ''
cfg.dataset.test.batch_sampler = ''
cfg.dataset.test.drop_last = False
cfg.dataset.test.shuffle = False
cfg.dataset.test.chunk = 2000


# network render
cfg.render = CN()
cfg.render.file = 'BaseRender'
cfg.render.audio_encoder_path = ''
cfg.render.resume_path = ''


# nerfhead
cfg.model = CN()
cfg.model.file = ''

# train
cfg.train = CN()

cfg.train.file = 'BaseTrainer'
cfg.train.criterion_file = 'BaseCriterion'

cfg.train.resume = False

cfg.train.ep_iter = 500
cfg.train.lr = 1e-4
cfg.train.gamma = 0.1
cfg.train.decay_epochs = 1000
cfg.train.weight_decay = 0.0001
cfg.train.max_epoch = 1000

cfg.train.print_freq = 10
cfg.train.save_every_checkpoint = True
cfg.train.save_interval = 1
cfg.train.valiter_interval = 100
cfg.train.val_when_train = False

# test
cfg.test = CN()

cfg.test.save_imgs = True
cfg.test.is_vis = False


def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()