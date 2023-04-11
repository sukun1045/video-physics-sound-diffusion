import os
import os.path as osp
import time
import logging
import torch

def resource_path(relative_path):
    """To get the absolute path"""
    base_path = osp.abspath(".")

    return osp.join(base_path, relative_path)


def ensure_dir(root_dir, rank=0):
    if not osp.exists(root_dir) and rank == 0:
        print(f'=> creating {root_dir}')
        os.mkdir(root_dir)
    else:
        while not osp.exists(root_dir):
            print(f'=> wait for {root_dir} created')
            time.sleep(10)

    return root_dir

def create_logger(cfg, rank=0):
    # working_dir root
    abs_working_dir = resource_path('work_dirs')
    working_dir = ensure_dir(abs_working_dir, rank)
    # output_dir root
    output_root_dir = ensure_dir(os.path.join(working_dir, cfg.output_dir), rank)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    final_output_dir = ensure_dir(os.path.join(output_root_dir, time_str), rank)
    # set up logger
    logger = setup_logger(final_output_dir, time_str, rank)

    return logger, final_output_dir


def setup_logger(final_output_dir, time_str, rank, phase='train'):
    log_file = f'{phase}_{time_str}_rank{rank}.log'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def load_checkpoint(cfg, model, optimizer, lr_scheduler, device, module_name='model'):
    last_iter = -1
    resume_path = cfg.render.resume_path
    resume = cfg.train.resume
    if resume_path and resume:
        if osp.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location='cpu')
            # resume
            if 'state_dict' in checkpoint:
                # model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                logging.info(f'==> model pretrained from {resume_path} \n')
            elif 'model' in checkpoint:
                if module_name == 'detr':
                    model.module.detr_head.load_state_dict(checkpoint['model'], strict=False)
                    logging.info(f'==> detr pretrained from {resume_path} \n')
                else:
                    model.module.load_state_dict(checkpoint['model'], strict=False)
                    logging.info(f'==> model pretrained from {resume_path} \n')
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer.param_groups[0]['capturable'] = True
                logging.info(f'==> optimizer resumed, continue training')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                last_iter = checkpoint['epoch']
                logging.info(f'==> last_epoch = {last_iter}')
            if 'epoch' in checkpoint:
                last_iter = checkpoint['epoch']
                logging.info(f'==> last_epoch = {last_iter}')
            # pre-train
        else:
            logging.error(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    else:
        logging.info("==> train model without resume")

    return model, optimizer, lr_scheduler, last_iter


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(states, os.path.join(output_dir, filename))
    logging.info(f'save model to {output_dir}')
    if is_best:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


def load_eval_model(resume_path, model):
    if resume_path != '':
        if osp.exists(resume_path):
            print(f'==> model load from {resume_path}')
            checkpoint = torch.load(resume_path)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"==> checkpoint do not exists: \"{resume_path}\"")
            raise FileNotFoundError
    return model


def write_dict_to_json(mydict, f_path):
    import json
    import numpy
    class DateEnconding(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                numpy.uint16,numpy.uint32, numpy.uint64)):
                return int(obj)
            elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                numpy.float64)):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)): # add this line
                return obj.tolist() # add this line
            return json.JSONEncoder.default(self, obj)
    with open(f_path, 'w') as f:
        json.dump(mydict, f, cls=DateEnconding)
        print("write down det dict to %s!" %(f_path))