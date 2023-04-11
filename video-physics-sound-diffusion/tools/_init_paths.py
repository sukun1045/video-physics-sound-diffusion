import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

project_path = osp.dirname(this_dir)
add_path(project_path)

add_path(osp.join(project_path, 'libs', 'criterions'))
add_path(osp.join(project_path, 'libs', 'datasets'))
add_path(osp.join(project_path, 'libs', 'models'))
add_path(osp.join(project_path, 'libs', 'renders'))
add_path(osp.join(project_path, 'libs', 'trainers'))