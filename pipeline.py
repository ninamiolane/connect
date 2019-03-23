"""Data processing pipeline."""

import logging
import luigi
import matplotlib
matplotlib.use('Agg')  # NOQA
import os
import jinja2

HOME_DIR = '/home/nina/code/sommet'
DATA_DIR = os.path.join('/neuro/recordings', '2018-05-31_15-43-39')


OUTPUT_DIR = os.path.join(HOME_DIR, 'output0206')
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'training')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

DEBUG = False

LOADER = jinja2.FileSystemLoader('./templates/')
TEMPLATE_ENVIRONMENT = jinja2.Environment(
    autoescape=False,
    loader=LOADER)
TEMPLATE_NAME = 'report.jinja2'


class Filter(luigi.Task):
    file_list_path = './datasets/openneuro_files.txt'
    target_dir = os.path.join(NEURO_DIR, 't1scans')

    def requires(self):
        pass

    def run(self):
        with open(self.file_list_path) as f:
            all_files = f.readlines()

        Parallel(n_jobs=10)(delayed(self.dl_file)(f) for f in all_files)

    def output(self):
        return luigi.LocalTarget(self.target_dir)


class RunAll(luigi.Task):
    def requires(self):
        return {'filter': Filter()}

    def output(self):
        return luigi.LocalTarget('dummy')


def init():
    for directory in [OUTPUT_DIR, TRAIN_DIR, REPORT_DIR]:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            os.chmod(directory, 0o777)

    logging.basicConfig(level=logging.INFO)
    logging.info('start')
    luigi.run(
        main_task_cls=RunAll(),
        cmdline_args=[
            '--local-scheduler',
        ])


if __name__ == "__main__":
    init()
