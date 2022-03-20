#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
from utils.utils import *
from modules.user import *
from modules.user_refine import *


if __name__ == '__main__':
    
    parser = get_args(flags="infer")
    FLAGS, unparsed = parser.parse_known_args()

    print("----------")
    print("INTERFACE:")
    print("  dataset", FLAGS.dataset)
    print("  log", FLAGS.log)
    print("  model", FLAGS.model)
    print("  infering", FLAGS.split)
    print("  pointrefine", FLAGS.pointrefine)
    print("----------\n")

    # open arch / data config file
    ARCH = load_yaml(FLAGS.model + "/arch_cfg.yaml")
    DATA = load_yaml(FLAGS.model + "/data_cfg.yaml")

    make_predictions_dir(FLAGS, DATA) # create predictions file folder
    check_model_dir(FLAGS.model)      # does model folder exist?

    # create user and infer dataset
    if not FLAGS.pointrefine:
        user = User(ARCH, DATA, datadir=FLAGS.dataset, outputdir=FLAGS.log,
                    modeldir=FLAGS.model, split=FLAGS.split)
    else:
        user = UserRefine(ARCH, DATA, datadir=FLAGS.dataset, outputdir=FLAGS.log,
                          modeldir=FLAGS.model, split=FLAGS.split)
    user.infer()
