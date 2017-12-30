import trainer
import trainer_parser
import copy
import traceback
import uuid
import os
import json
from datetime import datetime
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('bulk_json', None, "bulk_json")
DEFAULT_PARAMS = {
    'dataset_name': None,
    'dataset_dir': None,
    'num_dataset': None,
    'model_name': None,
    'num_classes': None,
    'num_channel': None,
    'num_shards': None,
    'train_name': "train",
    "test_name": "validation",
    'vis_epoch': 10,
    'num_vis_steps': 3,
    'batch_size': 16,
    'model_image_size': None,
    'deconv_image_size': 30,
    'summary_interval': 10,
    'summary_images': 32,
    'epoch': 20,
    'train': True,
    'eval': True,
    'bn': False,
    'add_image': True,
    'add_image_interval': 2,
    'pooling': True,
    'pool_size': "[3,3,3,3,3]",
    'pool_stride': "[2,2,2,2,2]",
    'strides': "[1,1,2,2,2]",
    'filter_size': 64,
    'shuffle_buffer': 50,
    'num_layers': 5,
    'num_dataset_parallel': 4,
    'restore_model_path': None,
    'preprocessing_name': None,
    'filters': "[9,9,9,9,9,9,9]",
    'weight_decay': 0.00004,
    'optimizer': 'rmsprop',
    'adadelta_rho': 0.95,
    'adagrad_initial_accumulator_value': 0.1,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'opt_epsilon': 1.0,
    'ftrl_learning_rate_power': -0.5,
    'ftrl_initial_accumulator_value': 0.1,
    'ftrl_l1': 0.0,
    'ftrl_l2': 0.0,
    'momentum': 0.9,
    'rmsprop_momentum': 0.9,
    'rmsprop_decay': 0.9,
    'learning_rate_decay_type': 'exponential',
    'learning_rate': 0.01,
    'end_learning_rate': 0.0001,
    'label_smoothing': 0.0,
    'learning_rate_decay_factor': 0.94,
    'num_epochs_per_decay': 2.0,
    'moving_average_decay': None,
    'cycle_learning_rate': True,
    'train_fraction': 0.9,
    'num_save_interval': 1
}

# grid_params = {
#     "dataset_name":
#         {
#             "block": {"dataset_dir": "/home/data/block"},
#             "direction": {"dataset_dir": "/home/data/direction"},
#             "flowers": {"dataset_dir": "/home/data/flowers"},
#             "cifar10": {"dataset_dir": "/home/data/cifar10"},
#             "new_dataset": {"dataset_dir": "/home/data/new_dataset", "train_fraction": 0.9, "num_classes": 3,
#                             "num_channel": 1, 'num_dataset': 4000, 'num_shards': 4},
#             "mnist": {"dataset_dir": "/home/data/mnist"}
#         }
#     ,
#     "model_name":
#         {
#             "ed":
#                 {
#                     "num_layers": 5,
#                     "bn": True,
#                     "add_image": True,
#                     "add_image_interval": 2,
#                     "strides": "[2,2,1,1,1]",
#                     "filter_size": "[64,64,64,64,64]",
#                     "filters": "[9,9,9,9,9]",
#                     "deconv_image_size": 30,
#                     "batch_size": 16,
#                     "preprocessing_name": None
#                 },
#             "conv":
#                 {
#                     "num_layers": [5, 3],
#                     "bn": True,
#                     "pooling": True,
#                     "pool_size": "[3,3,3,3,3]",
#                     "pool_stride": "[2,2,2,2,2]",
#                     "strides": "[2,2,1,1,1]",
#                     "filter_size": "[64,64,64,64,64]",
#                     "filters": "[9,9,9,9,9]",
#                     "batch_size": 16,
#                     "preprocessing_name": None
#                 },
#             "alexnet_v2": {"preprocessing_name": None},
#             "resnet_v2_152": {"preprocessing_name": None},
#             "resnet_v2_101": {"preprocessing_name": None},
#             "inception_resnet_v2": {"preprocessing_name": None},
#             "nasnet_large": {"preprocessing_name": None},
#             "cifarnet": {"preprocessing_name": [None, "cifarnet"]},
#             "vgg_19": {"preprocessing_name": [None, "vgg"]},
#             "inception_v1": {"preprocessing_name": None},
#             "inception_v2": {"preprocessing_name": None},
#             "inception_v3": {"preprocessing_name": None},
#             "inception_v4": {"preprocessing_name": "vgg"}
#         },
#     "optimizer": {
#         "sgd": None,
#         "rmsprop":
#             {
#                 "rmsprop_momentum": 0.9,
#                 "rmsprop_decay": 0.9,
#                 "opt_epsilon": 1e-10,
#             },
#         "adadelta":
#             {
#                 "adadelta_rho": 0.95,
#                 "opt_epsilon": 1.0,
#             },
#         "adagrad":
#             {
#                 "adagrad_initial_accumulator_value": 0.1
#             },
#         "adam":
#             {
#                 "adam_beta1": 0.9,
#                 "adam_beta2": 0.999,
#                 "opt_epsilon": 1e-8,
#             },
#         "ftrl":
#             {
#                 "ftrl_learning_rate_power": -0.5,
#                 "ftrl_initial_accumulator_value": 0.1,
#                 "ftrl_l1": 0.0,
#                 "ftrl_l2": 0.0
#             },
#         "momentum": {
#             "momentum": 0.9,
#         }
#     },
#     "learning_rate_decay_type":
#         {
#             "exponential":
#                 {
#                     "learning_rate": [0.01, 0.001],
#                     "label_smoothing": 0.0,
#                     "learning_rate_decay_factor": 0.94,
#                     "num_epochs_per_decay": 2.0,
#                     "moving_average_decay": None
#                 },
#             "polynomial":
#                 {
#                     "learning_rate": [0.01, 0.001],
#                     "end_learning_rate": 0.0001,
#                     "label_smoothing": 0.0,
#                     "num_epochs_per_decay": 2.0,
#                     "moving_average_decay": None,
#                     "cycle_learning_rate": True
#                 }
#         },
#     "epoch": 1,
#     "weight_decay": 0.00004,
#     "summary_interval": 10,
#     "summary_images": 32,
#     "shuffle_buffer": 100,
#     "num_dataset_parallel": 4
# }

grid_params = {
    # "train": False,
    # "eval": False,
    "dataset_name":
        {
            "mustang": {"dataset_dir": "/home/data/mustang/grayscale", "epoch": 5, 'vis_epoch': None,
                        'num_save_interval': 5, 'num_channel': 1, 'train_fraction': 0.9},
            # "direction": {"dataset_dir": "F:\data\grading\\direction", "epoch": 12, 'vis_epoch': 1,
            #               'num_save_interval': 4}
        }
    ,
    "model_name":
        {
            # "alexnet_v2": {"preprocessing_name": None, "batch_size": 128},
            # "conv":
            #     {
            #         "num_layers": [1, 2, 3, 4, 5],
            #         "bn": True,
            #         "pooling": True,
            #         "pool_size": "[3,3,3,3,3]",
            #         "pool_stride": "[2,2,2,2,2]",
            #         "strides": "[2,2,2,2,2]",
            #         "filter_size": "[128,128,128,128,128]",
            #         "filters": "[5,5,5,5,5]",
            #         "batch_size": 8,
            #         "preprocessing_name": None
            #     },
            # # "resnet_v2_152": {"preprocessing_name": None},
            # "resnet_v2_101": {"preprocessing_name": None, "batch_size": 8},
            # "resnet_v1_50": {"preprocessing_name": None, "batch_size": 32},
            # "resnet_v2_50": {"preprocessing_name": None, "batch_size": 32},
            # "inception_resnet_v2": {"preprocessing_name": None, "batch_size": 12},
            # "nasnet_large": {"preprocessing_name": None, "batch_size": 10},
            # "lenet": {"preprocessing_name": None, "batch_size": 32},
            # "nasnet_mobile": {"preprocessing_name": None, "batch_size": 12},

            # "cifarnet": {"preprocessing_name": [None, "cifarnet"]},
            # "vgg_19": {"preprocessing_name": None, "batch_size": 12},
            # "vgg_16": {"preprocessing_name": None, "batch_size": 12},
            # "vgg_a": {"preprocessing_name": None, "batch_size": 12},
            # "inception_v1": {"preprocessing_name": None, "batch_size": 16},
            # "inception_v2": {"preprocessing_name": None, "batch_size": 14},
            # "inception_v3": {"preprocessing_name": None, "batch_size": 12},
            "inception_v4": {"preprocessing_name": "crop_or_pad", "batch_size": 10}
        },
    "optimizer": {
        # "sgd": None,
        "rmsprop":
            {
                "rmsprop_momentum": 0.9,
                "rmsprop_decay": 0.9,
                "opt_epsilon": 1.0,
            },
    },
    "learning_rate_decay_type":
        {
            "exponential":
                {
                    "learning_rate": 0.01,
                    "label_smoothing": 0.0,
                    "learning_rate_decay_factor": 0.94,
                    "num_epochs_per_decay": 2.0,
                    "moving_average_decay": None
                },
        }
}

if FLAGS.bulk_json:
    grid_params = json.load(open(FLAGS.bulk_json))
    print("bulk json load")
else:
    json.dump(grid_params, open("grid_params.json", "w+"))

results = trainer_parser.parse_train_conf(grid_params)


# for result in results:
#     keys = list(result.keys())
#     keys.sort()
#     values = []
#     for key in keys:
#         values.append(result[key])
#
#     print(str(list(zip(keys, values))))


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


for i, result in enumerate(results):
    def_params = copy.deepcopy(DEFAULT_PARAMS)
    def_params.update(result)
    results[i] = def_params

print(len(results))
now = datetime.now().strftime('%Y%m%d%H%M%S')
for params in results:
    try:
        params["log_dir"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log_" + now,
                                         params["dataset_name"], params["model_name"], str(uuid.uuid4()))

        print("model_name", params["model_name"])
        print("dataset_name", params["dataset_name"])
        print("log_dir", params["log_dir"])
        print(params)
        if not os.path.exists(params["log_dir"]):
            os.makedirs(params["log_dir"])
        json.dump(params, open(os.path.join(params["log_dir"], "train_info.json"), mode="w"))
        params = Dict2Obj(params)
        trainer.train(params)
    except Exception as e:
        traceback.print_exc()
        # print(e)
    tf.reset_default_graph()
