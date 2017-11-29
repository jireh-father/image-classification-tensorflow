import copy

TRAIN_KEYS = [
    'dataset_name', 'dataset_dir', 'model_name', 'batch_size', 'model_image_size', 'deconv_image_size',
    'summary_interval', 'summary_images', 'epoch', 'train', 'eval', 'bn', 'add_image', 'add_image_interval', 'pooling',
    'pool_size', 'pool_stride', 'strides', 'filter_size', 'shuffle_buffer', 'num_layers', 'num_dataset_parallel',
    'restore_model_path', 'preprocessing_name', 'filters', 'weight_decay', 'optimizer', 'adadelta_rho',
    'adagrad_initial_accumulator_value', 'adam_beta1', 'adam_beta2', 'opt_epsilon', 'ftrl_learning_rate_power',
    'ftrl_initial_accumulator_value', 'ftrl_l1', 'ftrl_l2', 'momentum', 'rmsprop_momentum', 'rmsprop_decay',
    'learning_rate_decay_type', 'learning_rate', 'end_learning_rate', 'label_smoothing', 'learning_rate_decay_factor',
    'num_epochs_per_decay', 'moving_average_decay', 'cycle_learning_rate', ]


def set_values(results, key, value):
    start_idx = 0
    if key in results[start_idx]:
        start_idx = len(results)
        results = results + copy.deepcopy(results)
    for i in range(start_idx, len(results)):
        results[i][key] = value
    return results, start_idx


def parse_train_conf(params, results=[{}]):
    for key in params:
        value = params[key]
        if isinstance(value, dict):
            for dict_key in value:
                results, start_idx = set_values(results, key, dict_key)
                if isinstance(value[dict_key], dict):
                    for inner_key in value[dict_key]:
                        if not isinstance(value[dict_key][inner_key], list):
                            for i in range(start_idx, len(results)):
                                results[i][inner_key] = value[dict_key][inner_key]
                        else:
                            for inner_value in value[dict_key][inner_key]:
                                if inner_key in results[start_idx]:
                                    inner_start_idx = len(results)
                                    results = results + copy.deepcopy(results[start_idx:])
                                    start_idx = inner_start_idx
                                for i in range(start_idx, len(results)):
                                    results[i][inner_key] = inner_value
        elif isinstance(value, list):
            for list_value in value:
                results, start_idx = set_values(results, key, list_value)
        else:
            results, start_idx = set_values(results, key, value)
    return results
