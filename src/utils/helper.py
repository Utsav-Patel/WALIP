import os
from torchvision.utils import save_image


def get_lr(optimizer):
    """Returns the current learning rate of the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer object.

    Returns:
        float: The current learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def flatten_dict(initial_dict):
    """Flattens a dictionary of nested dictionaries into a single-level dictionary.

    Args:
        initial_dict (dict): The dictionary to be flattened.

    Returns:
        dict: The flattened dictionary.
    """
    result_dict = dict()
    if not isinstance(initial_dict, dict):
        return result_dict

    for key, value in initial_dict.items():
        if isinstance(value, dict):
            result_dict.update(flatten_dict(value))
        else:
            result_dict[key] = value
    return result_dict


def generate_path(file_type: str, options: dict) -> str:
    """
    Generates a path to a file based on the file type and options.

    Args:
        file_type (str): The type of file to generate a path for.
        options (dict): A dictionary containing options for the path generation.

    Returns:
        str: The path to the file.
    """
    root_dir = '../dicts/'

    if 'selected' in options:
        selection = 's' if options["selected"] else 'u'

    if file_type.startswith('emb'):
        root_dir += 'embeddings/'

        if file_type == 'emb_txt':
            subdir = f'txt_emb/{options["word_data"]}/'
            file_name = f'txt_emb_{options["word_data"]}_{options["src_lang"]}_{options["tgt_lang"]}_{options["lang"]}_{options["data_mode"]}.npy'

        elif file_type == 'emb_img':
            subdir = f'img_emb/{options["image_data"]}/'
            file_name = f'img_emb_{options["image_data"]}_{options["lang"]}_k{options["num_images"]}_{selection}.npy'

        elif file_type == 'emb_fp':
            prefix = f'{options["image_data"]}_{selection}_{options["word_data"]}'
            subdir = f'fp/{prefix}/'
            file_name = f'fp_{prefix}_{options["src_lang"]}_{options["tgt_lang"]}_{options["lang"]}_{options["data_mode"]}_k{options["num_images"]}.npy'

        elif file_type == 'emb_fasttext':
            subdir = f'fasttext/{options["word_data"]}/'
            file_name = f'fasttext_{options["word_data"]}_{options["src_lang"]}_{options["tgt_lang"]}_{options["lang"]}_{options["data_mode"]}.npy'

        elif file_type == 'emb_htw':
            subdir = f'htw/{options["word_data"]}/'
            file_name = f'htw_{options["word_data"]}_{options["src_lang"]}_{options["tgt_lang"]}_{options["lang"]}_{options["data_mode"]}.npy'

        elif file_type == 'emb_globe':
            subdir = f'globe/{options["word_data"]}/'
            file_name = f'globe_{options["word_data"]}_{options["src_lang"]}_{options["tgt_lang"]}_{options["lang"]}_{options["data_mode"]}.npy'

    elif file_type.startswith('img'):
        subdir = f'images/{options["image_data"]}/'

        if file_type == 'img':
            file_name = f'img_{options["image_data"]}_k{options["num_images"]}_{selection}.npy'

        elif file_type == 'img_label':
            file_name = f'label_{options["image_data"]}.npy'

        elif file_type == 'img_index':
            file_name = f'index_{options["image_data"]}_{options["lang"]}.npy'

        elif file_type == 'img_shared_index':
            file_name = f'shared_index_{options["image_data"]}.npy'

    elif file_type == 'txt_single':
        subdir = f'texts/{options["word_data"]}/'
        file_name = f'{options["word_data"]}_{options["src_lang"]}_{options["tgt_lang"]}_{options["lang"]}_{options["data_mode"]}.txt'

    elif file_type == 'txt_pair':
        subdir = f'texts/{options["word_data"]}/'
        file_name = f'{options["word_data"]}_{options["src_lang"]}_{options["tgt_lang"]}_{options["data_mode"]}.txt'

    folder_path = os.path.join(root_dir, subdir)
    try_make_dir(folder_path)
    return os.path.join(folder_path, file_name)


def dict2clsattr(train_configs, model_configs):
    """
    Takes two dictionaries as input, combines them and creates a class object containing their key-value pairs as
    attributes.

    Args:
    - train_configs (dict): A dictionary containing training configurations.
    - model_configs (dict): A dictionary containing model configurations.

    Returns:
    - cfg_container (class): A class object containing attributes that are the same as the keys and values of the
                             input dictionaries.
    """

    cfgs = {**train_configs, **model_configs}

    class cfg_container:
        train_configs = train_configs
        model_configs = model_configs

    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container


class AverageMeter:
    """
    Computes and stores the average value of some quantity across time.

    Attributes:
    - value (float): The most recent value of the quantity.
    - average (float): The average value of the quantity across time.
    - total_sum (float): The sum of all values of the quantity.
    - count (int): The number of times the quantity has been recorded.
    """

    def __init__(self):
        """
        Initializes a new instance of the AverageMeter class.
        """
        self.reset()

    def reset(self):
        """
        Resets all attributes of the AverageMeter instance to their initial values.
        """
        self.value = 0
        self.average = 0
        self.total_sum = 0
        self.count = 0

    def update(self, value, n=1):
        """
        Updates the AverageMeter instance with a new value of the quantity.

        Args:
        - value (float): The new value of the quantity.
        - n (int, optional): The number of times the quantity was observed to take on the value "value". Defaults to 1.
        """
        self.value = value
        self.total_sum += value * n
        self.count += n
        self.average = self.total_sum / self.count



def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy of the model's predictions given the predicted output and target.

    Args:
        output (torch.Tensor): the model's predicted output of shape (batch_size, num_classes).
        target (torch.Tensor): the true labels of shape (batch_size).
        topk (tuple): the tuple of integers indicating the top-k values to compute.

    Returns:
        list: the list of accuracy values for each value in topk.

    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_accuracy(data, col_ind):
    """
    Computes the accuracy of the model given the input data and the column index.

    Args:
        data (list): the input data of the form [(input_1, label_1), (input_2, label_2), ..., (input_n, label_n)]
        col_ind (list): the list of predicted column indices.

    Returns:
        tuple: the accuracy value and a list of the wrong pairs in the form [(index, predicted_index, true_index), ...]

    """
    s = 0
    wrong_pairs = []
    for i in range(len(data)):
        if col_ind[i] == data[i][1]:
            s += 1
        else:
            wrong_pairs.append([i, col_ind[i], data[i][1]])
    acc = s / len(data) * 100
    print('Accuracy: {:.4f}/100'.format(acc))
    return acc, wrong_pairs


def log(logf, msg, print_to_console=True):
    """
    Write a message `msg` to the file `logf`, and optionally print it to the console.

    Args:
        logf: File object to write the log message to.
        msg: The message to be logged.
        print_to_console (optional): If True, the message will also be printed to the console. Default is True.
    """

    logf.write(msg + '\n')
    if print_to_console:
        print(msg)


def save_images(samples, save_path, nrows=0):
    """
    Save a batch of image `samples` to the specified `save_path`.

    Args:
        samples: The batch of images to be saved.
        save_path: The full path to the file where the images will be saved.
        nrows (optional): The number of rows to use when displaying the saved images.
            If not specified or set to 0, the number of rows is calculated based on the batch size.

    Returns:
        None
    """
    if nrows == 0:
        batch_size = len(samples)
        nrows = int(batch_size ** .5)
    save_image(samples.cpu(), save_path, nrow=nrows, normalize=True)


def setattr_cls_from_kwargs(cls, kwargs):
    """
    Set attributes on the specified `cls` object based on the key-value pairs in the `kwargs` dictionary.

    Args:
        cls: The object whose attributes will be modified.
        kwargs: A dictionary of key-value pairs where each key represents the name of an attribute to set,
            and the corresponding value represents the value to set it to.

    Returns:
        None
    """
    for key, value in kwargs.items():
        setattr(cls, key, value)


def try_make_dir(d):
    """
    Attempt to create a new directory at the specified `d` path, if it does not already exist.

    Args:
        d: The full path to the directory to create.

    Returns:
        None
    """
    os.makedirs(d, exist_ok=True)


def get_basename(fpath):
    """
    Get the base name of the file at the specified `fpath` path, without the file extension.

    Args:
        fpath: The full path to the file whose base name will be returned.

    Returns:
        The base name of the file, without the file extension.
    """
    return os.path.splitext(os.path.basename(fpath))[0]
