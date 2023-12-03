import logging

import numpy as np
import torch
from torch.nn import functional as F
# from torch.utils.tensorboard import SummaryWriter

import configs.classification.class_parser_eval as class_parser_eval
import datasets.datasetfactory as df
import model.learner as Learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment

logger = logging.getLogger('experiment')

# This will be changed to args for the official release
global NUM_ITER
NUM_ITER = 8

global best_lr
best_lr = 0.0003

def load_model(args, config):
    if args['model_path'] is not None:
        net_old = Learner.Learner(config)
        net = torch.load(args['model_path'],
                         map_location="cpu")
        net.config = net_old.config

        for (n1, old_model), (n2, loaded_model) in zip(net_old.named_parameters(), net.named_parameters()):
            loaded_model.adaptation = old_model.adaptation
            loaded_model.meta = old_model.meta

        net.reset_vars()
    else:
        net = Learner.Learner(config)
    return net

def eval_iterator(split_mnist_test_loaders, device, maml):
    acc_list = []
    for task_id in range(5):
        correct = 0
        total = 0

        for _, batch in enumerate(split_mnist_test_loaders[task_id]):

            img = batch[0].to(device)
            target = batch[1].to(device)

            bsz = img.shape[0]
            logits_q = maml(img)[:, :2]

            pred_q = (logits_q).argmax(dim=1)

            correct += torch.eq(pred_q, target).sum().item()
            total += bsz
        acc_list.append(correct / total)
    return acc_list


def train_iterator(extra_dataset, device, maml, opt, run_id=0):
    loss_fn = torch.nn.CrossEntropyLoss()
    k_shot_train = 5
    compat_shape = [1, 84, 84]
    for split_id in range(5):
        extra_train_data = []
        extra_train_labels = []

        for class_id in range(split_id * 2, split_id * 2 + 2):
            indices = extra_dataset.targets == class_id
            extra_train_data.append(extra_dataset.data[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))
            extra_train_labels.append(extra_dataset.targets[indices][k_shot_train*run_id:k_shot_train*(run_id+1)].to(device))

        # class appears nth time only once all classes were seen for n-1 times for all n
        # i.e. classes appear shot-wise like 0, 1, 2, ..., 8, 9, 1, 2, ...
        extra_train_data = torch.stack(extra_train_data, dim=1)
        img = extra_train_data.reshape(2 * k_shot_train, *compat_shape)

        extra_train_labels = torch.stack(extra_train_labels, dim=1)
        y = extra_train_labels.reshape(2 * k_shot_train) - split_id * 2

        for iter in range(NUM_ITER):  # inner loop
            pred = maml(img)[:, :2]

            opt.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()


def main():
    p = class_parser_eval.Parser()
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args['seed'])

    my_experiment = experiment(args['name'], args, "../results/", commit_changes=False, rank=0, seed=1)

    data_train = df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args['path'])
    data_test = df.DatasetFactory.get_dataset("omniglot", train=False, background=False, path=args['path'])
    final_results_train = []
    final_results_test = []
    lr_sweep_results = []

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    config = mf.ModelFactory.get_model("na", args['dataset'], output_dimension=1000)

    maml = load_model(args, config)

    maml = maml.to(device)

    args['schedule'] = [int(x) for x in args['schedule'].split(":")]
    no_of_classes_schedule = args['schedule']
    print(args["schedule"])
    total_classes = 10

    import torchvision
    from torchvision.transforms import Compose, Resize, Lambda, ToTensor, Normalize
    print("Preparing Split-MNIST...")
    norm_params = {'mean': [0.1307], 'std': [0.3081]}
    compat_shape = [1, 84, 84]
    # no normalize is better
    # mnist_transform = Compose(
    #     [Resize(84), ToTensor(), Normalize(**norm_params)])
    mnist_transform = Compose(
        [Resize(84), ToTensor()])

    extra_dataset = torchvision.datasets.MNIST(
        download=True, root='./data', train=True)

    idx = np.arange(extra_dataset.__len__())
    train_indices= idx[:-5000]

    extra_dataset.targets = extra_dataset.targets[train_indices]
    extra_dataset.data = extra_dataset.data[train_indices]

    from torch.utils.data import Dataset
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            data_list = []
            targets_list = []
            self.transform = transform

            for index in range(len(dataset)):
                raw_data, _ = dataset[index]
                label = dataset.targets[index]
                transformed_data = self.transform(raw_data)
                data_list.append(transformed_data)
                if isinstance(label, int):
                    label = torch.tensor(label)
                targets_list.append(label)
            self.data = torch.stack(data_list, dim=0)
            self.targets = torch.stack(targets_list, dim=0)

        def __len__(self):
            return self.data

    extra_dataset = TransformedDataset(extra_dataset, mnist_transform)

    # test loader
    split_mnist_test_loaders = {}

    for split_id in range(5):  # 5 tasks
        # test set
        test_set = torchvision.datasets.MNIST(
            root='./data', train=False, transform=mnist_transform, download=True)
        idx_0 = test_set.targets == (split_id * 2)
        idx_1 = test_set.targets == (split_id * 2+1)
        idx_0_np = idx_0.numpy()
        idx_1_np = idx_1.numpy()
        idx_np = np.logical_or(idx_0_np, idx_1_np).astype(int)
        idx_np = np.argwhere(np.asarray(idx_np))
        idx = torch.from_numpy(idx_np).view(-1)
        test_set.targets = test_set.targets[idx] - split_id * 2
        test_set.data = test_set.data[idx]

        extra_test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=128, shuffle=False,
            pin_memory=True, num_workers=2, drop_last=True)

        split_mnist_test_loaders[split_id] = extra_test_loader

    results = {}
    for task_id in range(5):
        results[task_id] = []
    all_task_mean = []
    if True:
        logger.info("BEST LR %s= ", str(best_lr))

        for current_run in range(0, args['runs']):
            print('===============================')
            print(current_run)
            print('===============================')

            lr = best_lr

            maml.reset_vars()

            opt = torch.optim.Adam(maml.get_adaptation_parameters(), lr=lr)

            train_iterator(extra_dataset, device,maml, opt, current_run)

            logger.info("Result after one epoch for LR = %f", lr)

            correct_test_list = eval_iterator(split_mnist_test_loaders, device, maml)

            for i, acc in enumerate(correct_test_list):
                results[i].append(acc * 100)
                logger.debug(f'[run {current_run}] Acc Task {i}: {acc:.1f}')
            logger.debug(f'[run {current_run}] Mean over Tasks: {np.mean(correct_test_list) * 100:.2f}')
            all_task_mean.append(np.mean(correct_test_list) * 100)

    for task_id in range(5):
        logger.debug(f'Mean Acc Task {task_id}: {np.mean(results[task_id]):.2f} STD: {np.std(results[task_id]):.2f}')

    logger.debug(f'Total Mean Acc: {np.mean(all_task_mean):.2f} STD: {np.std(all_task_mean):.2f}')


if __name__ == '__main__':
    main()
