'''Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti).'''

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from hack_utils import NUM_PTS, CROP_SIZE
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from hack_utils import ThousandLandmarksDataset
from hack_utils import restore_landmarks_batch, create_submission

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument('--name', '-n', help='Experiment name (for saving checkpoints and submits).',
                        default='baseline')
    parser.add_argument('--data', '-d', help='Path to dir with target images & landmarks.', default=None)
    parser.add_argument('--batch-size', '-b', default=512, type=int)  # 512 is OK for resnet18 finetune @ 6Gb of VRAM
    parser.add_argument('--epochs', '-e', default=1, type=int)
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', '-c', type=int)
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device, writer, epoch):
    model.train()
    train_loss = []
    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc='training...'):
        images = batch['image'].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch['landmarks']  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction='mean')
        train_loss.append(loss.item())
        writer.add_scalar('Train batch MSE', loss.item(), epoch * len(loader) + i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(train_loss)
    writer.add_scalar('Train epoch MSE', epoch_loss, epoch)
    return epoch_loss


def validate(model, loader, loss_fn, device, writer, epoch, scheduler=None):
    model.eval()
    val_loss = []
    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc='validation...'):
        images = batch['image'].to(device)
        landmarks = batch['landmarks']

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction='mean')
        val_loss.append(loss.item())
        writer.add_scalar('Val batch MSE', loss.item(), epoch * len(loader) + i)

    epoch_loss = np.mean(val_loss)
    writer.add_scalar('Val epoch MSE', epoch_loss, epoch)

    if scheduler is not None:
        scheduler.step(epoch_loss)
    return epoch_loss


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc='test prediction...')):
        images = batch['image'].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch['scale_coef'].numpy()  # B
        margins_x = batch['crop_margin_x'].numpy()  # B
        margins_y = batch['crop_margin_y'].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    print(torch.cuda.device_count(), 'gpus available')
    # 0. Initializing training
    if args.checkpoint is None:
        for i in range(100):
            folder_name = f'{args.name}_{i // 10}{i % 10}'
            checkpoint_path = os.path.join(args.data, 'checkpoints', folder_name)
            log_path = os.path.join(args.data, 'logs', folder_name)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)
                break
        training_state = {
            'best_checkpoints': [],
            'best_scores': [],
            'epoch': []
        }
    else:
        folder_name = f'{args.name}_{args.checkpoint // 10}{args.checkpoint % 10}'
        checkpoint_path = os.path.join(args.data, 'checkpoints', folder_name)
        log_path = os.path.join(args.data, 'logs', folder_name)
        training_state = torch.load(os.path.join(checkpoint_path, 'training_state.pth'))
    print(f'Results can be found in {folder_name}')
    writer = SummaryWriter(log_dir=log_path)

    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ('image',)),
        TransformByKeys(transforms.ToTensor(), ('image',)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ('image',)),
    ])

    print('Reading data...')
    datasets = torch.load(os.path.join(args.data, 'datasets.pth'))
    train_dataset = datasets['train_dataset']
    val_dataset = datasets['val_dataset']
    test_dataset = datasets['test_dataset']
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True,
                                       shuffle=True, drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True,
                                     shuffle=False, drop_last=False)

    print('Creating model...')
    device = torch.device('cuda: 0') if args.gpu else torch.device('cpu')
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} gpus')
        model = nn.DataParallel(model)
    if args.checkpoint is not None:
        model.load_state_dict(training_state['best_checkpoints'][0])
    model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = fnn.mse_loss

    # 2. train & validate
    print('Ready for training...')
    if args.checkpoint is None:
        start_epoch = 0
        best_val_loss = np.inf
    else:
        start_epoch = training_state['epoch'][0]
        best_val_loss = training_state['best_scores'][0]

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device, writer, epoch)
        val_loss = validate(model, val_dataloader, loss_fn, device, writer, epoch, scheduler)
        print('Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}'.format(epoch, train_loss, val_loss))
        print(f'Learning rate = {optimizer.param_groups[0]["lr"]}')
        if len(training_state['best_scores']) == 0:
            training_state['best_checkpoints'].append(model.state_dict())
            training_state['best_scores'].append(val_loss)
            training_state['epoch'].append(epoch)
            with open(os.path.join(checkpoint_path, 'training_state.pth'), 'wb') as fp:
                torch.save(training_state, fp)
        elif len(training_state['best_scores']) < 3 or val_loss < training_state['best_scores'][-1]:
            cur_val_index = 0
            for cur_val_index in range(len(training_state['best_scores'])):
                if val_loss < training_state['best_scores'][cur_val_index]:
                    break
            training_state['best_scores'].insert(cur_val_index, val_loss)
            training_state['best_checkpoints'].insert(cur_val_index, model.state_dict())
            training_state['epoch'].insert(cur_val_index, epoch)
            if len(training_state['best_scores']) > 3:
                training_state['best_scores'] = training_state['best_scores'][:3]
                training_state['best_checkpoints'] = training_state['best_checkpoints'][:3]
                training_state['epoch'] = training_state['epoch'][:3]
            with open(os.path.join(checkpoint_path, 'training_state.pth'), 'wb') as fp:
                torch.save(training_state, fp)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join(checkpoint_path, f'{args.name}_best.pth'), 'wb') as fp:
                torch.save(model.state_dict(), fp)


    # 3. predict
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True,
                                      shuffle=False, drop_last=False)

    with open(os.path.join(checkpoint_path, f'{args.name}_best.pth'), 'rb') as fp:
        best_state_dict = torch.load(fp, map_location='cpu')
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join(checkpoint_path, f'{args.name}_test_predictions.pkl'), 'wb') as fp:
        pickle.dump({'image_names': test_dataset.image_names,
                     'landmarks': test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join(f'{args.name}_submit.csv'))


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
