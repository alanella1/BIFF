import os
from os import path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DatabaseObjects import LoadTrainingSet
from SpreadModels import GameSpreadModel
from WinnerModels import GameWinnerModel

device = 'cuda'


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets):
        # predictions: [batch_size, 40] (logits)
        # targets: [batch_size] (true indices)

        # Step 1: Compute standard cross entropy loss
        ce_loss = F.cross_entropy(predictions, targets)

        # Step 2: Compute proximity rewards
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)

        # Create a tensor of indices [0, 1, 2, ..., 40] with shape [1, num_classes]
        indices = torch.arange(num_classes).unsqueeze(0).to(predictions.device)

        # Expand targets to shape [batch_size, 41] for broadcasting
        targets_expanded = targets.unsqueeze(1)

        # Calculate the absolute distance between each prediction index and the true index
        abs_dist = torch.abs(indices - targets_expanded)

        # Compute proximity weights: 1 / (1 + abs_dist^2)
        proximity_weights = 1 / (1 + abs_dist.float() ** 2)

        # Step 3: Apply softmax to the predictions to get probabilities
        probs = F.softmax(predictions, dim=1)

        # Multiply the probabilities by proximity weights
        weighted_probs = probs * proximity_weights

        # Step 4: Compute the proximity reward by summing weighted probabilities over the class dimension
        proximity_reward = torch.sum(weighted_probs, dim=1)

        # Step 5: Combine the cross-entropy loss and proximity reward
        # We minimize cross-entropy, but maximize proximity reward, so subtract it
        loss = ce_loss - torch.mean(proximity_reward)

        return loss


class NormalNLLLoss(nn.Module):
    def __init__(self):
        super(NormalNLLLoss, self).__init__()

    def forward(self, mu, log_sigma, y):
        sigma = torch.exp(log_sigma)  # Ensure sigma is positive
        nll = log_sigma + ((y - mu) ** 2) / (2 * sigma ** 2)

        hard_diff = abs(y - mu)
        return nll, hard_diff, sigma


def evaluate_model(model, test_set, criterion, training_spread):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for game_dict, target in test_set:
            try:
                if training_spread:
                    logits = model(game_dict).unsqueeze(0)
                    loss = criterion(logits, torch.tensor(target, dtype=torch.long, device=device).unsqueeze(0))
                    total_loss = total_loss + loss.item()
                else:
                    winner = torch.tensor([1.0 if target > 0 else 0.0], dtype=torch.float32).to(device)
                    logit = model(game_dict).view(-1)
                    correct = (torch.round(torch.sigmoid(logit)) == winner).float()
                    loss = criterion(logit, winner)
                    total_loss = total_loss + loss.item()
                    total_correct = total_correct + correct.item()
            except Exception as e:
                print(f"Error processing game ID {game_dict.get('game_id', 'Unknown')}: {e}")

    if not training_spread:
        print("Correct Rate: " + str(total_correct / len(test_set)))
    avg_loss = total_loss / len(test_set)
    return avg_loss


def custom_collate_fn(batch):
    # Since we're using batch size of 1, batch will always be a list with a single element
    return batch[0]


def train_model(base_name, num_epochs, model, optimizer, scheduler, criterion):
    if isinstance(model, GameSpreadModel):
        training_spread = True
    else:
        training_spread = False

    logger_folder = 'loggin_stuff' if training_spread else 'winner_loggin_stuff'

    train_logger = SummaryWriter(path.join(logger_folder, base_name, "train"), flush_secs=1)

    global_step = 0

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        model.train()
        train_set = LoadTrainingSet(training_spread)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        total_loss = 0
        total_correct = 0
        n = 0
        for i, (game_dict, target) in enumerate(train_loader):
            try:
                if training_spread:
                    logits = model(game_dict).unsqueeze(0)
                    loss = criterion(logits, torch.tensor(target, dtype=torch.long, device=device).unsqueeze(0))
                else:
                    logit = model(game_dict).view(-1)
                    winner = torch.tensor([1.0 if target > 0 else 0.0], dtype=torch.float32).to(device)
                    correct = (torch.round(torch.sigmoid(logit)) == winner).float()
                    loss = criterion(logit, winner)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                n = n + 1
                total_loss = total_loss + loss.item()
                if training_spread:
                    pass
                else:
                    total_correct = total_correct + correct.item()

            except Exception as e:
                print(str(e))

            global_step = global_step + 1

            if (i + 1) % 1000 == 0:
                avg_loss = total_loss / n
                train_logger.add_scalar('loss', avg_loss, global_step)
                scheduler.step(avg_loss)
                total_loss = 0

                if training_spread:
                    pass
                else:
                    avg_correct = total_correct / n
                    train_logger.add_scalar('correct', avg_correct, global_step)
                    total_correct = 0
                n = 0

        # Evaluate the model on the test set !! SKIPPING CAUSE FINAL TRAINING NOW ON ALL GAMES
        """test_set = LoadTestSet(training_spread)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        test_loss = evaluate_model(model, test_loader, criterion, training_spread)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}")
        train_logger.add_scalar('Test Loss', test_loss, global_step)"""

    train_logger.close()
    return model


def SaveModel(model_to_save):
    if isinstance(model_to_save, GameSpreadModel):
        is_spread_model = True
    else:
        is_spread_model = False

    if is_spread_model:
        name = 'BIFF_FINAL_SPREAD_MODEL.pth'
    else:
        name = 'BIFF_FINAL_WINNER_MODEL.pth'

    # Define the save directory and create it if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_MODELS")
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the save path
    save_path = os.path.join(save_dir, name)
    torch.save(model_to_save.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    spread_model = GameSpreadModel().to(device)
    spread_adam = torch.optim.Adam(params=spread_model.parameters(), lr=1e-4, weight_decay=1e-6)
    spread_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=spread_adam, mode='min', factor=0.5, patience=10)
    spread_criterion = CustomCrossEntropyLoss()
    # Only for 7 epochs in training
    # Dropout rate 0.1
    final_spread_model = train_model('FINAL_SPREAD_MODEL_3', 7, spread_model, spread_adam, spread_scheduler, spread_criterion)
    SaveModel(final_spread_model)

    winner_model = GameWinnerModel().to(device)
    winner_adam = torch.optim.Adam(params=winner_model.parameters(), lr=1e-4, weight_decay=1e-2)
    winner_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=winner_adam, mode='min', factor=0.2, patience=10)
    winner_criterion = nn.BCEWithLogitsLoss()
    # 5 Epochs
    # DROPOUT RATE 0.1
    final_winner_model = train_model('FINAL_WINNER_MODEL_3', 5, winner_model, winner_adam, winner_scheduler, winner_criterion)
    SaveModel(final_winner_model)
