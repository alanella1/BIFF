import torch.nn.functional as F

from SpreadModels import *


class DistributedCrossEntropy(nn.Module):
    def __init__(self):
        super(DistributedCrossEntropy, self).__init__()

    def forward(self, output, target):
        num_classes = output.size(1)
        target_item = target.item()
        # Compute softmax probabilities
        softmax_probs = F.log_softmax(output, dim=1)[0]

        total_weight = 1
        # Calculate the base cross-entropy loss for the correct class
        main_loss = -softmax_probs[target_item]
        total_loss = main_loss

        # Calculate weighted losses for neighboring classes
        for dist in range(1, 5):
            weight = 1.0 / (dist * 2) + 1

            # Left neighbor
            if target_item - dist >= 0:
                left_loss = -softmax_probs[target_item - dist]
                total_loss = total_loss + (weight * left_loss)
                total_weight = total_weight + weight
            # Right neighbor
            if target_item + dist < num_classes:
                right_loss = -softmax_probs[target_item + dist]
                total_loss = total_loss + (weight * right_loss)
                total_weight = total_weight + weight

        return total_loss / total_weight


# NOT USED ANYMORE, but save
class OnePreviousSeasonModel(nn.Module):
    def __init__(self):
        super(OnePreviousSeasonModel, self).__init__()
        self.lstm = nn.LSTM(input_size=NUM_PLAYER_FEATURES * NUM_OPP_META_FEATURES, hidden_size=ONE_SEASON_OUTPUT_N, batch_first=True)
        self.layer_norm = nn.LayerNorm(ONE_SEASON_OUTPUT_N)

    # season is tensor (batch_size, season_len, 18, 12)
    def forward(self, season):
        batch_size, seq_len, num_player_features, num_opp_features = season.size()
        season = season.view(batch_size, seq_len, num_player_features * num_opp_features)
        lstm_out, (h_n, c_n) = self.lstm(season)
        x = lstm_out[:, -1, :]  # Use the output of the last LSTM cell
        x = torch.relu(self.layer_norm(x))
        return x  # Want every sub-model to be norm/relu'd


def get_adj_loss(game_id, loss_dict):
    if 'first_time' in loss_dict.keys():
        return 1.0
    else:
        if game_id in loss_dict.keys():
            return loss_dict[game_id]
        else:
            return 1.0


def get_big_model_loss(dataset_loader, criterion):
    mini_models = get_mini_models()
    for model in mini_models:
        model.to(device)
        model.eval()

    total_loss = 0
    total_diff = 0
    total_sig = 0
    n = 0
    loss_dict = {}
    with torch.no_grad():
        for game_dict, result in dataset_loader:
            mu, log_sigma, error = get_ensemble_prediction(mini_models, game_dict)
            if error is None:
                loss, hard_diff, sigma = criterion(mu, log_sigma, result)
                total_loss += loss.item()
                total_diff += hard_diff.item()
                total_sig += sigma.item()
                n = n + 1
                loss_dict[game_dict['game_id']] = loss
            else:
                print(error)

    avg_loss = total_loss / n
    avg_diff = total_diff / n
    avg_sig = total_sig / n
    for game_id in loss_dict.keys():
        loss_dict[game_id] = loss_dict[game_id] / avg_loss

    return loss_dict, avg_loss, avg_diff, avg_sig


def save_mini_model(model, epoch):
    from torch import save
    model_name = 'mini_model_' + str(epoch) + '.pth'

    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'mini_models', model_name))


def get_mini_models():
    from torch import load
    mini_models_folder = path.join(path.dirname(path.abspath(__file__)), "mini_models")
    mini_models = []
    for filename in os.listdir(mini_models_folder):
        if filename.endswith(".pth"):
            # Create a new instance of the model class
            model = GameModel()  # Replace GameModel with your model class
            # Load the state dict into the model
            model.load_state_dict(load(path.join(mini_models_folder, filename), map_location=device))
            # Add the model to the list
            mini_models.append(model)

    return mini_models


def get_ensemble_prediction(mini_models, game_dict):
    try:
        mus = []
        sigmas = []
        for model in mini_models:
            mu, log_sigma = model(game_dict)
            mus.append(mu.cpu())
            sigmas.append(torch.exp(log_sigma).cpu())

        mus = np.array(mus)
        sigmas = np.array(sigmas)

        weights = 1.0 / (sigmas ** 2)
        weights = weights / weights.sum(axis=0)

        mu = np.sum(weights * mus, axis=0)

        combined_variance = np.sum(weights * (sigmas ** 2), axis=0)
        sigma = np.sqrt(combined_variance)
        log_sig = np.log(sigma)

        mu = torch.tensor(mu).to(device)
        log_sigma = torch.tensor(log_sig).to(device)

        return mu, log_sigma, None


    except Exception as e:
        return 0, 0, e


class GameDataset(Dataset):
    def __init__(self, list_name):
        self.list_name = list_name
        self.chunk_files = os.listdir(list_name)
        self.chunk_size = self.get_chunk_size()
        self.total_size = len(self.chunk_files) * self.chunk_size

    def get_chunk_size(self):
        with open(f'{self.list_name}/{self.chunk_files[0]}', 'rb') as f:
            chunk = pickle.load(f)
        return len(chunk)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        game_idx = idx % self.chunk_size
        with open(f'{self.list_name}/{self.chunk_files[chunk_idx]}', 'rb') as f:
            chunk = pickle.load(f)
        return chunk[game_idx]


def GetTrainingGames():
    return GetGames('train_games')


def GetTestGames():
    return GetGames('test_games')
