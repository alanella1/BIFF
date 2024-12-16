import numpy as np
import torch

from CreatePlayerTensors import *

OPPONENT_STATS_MEANS = np.array([13.0604, 3.1462, 25.4623, 6.9134, 105.2187, 10.2970, 0.4896, 0.0587, 0.2922, 0.0919, 26.6513])
OPPONENT_STATS_STDS = np.array([2.0517, 0.7671, 4.3474, 1.6937, 12.7862, 2.5155, 0.0621, 0.0142, 0.0574, 0.0252, 4.2732])

PLAYER_STATS_MEANS = np.array(
    [23.6897, 3.6470, 8.0360, 0.7350, 1.5986, 1.7549, 2.3129, 1.0712, 3.0635, 2.1515, 0.7369, 0.4749, 1.3392, 2.0319, 7.4866, 0.0000])
PLAYER_STATS_STDS = np.array(
    [26.5831, 3.0620, 5.7535, 1.1990, 2.5160, 2.3658, 2.9052, 1.4302, 2.7039, 2.5148, 0.9875, 0.8877, 1.4091, 1.5088, 7.2119, 10.5796])


def GetPIGDict():
    with open('player_individual_games.pkl', 'rb') as f:
        pig_dict = pickle.load(f)
        return pig_dict


def GetMeanSTDForStats():
    pig_dict = GetPIGDict()
    all_player_stats = []
    all_opponent_stats = []

    for player_name, player_dict in pig_dict.items():
        for year, year_dict in player_dict.items():
            player_stats = year_dict[PLAYER_STATS_STR]
            opponent_stats = year_dict[OPP_STATS_STR]

            for p_stats, o_stats in zip(player_stats, opponent_stats):
                p_stats = np.nan_to_num(np.array(p_stats), nan=0.0)
                o_stats = np.nan_to_num(np.array(o_stats), nan=0.0)
                # Only care about non-zero games, need to preserve 0's
                if not np.all(np.array(p_stats[2:]) == 0):  # Exclude result and game_started [0,1]
                    all_player_stats.append(p_stats[2:])
                if not np.all(np.array(o_stats[1:]) == 0):  # Exclude home/away (1 or o)
                    all_opponent_stats.append(o_stats[1:])

    all_player_stats = np.array(all_player_stats)
    all_opponent_stats = np.array(all_opponent_stats)

    # Compute Means
    player_mean = np.mean(all_player_stats, axis=0)
    player_std = np.std(all_player_stats, axis=0)

    opponent_mean = np.mean(all_opponent_stats, axis=0)
    opponent_std = np.std(all_opponent_stats, axis=0)

    return player_mean, player_std, opponent_mean, opponent_std


def StandardizePlayerStats(player_stats):
    player_stats = np.nan_to_num(player_stats, nan=0.0)
    game_result = player_stats[0]
    game_started = player_stats[1]
    concerned_stats = player_stats[2:]
    if np.all(concerned_stats == 0):
        return player_stats

    standardized_stats = (concerned_stats - PLAYER_STATS_MEANS) / PLAYER_STATS_STDS
    return np.concatenate(([game_result, game_started], standardized_stats))


def StandardizeOpponentStats(opponent_stats):
    opponent_stats = np.nan_to_num(opponent_stats, nan=0.0)
    home_away = opponent_stats[0]
    concerned_stats = opponent_stats[1:]

    if np.all(concerned_stats == 0):
        return opponent_stats

    standardized_stats = (concerned_stats - OPPONENT_STATS_MEANS) / OPPONENT_STATS_STDS
    return np.concatenate(([home_away], standardized_stats))


def GetOneAggroTensor(p_stats, o_stats):
    # Get one 3D array # Shape: (num_games, num_player_features, num_opponent_features)
    game_tensors = []
    for i in range(len(p_stats)):
        p_tensor = torch.tensor(StandardizePlayerStats(p_stats[i]), dtype=torch.float32)
        o_tensor = torch.tensor(StandardizeOpponentStats(o_stats[i]), dtype=torch.float32)

        cross_multiplied = p_tensor.unsqueeze(1) * o_tensor.unsqueeze(0)
        game_tensors.append(cross_multiplied)

    return torch.stack(game_tensors)


def SavePlayerSeasonTensor(name, year, tensor):
    with open('player_individual_games.pkl', 'rb') as f:
        pig_dict = pickle.load(f)

    pig_dict[name][year]["WHOLE_SEASON_TENSOR"] = tensor
    with open('player_individual_games.pkl', 'wb') as f:
        pickle.dump(pig_dict, f)


def BuildSeasonAggroTensors():
    pig_dict = GetPIGDict()
    for player_name, player_dict in pig_dict.items():
        print(player_name)
        for year, year_dict in player_dict.items():
            if int(year) <= 2022:
                continue
            # Build Overall Tensor for whole season
            # TODO this again
            whole_season_tensor = GetOneAggroTensor(year_dict[PLAYER_STATS_STR], year_dict[OPP_STATS_STR])
            SavePlayerSeasonTensor(player_name, year, whole_season_tensor)

    return


def SaveIndividualPlayerPickle(name, player_dict):
    file_name = 'Player_Individual_Pickles/' + name + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(player_dict, f)
    return


def BuildPlayerIndividualPickles():
    pig_dict = GetPIGDict()
    for player_name, player_dict in pig_dict.items():
        print(player_name)
        SaveIndividualPlayerPickle(player_name, player_dict)


def GetMeanSTDForGameDescription():
    pig_dict = GetPIGDict()
    all_games_ever = []

    for player_name, player_dict in pig_dict.items():
        for year, year_dict in player_dict.items():
            whole_season_tensor = year_dict['WHOLE_SEASON_TENSOR']  # Shape is (seq_len, 18, 12)

            # Reshape each game in the season to (seq_len, 216)
            flattened_season = whole_season_tensor.view(whole_season_tensor.size(0), -1)  # Shape is (seq_len, 216)

            # Add this season's games to all_games_ever
            all_games_ever.append(flattened_season)

    # Stack all games from every season into one tensor
    all_games_ever = torch.cat(all_games_ever, dim=0)  # Shape is (total_games, 216)

    # Calculate the mean and std across all games for each of the 216 features
    mean = torch.mean(all_games_ever, dim=0)  # Shape is (216,)
    std = torch.std(all_games_ever, dim=0)  # Shape is (216,)

    # Print mean and std values as lists for easy hard-coding
    print("mean = [", ", ".join(f"{m.item():.6f}" for m in mean), "]")
    print("std = [", ", ".join(f"{s.item():.6f}" for s in std), "]")


if __name__ == "__main__":
    # GetMeanSTDForGameDescription()
    BuildSeasonAggroTensors()
