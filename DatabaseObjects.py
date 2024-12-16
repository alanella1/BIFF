import os
import pickle
import random
import threading

import torch
from torch.utils.data import Dataset

EMPTY_PLAYER_NAME = "XXXXXXXX01"


def result_to_class(result):
    # Clip the result to the range [-20, 20]
    clipped_result = max(-20, min(20, result))

    if clipped_result < 0:
        return clipped_result + 20  # Shifts -20 to 0 and -1 to 19
    else:
        return clipped_result + 19  # Shifts 1 to 20 and 20 to 39


class GameDataLoader(Dataset):
    def __init__(self, list_name, spread=True):
        self.list_name = list_name
        self.train_set = list_name.split('_')[0] == 'train'
        self.chunk_files = os.listdir(list_name)
        random.shuffle(self.chunk_files)
        self.current_chunk = None
        self.current_chunk_idx = -1
        self.current_item_idx = 0
        self.chunk = None
        self.chunk_size = 0
        self.shuffled_indices = []

        self.next_chunk = None
        self.prefetch_thread = None
        self.prefetch_ready = False
        self.for_spread = spread

    def __len__(self):
        if self.train_set:
            return 26103
        else:
            return 1068

    def load_chunk(self, chunk_idx):
        # print('loading: regular (' + str(chunk_idx) + ')')
        with open(f'{self.list_name}/{self.chunk_files[chunk_idx]}', 'rb') as f:
            self.current_chunk = pickle.load(f)
        self.current_chunk_idx = chunk_idx
        self.current_item_idx = 0
        self.shuffled_indices = list(range(len(self.current_chunk)))
        random.shuffle(self.shuffled_indices)
        self.prefetch_ready = False

    def prefetch_chunk(self, chunk_idx):
        if chunk_idx < len(self.chunk_files):
            # print('loading: prefetch (' + str(chunk_idx) + ')')
            with open(f'{self.list_name}/{self.chunk_files[chunk_idx]}', 'rb') as f:
                self.next_chunk = pickle.load(f)
            self.prefetch_ready = True  # Prefetch completed

    def start_prefetch(self):
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()  # Wait for previous thread to complete

        next_chunk_idx = self.current_chunk_idx + 1
        if next_chunk_idx < len(self.chunk_files):
            self.prefetch_thread = threading.Thread(target=self.prefetch_chunk, args=(next_chunk_idx,))
            self.prefetch_thread.start()

    def use_prefetched_chunk(self):
        """Use the preloaded chunk if it's available."""
        if self.prefetch_ready:
            self.current_chunk = self.next_chunk
            self.current_chunk_idx = self.current_chunk_idx + 1
            self.current_item_idx = 0
            self.shuffled_indices = list(range(len(self.current_chunk)))
            random.shuffle(self.shuffled_indices)
            self.prefetch_ready = False

    def __getitem__(self, idx):
        if self.current_chunk is None or self.current_item_idx >= len(self.current_chunk):
            if self.prefetch_ready:
                self.use_prefetched_chunk()
            else:
                self.current_chunk_idx += 1
                if self.current_chunk_idx >= len(self.chunk_files):
                    raise StopIteration

                self.load_chunk(self.current_chunk_idx)

            self.start_prefetch()

        item = self.current_chunk[self.shuffled_indices[self.current_item_idx]]
        self.current_item_idx += 1
        game_dict = to_device(item, 'cuda')

        target_result = game_dict['result']
        if self.for_spread:
            return game_dict, result_to_class(target_result)
        else:
            return game_dict, target_result


def GetGames(file_path):
    games_list = []
    chunk_files = sorted(os.listdir(file_path))
    for chunk_file in chunk_files:
        with open(file_path + '/' + chunk_file, 'rb') as f:
            chunk = pickle.load(f)
            games_list.extend(chunk)

    return games_list


def LoadTrainingSet(spread=True):
    return GameDataLoader('train_games', spread)


def LoadTestSet(spread=True):
    return GameDataLoader('test_games', spread)


class Player:
    def __init__(self, name="", prev_tensors=None, this_year_tensor=None, is_dummy=False):
        if prev_tensors is None:
            prev_tensors = []
        self.name = name
        self.prev_year_tensors = prev_tensors
        self.this_year_tensor = this_year_tensor
        self.is_dummy = is_dummy

    def add_prev_tensor(self, prev_tensor):
        self.prev_year_tensors.append(prev_tensor)

    def add_this_year_tensor(self, this_year_tensor):
        self.this_year_tensor = this_year_tensor

    def to_dict(self):
        return {
            'name': self.name,
            'prev_year_tensors': self.prev_year_tensors,
            'this_year_tensor': self.this_year_tensor,
            'is_dummy': self.is_dummy
        }


class Team:
    def __init__(self):
        self.home = None
        self.roster = []
        self.minutes_played_order = []
        self.wins = None
        self.losses = None
        self.conference_seed = None
        self.conference = None
        self.OTOVm = None
        self.OTOVo = None
        self.OORBm = None
        self.OORBo = None
        self.OEFGPm = None
        self.OEFGPo = None
        self.OFTRm = None
        self.OFTRo = None
        self.OORTGm = None
        self.OORTGo = None
        self.adjusted_game_time = None
        self.name = None

    def set_roster_info(self, minutes_played_order, players):
        self.minutes_played_order = minutes_played_order
        self.roster = players

    def set_meta_info(self, home, wins, losses, conference_seed, conference, otovm, otovo, oorbm, oorbo, oortgm, oortgo, oefgpm, oefgpo, oftrm, oftro,
                      adjusted_game_time):
        self.home = home
        self.wins = wins
        self.losses = losses
        self.conference_seed = conference_seed
        self.conference = conference
        self.OTOVm = otovm
        self.OTOVo = otovo
        self.OORBm = oorbm
        self.OORBo = oorbo
        self.OORTGm = oortgm
        self.OORTGo = oortgo
        self.OEFGPm = oefgpm
        self.OEFGPo = oefgpo
        self.OFTRm = oftrm
        self.OFTRo = oftro
        self.adjusted_game_time = adjusted_game_time

    def set_meta_info_from_df(self, meta_df):
        self.home = int(meta_df['Home/Away'].iloc[0])
        self.wins = int(meta_df['Wins'].iloc[0])
        self.losses = int(meta_df['Losses'].iloc[0])
        self.conference_seed = int(meta_df['Conference Standing'].iloc[0])
        self.conference = int(meta_df['Conference'].iloc[0])
        self.OTOVm = float(meta_df['OTOVm'].iloc[0])
        self.OTOVo = float(meta_df['OTOVo'].iloc[0])
        self.OORBm = float(meta_df['OORBm'].iloc[0])
        self.OORBo = float(meta_df['OORBo'].iloc[0])
        self.OEFGPm = float(meta_df['OEFGPm'].iloc[0])
        self.OEFGPo = float(meta_df['OEFGPo'].iloc[0])
        self.OFTRm = float(meta_df['OFTRm'].iloc[0])
        self.OFTRo = float(meta_df['OFTRo'].iloc[0])
        self.OORTGm = float(meta_df['OORTGm'].iloc[0])
        self.OORTGo = float(meta_df['OORTGo'].iloc[0])
        self.adjusted_game_time = float(meta_df['Game Time Adjusted'].iloc[0])

    def set_name(self, name):
        self.name = name

    def to_dict(self):
        return {
            'home': self.home,
            'roster': [player.to_dict() for player in self.roster],
            'minutes_played_order': self.minutes_played_order,
            'wins': self.wins,
            'losses': self.losses,
            'conference_seed': self.conference_seed,
            'conference': self.conference,
            'OTOVm': self.OTOVm,
            'OTOVo': self.OTOVo,
            'OORBm': self.OORBm,
            'OORBo': self.OORBo,
            'OORTGm': self.OORTGm,
            'OORTGo': self.OORTGo,
            'OEFGPm': self.OEFGPm,
            'OEFGPo': self.OEFGPo,
            'OFTRm': self.OFTRm,
            'OFTRo': self.OFTRo,
            'adjusted_game_time': self.adjusted_game_time,
            'name': self.name,
        }


class Game:
    def __init__(self):
        self.home_team = None
        self.away_team = None
        self.game_year = None
        self.game_month = None
        self.game_day = None
        self.result = None
        self.game_id = None

    def set_home_team(self, home_team):
        self.home_team = home_team

    def set_away_team(self, away_team):
        self.away_team = away_team

    def set_meta_info(self, game_year, game_month, game_day):
        self.game_year = game_year
        self.game_month = game_month
        self.game_day = game_day

    def set_game_id(self, game_id):
        self.game_id = game_id

    def set_meta_info_from_df(self, df):
        self.game_year = df["Game Year"].iloc[0]
        self.game_month = df["Game Month"].iloc[0]
        self.game_day = df["Game Day"].iloc[0]

    def set_result(self, result):
        self.result = result

    def to_dict(self):
        return {
            'home_team': self.home_team.to_dict(),
            'away_team': self.away_team.to_dict(),
            'game_year': self.game_year,
            'game_month': self.game_month,
            'game_day': self.game_day,
            'result': self.result,
            'game_id': self.game_id
        }


def to_device(game_dict, device):
    def tensor_to_device(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device)
        elif isinstance(tensor, list):
            return [tensor_to_device(t) for t in tensor]
        elif isinstance(tensor, dict):
            return {k: tensor_to_device(v) for k, v in tensor.items()}
        elif isinstance(tensor, int):
            return torch.tensor(tensor, dtype=torch.float32).to(device)
        elif isinstance(tensor, float):
            return torch.tensor(tensor, dtype=torch.float32).to(device)
        elif isinstance(tensor, bool):
            return torch.tensor(tensor, dtype=torch.float32).to(device)
        else:
            return tensor

    return tensor_to_device(game_dict)


def assert_roster_order_ok(team):
    roster = team['roster']
    roster_names_order = [player['name'] for player in roster]
    minutes_order = team['minutes_played_order']
    for idx in range(10):
        if roster_names_order[idx] != EMPTY_PLAYER_NAME:
            if roster_names_order[idx] != minutes_order[idx]:
                print('uh oh')


if __name__ == "__main__":
    train_set = LoadTrainingSet()
    print(len(train_set))
    test_set = LoadTestSet()
    print(len(test_set))
