import numpy as np
import pandas as pd

from CreatePlayerTensors import GAME_IDS_STR, ExtractGameResultInt
from DatabaseObjects import *

CWD = os.getcwd()
EMPTY_PLAYER_NAME = "XXXXXXXX01"

with open('player_individual_games.pkl', 'rb') as f:
    PIG_DICT = pickle.load(f)


def SaveGamesList(games_list, list_name, chunk_size=200):
    os.makedirs(list_name, exist_ok=True)
    for i in range(0, len(games_list), chunk_size):
        chunk = games_list[i:i + chunk_size]
        with open(f'{list_name}/chunk_{i // chunk_size}.pkl', 'wb') as f:
            pickle.dump(chunk, f, protocol=4)


def LoadAndSaveTrainGames():
    from BuildDatabaseFromScraping import get_game_objects_for_whole_year
    train_games = []
    games_folder = os.path.join(CWD, 'DATA', 'Games')
    years = os.listdir(games_folder)

    # All previous games (not scraped)
    for year in years:
        print(year)
        year_games = os.listdir(os.path.join(games_folder, year))
        for game in year_games:
            game_obj = CreateGameObject(year, game)
            if game_obj != False:
                game_dict = GameToDictionary(game_obj)
                train_games.append(game_dict)

    # Do 2023 special because it's scraped
    last_year = '2023'
    game_objs = get_game_objects_for_whole_year(last_year)
    for game_obj in game_objs:
        game_dict = GameToDictionary(game_obj)
        train_games.append(game_dict)

    random.shuffle(train_games)
    print('LEN FINAL_TRAIN: ' + str(len(train_games)))
    SaveGamesList(train_games, 'train_games')


def LoadAndSaveTestGames():
    from BuildDatabaseFromScraping import get_game_objects_for_whole_year
    test_games = []
    game_objs = get_game_objects_for_whole_year('2024')
    for game_obj in game_objs:
        test_games.append(GameToDictionary(game_obj))
    print('done')
    SaveGamesList(test_games, 'test_games')


def LoadAndSaveFinalTrainSet():
    from BuildDatabaseFromScraping import get_game_objects_for_whole_year
    train_games = []
    games_folder = os.path.join(CWD, 'DATA', 'Games')
    years = os.listdir(games_folder)

    # All previous games (not scraped)
    for year in years:
        print(year)
        year_games = os.listdir(os.path.join(games_folder, year))
        for game in year_games:
            game_obj = CreateGameObject(year, game)
            if game_obj != False:
                game_dict = GameToDictionary(game_obj)
                train_games.append(game_dict)

    # Do 2023 and 2024 special because it's scraped
    scraped_years = ['2023', '2024']
    for scraped_year in scraped_years:
        print(scraped_year)
        game_objs = get_game_objects_for_whole_year(scraped_year)
        for game_obj in game_objs:
            game_dict = GameToDictionary(game_obj)
            train_games.append(game_dict)

    random.shuffle(train_games)
    print('LEN FINAL_TRAIN: ' + str(len(train_games)))
    SaveGamesList(train_games, 'train_games')


def LoadRandomGame():
    games_folder = os.path.join(CWD, 'DATA', 'Games')
    years = os.listdir(games_folder)
    random_year = random.choice(years)
    year = os.path.join(games_folder, random_year)
    year_games = os.listdir(year)
    random_game = random.choice(year_games)
    game_obj = CreateGameObject(random_year, random_game)
    return game_obj


def IsTeamHome(meta_df):
    return int(meta_df['Home/Away'].iloc[0]) == 1


def GetOrderedTeamDataFrames(game_folder, team_folders):
    team_0_folder = os.path.join(game_folder, team_folders[0])
    team_1_folder = os.path.join(game_folder, team_folders[1])

    team_0_meta_file_path = os.path.join(team_0_folder, "meta.csv")
    team_0_roster_file_path = os.path.join(team_0_folder, "roster.csv")
    team_0_meta_df = pd.read_csv(team_0_meta_file_path)
    team_0_roster_df = pd.read_csv(team_0_roster_file_path)

    team_1_meta_file_path = os.path.join(team_1_folder, "meta.csv")
    team_1_roster_file_path = os.path.join(team_1_folder, "roster.csv")
    team_1_meta_df = pd.read_csv(team_1_meta_file_path)
    team_1_roster_df = pd.read_csv(team_1_roster_file_path)

    if IsTeamHome(team_0_meta_df):
        return team_0_meta_df, team_0_roster_df, team_1_meta_df, team_1_roster_df, 0
    else:
        return team_1_meta_df, team_1_roster_df, team_0_meta_df, team_0_roster_df, 1


def GetPlayerDict(player_id):
    # IT's way faster to load whole pig dict
    return PIG_DICT[player_id]


def GetPlayerObject(player_id, this_game_year, game_id):
    prev_season_tensors = []
    player_dict = GetPlayerDict(player_id)

    if this_game_year not in player_dict.keys():
        return Player(EMPTY_PLAYER_NAME, None, None, True)
    for year, year_dict in player_dict.items():
        if year < this_game_year:
            prev_season_tensors.append(year_dict["WHOLE_SEASON_TENSOR"])

    this_season_dict = player_dict[this_game_year]
    if game_id in this_season_dict[GAME_IDS_STR]:
        this_season_tensor = this_season_dict["WHOLE_SEASON_TENSOR"][:this_season_dict[GAME_IDS_STR].index(game_id)]
    else:
        this_season_tensor = this_season_dict["WHOLE_SEASON_TENSOR"]

    return Player(player_id, prev_season_tensors, this_season_tensor, False)


def OrderPlayersByMinutesPlayed(top_ten_player_arr, year, game_id):
    minutes_played_idx = 2
    avg_minutes_arr = []
    for player in top_ten_player_arr:
        if player == EMPTY_PLAYER_NAME:
            avg_minutes_arr.append(0.0)
        else:
            if player not in PIG_DICT.keys():
                avg_minutes_arr.append(0.0)
            elif year not in PIG_DICT[player].keys():
                avg_minutes_arr.append(0.0)
            else:
                year_dict = PIG_DICT[player][year]
                if game_id in year_dict[GAME_IDS_STR]:
                    game_idx = year_dict[GAME_IDS_STR].index(game_id)
                else:
                    game_idx = -1
                stats_to_date = year_dict["Player Stats"][:game_idx]
                minutes_played = [game_stats[minutes_played_idx] for game_stats in stats_to_date]
                avg_minutes_played = np.mean(minutes_played)
                avg_minutes_arr.append(avg_minutes_played)

    sorted_pair = sorted(zip(avg_minutes_arr, top_ten_player_arr), reverse=True)
    _, sorted_names = zip(*sorted_pair)
    return list(sorted_names)


def BuildRosterFromDF(roster_df, year, game_id):
    player_obj_arr = []
    top_ten_players = roster_df["Player_ID"].head(10).to_numpy()
    top_ten_players = OrderPlayersByMinutesPlayed(top_ten_players, year, game_id)
    for player_id in top_ten_players:
        if player_id == EMPTY_PLAYER_NAME:
            player_obj_arr.append(Player(EMPTY_PLAYER_NAME, None, None, True))
        else:
            player_obj_arr.append(GetPlayerObject(player_id, year, game_id))

    return top_ten_players, player_obj_arr


def GetResultForGame(year, game_id, home_player):
    letter = home_player[0]
    player_file_name = home_player + str(year) + '.csv'
    full_csv_file_path = os.path.join(CWD, 'Data', 'PlayersPerGame', letter, home_player, player_file_name)
    df = pd.read_csv(full_csv_file_path)
    try:
        row = df[df["Game ID"] == game_id].iloc[0]
    except Exception as e:
        print("bad game: " + game_id)
        return False

    result = ExtractGameResultInt(row['Game Result'])
    return result


def GameToDictionary(game):
    game_dict = game.to_dict()
    return game_dict


def CreateGameObject(year, game_id):
    # Initialize Objects
    game = Game()
    game.set_game_id(game_id)
    home_team = Team()
    away_team = Team()

    game_folder = os.path.join(CWD, 'DATA', 'Games', year, game_id)
    team_folders = os.listdir(game_folder)
    # Get folder for teams
    (home_team_meta_df,
     home_team_roster_df,
     away_team_meta_df,
     away_team_roster_df,
     home_folder_idx) = GetOrderedTeamDataFrames(game_folder, team_folders)

    # Set Names
    home_team.set_name(team_folders[home_folder_idx])
    away_team.set_name(team_folders[home_folder_idx ^ 1])  # XOR (Opposite idx of home)

    # Set Meta Info
    home_team.set_meta_info_from_df(home_team_meta_df)
    away_team.set_meta_info_from_df(away_team_meta_df)

    # Not concerned with games where teams have played less than three games
    if home_team.wins + home_team.losses < 3 or away_team.wins + away_team.losses < 3:
        return False

    # Set Roster info
    home_min_played_order, home_roster_arr = BuildRosterFromDF(home_team_roster_df, year, game_id)
    home_team.set_roster_info(home_min_played_order, home_roster_arr)
    away_min_played_order, away_roster_arr = BuildRosterFromDF(away_team_roster_df, year, game_id)
    away_team.set_roster_info(away_min_played_order, away_roster_arr)

    game.set_home_team(home_team)
    game.set_away_team(away_team)
    game.set_meta_info_from_df(home_team_meta_df)

    # Remove April Games, they're weird
    if game.game_month == 4:
        return False

    result = GetResultForGame(year, game_id, home_min_played_order[0])
    if result == False:
        return False

    game.set_result(result)
    return game


if __name__ == '__main__':
    LoadAndSaveFinalTrainSet()
    # LoadAndSaveTestGames()
    # pig_dict = GetPIGDict()
    pass
