import os
import pickle
from datetime import datetime, timedelta

import pandas as pd

# GLOBALS
CWD = os.getcwd()
NUM_PLAYER_FEATURES = 18
NUM_OPP_META_FEATURES = 12
PLAYER_STATS_STR = "Player Stats"
OPP_STATS_STR = "Opponent Stats"
DATES_STR = "Dates"
GAME_IDS_STR = "Game Ids"
MAX_GAMES_ACCOUNTED = 86
NAN_DATE_STR = "NAN"


def CreateEmptyPIGDict():
    temp = {}
    with open('player_individual_games.pkl', 'wb') as f:
        pickle.dump(temp, f)

    print("done")


def GetPlayerCurrentDict(player_name):
    """
    Get the dictionary of all players' individual game data
        dict[player_name] = [years]
        dict[player_name][year] = {player_stats: [], opponent_stats: [], dates:[], game_ids: []}
    Args:
        player_name (): player's abbreviated name

    Returns:
        current state of that player's dictionary in the database or new empty one
    """
    with open('player_individual_games.pkl', 'rb') as f:
        pig_dict = pickle.load(f)
    if player_name not in pig_dict:
        return {}
    else:
        return pig_dict[player_name]


def SavePlayerDict(name, player_dict):
    with open('player_individual_games.pkl', 'rb') as f:
        pig_dict = pickle.load(f)

    if name not in pig_dict:
        pig_dict[name] = player_dict

    else:
        pig_dict[name].update(player_dict)

    with open('player_individual_games.pkl', 'wb') as f:
        pickle.dump(pig_dict, f)


def FixNANDates(dates_str_list):
    """
    Fix dates not found, because game id was not in games folder for some reason
    Args:
        dates_str_list (): list of date strings or NAN for not found dates

    Returns:
        list of dates, where NANs are simply the day after the date before it
    """
    if NAN_DATE_STR not in dates_str_list:
        return dates_str_list

    date_obj_list = []
    for date in dates_str_list:
        if date != NAN_DATE_STR:
            date_obj_list.append(datetime.strptime(date, '%Y-%m-%d'))
        else:
            date_obj_list.append(date)

    for i in range(len(date_obj_list)):
        if date_obj_list[i] == NAN_DATE_STR:
            date_obj_list[i] = date_obj_list[i - 1] + timedelta(days=1)

    return [date_obj.strftime('%Y-%m-%d') for date_obj in date_obj_list]


def GetPlayerNameYearFromFile(file):
    """
    Get an abbreviated name of a player and year from PPG file name
    Args:
        file (): aaaron2000.csv

    Returns:
        (abbreviated name, year)
    """
    file_name = file.split('.')[0]
    year = file_name[-4:]
    name = file_name[:-4]
    return name, year


def GetDateFromID(year, game_id):
    """
    From a game id and season year, get the exact date of a game
    Args:
        year (): season year
        game_id (): unique game id

    Returns:
        Date in string format MM-DD-YYYY
    """
    game_folder = os.path.join(CWD, "DATA", "Games", year, game_id)
    try:
        first_team = os.listdir(game_folder)[0]
    except FileNotFoundError as e:
        return NAN_DATE_STR
    meta_file = os.path.join(game_folder, first_team, "meta.csv")
    meta_df = pd.read_csv(meta_file)
    month = str(meta_df["Game Month"].iloc[0])
    day = str(meta_df["Game Day"].iloc[0])
    year = str(meta_df["Game Year"].iloc[0])
    date_str = year + "-" + month + "-" + day
    return date_str


def GetNumberOfBlanks(team, year, gameID, last_idx):
    """
    Given a last index checked and a new game ID, see how many games exist in between
    Args:
        team (): Team abbreviation (MIN)
        year (): Year to check
        gameID (): "Current" game ID to search up until
        last_idx (): Index of last game acknowledged  (< index of gameID no matter what)

    Returns:
        The number of games that exist IN BETWEEN last index checked and index of passed in gameID,
        and index of the gameID, to be used in the next call
    """
    file = os.path.join(CWD, "DATA", "GameOrders", year + ".csv")
    df = pd.read_csv(file)
    game_order = df[team].tolist()  # [game_id1, game_id2, ... last_game_id]
    blank_game_dates = []
    blank_game_ids = []
    last_idx += 1  # Increment last index

    if gameID == "":  # Checking until end of season
        for i in range(last_idx, len(game_order)):
            # Add dates and game ids to reference in players dictionary
            if str(game_order[i]) == 'nan':
                return i - last_idx, 0, blank_game_dates, blank_game_ids
            blank_game_dates.append(GetDateFromID(year, str(game_order[i])))
            blank_game_ids.append(game_order[i])
        return len(game_order) - last_idx, 0, blank_game_dates, blank_game_ids

    if gameID not in game_order:  # just going to take player's file as source of truth
        return 0, last_idx, [], []

    this_game_idx = game_order.index(gameID)

    if this_game_idx > last_idx:  # There must be some games in between
        for i in range(last_idx, this_game_idx):
            # Add dates and game ids to reference in players dictionary
            blank_game_dates.append(GetDateFromID(year, str(game_order[i])))
            blank_game_ids.append(game_order[i])

        return this_game_idx - last_idx, this_game_idx, blank_game_dates, blank_game_ids
    else:
        return 0, this_game_idx, [], []


def TimeToDecimalMinutes(time_str):
    # Split the time string by colon
    parts = time_str.split(':')
    # If time is in HH:MM:SS format
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        total_minutes = hours * 60 + minutes + seconds / 60
    # If time is in MM:SS format
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        total_minutes = minutes + seconds / 60
    else:
        total_minutes = float(time_str)  # Handle any unexpected format
    return total_minutes


def ExtractGameResultInt(game_result_str):
    """
    Turn game result string to int
    "W (+10)" -> 10
    "L (-20)" -> -20
    Args:
        game_result_str (): input game result string

    Returns:
        Int of game result
    """
    return int(game_result_str.split('(')[-1].strip(')').replace('+', ''))


def GetOppMetaStatsFromRow(row):
    opp_meta_stats = [0.0] * NUM_OPP_META_FEATURES
    opp_meta_stats[0] = row["Home/Away"]
    opp_meta_stats[1] = row["OTOVm"]
    opp_meta_stats[2] = row["OTOVo"]
    opp_meta_stats[3] = row["OORBm"]
    opp_meta_stats[4] = row["OORBo"]
    opp_meta_stats[5] = row["OORTGm"]
    opp_meta_stats[6] = row["OORTGo"]
    opp_meta_stats[7] = row["OEFGPm"]
    opp_meta_stats[8] = row["OEFGPo"]
    opp_meta_stats[9] = row["OFTRm"]
    opp_meta_stats[10] = row["OFTRo"]
    opp_meta_stats[11] = row["Player Age Years"]
    return opp_meta_stats


def GetPlayerStatsFromRow(row):
    player_stats = [0.0] * NUM_PLAYER_FEATURES
    player_stats[0] = ExtractGameResultInt(row["Game Result"])
    player_stats[1] = row["Game Started"]
    player_stats[2] = TimeToDecimalMinutes(row["Minutes Played"])
    player_stats[3] = row["Field Goals Made"]
    player_stats[4] = row["Field Goals Attempted"]
    player_stats[5] = row["3 Pointers Made"]
    player_stats[6] = row["3 Pointers Attepmted"]
    player_stats[7] = row["Free Throws Made"]
    player_stats[8] = row["Free Throws Attempted"]
    player_stats[9] = row["Offensive Rebounds"]
    player_stats[10] = row["Defensive Rebounds"]
    player_stats[11] = row["Assists"]
    player_stats[12] = row["Steals"]
    player_stats[13] = row["Blocks"]
    player_stats[14] = row["Turnovers"]
    player_stats[15] = row["Fouls"]
    player_stats[16] = row["Game Score"]
    player_stats[17] = row["Plus/Minus"]
    return player_stats


def TranslateDateToDash(date_str):
    try:
        dash_date = pd.to_datetime(date_str, format="%m/%d/%Y").strftime('%Y-%m-%d')
    except ValueError:
        dash_date = date_str
    return dash_date


def MakeOnePlayerDictionary(file_name, df):
    """
    Make a dictionary for one player from a file for one year's data
    Args:
        file_name (): "aaron2000.csv"
        df (): Data frame of that file's information

    Returns:
        Nothing
    """
    df.reset_index()
    name, year = GetPlayerNameYearFromFile(file_name)

    # Get player's existing dictionary and add a new node for this year
    player_dict = GetPlayerCurrentDict(name)
    player_dict[year] = {}
    player_dict[year][PLAYER_STATS_STR] = []
    player_dict[year][OPP_STATS_STR] = []
    player_dict[year][DATES_STR] = []
    player_dict[year][GAME_IDS_STR] = []

    last_idx_checked = -1
    last_team = ''

    blank_game_player_info = [0.0] * NUM_PLAYER_FEATURES
    blank_game_opp_info = [0.0] * NUM_OPP_META_FEATURES
    for index, row in df.iterrows():
        team = row['Team']
        last_team = team
        game_id = row['Game ID']
        num_blanks, last_idx_checked, blank_dates, blank_game_ids = GetNumberOfBlanks(team, year, game_id, last_idx_checked)
        for i in range(num_blanks):  # Number of games missed since last game
            player_dict[year][DATES_STR].append(blank_dates[i])
            player_dict[year][GAME_IDS_STR].append(blank_game_ids[i])
            player_dict[year][PLAYER_STATS_STR].append(blank_game_player_info.copy())
            player_dict[year][OPP_STATS_STR].append(blank_game_opp_info.copy())

        # Now Add this game's information
        player_dict[year][DATES_STR].append(TranslateDateToDash(row['Game Date']))
        player_dict[year][GAME_IDS_STR].append(game_id)
        player_dict[year][PLAYER_STATS_STR].append(GetPlayerStatsFromRow(row))
        player_dict[year][OPP_STATS_STR].append(GetOppMetaStatsFromRow(row))

    # Now check until the end of the season
    rest_of_season_games, _, blank_dates, blank_game_ids = GetNumberOfBlanks(last_team, year, "", last_idx_checked)
    for i in range(rest_of_season_games):
        player_dict[year][DATES_STR].append(blank_dates[i])
        player_dict[year][GAME_IDS_STR].append(blank_game_ids[i])
        player_dict[year][PLAYER_STATS_STR].append(blank_game_player_info.copy())
        player_dict[year][OPP_STATS_STR].append(blank_game_opp_info.copy())

    player_dict[year][DATES_STR] = FixNANDates(player_dict[year][DATES_STR])
    return player_dict


def MakePlayerTensors():
    """
    Make a dictionary of tensors for players each game stats and opponents defensive stats
    Returns:
        Nothing
    """
    players_per_game_folder = CWD + "/Data/PlayersPerGame"
    letters = os.listdir(players_per_game_folder)
    for letter in letters:
        print(letter)
        players_of_letter_folder = os.path.join(players_per_game_folder, letter)
        players = os.listdir(players_of_letter_folder)
        for player in players:
            individual_player_folder = os.path.join(players_of_letter_folder, player)
            years = os.listdir(individual_player_folder)
            for year_file in years:
                df = pd.read_csv(os.path.join(individual_player_folder, year_file), dtype={"Game Date": str})
                player_dict = MakeOnePlayerDictionary(year_file, df)
                SavePlayerDict(player, player_dict)


# _____________ TEST METHODS _______________


def TestNameYearWorks(name, year):
    """
    Test to make sure formatting a player's name and year works
    from just a file name
    Args:
        name (): player abbreviated name
        year (): year to test

    Returns:
    True if player name / year works
    """
    players_per_game_folder = CWD + "/Data/PlayersPerGame"
    letter = name[0]
    players_in_letter = os.listdir(players_per_game_folder + "/" + letter)
    if name in players_in_letter:
        pass
    else:
        print(name + " doesn't exist")
        return False

    year = int(year)
    if year < 2000 or year > 2022:
        print(str(year) + " bad year")
        return False

    return True


def TestGamesAccountedFor(num_games, year):
    """
    Test to make sure after adding blanks, players are accounted for every game that season
    Args:
        num_games (): num games calculated by MakePlayerDictionary
        year (): year to test\
    Returns:
        True if correct number of games else False
    """
    year = int(year)
    if year == 2012:
        return num_games == 66
    elif year == 2013:
        return num_games == 81 or num_games == 82
    elif year == 2020:
        return num_games in [64, 65, 66, 67]
    elif year == 2021:
        return num_games == 72
    elif year == 2022:
        return num_games in [82, 83, 84]
    else:
        return num_games == 82

# MakePlayerTensors()
