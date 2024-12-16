import datetime

import numpy as np

from BuildDatabase import OrderPlayersByMinutesPlayed, GetPlayerObject
from DatabaseObjects import *

NUM_PLAYER_FEATURES = 18
NUM_OPP_META_FEATURES = 12
PLAYER_STATS_STR = "Player Stats"
OPP_STATS_STR = "Opponent Stats"
DATES_STR = "Dates"
GAME_IDS_STR = "Game Ids"

EAST = ['BOS',
        'NYK',
        'MIL',
        'CLE',
        'ORL',
        'IND',
        'PHI',
        'MIA',
        'CHI',
        'ATL',
        'BRK',
        'TOR',
        'CHO',
        'WAS',
        'DET', ]

WEST = ['OKC',
        'DEN',
        'MIN',
        'LAC',
        'DAL',
        'PHO',
        'NOP',
        'LAL',
        'SAC',
        'GSW',
        'HOU',
        'UTA',
        'MEM',
        'SAS',
        'POR', ]


def get_player_current_dict(player_name):
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


def save_player_dict(name, player_dict):
    with open('player_individual_games.pkl', 'rb') as f:
        pig_dict = pickle.load(f)

    if name not in pig_dict:
        pig_dict[name] = player_dict

    else:
        pig_dict[name].update(player_dict)

    with open('player_individual_games.pkl', 'wb') as f:
        pickle.dump(pig_dict, f)


def get_game_id_dict(year):
    file_name = os.path.join('Scraping', 'game_id_dict_' + year + '.pkl')
    with open(file_name, 'rb') as f:
        game_id_dict = pickle.load(f)
    return game_id_dict


def get_player_stats_dict(year):
    file_name = os.path.join('Scraping', 'player_stats_dict_' + year + '.pkl')
    with open(file_name, 'rb') as f:
        player_stats_dict = pickle.load(f)

    return player_stats_dict


def save_player_stats_dict(year, name, player_info):
    file_name = os.path.join('Scraping', 'player_stats_dict_' + year + '.pkl')

    with open(file_name, 'rb') as f:
        player_stats_dict = pickle.load(f)

    player_stats_dict[name].update(player_info)

    with open(file_name, 'wb') as f:
        pickle.dump(player_stats_dict, f)

    return


def get_team_meta_dict(year):
    file_name = os.path.join('Scraping', 'team_meta_dict_' + year + '.pkl')
    with open(file_name, 'rb') as f:
        team_meta_dict = pickle.load(f)
    return team_meta_dict


def get_date_from_game_id(team_meta_dict, team, game_id):
    """
    Get the date as a string "YYYY-MM-DD" from a game_id
    """
    index = team_meta_dict[team]['game_id_order'].index(game_id)
    date_tup = team_meta_dict[team]['game_dates'][index]
    date_obj = datetime.date(int(date_tup[2]), int(date_tup[1]), int(date_tup[0]))
    date_str = date_obj.strftime("%Y-%m-%d")
    return date_str


def is_player_on_roster(team, game_id, team_meta_dict, player):
    """
    Check if player was on roster for this team for this game
    Returns:
        True if yes, false otherwise
    """
    team_info = team_meta_dict[team]
    game_idx = team_info['game_id_order'].index(game_id)
    roster = team_info['full_roster'][game_idx]
    if player in roster:
        return True
    else:
        return False


def get_players_team_for_game(game_id, player, team_meta_dict, game_id_dict):
    """
    Get the team the player was on for a specific game (whether they played or not)
    """
    possible_teams = game_id_dict[game_id]

    team_0 = possible_teams[0]
    team_1 = possible_teams[1]

    if is_player_on_roster(team_0, game_id, team_meta_dict, player):
        return team_0, team_1
    elif is_player_on_roster(team_1, game_id, team_meta_dict, player):
        return team_1, team_0
    else:
        pass


def get_defensive_stats_array_from_games(defensive_stats_list):
    """
    Given a list of dictionaries for individual game stats, make the mean/std list
    as seen in CreatePlayerTensors.GetOppMetaStatsFromRow
    """
    if len(defensive_stats_list) == 0:
        return [0.0] * 10
    otov = [item['OTOV'] for item in defensive_stats_list]
    oorb = [item['OORB'] for item in defensive_stats_list]
    oortg = [item['OORTG'] for item in defensive_stats_list]
    oefgp = [item['OEFGP'] for item in defensive_stats_list]
    oftr = [item['OFTR'] for item in defensive_stats_list]

    mean_std_array = [
        np.mean(otov), np.std(otov),
        np.mean(oorb), np.std(oorb),
        np.mean(oortg), np.std(oortg),
        np.mean(oefgp), np.std(oefgp),
        np.mean(oftr), np.std(oftr)
    ]

    return mean_std_array


def get_opp_stats_for_game(team_meta_dict, game_id, my_team, opp_team):
    """
    Get the opponent stats array - like CreatePlayerTensors.GetOppMetaStatsFromRow
    """
    # Home/Away
    my_team_idx = team_meta_dict[my_team]['game_id_order'].index(game_id)
    home_away = team_meta_dict[my_team]['home_away'][my_team_idx]

    # Opponent Defensive Stats
    opp_team_idx = team_meta_dict[opp_team]['game_id_order'].index(game_id)
    opp_defensive_stats = team_meta_dict[opp_team]['four_factors'][:opp_team_idx]
    opp_defensive_stats_arr = get_defensive_stats_array_from_games(opp_defensive_stats)

    opp_defensive_stats_arr.insert(0, home_away)

    return opp_defensive_stats_arr


def make_one_player_dict(year, name, player_info, team_meta_dict):
    # SEE CreatePlayerTensors.MakeOnePlayerDictionary - copying that with scraped data

    # Get current dictionary to add to (could be blank for rookie)
    player_dict = get_player_current_dict(name)

    # Create empty placeholders to fill out
    player_dict[year] = {}
    player_dict[year][GAME_IDS_STR] = []
    player_dict[year][DATES_STR] = []
    player_dict[year][PLAYER_STATS_STR] = []
    player_dict[year][OPP_STATS_STR] = []

    for i, game_id in enumerate(player_info['game_ids']):
        my_team = player_info['team_names'][i][0]
        opp_team = player_info['team_names'][i][1]
        date_str = get_date_from_game_id(team_meta_dict, my_team, game_id)
        player_stats = player_info['game_stats'][i]
        opp_stats = get_opp_stats_for_game(team_meta_dict, game_id, my_team, opp_team)
        opp_stats.append(player_info['age'])

        player_dict[year][GAME_IDS_STR].append(game_id)
        player_dict[year][DATES_STR].append(date_str)
        player_dict[year][PLAYER_STATS_STR].append(player_stats)
        player_dict[year][OPP_STATS_STR].append(opp_stats)

    return player_dict


def make_player_individual_tensor_dicts(year):
    player_stats_dict = get_player_stats_dict(year)
    game_id_dict = get_game_id_dict(year)
    team_meta_dict = get_team_meta_dict(year)

    players = player_stats_dict.keys()
    for player in players:
        print(player)
        player_info = player_stats_dict[player]
        player_dict = make_one_player_dict(year, player, player_info, team_meta_dict)
        save_player_dict(player, player_dict)


def set_team_and_save_player(year, player, player_info, game_id_dict, team_meta_dict):
    teams_info = []
    for game_id in player_info['game_ids']:
        my_team, opp_team = get_players_team_for_game(game_id, player, team_meta_dict, game_id_dict)
        teams_info.append((my_team, opp_team))
    player_info['team_names'] = teams_info
    save_player_stats_dict(year, player, player_info)


def put_teams_on_player_dict(year):
    player_stats_dict = get_player_stats_dict(year)
    game_id_dict = get_game_id_dict(year)
    team_meta_dict = get_team_meta_dict(year)
    players = player_stats_dict.keys()
    for player in players:
        print(player)
        set_team_and_save_player(year, player, player_stats_dict[player], game_id_dict, team_meta_dict)
    pass


def get_game_objects_for_whole_year(year):
    """
    Get list of game objects for whole year
    Returns:
        Array of games
    """
    # Load Dictionaries
    player_stats_dict = get_player_stats_dict(year)
    game_id_dict = get_game_id_dict(year)
    team_meta_dict = get_team_meta_dict(year)

    game_objs = []
    # Loop all games
    for game_id in game_id_dict.keys():
        game_obj = get_game_object(year, game_id, player_stats_dict, game_id_dict, team_meta_dict)
        if game_obj != False:
            game_objs.append(game_obj)

    return game_objs


def get_game_object(year, game_id, player_stats_dict, game_id_dict, team_meta_dict):
    """
    Create Game object from scraped data (Per - BuildDatabase.CreateGameObject)
    Returns:
        Game object from DatabaseObjects.Game()
    """

    # Initialize Objects
    game = Game()
    game.set_game_id(game_id)
    home_team = Team()
    away_team = Team()

    teams = game_id_dict[game_id]
    # GAME_ID_DICT[game_id] = (home_team, away_team) As built
    home_team.set_name(teams[0])
    away_team.set_name(teams[1])
    pass

    home_meta_info = get_meta_info_for_team(home_team.name, team_meta_dict[home_team.name], game_id)
    if home_meta_info == False:
        return False

    away_meta_info = get_meta_info_for_team(away_team.name, team_meta_dict[away_team.name], game_id)
    if away_meta_info == False:
        return False

    home_team.set_meta_info(*home_meta_info)
    away_team.set_meta_info(*away_meta_info)

    top_ten, home_roster = get_roster_for_team(team_meta_dict[home_team.name], game_id, year)
    home_team.set_roster_info(top_ten, home_roster)

    top_ten, away_roster = get_roster_for_team(team_meta_dict[away_team.name], game_id, year)
    away_team.set_roster_info(top_ten, away_roster)

    game.set_home_team(home_team)
    game.set_away_team(away_team)

    date_params = get_game_date(team_meta_dict[home_team.name], game_id)
    game.set_meta_info(*date_params)

    game.set_result(get_result_for_game(player_stats_dict, home_roster[0].name, game_id))

    return game


def get_result_for_game(player_stats_dict, player_name, game_id):
    player_dict = player_stats_dict[player_name]
    game_idx = player_dict['game_ids'].index(game_id)
    game_stats = player_dict['game_stats'][game_idx]
    result = game_stats[0]
    if result == 0.0:
        raise Exception("Fuck no result")
    return result


def get_game_date(team_dict, game_id):
    game_idx = team_dict['game_id_order'].index(game_id)
    dates_tup = team_dict['game_dates'][game_idx]
    return [dates_tup[2], dates_tup[1], dates_tup[0]]


def get_roster_for_team(team_dict, game_id, year):
    player_obj_arr = []
    game_idx = team_dict['game_id_order'].index(game_id)
    active_players = team_dict['active_roster'][game_idx]
    top_ten_players = OrderPlayersByMinutesPlayed(active_players, year, game_id)
    for player_id in top_ten_players:
        if player_id == EMPTY_PLAYER_NAME:
            player_obj_arr.append(Player(EMPTY_PLAYER_NAME, None, None, True))
        else:
            player_obj_arr.append(GetPlayerObject(player_id, year, game_id))

    return top_ten_players, player_obj_arr
    pass


def get_meta_info_for_team(team_name, team_dict, game_id):
    info_arr_len = 16  # SEE DatabaseObjects.Team.set_meta_info() getting all those params
    info_arr = [0.0 * info_arr_len]
    game_idx = team_dict['game_id_order'].index(game_id)
    if game_idx == 0:
        return False
    record_at_tip = team_dict['record'][game_idx - 1]
    if sum(record_at_tip) < 3:
        return False
    home = team_dict['home_away'][game_idx]
    wins = record_at_tip[0]
    losses = record_at_tip[1]
    conference_seed = 0.0  # NOT USED IN MODEL ANYMORE
    conference = 1.0 if team_name in EAST else 0.0
    def_stats_list = get_defensive_stats_array_from_games(team_dict['four_factors'][:game_idx])
    adj_game_time = team_dict['adjusted_game_time'][game_idx]
    params = [home, wins, losses, conference_seed, conference, *def_stats_list, adj_game_time]
    return params


if __name__ == '__main__':
    # make_player_individual_tensor_dicts('2024')
    pass
