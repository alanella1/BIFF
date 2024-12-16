import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from bs4 import Comment, BeautifulSoup

from BIFFEMAIL import send_email
from BIFFScrapingHelpers import (get_soup, get_team_info_dict, get_player_individual_games_dict, get_defensive_stats_array_from_games, PST_TZ,
                                 MST_TZ, \
                                 CST_TZ, EST_TZ, BASE_URL, get_player_name_from_href, EMPTY_PLAYER_NAME)
from BuildDatabase import GameToDictionary
from BuildDatabaseFromScraping import EAST
from DatabaseObjects import to_device, Game, Team, Player
from DictToTensor import GetOneAggroTensor
from SpreadModels import GameSpreadModel
from WinnerModels import GameWinnerModel


def get_booms(american_odds, percentage_win):
    if american_odds > 0:
        decimal_odds = (american_odds / 100)
    else:
        decimal_odds = (100 / abs(american_odds))

    kelly = ((decimal_odds * percentage_win) - (1 - percentage_win)) / (decimal_odds * 1.87)

    if kelly < 0.04:
        return 0
    elif 0.04 < kelly < 0.08:
        return 1
    elif 0.08 < kelly < 0.12:
        return 2
    elif kelly > 0.12:
        return 3


class NBAGame:
    def __init__(self, home_team_name, away_team_name, game_time_etc):
        # Always needed
        self.home_team = home_team_name
        self.away_team = away_team_name
        self.game_time_etc = game_time_etc

        # Predictions
        self.spread_prediction = None
        # Odds
        self.home_spread = None
        self.home_ml = None
        self.away_ml = None
        # Bets
        self.bet_home_spread = 0
        self.bet_away_spread = 0
        self.bet_home_ml = 0
        self.bet_away_ml = 0

    def add_predictions(self, spread_prediction):
        self.spread_prediction = spread_prediction

    def add_game_odds(self, home_spread, home_ml, away_ml):
        self.home_spread = home_spread
        self.home_ml = home_ml
        self.away_ml = away_ml

    def build_bets(self):
        # Spread First
        concerned_idx = 20 - int(self.home_spread)
        if self.home_spread % 1 == 0.5:
            total = 1.0
            perc_cover = np.sum(self.spread_prediction[concerned_idx:]) / total
        else:
            total = 1.0 - self.spread_prediction[concerned_idx]
            perc_cover = np.sum(self.spread_prediction[concerned_idx + 1:]) / total

        if perc_cover > 0.5:
            self.bet_home_spread = get_booms(-110, perc_cover)
        else:
            self.bet_away_spread = get_booms(-110, (1 - perc_cover))

        # MoneyLines
        self.bet_home_ml = get_booms(self.home_ml, np.sum(self.spread_prediction[20:]))
        self.bet_away_ml = get_booms(self.away_ml, np.sum(self.spread_prediction[:20]))


# TODO turn spread into odds
# Todo figure out percentages to bets
# todo tweet?
def load_model(is_spread_model):
    if is_spread_model:
        name = 'BIFF_FINAL_SPREAD_MODEL.pth'
        model_class = GameSpreadModel()
    else:
        name = 'BIFF_FINAL_WINNER_MODEL.pth'
        model_class = GameWinnerModel()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_MODELS")

    load_path = os.path.join(save_dir, name)

    model_class.load_state_dict(torch.load(load_path))
    model_class = model_class.to('cuda')
    model_class.eval()
    return model_class


def get_model_input_from_game_obj(game_obj):
    game_dict = GameToDictionary(game_obj)
    tensor_dict = to_device(game_dict, 'cuda')

    return tensor_dict


def get_spread_real_probs(input_tensor):
    probs = F.softmax(input_tensor, dim=0)
    return probs.detach().cpu().numpy()


def get_winner_real_probs(input_tensor):
    probs = torch.sigmoid(input_tensor)
    return probs.detach().cpu().numpy()


# region GET GAMES TODAY

def get_games_today():
    def get_team_from_anchor(anchor):
        href = anchor.get('href')
        return href.split('/')[-2]

    games_today = []
    url = "https://www.basketball-reference.com/"
    ref_soup = get_soup(url)
    all_rows = ref_soup.find('div', id='scheduled-games').find('table').find_all('tr')
    for row in all_rows:
        if row.find('th') is not None:
            if row.find('th').text == 'Today':
                continue
            else:
                break

        tds = row.find_all('td')

        if len(tds) == 1:  # Blank row at end of today's game list
            break
        teams = tds[0].find('small').find_all('a')
        game_time = tds[1].find('small').text

        away_team = get_team_from_anchor(teams[0])
        home_team = get_team_from_anchor(teams[1])
        pass
        # Create Obj
        this_game = NBAGame(home_team, away_team, game_time)
        games_today.append(this_game)

    return games_today


# endregion

# region time stuff
def time_to_decimal(time_str: str) -> float:
    # Parse the time string using strptime
    dt = datetime.strptime(time_str, '%I:%M %p')  # %I is 12-hour format, %p is AM/PM
    # Convert the hour and minutes to decimal format
    decimal_time = dt.hour + dt.minute / 60.0
    return decimal_time


def get_etc_adjusted_time(team, time):
    """
    Get the adjusted game start time for a team based on their home time zone
    """
    if team in PST_TZ:
        return time - 3
    elif team in MST_TZ:
        return time - 2
    elif team in CST_TZ:
        return time - 1
    elif team in EST_TZ:
        return time
    else:
        print("TEAM NO EXIST")


# end region


# region roster stuff
def get_player_object(player_name, player_dict):
    prev_season_tensors = []

    if '2025' not in player_dict.keys():
        return Player(EMPTY_PLAYER_NAME, None, None, True)

    for year, year_dict in player_dict.items():
        if int(year) < 2025:
            prev_season_tensors.append(year_dict['WHOLE_SEASON_TENSOR'])

    this_year_dict = player_dict['2025']
    p_stats = this_year_dict["Player Stats"]
    o_stats = this_year_dict["Opponent Stats"]
    this_year_tensor = GetOneAggroTensor(p_stats, o_stats)

    return Player(player_name, prev_season_tensors, this_year_tensor, False)


def get_players_avg_minutes(name, pig_dict):
    minutes_played_idx = 2

    if name not in pig_dict:
        return 0.0

    if '2025' not in pig_dict[name]:
        return 0.0

    year_dict = pig_dict[name]['2025']
    player_stats = year_dict['Player Stats']
    minutes_played = [game_stats[minutes_played_idx] for game_stats in player_stats]
    return np.mean(minutes_played)


def get_ordered_roster_for_team(team_name, pig_dict):
    all_players = []
    url = BASE_URL + "/teams/" + team_name + "/2025.html"
    team_soup = get_soup(url)
    roster_tbody = team_soup.find('div', id='div_roster').find('tbody')

    for row in roster_tbody.find_all('tr'):
        td = row.find('td', attrs={'data-stat': 'player'})
        href = td.find('a').get('href')
        name = get_player_name_from_href(href)
        all_players.append(name)

    avg_minutes_played = [get_players_avg_minutes(name, pig_dict) for name in all_players]
    sorted_players = [player for player, _ in sorted(zip(all_players, avg_minutes_played), key=lambda x: x[1], reverse=True)]

    # remove_injuries
    injured_players = []
    injury_div = (team_soup.find('div', id='all_injuries'))
    if injury_div:
        injury_comment = injury_div.find(string=lambda text: isinstance(text, Comment))
        injury_soup = BeautifulSoup(injury_comment, 'html.parser')
        injury_tbody = injury_soup.find('tbody')
        for row in injury_tbody.find_all('tr'):
            # check for probable
            note = row.find('td', attrs={'data-stat': 'note'}).text
            if 'probable' in note.lower():
                continue
            th = row.find('th')
            href = th.find('a').get('href')
            name = get_player_name_from_href(href)
            injured_players.append(name)

    injury_updated_list = [player for player in sorted_players if player not in injured_players]

    top_ten = injury_updated_list[:10]
    top_ten_objects = [get_player_object(player, pig_dict[player]) for player in top_ten]

    return top_ten, top_ten_objects


# endregion

# region Scraping Odds

def get_odds_soup():
    url = "https://sportsbook.draftkings.com/leagues/basketball/nba"
    return get_soup(url, True)


def get_odds_for_one_game(odds_soup, home_team, away_team):
    def is_team_name_on_kings(team_name, kings_text):
        if team_name == "CHO":
            return kings_text[:3] == "CHA"
        elif team_name == "LAC":
            return kings_text[:4] == "LA C"
        elif team_name == "LAL":
            return kings_text[:4] == "LA L"
        elif team_name == "SAS":
            return kings_text[:4] == "SA S"
        elif team_name == "NYK":
            return kings_text[:4] == "NY K"
        elif team_name == "BRK":
            return kings_text[:3] == "BKN"
        elif team_name == "NOP":
            return kings_text[:4] == "NO P"
        elif team_name == "GSW":
            return kings_text[:4] == "GS W"
        else:
            return team_name == kings_text[:3]

    # Hornets fix
    if home_team == "CHO":
        home_team = "CHA"

    if away_team == "CHO":
        away_team = "CHA"

    all_games_tbody = odds_soup.find('tbody', class_='sportsbook-table__body')
    all_rows = all_games_tbody.find_all('tr')
    away_row = None
    home_row = None
    for row in all_rows:
        team_name_div = row.find('div', class_='event-cell__name-text')
        if team_name_div:
            if is_team_name_on_kings(home_team, team_name_div.text):
                home_row = row
            elif is_team_name_on_kings(away_team, team_name_div.text):
                away_row = row

    home_tds = home_row.find_all('td')
    away_tds = away_row.find_all('td')

    home_spread = home_tds[0].find('span', class_="sportsbook-outcome-cell__line").text
    home_ml = home_tds[2].find('span').text
    away_ml = away_tds[2].find('span').text

    home_spread = home_spread.replace('−', '-')
    home_ml = home_ml.replace('−', '-')
    away_ml = away_ml.replace('−', '-')
    return float(home_spread), float(home_ml), float(away_ml)


# endregion


# region Process a Game


def build_meta_info_for_team(team_name, team_meta_dict, home_away, etc_game_time):
    home = home_away

    record_at_tip = team_meta_dict[team_name]['record'][-1]
    wins = record_at_tip[0]
    losses = record_at_tip[1]

    conference_seed = 0.0
    conference = 1.0 if team_name in EAST else 0.0

    def_stats_list = get_defensive_stats_array_from_games(team_meta_dict[team_name]['four_factors'])

    time_decimal = time_to_decimal(etc_game_time)
    adjusted_time = get_etc_adjusted_time(team_name, time_decimal)

    params = [home, wins, losses, conference_seed, conference, *def_stats_list, adjusted_time]
    return params


def build_game_object(home_team_name, away_team_name, etc_game_time, team_meta_dict, pig_dict):
    game = Game()
    game.set_game_id('realgame')

    # Make teams / set names
    home_team = Team()
    home_team.set_name(home_team_name)
    away_team = Team()
    away_team.set_name(away_team_name)

    # Get Meta Info
    home_meta_info = build_meta_info_for_team(home_team_name, team_meta_dict, 1, etc_game_time)
    home_team.set_meta_info(*home_meta_info)
    away_meta_info = build_meta_info_for_team(away_team_name, team_meta_dict, 0, etc_game_time)
    away_team.set_meta_info(*away_meta_info)

    # Roster Info
    home_top_ten, home_roster = get_ordered_roster_for_team(home_team_name, pig_dict)
    home_team.set_roster_info(home_top_ten, home_roster)
    away_top_ten, away_roster = get_ordered_roster_for_team(away_team_name, pig_dict)
    away_team.set_roster_info(away_top_ten, away_roster)

    # Add teams to game
    game.set_home_team(home_team)
    game.set_away_team(away_team)

    # Date stuff
    today = datetime.today()
    year = today.year
    month = today.month
    day = today.day
    game.set_meta_info(year, month, day)

    return game


def process_one_game(away_team, home_team, etc_game_time, team_meta_dict, pig_dict):
    game_obj = build_game_object(home_team, away_team, etc_game_time, team_meta_dict, pig_dict)

    spread_model = load_model(True)

    game_model_input = get_model_input_from_game_obj(game_obj)

    spread_output = spread_model(game_model_input)

    real_spread_output = get_spread_real_probs(spread_output)

    return real_spread_output


def process_all_games_today():
    # Load Dictionaries
    TEAM_META_DICT = get_team_info_dict()
    PIG_DICT = get_player_individual_games_dict()

    # Get games being played today
    games_today = get_games_today()

    # Get Odds soup to share
    odds_soup = get_odds_soup()

    # build each game object
    for game in games_today:
        spread_output = process_one_game(game.away_team, game.home_team, game.game_time_etc, TEAM_META_DICT, PIG_DICT)
        game.add_predictions(spread_output)

        home_spread, home_ml, away_ml = get_odds_for_one_game(odds_soup, game.home_team, game.away_team)

        game.add_game_odds(home_spread, home_ml, away_ml)

    for game in games_today:
        game.build_bets()

    send_email(games_today)


# endregion
if __name__ == '__main__':
    process_all_games_today()
