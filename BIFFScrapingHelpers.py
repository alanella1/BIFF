import pickle
import time
import uuid
from datetime import datetime

import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# region CONSTANTS

#  Pickle Names
PIG_DICT_NAME = 'player_individual_games.pkl'
TEAM_INFO_DICT_NAME = 'biff_2025_team_info.pkl'
GAME_ID_DICT_NAME = 'biff_2025_game_ids.pkl'

# Scraping Constants
BASE_URL = 'https://www.basketball-reference.com'
EMPTY_PLAYER_NAME = "XXXXXXXX01"
NUM_PLAYER_FEATURES = 18
DNP_PLAYER_STATS = [0 for i in range(NUM_PLAYER_FEATURES)]
SEASON_MONTHS = ['october', 'november', 'december', 'january', 'february', 'march']
namespace = uuid.NAMESPACE_DNS
MIN_DELAY = 2.07
last_request_time = None

PLAYER_STATS_STR = "Player Stats"
OPP_STATS_STR = "Opponent Stats"
DATES_STR = "Dates"
GAME_IDS_STR = "Game Ids"

# TIME ZONES
PST_TZ = [
    "GSW",
    "LAC",
    "SDC",
    "LAL",
    "VAN",
    "SEA",
    "POR",
    "SAC",
]

MST_TZ = ["DEN", "PHO", "UTA"]

CST_TZ = [
    "NOK",
    "CHI",
    "DAL",
    "HOU",
    "MEM",
    "MIL",
    "MIN",
    "NOP",
    "NOH",
    "OKC",
    "KCK",
    "SAS",
]

EST_TZ = [
    "ATL",
    "BOS",
    "BRK",
    "NJN",
    "CHO",
    "CHA",
    "CHH",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
    "WSB",
]


# endregion

# region LOAD DICTIONARIES
def get_pickle(pickle_name):
    """
    wrapper to load pickles
    """
    with open(pickle_name, 'rb') as f:
        return pickle.load(f)


def get_player_individual_games_dict():
    """
    Get the dictionary of individual games played by players
    Returns:
        dictionary = { 'playname' :
        [year] : {'Dates' : ['YYYY-MM_DD], 'Player Stats': [SEE CreatePlayerTensor.GetPlayerStatsFromRow],
        'Opponent Stats': [SEE CreatePlayerTensors.GetOppMetaStatsFromRow], 'Game Ids': [xxxx222,...]
    """
    return get_pickle(PIG_DICT_NAME)


def get_team_info_dict():
    """
    Layout of dict

    team_meta_dict = {
        'ATL' = {
            game_id_order = [id1, id2, id3]
            record_after_game_complete = [(win,loss),(win,loss)..]
            game_dates = [(DD,MM,YYYY),...]
            four_factors = [
                {
                    OTOV: 1.0,
                    OORB: 1.0,
                    OORTG: 1.0,
                    OEFGP: 1.0,
                    OFTR: 1.0,
                }
            ]
            adjusted_game_time = [game_time1, game_time2, game_time3]
            rosters = [ [guy1,guy2,..guy10],[guy1..],..]
        }
    }
    """
    return get_pickle(TEAM_INFO_DICT_NAME)


def get_game_id_dict():
    """
    Layout of dict
    game_id_dict = {
        1111eee: ("ATL",'PHI")
    }
    """
    return get_pickle(GAME_ID_DICT_NAME)


PLAYER_STATS_DICT = get_player_individual_games_dict()
GAME_ID_DICT = get_game_id_dict()
TEAM_META_DICT = get_team_info_dict()


def save_all_dictionaries():
    """
    Save all the pickle dictionaries after processing
    """
    with open(PIG_DICT_NAME, 'wb') as f:
        pickle.dump(PLAYER_STATS_DICT, f)
    print(f"Saved PLAYER_STATS_DICT to {PIG_DICT_NAME}")

    with open(GAME_ID_DICT_NAME, 'wb') as f:
        pickle.dump(GAME_ID_DICT, f)
    print(f"Saved GAME_ID_DICT to {GAME_ID_DICT_NAME}")

    with open(TEAM_INFO_DICT_NAME, 'wb') as f:
        pickle.dump(TEAM_META_DICT, f)
    print(f"Saved TEAM_META_DICT to {TEAM_INFO_DICT_NAME}")


def reset_dictionaries():
    """
    Reset the dictionaries as they were before the any games processed
    """
    with open(TEAM_INFO_DICT_NAME, 'wb') as f:
        pickle.dump({}, f)
    print(f"Reset {TEAM_INFO_DICT_NAME} to empty.")

    with open(GAME_ID_DICT_NAME, 'wb') as f:
        pickle.dump({}, f)
    print(f"Reset {GAME_ID_DICT_NAME} to empty.")

    for player in list(PLAYER_STATS_DICT.keys()):
        if '2025' in PLAYER_STATS_DICT[player]:
            del PLAYER_STATS_DICT[player]['2025']

    with open(PIG_DICT_NAME, 'wb') as f:
        pickle.dump(PLAYER_STATS_DICT, f)

    print(f"Updated {PIG_DICT_NAME} after clearing '2025' data.")


# endregion

# region STRING BUILDING

def get_year_month_schedule_results_url(month):
    """
    Build the URL for NBA game schedule for a given month
    """
    return f'/leagues/NBA_2025_games-{month}.html'


def get_team_basic_box_id(team):
    """
    Build the box score element ID for a given team
    """
    return f'box-{team}-game-basic'


def get_url_from_player_name(name):
    """
    Build the URL for a player's page based on their name
    """
    first_letter = name[0]
    return f'/players/{first_letter}/{name}.html'


# endregion

# region STRING PARSING
def get_date_info_from_href(href):
    query_string = href.split('?')[-1]
    params = query_string.split('&')
    day = None
    month = None
    year = None

    for param in params:
        key, value = param.split('=')
        if key == 'day':
            day = value
        elif key == 'month':
            month = value
        elif key == 'year':
            year = value
    return day, month, year


def convert_to_24hr_decimal(time_str):
    # Add 'm' to create a valid AM/PM time string (e.g., "8:00p" becomes "8:00pm")
    if 'p' in time_str:
        time_str = time_str.replace('p', 'pm')
    elif 'a' in time_str:
        time_str = time_str.replace('a', 'am')

    # Parse the time string into a datetime object
    time_obj = datetime.strptime(time_str, '%I:%M%p')

    # Convert to 24-hour decimal time
    hours = time_obj.hour
    minutes = time_obj.minute

    # Decimal hour = hour + (minutes / 60)
    decimal_time = hours + minutes / 60.0

    return decimal_time


def makeUUID(date_string):
    id = str(uuid.uuid5(namespace, date_string))[:8]
    return id


def get_record_from_str(record_str):
    wins = float(record_str.split('-')[0])
    losses = float(record_str.split('-')[1])
    return (wins, losses)


def get_player_name_from_href(href):
    href = href.split('.')[0]
    name = href.split('/')[-1]
    return name


# endregion

# region SOUP STUFF
def get_soup(url, is_draft_kings=False):
    """
    Fetches a webpage and returns a BeautifulSoup object for parsing
    """
    global last_request_time
    retries = 0

    while retries < 3:
        # Get the current time
        current_time = time.time()

        # If this is not the first request, check if enough time has passed
        if last_request_time:
            time_since_last_request = current_time - last_request_time
            if time_since_last_request < MIN_DELAY:
                # If not enough time has passed, wait for the remaining time
                time_to_wait = MIN_DELAY - time_since_last_request
                time.sleep(time_to_wait)

        try:  # Try and get page
            if is_draft_kings:
                chrome_options = Options()
                chrome_options.add_argument("--headless")  # Run Chrome in headless mode
                # Set up the WebDriver (make sure ChromeDriver is in your PATH)
                service = Service("D:\chromedriver-win64\chromedriver.exe")
                driver = webdriver.Chrome(service=service, options=chrome_options)

                # Open the webpage
                driver.get(url)

                # Scroll down to the bottom of the page to load all content
                scroll_pause_time = 2  # Adjust time as needed
                last_height = driver.execute_script("return document.body.scrollHeight")

                while True:
                    # Scroll down to the bottom
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(scroll_pause_time)  # Wait to load the new content

                    # Calculate new scroll height and compare with last scroll height
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break  # Break the loop if no new content is loaded
                    last_height = new_height

                # Once fully loaded, get the page source
                page_source = driver.page_source

                # Parse the page source with BeautifulSoup
                soup = BeautifulSoup(page_source, 'html.parser')
                return soup

            # Send the request
            response = requests.get(url)
            last_request_time = time.time()

            # request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                return soup

            # Handle retryable errors
            elif response.status_code in [502, 404]:
                retries += 1
                time.sleep(retries ** 2)

            # Some other random issue
            else:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching {url}: {e}")
            retries += 1
            time.sleep(retries ** 2)

    return None


def get_four_factors_dict_from_soup(four_factors_soup):
    """
    Get the actual float values from a row in the four factors box
    Returns:
        Dict[str, float]: A dictionary with the four factors statistics, formatted as:
            {
                'OTOV': <turnover percentage>,
                'OORB': <offensive rebound percentage>,
                'OORTG': <offensive rating>,
                'OEFGP': <effective field goal percentage>,
                'OFTR': <free throw rate>
            }
    """
    try:
        ff_dict = {
            'OTOV': float(four_factors_soup.find('td', {'data-stat': 'tov_pct'}).text),
            'OORB': float(four_factors_soup.find('td', {'data-stat': 'orb_pct'}).text),
            'OORTG': float(four_factors_soup.find('td', {'data-stat': 'off_rtg'}).text),
            'OEFGP': float(four_factors_soup.find('td', {'data-stat': 'efg_pct'}).text),
            'OFTR': float(four_factors_soup.find('td', {'data-stat': 'ft_rate'}).text)
        }
    except AttributeError as e:
        raise ValueError(f"Missing data in the four factors table: {e}")
    except ValueError as e:
        raise ValueError(f"Error converting data to float in the four factors table: {e}")

    return ff_dict


def get_roster_from_basic_box(basic_box_table_soup):
    """
    Get the list of players who played, from the basic box table
    Returns:
        array of player shorthand names
    """
    roster = []  # List of all players (active and inactive)
    active_roster = []  # List of players to consider playing in this game

    # Get table containing player rows
    body = basic_box_table_soup.find('tbody')
    rows = body.find_all('tr')

    for row in rows:
        if 'thead' not in row.get('class', []):  # Skip header rows
            name = row.find('th', {'data-stat': 'player'}).get('data-append-csv')

            if row.find('td', {'data-stat': 'reason'}) is not None:  # Player is inactive for some reason
                active_roster.append(EMPTY_PLAYER_NAME)
            else:
                active_roster.append(name)

            # Add to full roster regardless
            roster.append(name)

    return active_roster[:10], roster


def get_inactive_players(home_team, away_team, soup):
    """
    Get list of inactive players for both home and away team
    Returns:
        tuple, of inactive players for home and away respectively
    """
    home_inactive = []
    away_inactive = []

    # Find the 'inactive' tag in the HTML
    inactive_tag = soup.find('strong', string=lambda text: text and text.startswith('Inactive'))

    if inactive_tag is not None:
        parent_div = inactive_tag.parent

        home_hrefs = get_inactive_player_hrefs_for_team(parent_div, home_team)
        away_hrefs = get_inactive_player_hrefs_for_team(parent_div, away_team)

        # Convert hrefs to player names
        home_inactive = [get_player_name_from_href(href) for href in home_hrefs]
        away_inactive = [get_player_name_from_href(href) for href in away_hrefs]

    return home_inactive, away_inactive


def get_player_stats_arr(row, result, starting):
    player_stats = [0.0] * NUM_PLAYER_FEATURES

    # Helper function to extract a float value from a row based on the 'data-stat' key
    def extract_stat(row: dict, stat_key: str) -> float:
        value = get_val_from_data_stat(row, stat_key)
        return float(value) if value else 0.0

    # Create array (LIKE CreatePlayerTensors.py GetPlayerStatsFromRow)
    player_stats[0] = result
    player_stats[1] = float(starting)
    player_stats[2] = time_str_to_decimal_minutes(get_val_from_data_stat(row, 'mp'))
    player_stats[3] = extract_stat(row, 'fg')
    player_stats[4] = extract_stat(row, 'fga')
    player_stats[5] = extract_stat(row, 'fg3')
    player_stats[6] = extract_stat(row, 'fg3a')
    player_stats[7] = extract_stat(row, 'ft')
    player_stats[8] = extract_stat(row, 'fta')
    player_stats[9] = extract_stat(row, 'orb')
    player_stats[10] = extract_stat(row, 'drb')
    player_stats[11] = extract_stat(row, 'ast')
    player_stats[12] = extract_stat(row, 'stl')
    player_stats[13] = extract_stat(row, 'blk')
    player_stats[14] = extract_stat(row, 'tov')
    player_stats[15] = extract_stat(row, 'pf')
    player_stats[16] = extract_stat(row, 'game_score')
    player_stats[17] = extract_stat(row, 'plus_minus')

    return player_stats


def get_val_from_data_stat(row, data_stat):
    """
    Get the value for a specific stat for a player in one game
    """
    data = row.find('td', {'data-stat': data_stat}).text
    if data == '':
        data = 0.0
    return data


def time_str_to_decimal_minutes(time_str):
    """
    Turn a time string like MM:SS into a decimal minutes for players play time
    """

    # Split the time string by colon
    parts = time_str.split(':')

    if len(parts) == 3:  # HH:MM:SS format
        hours, minutes, seconds = map(int, parts)
        total_minutes = hours * 60 + minutes + seconds / 60

    elif len(parts) == 2:  # MM:SS format
        minutes, seconds = map(int, parts)
        total_minutes = minutes + seconds / 60

    else:
        total_minutes = float(time_str)  # Handle any unexpected format
    return total_minutes


def get_inactive_player_hrefs_for_team(parent_div, team_abbreviation):
    """
    Get the href links for each team from the parent div
    Args:
        parent_div:
        team_abbreviation:

    Returns:

    """
    # Find the <strong> tag with the given team abbreviation
    team_tag = parent_div.find('strong', string=team_abbreviation)

    if not team_tag:
        print(f"Team {team_abbreviation} not found.")
        return []

    # Find the <span> that contains the team's <strong> tag
    team_span = team_tag.find_parent('span')

    # If no span is found, exit early
    if not team_span:
        print(f"No parent <span> found for team {team_abbreviation}.")
        return []

    hrefs = []

    # Iterate through the following siblings of the team's <span> tag
    for sibling in team_span.find_next_siblings():
        # Stop when encountering the next <strong> tag
        if sibling.find('strong'):
            break
        # Find all <a> tags and extract their href attributes
        if sibling.name == 'a':
            hrefs.append(sibling['href'])

    return hrefs


def get_player_age_in_year(name):
    """
    Get a player's age right now from their homepage
    """
    soup = get_soup(BASE_URL + get_url_from_player_name(name))
    if not soup:  # could be for dummy player name
        return 99

    parent_p = soup.find('strong', string=lambda text: text and text.startswith('Born')).parent
    born_year = int(parent_p.find('span', id='necro-birth').get('data-birth').split('-')[0])

    return 2025 - born_year


def try_get_player_age(name):
    """
    Try and get player age from available data, else search the web
    """
    if name not in PLAYER_STATS_DICT:
        return get_player_age_in_year(name)

    player_dict = PLAYER_STATS_DICT[name]

    if '2025' not in player_dict:
        return get_player_age_in_year(name)

    this_year_stats = player_dict['2025']

    if len(this_year_stats[OPP_STATS_STR]) == 0:
        return get_player_age_in_year(name)

    return this_year_stats[OPP_STATS_STR][-1][-1]  # player age last game


# endregion


def save_player_info_to_dict(name, game_id, date_str, player_stats, opp_stats):
    """
    Save a game for a player into their dictionary
    """
    # Initialize player's dictionary if it doesn't exist already
    if name not in PLAYER_STATS_DICT:
        PLAYER_STATS_DICT[name] = {
            '2025': {
                PLAYER_STATS_STR: [],
                OPP_STATS_STR: [],
                DATES_STR: [],  # YYYY-MM-DD
                GAME_IDS_STR: [],
            }
        }

    if '2025' not in PLAYER_STATS_DICT[name]:
        PLAYER_STATS_DICT[name]['2025'] = {
            PLAYER_STATS_STR: [],
            OPP_STATS_STR: [],
            DATES_STR: [],  # YYYY-MM-DD
            GAME_IDS_STR: [],
        }
    # If the game id is already accounted for just skip
    if game_id in PLAYER_STATS_DICT[name]['2025'][GAME_IDS_STR]:
        return

    player_age = try_get_player_age(name)
    opponent_stats_to_add = opp_stats + [player_age]

    PLAYER_STATS_DICT[name]['2025'][PLAYER_STATS_STR].append(player_stats)
    PLAYER_STATS_DICT[name]['2025'][OPP_STATS_STR].append(opponent_stats_to_add)
    PLAYER_STATS_DICT[name]['2025'][DATES_STR].append(date_str)
    PLAYER_STATS_DICT[name]['2025'][GAME_IDS_STR].append(game_id)


def save_game_id_to_dict(game_id, home_team, away_team):
    """
    Save a game to the game_id dictionary (if not already in there)
    """
    if game_id in GAME_ID_DICT.keys():
        return
    else:
        GAME_ID_DICT[game_id] = (home_team, away_team)


def get_adjusted_time(team, time):
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


def add_team_meta_info_to_dict(home_away, team, game_id, four_factors_dict, adj_time, record, active_roster, full_roster, date_tuple):
    """
    Add a team's meta information to the dictionary
    """

    # If team hasn't been entered yet (first game) add the dictionary format
    if team not in TEAM_META_DICT:
        TEAM_META_DICT[team] = {
            'game_id_order': [],
            'home_away': [],
            'game_dates': [],
            'four_factors': [],
            'adjusted_game_time': [],
            'active_roster': [],
            'full_roster': [],
            'record': [],
        }

    TEAM_META_DICT[team]['game_id_order'].append(game_id)
    TEAM_META_DICT[team]['home_away'].append(home_away)
    TEAM_META_DICT[team]['game_dates'].append(date_tuple)
    TEAM_META_DICT[team]['four_factors'].append(four_factors_dict)
    TEAM_META_DICT[team]['adjusted_game_time'].append(adj_time)
    TEAM_META_DICT[team]['active_roster'].append(active_roster)
    TEAM_META_DICT[team]['full_roster'].append(full_roster)
    TEAM_META_DICT[team]['record'].append(record)


# region DICTIONARY HELPERS

def get_opp_stats_for_game(game_id, opp_team, home_away):
    """
    Get the opponent stats array - like CreatePlayerTensors.GetOppMetaStatsFromRow
    """
    # Opponent hasn't played a game yet
    if opp_team not in TEAM_META_DICT:
        opp_defensive_stats_arr = [0.0] * 10
    else:
        opp_team_meta = TEAM_META_DICT[opp_team]
        opp_defensive_stats = opp_team_meta['four_factors']

        # If this game_id exists for opp team make sure to exclude it
        if game_id in opp_team_meta['game_id_order']:
            opp_team_idx = opp_team_meta['game_id_order'].index(game_id)
            opp_defensive_stats = opp_team_meta['four_factors'][:opp_team_idx]

        # Convert stats to mean/std format
        opp_defensive_stats_arr = get_defensive_stats_array_from_games(opp_defensive_stats)

    flipped_home_away = 1 - home_away  # Flip home away because it's from the opposing team's players perspective
    opp_defensive_stats_arr.insert(0, flipped_home_away)

    return opp_defensive_stats_arr


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


def check_game_id_exists(game_id, home_team, away_team):
    """
    Check if this game_id has already been saved in the dictionaries
    Returns:
        True if already entered else False
    """
    if game_id not in GAME_ID_DICT:
        return False

    # Check if both teams exist in TEAM_META_DICT
    if home_team not in TEAM_META_DICT or away_team not in TEAM_META_DICT:
        return False

    # Check if the game_id is listed for both home and away teams
    return (game_id in TEAM_META_DICT[home_team]['game_id_order'] and
            game_id in TEAM_META_DICT[away_team]['game_id_order'])
# endregion
