# SCRAPING STUFF Goes down in specificity
from bs4 import Comment

from BIFFScrapingHelpers import *


# TODO reset and redo

def main():
    """
    Process all the games this season
    """
    did_all_games = False
    for month in SEASON_MONTHS:
        if did_all_games:
            break
        did_all_games = process_one_month_schedule(month)

    save_all_dictionaries()


def process_one_month_schedule(month):
    """
    Process one month schedule for each game
    """
    schedule_url = BASE_URL + get_year_month_schedule_results_url(month)
    month_schedule_soup = get_soup(schedule_url)
    schedule_table = month_schedule_soup.find('table', id='schedule')
    table_body = schedule_table.find('tbody')

    # For each game this month
    for row in table_body.find_all('tr'):
        if 'thead' not in row.get('class', []):  # If thead, just a spacer row
            home_points = row.find('td', {'data-stat': 'home_pts'}).text
            if home_points == "":  # Stop if we reached games that haven't been played yet
                return True

            box_score_td = row.find('td', {'data-stat': 'box_score_text'})
            if box_score_td:

                # Date Stuff
                date_str = row.find('th', {'data-stat': 'date_game'}).find('a').get('href')
                day, month, year = get_date_info_from_href(date_str)

                # Time Stuff
                time_decimal = convert_to_24hr_decimal(row.find('td', {'data-stat': 'game_start_time'}).text)

                # Teams
                away_team = row.find('td', {'data-stat': 'visitor_team_name'}).get('csk').split('.')[0]
                home_team = row.find('td', {'data-stat': 'home_team_name'}).get('csk').split('.')[0]

                # Make Game ID
                uuid_input = str(year) + '-' + str(month) + '-' + str(day) + home_team + away_team
                game_id = makeUUID(uuid_input)

                # Check if already entered
                if check_game_id_exists(game_id, home_team, away_team):
                    continue

                # Else process game
                box_score_link = box_score_td.find('a')
                href = box_score_link.get('href')
                box_score_url = BASE_URL + href
                box_score_soup = get_soup(box_score_url)
                # Process the game
                process_one_game(box_score_soup, home_team, away_team, game_id, day, month, year, time_decimal)

    return False


def process_one_game(box_score_soup, home_team, away_team, game_id, day, month, year, time_decimal):
    """
    Process an entire game -team meta information and player information
    Args:
        box_score_soup:
        home_team:
        away_team:
        game_id:
        day:
        month:
        year:
        time_decimal:

    Returns:

    """

    home_result, away_result, home_inactive, away_inactive = parse_and_save_team_meta_data(box_score_soup, home_team,
                                                                                           away_team, game_id, day, month,
                                                                                           year, time_decimal)

    date_str_player_input = f"{year}-{month}-{day}"

    away_as_opp_stats = get_opp_stats_for_game(game_id, away_team, 0)
    home_as_opp_stats = get_opp_stats_for_game(game_id, home_team, 1)

    away_basic_box_table = box_score_soup.find('table', id=get_team_basic_box_id(away_team))
    home_basic_box_table = box_score_soup.find('table', id=get_team_basic_box_id(home_team))

    # Process all the players from away and home team
    process_players_from_team_box_score(away_basic_box_table, game_id, away_result, date_str_player_input, home_as_opp_stats)
    process_players_from_team_box_score(home_basic_box_table, game_id, home_result, date_str_player_input, away_as_opp_stats)

    # Save inactive players as dummy games
    for player in home_inactive:
        save_player_info_to_dict(player, game_id, date_str_player_input, DNP_PLAYER_STATS.copy(), away_as_opp_stats)
    for player in away_inactive:
        save_player_info_to_dict(player, game_id, date_str_player_input, DNP_PLAYER_STATS.copy(), home_as_opp_stats)

    # Save game_id to dictionary
    save_game_id_to_dict(game_id, home_team, away_team)


def parse_and_save_team_meta_data(box_score_soup, home_team, away_team, game_id, day, month, year, time):
    """
    Parses metadata from the box score and adds it to the respective team dictionaries.
    """
    try:
        # Records and scores
        scorebox_div = box_score_soup.find('div', class_='scorebox')
        scores_arr = scorebox_div.find_all('div', class_='scores')

        # Parse the scores
        away_score = float(scores_arr[0].find('div', class_='score').text)
        home_score = float(scores_arr[1].find('div', class_='score').text)

        away_result = away_score - home_score
        home_result = home_score - away_score

        # Parse team records
        away_record = get_record_from_str(scores_arr[0].next_sibling.text)
        home_record = get_record_from_str(scores_arr[1].next_sibling.text)

        # Calculate adjusted time
        home_adjusted_time = get_adjusted_time(home_team, time)
        away_adjusted_time = get_adjusted_time(away_team, time)

        # Parse Four Factors
        four_factors_comment = box_score_soup.find('div', id='all_four_factors').find(string=lambda text: isinstance(text, Comment))
        four_factors_soup = BeautifulSoup(four_factors_comment, 'html.parser')
        ff_rows = four_factors_soup.find('tbody').find_all('tr')

        away_ff_dict = get_four_factors_dict_from_soup(ff_rows[0])
        home_ff_dict = get_four_factors_dict_from_soup(ff_rows[1])

        # Rosters
        away_basic_box_table = box_score_soup.find('table', id=get_team_basic_box_id(away_team))
        home_basic_box_table = box_score_soup.find('table', id=get_team_basic_box_id(home_team))

        away_active_roster, away_full_roster = get_roster_from_basic_box(away_basic_box_table)
        home_active_roster, home_full_roster = get_roster_from_basic_box(home_basic_box_table)

        # Parse inactive players
        home_inactive, away_inactive = get_inactive_players(home_team, away_team, box_score_soup)
        home_full_roster.extend(home_inactive)
        away_full_roster.extend(away_inactive)

        # Date as a tuple
        date_tuple = (float(day), float(month), float(year))

        # Adding all metadata to teams dictionaries
        # Away team
        add_team_meta_info_to_dict(0, away_team, game_id, home_ff_dict, away_adjusted_time, away_record,
                                   away_active_roster, away_full_roster, date_tuple)
        # Home team
        add_team_meta_info_to_dict(1, home_team, game_id, away_ff_dict, home_adjusted_time, home_record,
                                   home_active_roster, home_full_roster, date_tuple)

        return home_result, away_result, home_inactive, away_inactive

    except Exception as e:
        print(f"Error: {e}")
        return 'BAD', 1, 1, 1, 1


def process_players_from_team_box_score(basic_box_table_soup, game_id, result, date_str, opp_stats):
    """
    Extracts player information from the basic box score table and saves it
    """

    body = basic_box_table_soup.find('tbody')
    rows = body.find_all('tr')

    starters = True

    for row in rows:
        if 'thead' not in row.get('class', []):
            name = row.find('th', {'data-stat': 'player'}).get('data-append-csv')
            # IF DNP
            if row.find('td', {'data-stat': 'reason'}) is not None:
                player_stats = DNP_PLAYER_STATS.copy()
            else:
                player_stats = get_player_stats_arr(row, result, starters)

            # Save off player info to the dictionary
            save_player_info_to_dict(name, game_id, date_str, player_stats, opp_stats)
        else:
            starters = False  # Players after the thead did not start


if __name__ == '__main__':
    main()
