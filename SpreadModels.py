import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from ModelHelperFunctions import *

SPREAD_DROPOUT_RATE = 0.10


class LinearCell(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearCell, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_size, output_size),
                                   nn.ReLU(),
                                   nn.LayerNorm(output_size),
                                   nn.Dropout(SPREAD_DROPOUT_RATE))

    def forward(self, x):
        return self.layer(x)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x is expected to have shape [batch_size, seq_len, input_dim]
        attn_weights = F.softmax(self.attn(x), dim=1)  # Shape: [batch_size, seq_len, 1]
        attn_output = torch.sum(attn_weights * x, dim=1)  # Shape: [batch_size, input_dim]
        return attn_output


class OneSeasonModel(nn.Module):
    def __init__(self):
        super(OneSeasonModel, self).__init__()
        self.attention = AttentionPooling(ONE_SEASON_OUTPUT_N)
        self.embedding_layer = LinearCell(NUM_PLAYER_FEATURES * NUM_OPP_META_FEATURES, 64)
        self.attention = AttentionPooling(64)
        self.output_layer = LinearCell(64, ONE_SEASON_OUTPUT_N)

    # season is tensor (season_len,18,12)
    def forward(self, season):
        if season.size(1) == 0:
            return torch.zeros((1, ONE_SEASON_OUTPUT_N), device='cuda')

        # Turn season into "batches" of individual games for faster processing
        batch_size, seq_len, num_player_features, num_opp_features = season.size()
        season = season.view(batch_size * seq_len, num_player_features * num_opp_features)

        season = (season - SINGLE_GAME_EVERYTHING_MEANS) / SINGLE_GAME_EVERYTHING_STDS

        season = self.embedding_layer(season)
        pooled_season = self.attention(season.unsqueeze(0))
        output = self.output_layer(pooled_season)

        return output


class PreviousSeasonsModel(nn.Module):
    def __init__(self):
        super(PreviousSeasonsModel, self).__init__()
        self.embedding_layer = LinearCell(NUM_PLAYER_FEATURES * NUM_OPP_META_FEATURES, 32)
        self.attention = AttentionPooling(32)
        self.output_layer = LinearCell(32, PREVIOUS_SEASONS_OUTPUT_N)

    def forward(self, prev_seasons):
        season_averages = [
            ((season.view(season.size()[0], -1) - SINGLE_GAME_EVERYTHING_MEANS) / SINGLE_GAME_EVERYTHING_STDS).mean(dim=0)
            for season in prev_seasons
        ]

        season_averages = torch.stack(season_averages, dim=0)
        embedded_seasons = self.embedding_layer(season_averages)
        pooled_seasons = self.attention(embedded_seasons.unsqueeze(0))
        output = (self.output_layer(pooled_seasons) + pooled_seasons)
        return output


class PlayerModel(nn.Module):
    def __init__(self):
        super(PlayerModel, self).__init__()
        self.prev_seasons_model = PreviousSeasonsModel()
        self.this_season_model = OneSeasonModel()
        self.prev_seasons = []

        self.regular_fc = nn.Sequential(
            LinearCell(ONE_SEASON_OUTPUT_N + PREVIOUS_SEASONS_OUTPUT_N, 128),
            LinearCell(128, 64),
        )

    def forward(self, player_dict):
        # Check if dummy
        if GetIsDummy(player_dict):
            return torch.zeros((1, PLAYER_MODEL_OUTPUT_N), device='cuda')

        # Model previous seasons
        self.prev_seasons = GetPreviousSeasons(player_dict)
        if len(self.prev_seasons) == 0:
            prev_season_output = torch.zeros((1, PREVIOUS_SEASONS_OUTPUT_N), device='cuda')
        else:
            prev_season_output = self.prev_seasons_model(self.prev_seasons)

        # Model this season so far
        this_season_games = GetThisSeasonTensor(player_dict)
        this_season_output = self.this_season_model(this_season_games.unsqueeze(0))

        player_input = torch.cat((this_season_output, prev_season_output), dim=1)
        output = self.regular_fc(player_input)
        output = output + player_input
        return output


class TeamModel(nn.Module):
    def __init__(self):
        super(TeamModel, self).__init__()
        self.player_model = PlayerModel()
        self.player_outputs = []
        self.roster = []
        encoder_layers = TransformerEncoderLayer(d_model=PLAYER_MODEL_OUTPUT_N, nhead=4, dropout=SPREAD_DROPOUT_RATE)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

    def forward(self, team_dict):
        self.player_outputs = []
        self.roster = GetRoster(team_dict)
        for player in self.roster:
            self.player_outputs.append(self.player_model(player))

        combined_outputs = torch.cat(self.player_outputs, 0).unsqueeze(0)
        combined_outputs = combined_outputs.transpose(0, 1)
        simple_mean = combined_outputs.mean(dim=0)

        transformer_output = self.transformer_encoder(combined_outputs)
        pooled_output = transformer_output.mean(dim=0)
        output_with_skip = pooled_output + simple_mean

        return output_with_skip


class GameSpreadModel(nn.Module):
    def __init__(self):
        super(GameSpreadModel, self).__init__()
        self.game = {}
        self.team_model = TeamModel()

        self.single_team_fc = nn.Sequential(
            LinearCell(FINAL_LAYER_INPUT_N, 64),
            LinearCell(64, SINGLE_TEAM_FINAL_OUTPUT_N),
        )

        self.combined_layer = nn.Sequential(
            LinearCell(SINGLE_TEAM_FINAL_OUTPUT_N * 2, 64),
            nn.Linear(64, 40)
        )

    def build_final_team_input_tensor(self, team):
        if team == 'home_team':
            team_dict = GetHomeTeam(self.game)
        else:
            team_dict = GetAwayTeam(self.game)

        return self.team_model(team_dict)

    # (searching opposite team to get passed in teams stats)
    def get_defensive_stats_for_team(self, team):
        if team == 'away_team':
            team_to_search = GetHomeTeam(self.game)
        else:
            team_to_search = GetAwayTeam(self.game)

        def_stats = torch.tensor(GetDefensiveStatsArray(team_to_search), dtype=torch.float32).to('cuda')
        standardized_stats = (def_stats - MODEL_OPPONENT_STATS_MEANS) / MODEL_OPPONENT_STATS_STDS
        return standardized_stats

    def get_team_meta_stats(self, team):
        # [wins,losses,conference_seed,conference,adjusted_game_time]
        if team == 'home_team':
            team_to_search = GetHomeTeam(self.game)
        else:
            team_to_search = GetAwayTeam(self.game)

        adj_wins = GetTeamWins(team_to_search).unsqueeze(0) / 82
        adj_losses = GetTeamLosses(team_to_search).unsqueeze(0) / 82
        conference = GetTeamConference(team_to_search).unsqueeze(0)
        adj_adj_game_time = GetTeamAdjGameTime(team_to_search).unsqueeze(0) / 24

        return torch.cat((adj_wins, adj_losses, conference, adj_adj_game_time))

    def build_meta_features(self):
        # To return [game_month_adj,game_day_adj,home_team_defense_stats_adj,home_meta_stats, away_team_defensive_stats_adj, away_meta_stats]
        # Date Stuff
        game_month, game_day = GetGameMonthDay(self.game)
        # Assuming game_month and game_day are already tensors
        game_month_adj = torch.tensor((game_month / 12)).clone().detach().float().unsqueeze(0).to('cuda')
        game_day_adj = torch.tensor((game_day / 30)).clone().detach().float().unsqueeze(0).to('cuda')

        # Home Team
        home_adj_def_stats = self.get_defensive_stats_for_team('home_team')
        home_meta_stats = self.get_team_meta_stats('home_team')

        # Away Team
        away_adj_def_stats = self.get_defensive_stats_for_team('away_team')
        away_meta_stats = self.get_team_meta_stats('away_team')

        home_meta_features = torch.cat((game_month_adj, game_day_adj, home_adj_def_stats, home_meta_stats, away_adj_def_stats, away_meta_stats))
        away_meta_features = torch.cat((game_month_adj, game_day_adj, away_adj_def_stats, away_meta_stats, home_adj_def_stats, home_meta_stats))

        return home_meta_features, away_meta_features

    def forward(self, game):
        self.game = game
        # = 1 + 1 + 10 + 4 + 10 + 4 = 30
        home_meta_features, away_meta_features = self.build_meta_features()

        home_roster_features = self.build_final_team_input_tensor('home_team')  # Pooled roster info (so ,64)
        away_roster_features = self.build_final_team_input_tensor('away_team')

        # Concatenate all features into a single tensor
        home_features = torch.cat((home_meta_features, home_roster_features.squeeze(0)))  # 64 + 30 = 94
        away_features = torch.cat((away_meta_features, away_roster_features.squeeze(0)))

        away_final_input = self.single_team_fc(away_features) + away_roster_features
        home_final_input = self.single_team_fc(home_features) + home_roster_features

        final_input = torch.cat((home_final_input.squeeze(), away_final_input.squeeze()))
        final_output = self.combined_layer(final_input)
        return final_output
