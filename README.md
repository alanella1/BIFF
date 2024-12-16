# BIFF: NBA Prediction Model (2024-2025 Season)

**BIFF** is an AI model designed to predict the outcomes of NBA games for the 2024-2025 season. The model outputs a probability distribution over 40 possible outcomes, ranging from the home team losing by 20+ points to the home team winning by 20+ points. This can be used for gambling as betting against the spread or on the moneyline.

---

## Features

- **Outcome Prediction**: BIFF predicts the likelihood of all possible outcomes for NBA games.
- **Dynamic Updates**: The model continuously updates internal player and team states based on the latest game data for the most accurate possible predictions.
- **Kelly Criterion Integration**: BIFF identifies optimal bets for each game, balancing risk and reward.

---

## File Descriptions

Here’s an overview of the key files in this project:

- **`BIFF.py`**  
  Automates the process of gathering game and betting data for the day. Outputs recommended bets based on the Kelly Criterion.

- **`BIFFScraping.py` & `BIFFScrapingHelpers.py`**  
  Collects game information from completed games during the current season. Updates internal player and team states for accurate predictions.

- **`BuildDatabase.py`**  
  Constructs the training game database using previously gathered data stored in CSV format.

- **`BuildDatabaseFromScraping.py`**  
  Adds recent game data to the training database by scraping more recent seasons that were not downloaded initially.

- **`CreatePlayerTensors.py`**  
  Converts player information into tensors formatted for use as inputs to the prediction model.

- **`DatabaseObjects.py`**  
  Defines class representations for games, players, and teams.

- **`DictToTensor.py`**  
  Provides helper functions to convert games stored in dictionary format into tensors for model training.

- **`ModelHelperFunctions.py`**  
  Includes functions for dictionary access and transformations.

- **`SpreadModels.py`**  
  Contains PyTorch classes that define the architecture of the prediction model.

- **`Training.py`**  
  Handles training of the model on historical game data. Saves the trained parameters for use during the 2024-2025 season.

---

## How BIFF Works

1. **Data Collection**:  
   - Daily game and betting data are gathered for predictions (`BIFF.py`).
   - Historical game data is scraped and added to the training database (`BIFFScraping.py` and `BuildDatabase*.py`).

2. **Model Training**:  
   - Player and team statistics are processed into tensors (`CreatePlayerTensors.py`).
   - Historical games are used to train the model (`Training.py`) and optimize parameters.

3. **Game Predictions**:  
   - Predictions are made using the trained model (`SpreadModels.py`).
   - Outputs include a probability distribution over game outcomes, which informs the recommended bets.

4. **Betting Optimization**:  
   - Bets are calculated using the Kelly Criterion, balancing potential gains and risk.

---
