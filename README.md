# FPL Points Predictor

A project to predict how many points Fantasy Premier League players will score, and then use those predictions to pick a team.

I play FPL, and I wanted to see whether a machine learning model could pick players better than just going on recent form.

## What it does

1. Downloads past FPL data (every player's stats for every match since 2020-21)
2. Builds features for each player before each match, like recent points, minutes played, xG, and how good the opponent is
3. Trains a LightGBM model to predict points for each player in each match
4. Picks the best 15-man squad within the FPL rules (£100m budget, 2 GK / 5 DEF / 5 MID / 3 FWD, max 3 players per team) using linear programming

## Results

To test it fairly, I trained the model only on seasons *before* the one I was testing on, so it never saw the test season. I did this for 2023-24, 2024-25 and 2025-26 and compared it to some simple "form" predictions:

- **Form (last 8):** average points over a player's last 8 games
- **Form (weighted):** same idea but recent games count more

Averaged over the 3 test seasons:

| Method | Avg error (MAE) | R² | Avg points per GW of picked team |
|---|---|---|---|
| My model | 0.98 | 0.34 | 62.4 |
| Form (weighted) | 1.02 | 0.26 | 55.9 |
| Form (last 8) | 1.05 | 0.24 | 53.0 |

The model did better than the form methods in all 3 seasons.

For the last column, every gameweek I picked the best team using each method's predictions and then added up the points those players actually scored. This assumes you can change your whole team every week (like a free hit every week), so it isn't the score you'd get in a real season. It's just a way to compare the predictions.

![Squad backtest](reports/squad_backtest.png)

Some things I noticed:
- The most important features by far are about minutes, i.e. whether the player is actually going to play. Predicting *who plays* matters more than predicting who scores.
- It's still very hard to predict points for players who start every week. Goals and assists are pretty random, so the R² for those players on their own is low (around 0.09).
- I wanted to compare against FPL's own "xP" (expected points) column, but it lines up with actual points way too well. I think it gets recorded after the games, so it wouldn't be a fair comparison and I left it out.

<p float="left">
  <img src="reports/calibration.png" width="45%" />
  <img src="reports/feature_importance.png" width="50%" />
</p>

## How to run it

```bash
pip install -r requirements.txt

python -m fpl_predictor backtest                      # run the tests on past seasons
python -m fpl_predictor train                         # train the model on all the data
python -m fpl_predictor predict --horizon 3 --squad   # predict the next 3 GWs and pick a squad

pytest                                                # unit tests
```

The first run downloads the data (about 40MB). Add `--refresh` to get the latest gameweeks.

## Files

- `data.py`: downloads and cleans the data
- `features.py`: builds the features. Everything is shifted by one game so the model can't see the result of the match it's predicting
- `model.py`: the LightGBM model
- `evaluate.py`: testing on past seasons and making the charts
- `predict.py`: predictions for upcoming gameweeks
- `optimize.py`: picks the squad

## Things I'd like to add

- A transfer planner (right now it picks a new team every week and ignores the -4 for extra transfers)
- Betting odds as a feature
- A separate model just for predicting minutes
- Better matching of players between seasons (at the moment I match them by name)

## Data

From [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
