# winlock

**Is a Deadlock hero's win rate reliable, or is it statistical noise? And what does a game
update do to it?**

Win rate per hero with Wilson confidence intervals, before and after Deadlock's
**"Halloween 2026 + Gameplay Update"** (published 2026-10-30 20:51:49 UTC). It uses public match
data from the [Deadlock API](https://deadlock-api.com/data-dumps).

## Data

- Source: the `match_player` table of the public data dumps (one row per player per match).
- `uv run winlock extract` reads only the files and row groups that can hold matches started
  2026-10-28 → 10-31 UTC, and only 18 columns, over HTTPS. It then deduplicates rows that
  appear in more than one dump file.
  - Run of 2026-11-08 16:30 UTC: 11,420,407 rows and 1,016,323 matches, in
    `data/raw/patch_window/20261108T1630Z/` (not in git).
- **Late arrivals:** only ~55% of a day's matches reach the dumps within a day of starting,
  and 71–78% within 3 days. Rows were still arriving at extraction time.
- **The update in the data:** `game_mode_version` doesn't change (it follows the game mode).
  The update shows as a dip in match starts: from ~2,000 per 10 min to 219 at 20:50 UTC.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```sh
make setup   # install dependencies and the nbstripout filter
make test    # tests
make lint    # ruff
make nb      # open Jupyter Lab

uv run winlock extract     # read the window from the public data dumps (network, ~30 s)
uv run winlock quality     # population, window sizes, the dip in match starts at the update
uv run winlock figures     # Wilson CI per hero, 24 h before and 24 h after the update
uv run winlock patch       # how did win rates change across the update? (+ placebo)
uv run winlock model       # logistic regression: before vs after the update, retraining
```

Charts go to `reports/figures/` (8 charts), tables to `reports/tables/`. The notebooks
`01_winrates` and `02_model` show the charts in article order.

## Method

1. **Population** (`quality`):
   - Normal mode, Unranked + Ranked, `TeamWin`;
   - complete matches only (12 players, 6 winners);
   - new-player, low-priority and not-scored matches are kept.
2. **Windows**, fixed before any post-update rate was computed (`[windows]` in
   `config/analysis.toml`, `docs/decisions.md`):

   | Window | Start → end (UTC, by `start_time`) | Matches | Used for |
   |---|---|---|---|
   | `train` | 10-28 00:00 → 10-29 20:30 | 361,004 | fitting the model |
   | `placebo_pre` | 10-28 20:30 → 10-29 20:30 | 199,607 | placebo, first half |
   | `pre` | 10-29 20:30 → 10-30 20:30 | 203,085 | last 24 h of the old game; test |
   | `post` | 10-30 21:30 → 10-31 21:30 | 229,335 | first 24 h of the new game; test |

   Matches started 20:30–21:30 on 10-30 belong to no window (the buffer around the update).
3. **Win rates** (`figures`): 95% Wilson CI per hero, plus an exact binomial test against 50%
   with Holm/BH over 44 heroes.
4. **The update** (`patch`):
   - per hero, pre vs post: the change, its z, Holm/BH;
   - the **placebo**: the same comparison 24 h earlier, entirely before the update.
5. **Logistic regression** (`model`):
   - one row per match: +1 for each hero on Team0, −1 for each on Team1; target "Team0 won";
   - L2 penalty, C chosen on the last 20% of train in time;
   - compared with a coin flip and with summed win rates (no fitting);
   - tested on pre and on post, accuracy with a Wilson CI, plus log loss and calibration;
   - **retrain check:** refit on the first 70% of post, then score the old and the refit model
     on the same last 30%.

## What was discarded and why

Of 1,016,323 extracted matches:
- **Outside the windows' span** (10-31 21:30 → 24:00): 33,250 matches.
- **StreetBrawl** (another mode, ~8 players): 174,840 matches.
- **Not Unranked/Ranked** (CoopBot, PrivateLobby, Tutorial): 6,850 matches.
- **Incomplete** (not 12 rows and 6 winners): 1,589 matches, dropped whole.

Population: 799,794 matches (9,597,528 rows). Matches inside the 10-30 20:30–21:30 buffer are in
the population but in no window.

## Result

Numbers from `reports/tables/`. The full story is in `docs/article-outline.md`.

1. **Win rates are signal, not noise.**
   - In the 24 h before the update, they range from 44.2% (Pocket) to 57.7% (Graves).
   - 42 of 44 heroes differ from 50% after Holm.
   - With ~60k games per hero, the median 95% CI is ±0.4 pp.
2. **The update changed them.**
   - 20 of 44 heroes changed after Holm, SD of z 5.96.
   - The placebo, at a fake cut, finds 0 of 44 (SD of z 0.87).
   - Biggest moves: Violet −5.6 pp, Mo & Krill −5.5, Warden +3.4, McGinnis +3.3.
3. **Heroes alone predict ~60% of matches.**
   - Logistic regression on pre: 59.9% (95% CI 59.7–60.2%), log loss 0.663.
   - Summed win rates, no fitting: 59.7%. A coin flip: 50.1%.
4. **What it learned:** the weights correlate 0.95 with the win rates. The exception is
   McGinnis: a 48.3% win rate, but a positive weight.
5. **The update costs the model about 1 point, and retraining recovers it.**
   - The same model after the update: 58.8% (58.6–59.1%).
   - Refit on early post-update matches: 59.9%, against the old model's 58.8% on the same
     matches (log loss −0.0058, 95% CI −0.0066 to −0.0049).

## Limitations

- **Two days around one update.** The post side is Friday evening → Saturday evening; the pre
  side is Thursday → Friday. A weekday/weekend player mix isn't separated from the patch, and
  the placebo doesn't cover a weekend.
- **The Halloween event** (candy buffs) runs only on the post side.
- **New heroes** (Violet, Nurse Harrow, Deadman Danny…) were still being learned. The placebo
  shows no drift for them inside the old game, though.
- **Late arrivals:** the data was still filling in. A re-extraction can check that late
  matches don't change the result.
- **Players, not heroes.** A win rate measures who plays the hero as much as the hero, and
  `party` isn't available. Player clustering of the CIs isn't checked yet.
- **Skill** (`average_badge`) exists only for Ranked and isn't analysed.
