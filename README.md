# winlock

**Is a Deadlock hero's win rate reliable, or is it statistical noise?**

Win rate per hero with Wilson confidence intervals, using public data from the
[Deadlock API](https://api.deadlock-api.com/docs).

## Data

- Source: residual snapshot of the `match_player` table (one row per player per match).
- File: `data/raw/20261020T020300Z-20261020T221300Z-50a132fe33c6.parquet` (~1.6 GB, not in git).
- The file is cut by `created_at` (2026-10-20, 02:03–22:13 UTC), but only 55.5% of its rows are
  matches that **started** on 2026-10-20; the rest are older matches ingested late
  (back to 2024-06-24).
- Analysis period: matches with `start_time` on 2026-10-20 (UTC).
- Patch / game version: unknown. `game_mode_version` is 2 for the whole day (no change inside
  the window), but the column is undocumented. New heroes released in the week before (Violet
  on 10-16, Deadman Danny on 10-13, Solomon on 10-09) and on the day itself (Nurse Harrow)
  suggest a recently updated game.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```sh
make setup   # install dependencies and the nbstripout filter
make test    # tests
make lint    # ruff
make nb      # open Jupyter Lab

uv run winlock quality            # data checks: start days, start hours, complete matches
uv run winlock figures            # population report + train baseline charts in reports/
uv run winlock evaluate           # do the train win rates hold on test? + split-half control
uv run winlock model              # predict match winners from heroes, scored on test
uv run winlock synergy            # teammate pairs: do some win more than their parts predict?

uv run db-reader schema           # inspect the parquet from the terminal
uv run db-reader head -n 10
uv run db-reader sql "SELECT ... FROM match_player"
```

## Method

1. Population (`uv run winlock figures`): Normal, Unranked/Ranked matches that started on
   2026-10-20 UTC, complete matches only (12 players, 6 winners).
2. Temporal split: the first 70% of matches (00:00–13:35 UTC) are train, the last 30% are test.
3. Baseline: win rate per hero on train with a 95% Wilson CI, and an exact binomial test
   against 50% with Holm/BH correction over the 43 heroes.
4. Evaluation (`uv run winlock evaluate`): train vs test per hero (z-score, Holm/BH), and
   coverage of the train CIs compared with a "nothing changed" simulation. Split-half control:
   the same comparison between even and odd `match_id`s, which cover the same hours.
5. Model (`uv run winlock model`): predict the winner of test matches from the heroes alone
   (constant, win-rate model, logistic regression), with a Wilson CI on accuracy.
6. Synergy (`uv run winlock synergy`): per teammate pair, wins together vs the additive
   prediction (z, Holm/BH) on train, replication on test, and a model with pair terms.

Charts go to `reports/figures/`, tables to `reports/tables/`; `notebooks/` shows the charts.

## What was discarded and why

- Matches whose `start_time` is not on 2026-10-20 UTC (44.5% of rows): late-ingested older
  matches from other days and patches. See `docs/decisions.md`.
- StreetBrawl matches (8,499 of the remaining matches): another game mode with ~8 players.
- Matches outside Unranked/Ranked (433 matches).
- Incomplete matches (95 after the other filters): not exactly 12 rows and 6 winners. They are
  dropped whole.
- Final population: 41,685 matches (500,220 rows), split in time into 70% train / 30% test
  (`uv run winlock figures` prints this report).

## Result

Numbers from `reports/tables/`; all on 2026-10-20 UTC.

1. **Hero win rates are signal, not noise.** On train, win rates range from 42.9% (Mirage) to
   56.9% (Graves). **32 of 43 heroes differ from 50% after Holm correction** (36 by their
   individual CIs, 35 after BH).
2. **They hold over the day.** Train vs test: correlation 0.98, and no hero changed
   significantly (0 after Holm or BH). 83.7% of test rates fall inside their train CI; with no
   change at all the expected value is ~72%, not 95%, because the test rate is noisy too.
3. **The binomial standard error is honest here.** In the split-half control (same hours), the
   SD of z is 0.98 (1.00 = pure binomial noise) and coverage matches the simulation. Train →
   test is even *more* stable than chance (SD of z 0.76, borderline: 2.3% of simulations are
   that stable); this is not explained.
4. **Heroes predict matches.** Knowing only the 12 heroes, logistic regression calls 60.1% of
   test matches correctly (95% Wilson CI 59.3–61.0%; constant baseline 49.4%), log loss 0.663 vs
   0.693, well calibrated. The win rates alone, without fitting, reach 59.8%.
5. **Teammate synergy: there is some overall, but no single pair can be named.** Of 903 pairs, 0
   pass Holm or BH; 73 have p < 0.05 where ~45 are expected by chance, and pair z-scores
   replicate from train to test (correlation 0.24, noise band ±0.065). Adding 946 pair terms to
   the model makes test predictions slightly worse (log loss +0.0022, 95% CI +0.0010 to
   +0.0035).

## Limitations

- **One day, one (unknown) patch.** Results describe 2026-10-20 only; balance changes can move
  everything.
- **Train is morning, test is evening** (00:00–13:35 vs 13:35–22:00 UTC). Differences mix
  "time passed" with "different players online"; the split-half control checks noise but not
  this.
- **Players, not heroes.** A win rate measures who plays a hero as much as the hero: new heroes
  attract learners, strong players favour some picks, and `party` is empty, so groups who
  queue together can't be separated. The same applies to pair "synergy".
- **Non-independence.** 12 players per match and players repeating across matches; the
  split-half control suggests the binomial SE is still about right on this day.
- **The test set was used twice** (first models, then synergy). Nothing was tuned on it, but the
  synergy analysis was designed after seeing the first results.
- **Nurse Harrow** (released on the day) only appears in test and is not evaluated.
- **Skill** (`average_badge`) exists only for Ranked (24% of rows) and was not analysed.
