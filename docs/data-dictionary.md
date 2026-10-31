# Data dictionary: `match_player`

One row per player per match (157 columns). Mark **verified** only after checking it in the data,
with the query used under "How verified". Unless noted, checks are on matches that started on
2026-10-20 UTC in Normal mode (503,482 rows).

| Column | Type | Meaning | Verified? | How verified / notes |
|---|---|---|---|---|
| `match_id` | uint64 | Match ID; partitioned by `intDiv(match_id, 1000000)` | yes (2026-10-30) | `quality.match_completeness`: 98.9% of Normal matches have 12 rows and 6 winners; StreetBrawl ~8 rows/match |
| `account_id` | uint32 | Identifies the player. **Personal data: never publish** | yes (2026-10-31) | Aggregates only: 362 rows (0.07%) have no id (null or 0). Not used by the analysis; never enters the `population` table |
| `hero_id` | uint8 | Hero ID (names in `data/reference/heroes.csv`) | yes (2026-10-31) | 44 ids, all present in `heroes.csv`; no hero appears twice in a complete match |
| `team` | varchar | Player's team, `Team0` / `Team1` | yes (2026-10-31) | Population: 250,110 rows each; every complete match is 6 vs 6 |
| `won` | boolean | Player's team won the match | yes (2026-10-31) | Complete matches have exactly 6 `true`, all on the same team |
| `start_time` | timestamptz | Match start; defines days (in UTC) | yes (2026-10-30) | `quality.start_day_profile`: 2024-06-24 → 2026-10-20 22:01 UTC, 464 days; only 55.5% of rows start on 2026-10-20 |
| `created_at` | timestamptz | When the row reached the database (watermark) | yes (2026-10-30) | min 2026-10-20 02:03:24 UTC: the file is cut by this column |
| `duration_s` | uint32 | Match duration in seconds | yes (2026-10-31) | 121–4,891 s, median 2,171 s (~36 min); 43 rows under 5 min. Not used (it is an outcome of the match) |
| `game_mode` | varchar | Filter: `'Normal'` | yes (2026-10-30) | Whole file: Normal 920,280 rows, StreetBrawl 109,633 |
| `match_mode` | varchar | Filter: `'Unranked'`, `'Ranked'` | yes (2026-10-31) | Values: Unranked 380,673 rows, Ranked 120,590, PrivateLobby 1,481, CoopBot 738 |
| `match_outcome` | varchar | Filter: `'TeamWin'` | yes (2026-10-31) | Whole file: only `TeamWin` (90,843 matches); the filter removes nothing here |
| `party` | uint32 | Party; 0 = solo (manifest), nullable | yes (2026-10-31) | **100% null** in every match mode: solo vs group can't be analysed with this file |
| `average_badge` | uint32 | Skill tier | yes (2026-10-31) | Filled for Ranked only (24% of the population's rows, not ~14%); missing for every other mode |
| `game_mode_version` | uint32 | Possible patch proxy: **undocumented, do not assume** | yes (2026-10-31) | A single value (2) for the whole day, so no version change inside the analysis window; its meaning is unknown |
