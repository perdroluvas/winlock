from winlock.quality import match_completeness, start_day_profile, start_hour_profile


def test_start_day_profile_groups_old_days(ten_match_con):
    prof = start_day_profile(ten_match_con, recent_days=1)
    assert prof["start_day"].tolist() == ["older", "2026-10-20"]
    assert prof["pct_rows"].sum() == 100


def test_start_hour_profile_counts_matches(ten_match_con):
    prof = start_hour_profile(ten_match_con, "2026-10-20")
    # all modes on that day: 10 complete + 1 incomplete Normal + 1 StreetBrawl
    assert prof["matches"].sum() == 12
    assert prof.loc[prof.hour_utc == 0, "matches"].item() == 3


def test_match_completeness_finds_the_short_match(ten_match_con):
    comp = match_completeness(ten_match_con, "2026-10-20", "Normal")
    assert dict(zip(comp["rows"], comp["matches"], strict=True)) == {12: 10, 10: 1}
