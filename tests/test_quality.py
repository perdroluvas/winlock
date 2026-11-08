from winlock.quality import match_completeness, start_day_profile, starts_per_bin


def test_start_day_profile_groups_old_days(ten_match_con):
    prof = start_day_profile(ten_match_con, recent_days=1)
    assert prof["start_day"].tolist() == ["older", "2026-10-20"]
    assert prof["pct_rows"].sum() == 100


def test_starts_per_bin_counts_one_mode(ten_match_con):
    bins = starts_per_bin(
        ten_match_con, "2026-10-20T00:00:00Z", "2026-10-20T03:00:00Z", "Normal", 60
    )
    # 00:00 holds match 1 plus the incomplete Normal match 100 (StreetBrawl is another mode)
    assert bins["matches"].tolist() == [2, 1, 1]


def test_match_completeness_finds_the_short_match(ten_match_con):
    comp = match_completeness(
        ten_match_con, "2026-10-20T00:00:00Z", "2026-10-21T00:00:00Z", "Normal"
    )
    assert dict(zip(comp["rows"], comp["matches"], strict=True)) == {12: 10, 10: 1}
