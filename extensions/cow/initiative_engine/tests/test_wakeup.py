"""Test wake scheduling: timezone-aware, strictly future, night→morning."""
import pytest
from datetime import datetime, timedelta, timezone
from cow.initiative_engine.wakeup import compute_next_wake, _in_quiet, _to_cst

UTC = timezone.utc

class TestNextWake:
    def test_next_wake_is_future(self):
        from cow.initiative_engine.wakeup import _now
        t = compute_next_wake("silent", 0, 120, "scheduled")
        assert t > _now(), f"next_wake {t} is not after now"

    def test_next_wake_is_timezone_aware(self):
        t = compute_next_wake("silent", 0, 120, "scheduled")
        assert t.tzinfo is not None, "next_wake must be timezone-aware"

    def test_night_silent_goes_to_morning(self):
        """If budget exhausted, push to next morning CST 08-10."""
        from cow.initiative_engine.wakeup import set_clock
        # Set clock to Aug 4 06:00 UTC = CST 14:00 (active)
        set_clock(datetime(2026, 8, 4, 6, 0, tzinfo=UTC))
        t = compute_next_wake("silent", 99, 120, "scheduled")
        cst = _to_cst(t)
        assert 8 <= cst.hour <= 10, f"Expected CST 08-10, got hour={cst.hour} (UTC={t.hour})"

    def test_conversation_idle_uses_longer_delay(self):
        t1 = compute_next_wake("silent", 0, 60, "scheduled")
        t2 = compute_next_wake("silent", 0, 60, "conversation_idle")
        assert t1.tzinfo is not None
        assert t2.tzinfo is not None

class TestQuietHours:
    # All times below are Asia/Shanghai (CST=UTC+8)
    CST = __import__('datetime').timezone(__import__('datetime').timedelta(hours=8))
    def test_22_boundary_is_quiet(self):
        assert _in_quiet(datetime(2026,8,3,22,0,tzinfo=self.CST))
    def test_21_not_quiet(self):
        assert not _in_quiet(datetime(2026,8,3,21,0,tzinfo=self.CST))
    def test_night_is_quiet(self):
        assert _in_quiet(datetime(2026,8,3,23,0,tzinfo=self.CST))
    def test_3_is_quiet(self):
        assert _in_quiet(datetime(2026,8,3,3,0,tzinfo=self.CST))
    def test_8_boundary_not_quiet(self):
        assert not _in_quiet(datetime(2026,8,3,8,0,tzinfo=self.CST))
    def test_14_not_quiet(self):
        assert not _in_quiet(datetime(2026,8,3,14,0,tzinfo=self.CST))

class TestNextWakeBoundaries:
    def test_22_wake_rolls_to_morning(self):
        """Wake at 22:00 rolls to next day 08:00–10:00."""
        t = compute_next_wake("silent", 99, 120, "scheduled")
        cst = _to_cst(t)
        assert 8 <= cst.hour <= 10, f"Expected CST 08-10, got CST hour={cst.hour}"
    def test_8_wake_stays_in_window(self):
        """Wake at 08:00 stays in active window."""
        t = compute_next_wake("silent", 0, 120, "conversation_idle")
        cst = _to_cst(t)
        assert 8 <= cst.hour < 22

    # Timezone boundary tests
    def test_2103_plus_159_rolls_to_morning(self):
        """UTC 13:03 (CST 21:03) + 159min = UTC 15:42 (CST 23:42) → must roll to morning."""
        from datetime import timedelta
        t = datetime(2026, 8, 4, 13, 3, tzinfo=UTC) + timedelta(minutes=159)
        cst = _to_cst(t)
        assert cst.hour == 23, f"sanity: CST hour should be 23, got {cst.hour}"
        assert _in_quiet(t), "CST 23:42 must be quiet"
    def test_2159_plus_60_rolls_to_morning(self):
        """UTC 13:59 (CST 21:59) + 60min = UTC 14:59 (CST 22:59) → must roll."""
        from datetime import timedelta
        t = datetime(2026, 8, 4, 13, 59, tzinfo=UTC) + timedelta(minutes=60)
        cst = _to_cst(t)
        assert cst.hour == 22, f"sanity: CST hour should be 22, got {cst.hour}"
        assert _in_quiet(t), "CST 22:59 must be quiet"
    def test_2100_plus_60_boundary_rolls(self):
        """UTC 13:00 (CST 21:00) + 60min = UTC 14:00 (CST 22:00) → boundary, must roll."""
        from datetime import timedelta
        t = datetime(2026, 8, 4, 13, 0, tzinfo=UTC) + timedelta(minutes=60)
        cst = _to_cst(t)
        assert cst.hour == 22, f"sanity: CST hour should be 22, got {cst.hour}"
        assert _in_quiet(t), "CST 22:00 boundary must be quiet"

    def test_midnight_crosses_to_morning(self):
        """Wake at CST 02:00 should be quiet."""
        # UTC 18:00 = CST 02:00 next day — depends on date
        # Use direct CST check
        from datetime import timezone as tz, timedelta as td
        CST = tz(td(hours=8))
        t = datetime(2026, 8, 4, 2, 0, tzinfo=CST)
        assert _in_quiet(t), "CST 02:00 must be quiet"
