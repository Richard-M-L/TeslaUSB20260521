"""Tests for the mapping/geo-indexer service (mapping_service.py).

Covers: database schema, event detection rules, debouncing, trip queries,
background indexer with synthetic MP4 files.
"""

import json
import os
import struct
import sqlite3
import time
import pytest

from services.mapping_service import (
    _init_db,
    _detect_events,
    _debounce_events,
    _timestamp_from_filename,
    _find_front_camera_videos,
    _index_video,
    boot_catchup_scan,
    canonical_key,
    candidate_db_paths,
    index_single_file,
    IndexOutcome,
    IndexResult,
    start_daily_stale_scan,
    stop_daily_stale_scan,
    trigger_stale_scan_now,
    _initial_stale_scan_delay,
    _run_stale_scan_blocking,
    _reset_stale_scan_state_for_tests,
    DEFAULT_THRESHOLDS,
    _SCHEMA_VERSION,
)
from services.mapping_queries import (
    query_events,
    get_stats,
    get_driving_stats,
    get_event_chart_data,
)
from services.indexing_queue_service import (
    claim_next_queue_item,
    clear_all_queue,
    clear_pending_queue,
    clear_queue,
    complete_queue_item,
    compute_backoff,
    defer_queue_item,
    enqueue_for_indexing,
    enqueue_many_for_indexing,
    get_queue_status,
    priority_for_path,
    recover_stale_claims,
    release_claim,
    _PARSE_ERROR_MAX_ATTEMPTS,
    _PRIORITY_ARCHIVE,
    _PRIORITY_RECENT,
    _PRIORITY_SENTRY_SAVED,
)
from services.dashcam_pb2 import SeiMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box(name: str, content: bytes) -> bytes:
    size = 8 + len(content)
    return struct.pack('>I', size) + name.encode('ascii') + content


def _make_sei_protobuf(lat=37.7749, lon=-122.4194, speed=25.0,
                       gear=1, autopilot=0, heading=90.0,
                       accel_x=0.0, accel_y=0.0) -> bytes:
    msg = SeiMetadata()
    msg.latitude_deg = lat
    msg.longitude_deg = lon
    msg.heading_deg = heading
    msg.vehicle_speed_mps = speed
    msg.gear_state = gear
    msg.autopilot_state = autopilot
    msg.linear_acceleration_mps2_x = accel_x
    msg.linear_acceleration_mps2_y = accel_y
    msg.brake_applied = False
    msg.steering_wheel_angle = 0.0
    msg.frame_seq_no = 1
    return msg.SerializeToString()


def _make_sei_nal(protobuf_payload: bytes) -> bytes:
    nal_header = bytes([0x06, 0x05, 0x00])
    padding = bytes([0x42, 0x42, 0x42])
    marker = bytes([0x69])
    trailing = bytes([0x80])
    return nal_header + padding + marker + protobuf_payload + trailing


def _make_synthetic_mp4(sei_payloads, timescale=30000, frame_ticks=1001):
    """Build a minimal valid MP4 with SEI NAL units."""
    mdhd_content = struct.pack('>I', 0) + struct.pack('>I', 0) + struct.pack('>I', 0)
    mdhd_content += struct.pack('>I', timescale)
    mdhd_content += struct.pack('>I', frame_ticks * len(sei_payloads))
    mdhd_content += struct.pack('>I', 0)
    mdhd = _make_box('mdhd', mdhd_content)

    stts_content = struct.pack('>I', 0) + struct.pack('>I', 1)
    stts_content += struct.pack('>I', len(sei_payloads)) + struct.pack('>I', frame_ticks)
    stts = _make_box('stts', stts_content)

    avc1_inner = b'\x00' * 78
    avcc_content = bytes([0x01, 0x64, 0x00, 0x1F, 0xFF, 0xE1])
    avcc_content += struct.pack('>H', 4) + b'\x00' * 4
    avcc_content += bytes([0x01]) + struct.pack('>H', 4) + b'\x00' * 4
    avcc = _make_box('avcC', avcc_content)
    avc1 = _make_box('avc1', avc1_inner + avcc)
    stsd = _make_box('stsd', struct.pack('>I', 0) + struct.pack('>I', 1) + avc1)

    stbl = _make_box('stbl', stsd + stts)
    minf = _make_box('minf', stbl)
    mdia = _make_box('mdia', mdhd + minf)
    trak = _make_box('trak', mdia)
    moov = _make_box('moov', trak)

    mdat_content = bytearray()
    for pb in sei_payloads:
        sei_nal = _make_sei_nal(pb)
        mdat_content += struct.pack('>I', len(sei_nal)) + sei_nal
        idr = bytes([0x65, 0x00, 0x00, 0x01])
        mdat_content += struct.pack('>I', len(idr)) + idr

    mdat = _make_box('mdat', bytes(mdat_content))
    ftyp = _make_box('ftyp', b'mp42' + b'\x00' * 4)
    return ftyp + moov + mdat


# ---------------------------------------------------------------------------
# Database Schema Tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        assert 'trips' in tables
        assert 'waypoints' in tables
        assert 'detected_events' in tables
        assert 'indexed_files' in tables
        assert 'schema_version' in tables
        conn.close()

    def test_schema_version_stored(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row['version'] == _SCHEMA_VERSION
        conn.close()

    def test_idempotent_init(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn1 = _init_db(db_path)
        conn1.execute("INSERT INTO trips (start_time) VALUES ('2025-01-01T00:00:00')")
        conn1.commit()
        conn1.close()

        # Second init should not drop data
        conn2 = _init_db(db_path)
        count = conn2.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
        assert count == 1
        conn2.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == 'wal'
        conn.close()

    def test_foreign_keys_enabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()


class TestArchiveQueueSchema:
    """v9 → v10 migration adds the ``archive_queue`` table + ready index.

    These tests verify the migration is forward-compatible (creates the
    new table on a fresh DB), idempotent (re-running ``_init_db`` is a
    no-op), and non-destructive (existing rows in trips / waypoints /
    detected_events / indexed_files / indexing_queue survive the
    migration).
    """

    def test_archive_queue_table_exists_after_init(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert 'archive_queue' in tables
        conn.close()

    def test_archive_queue_ready_index_exists(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        indexes = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        assert 'archive_queue_ready' in indexes
        conn.close()

    def test_archive_queue_columns_match_spec(self, tmp_path):
        """All 13 columns from the issue spec must be present with the
        correct types and defaults."""
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        cols = {r[1]: r for r in conn.execute(
            "PRAGMA table_info(archive_queue)"
        ).fetchall()}
        # Column name: (type, notnull, dflt_value, pk)
        # cid(0), name(1), type(2), notnull(3), dflt_value(4), pk(5)
        assert 'id' in cols and cols['id'][5] == 1  # PK
        assert 'source_path' in cols
        assert cols['source_path'][2] == 'TEXT'
        assert cols['source_path'][3] == 1  # NOT NULL
        assert 'dest_path' in cols
        assert 'priority' in cols
        assert cols['priority'][4] == '3'  # default 3
        assert 'status' in cols
        assert cols['status'][4] == "'pending'"
        assert 'attempts' in cols
        assert cols['attempts'][4] == '0'
        assert 'last_error' in cols
        assert 'enqueued_at' in cols
        assert cols['enqueued_at'][3] == 1  # NOT NULL
        assert 'claimed_at' in cols
        assert 'claimed_by' in cols
        assert 'copied_at' in cols
        assert 'expected_size' in cols
        assert 'expected_mtime' in cols
        assert cols['expected_mtime'][2] == 'REAL'
        conn.close()

    def test_source_path_unique_constraint(self, tmp_path):
        """``source_path`` must enforce UNIQUE so dedup is automatic."""
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        conn.execute(
            "INSERT INTO archive_queue (source_path, enqueued_at) "
            "VALUES (?, ?)",
            ('/a/b.mp4', '2026-05-11T09:00:00+00:00'),
        )
        conn.commit()
        # Second insert with same source_path raises IntegrityError.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO archive_queue (source_path, enqueued_at) "
                "VALUES (?, ?)",
                ('/a/b.mp4', '2026-05-11T09:01:00+00:00'),
            )
        conn.close()

    def test_schema_version_is_v10(self, tmp_path):
        from services.mapping_service import _SCHEMA_VERSION
        # Phase 2a bumps the schema to v10.
        assert _SCHEMA_VERSION >= 10

    def test_migration_is_idempotent(self, tmp_path):
        """Running ``_init_db`` twice must not lose archive_queue rows."""
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        conn.execute(
            "INSERT INTO archive_queue (source_path, enqueued_at) "
            "VALUES (?, ?)",
            ('/keep/me.mp4', '2026-05-11T09:00:00+00:00'),
        )
        conn.commit()
        conn.close()

        # Re-init the same DB. Must preserve the row.
        conn2 = _init_db(db_path)
        n = conn2.execute(
            "SELECT COUNT(*) FROM archive_queue"
        ).fetchone()[0]
        assert n == 1
        row = conn2.execute(
            "SELECT source_path FROM archive_queue"
        ).fetchone()
        assert row[0] == '/keep/me.mp4'
        conn2.close()

    def test_migration_preserves_existing_data(self, tmp_path):
        """Pre-existing trips / waypoints / detected_events / indexed_files
        must survive the migration to v10 unchanged."""
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        # Seed data in the older tables.
        conn.execute(
            "INSERT INTO trips (start_time, end_time) VALUES (?, ?)",
            ('2025-01-01T00:00:00', '2025-01-01T01:00:00'),
        )
        trip_id = conn.execute(
            "SELECT id FROM trips ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO waypoints
                (trip_id, timestamp, lat, lon)
               VALUES (?, ?, ?, ?)""",
            (trip_id, '2025-01-01T00:30:00', 37.7749, -122.4194),
        )
        conn.execute(
            """INSERT INTO indexed_files
                (file_path, indexed_at)
               VALUES (?, ?)""",
            ('/tmp/x.mp4', '2025-01-01T00:00:00'),
        )
        conn.commit()
        conn.close()

        # Re-init (no-op for v10) and verify rows are untouched.
        conn2 = _init_db(db_path)
        assert conn2.execute(
            "SELECT COUNT(*) FROM trips"
        ).fetchone()[0] == 1
        assert conn2.execute(
            "SELECT COUNT(*) FROM waypoints"
        ).fetchone()[0] == 1
        assert conn2.execute(
            "SELECT COUNT(*) FROM indexed_files"
        ).fetchone()[0] == 1
        conn2.close()

    def test_indexing_queue_unaffected_by_v10(self, tmp_path):
        """The Phase 2a migration must not touch indexing_queue rows."""
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)
        conn.execute(
            """INSERT INTO indexing_queue
                (canonical_key, file_path, priority, enqueued_at)
               VALUES (?, ?, ?, ?)""",
            ('keyA', '/tmp/y.mp4', 50, 1700000000.0),
        )
        conn.commit()
        conn.close()

        conn2 = _init_db(db_path)
        n = conn2.execute(
            "SELECT COUNT(*) FROM indexing_queue"
        ).fetchone()[0]
        assert n == 1
        conn2.close()


# ---------------------------------------------------------------------------
# Event Detection Tests
# ---------------------------------------------------------------------------

class TestEventDetection:
    def _make_waypoint(self, **overrides):
        defaults = {
            'timestamp': '2025-11-08T08:15:44',
            'lat': 37.7749, 'lon': -122.4194,
            'speed_mps': 25.0,
            'acceleration_x': 0.0, 'acceleration_y': 0.0,
            'autopilot_state': 'NONE',
            'steering_angle': 0.0,
            'gear': 'DRIVE',
            'brake_applied': 0,
            'video_path': 'test.mp4',
            'frame_offset': 0,
        }
        defaults.update(overrides)
        return defaults

    def test_harsh_brake_detected(self):
        wps = [self._make_waypoint(acceleration_x=-5.0)]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert len(events) == 1
        assert events[0]['event_type'] == 'harsh_brake'
        assert events[0]['severity'] == 'warning'

    def test_emergency_brake_detected(self):
        wps = [self._make_waypoint(acceleration_x=-8.0)]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        types = [e['event_type'] for e in events]
        assert 'emergency_brake' in types

    def test_hard_acceleration_detected(self):
        wps = [self._make_waypoint(acceleration_x=4.0)]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert any(e['event_type'] == 'hard_acceleration' for e in events)

    def test_sharp_turn_detected(self):
        wps = [self._make_waypoint(acceleration_y=5.0)]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert any(e['event_type'] == 'sharp_turn' for e in events)

    def test_speeding_detected(self):
        wps = [self._make_waypoint(speed_mps=40.0)]  # ~89 mph
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert any(e['event_type'] == 'speeding' for e in events)

    def test_no_speeding_below_threshold(self):
        wps = [self._make_waypoint(speed_mps=30.0)]  # ~67 mph
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert not any(e['event_type'] == 'speeding' for e in events)

    def test_fsd_disengage_detected(self):
        wps = [
            self._make_waypoint(autopilot_state='AUTOSTEER'),
            self._make_waypoint(autopilot_state='NONE',
                                timestamp='2025-11-08T08:15:45'),
        ]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert any(e['event_type'] == 'fsd_disengage' for e in events)

    def test_fsd_engage_detected(self):
        wps = [
            self._make_waypoint(autopilot_state='NONE'),
            self._make_waypoint(autopilot_state='SELF_DRIVING',
                                timestamp='2025-11-08T08:15:45'),
        ]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert any(e['event_type'] == 'fsd_engage' for e in events)

    def test_no_fsd_event_when_state_unchanged(self):
        wps = [
            self._make_waypoint(autopilot_state='NONE'),
            self._make_waypoint(autopilot_state='NONE'),
        ]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert not any(e['event_type'] in ('fsd_disengage', 'fsd_engage')
                       for e in events)

    def test_normal_driving_no_events(self):
        wps = [self._make_waypoint(acceleration_x=0.5, speed_mps=20.0)]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        assert len(events) == 0

    def test_custom_thresholds(self):
        custom = dict(DEFAULT_THRESHOLDS)
        custom['harsh_brake_threshold'] = -2.0  # More sensitive
        wps = [self._make_waypoint(acceleration_x=-2.5)]
        events = _detect_events(wps, custom, 'test.mp4')
        assert any(e['event_type'] == 'harsh_brake' for e in events)

    def test_event_has_metadata_json(self):
        wps = [self._make_waypoint(acceleration_x=-5.0)]
        events = _detect_events(wps, DEFAULT_THRESHOLDS, 'test.mp4')
        metadata = json.loads(events[0]['metadata'])
        assert 'accel_x' in metadata
        assert 'speed_mps' in metadata


class TestDebounce:
    def test_deduplicates_within_window(self):
        events = [
            {'event_type': 'harsh_brake', 'timestamp': '2025-01-01T00:00:00'},
            {'event_type': 'harsh_brake', 'timestamp': '2025-01-01T00:00:02'},  # 2s later
            {'event_type': 'harsh_brake', 'timestamp': '2025-01-01T00:00:04'},  # 4s later
        ]
        result = _debounce_events(events, window_seconds=5.0)
        assert len(result) == 1

    def test_keeps_events_outside_window(self):
        events = [
            {'event_type': 'harsh_brake', 'timestamp': '2025-01-01T00:00:00'},
            {'event_type': 'harsh_brake', 'timestamp': '2025-01-01T00:00:10'},  # 10s later
        ]
        result = _debounce_events(events, window_seconds=5.0)
        assert len(result) == 2

    def test_different_types_not_debounced(self):
        events = [
            {'event_type': 'harsh_brake', 'timestamp': '2025-01-01T00:00:00'},
            {'event_type': 'sharp_turn', 'timestamp': '2025-01-01T00:00:01'},
        ]
        result = _debounce_events(events, window_seconds=5.0)
        assert len(result) == 2

    def test_empty_list(self):
        assert _debounce_events([], 5.0) == []


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_invalid_filename(self):
        assert _timestamp_from_filename('random_file.mp4') is None

    def test_short_filename(self):
        assert _timestamp_from_filename('short.mp4') is None


class TestFindFrontCameraVideos:
    def test_finds_recent_clips(self, tmp_path):
        recent = tmp_path / "RecentClips"
        recent.mkdir()
        (recent / "2025-11-08_08-15-44-front.mp4").write_bytes(b'')
        (recent / "2025-11-08_08-15-44-back.mp4").write_bytes(b'')
        (recent / "2025-11-08_08-16-44-front.mp4").write_bytes(b'')

        videos = list(_find_front_camera_videos(str(tmp_path)))
        assert len(videos) == 2
        assert all('-front' in v for v in videos)

    def test_finds_saved_clips(self, tmp_path):
        saved = tmp_path / "SavedClips" / "2025-11-08_08-15-44"
        saved.mkdir(parents=True)
        (saved / "2025-11-08_08-15-44-front.mp4").write_bytes(b'')
        (saved / "2025-11-08_08-15-44-back.mp4").write_bytes(b'')

        videos = list(_find_front_camera_videos(str(tmp_path)))
        assert len(videos) == 1

    def test_empty_directory(self, tmp_path):
        assert list(_find_front_camera_videos(str(tmp_path))) == []


class TestCanonicalKey:
    """The canonical key is the queue/dedup primary key. Two paths share a
    canonical key iff they refer to the same recording."""

    def test_recent_clips_keys_on_basename(self):
        assert canonical_key(
            '/mnt/gadget/part1-ro/TeslaCam/RecentClips/2026-01-01_12-00-00-front.mp4'
        ) == '2026-01-01_12-00-00-front.mp4'

    def test_archived_clips_keys_on_basename(self):
        assert canonical_key(
            '/home/pi/ArchivedClips/2026-01-01_12-00-00-front.mp4'
        ) == '2026-01-01_12-00-00-front.mp4'

    def test_recent_and_archived_collide(self):
        """The whole point: same basename in Recent and Archived → same key."""
        rec = canonical_key('RecentClips/2026-01-01_12-00-00-front.mp4')
        arc = canonical_key('ArchivedClips/2026-01-01_12-00-00-front.mp4')
        assert rec == arc

    def test_saved_clips_keys_include_event_folder(self):
        key = canonical_key(
            '/mnt/gadget/part1-ro/TeslaCam/SavedClips/'
            '2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'
        )
        assert key == 'SavedClips/2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'

    def test_sentry_clips_keys_include_event_folder(self):
        key = canonical_key(
            'SentryClips/2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'
        )
        assert key == 'SentryClips/2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'

    def test_different_events_dont_collide(self):
        """Two SavedClips events must not share a canonical key even if a
        clip basename happens to match (Tesla can use generic timestamps
        within an event)."""
        a = canonical_key(
            'SavedClips/2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'
        )
        b = canonical_key(
            'SavedClips/2026-02-15_09-30-00/2026-01-01_12-00-00-front.mp4'
        )
        assert a != b

    def test_bare_basename_collides_with_recent(self):
        """Legacy DB rows storing just the basename must dedupe with their
        Recent/Archived siblings."""
        bare = canonical_key('2026-01-01_12-00-00-front.mp4')
        rec = canonical_key('RecentClips/2026-01-01_12-00-00-front.mp4')
        assert bare == rec

    def test_handles_windows_separators(self):
        """File paths on Windows / cross-platform tooling may use backslashes."""
        key = canonical_key(
            r'C:\TeslaCam\SentryClips\2026-01-01_12-00-00\2026-01-01_12-00-00-front.mp4'
        )
        assert key == 'SentryClips/2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'


class TestCandidateDbPaths:
    def test_basename_key_expands_to_three_forms(self):
        paths = candidate_db_paths('2026-01-01_12-00-00-front.mp4')
        assert set(paths) == {
            '2026-01-01_12-00-00-front.mp4',
            'RecentClips/2026-01-01_12-00-00-front.mp4',
            'ArchivedClips/2026-01-01_12-00-00-front.mp4',
        }

    def test_event_folder_key_returns_only_itself(self):
        key = 'SavedClips/2026-01-01_12-00-00/2026-01-01_12-00-00-front.mp4'
        assert candidate_db_paths(key) == [key]


# ---------------------------------------------------------------------------
# Query API Tests
# ---------------------------------------------------------------------------

class TestQueryAPIs:
    @pytest.fixture
    def db_with_data(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        # Insert test trip
        conn.execute(
            """INSERT INTO trips (id, start_time, end_time, start_lat, start_lon,
               end_lat, end_lon, distance_km, duration_seconds, source_folder)
               VALUES (1, '2025-11-08T08:15:44', '2025-11-08T08:25:44',
               37.7749, -122.4194, 37.7850, -122.4100, 1.5, 600, 'RecentClips')"""
        )

        # Insert waypoints
        for i in range(5):
            conn.execute(
                """INSERT INTO waypoints (trip_id, timestamp, lat, lon, speed_mps,
                   autopilot_state, video_path, frame_offset)
                   VALUES (1, ?, ?, ?, 25.0, 'NONE', 'test.mp4', ?)""",
                (f'2025-11-08T08:1{5 + i}:44', 37.7749 + i * 0.001,
                 -122.4194 + i * 0.001, i * 30)
            )

        # Insert events
        conn.execute(
            """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
               event_type, severity, description, video_path)
               VALUES (1, '2025-11-08T08:17:44', 37.7769, -122.4174,
               'harsh_brake', 'warning', 'Harsh braking: -5.0 m/s²', 'test.mp4')"""
        )
        conn.commit()
        conn.close()
        return db_path

    advertise them, even though the map skips rendering null
    markers client-side.
    """

    def _make_db(self, tmp_path, name='events_date.db'):
        db_path = str(tmp_path / name)
        conn = _init_db(db_path)
        return db_path, conn

    def test_filter_returns_only_matching_day(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        for ts in ('2026-05-03T08:00:00', '2026-05-04T08:00:00',
                   '2026-05-04T20:00:00', '2026-05-05T08:00:00'):
            conn.execute(
                """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
                                                event_type, severity, description)
                   VALUES (NULL, ?, 37.7, -122.4, 'harsh_brake',
                           'warning', 'test')""",
                (ts,),
            )
        conn.commit(); conn.close()

        events = query_events(db_path, date='2026-05-04')
        assert len(events) == 2
        assert all(e['timestamp'].startswith('2026-05-04') for e in events)

    def test_filter_includes_null_lat_lon_rows(self, tmp_path):
        # Day card stats include null-coord events; the listing
        # endpoint must return them too so the client can show
        # "N events · location not available" guidance.
        db_path, conn = self._make_db(tmp_path)
        conn.execute(
            """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
                                            event_type, severity, description)
               VALUES (NULL, '2026-05-04T22:00:00', NULL, NULL,
                       'sentry', 'info', 'no gps')"""
        )
        conn.commit(); conn.close()

        events = query_events(db_path, date='2026-05-04')
        assert len(events) == 1
        assert events[0]['lat'] is None
        assert events[0]['lon'] is None

    def test_filter_combined_with_event_type(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        for evt in ('harsh_brake', 'sentry', 'speeding'):
            conn.execute(
                """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
                                                event_type, severity, description)
                   VALUES (NULL, '2026-05-04T08:00:00', 37.7, -122.4,
                           ?, 'warning', 'x')""",
                (evt,),
            )
        conn.commit(); conn.close()

        events = query_events(db_path, date='2026-05-04', event_type='sentry')
        assert len(events) == 1
        assert events[0]['event_type'] == 'sentry'

    def test_no_date_returns_all(self, tmp_path):
        # Backwards compat: omitting `date` returns everything.
        db_path, conn = self._make_db(tmp_path)
        for ts in ('2026-05-04T08:00:00', '2026-05-05T08:00:00'):
            conn.execute(
                """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
                                                event_type, severity, description)
                   VALUES (NULL, ?, 37.7, -122.4, 'harsh_brake',
                           'warning', 't')""",
                (ts,),
            )
        conn.commit(); conn.close()
        assert len(query_events(db_path)) == 2


# ---------------------------------------------------------------------------
# End-to-End Indexing Tests
# ---------------------------------------------------------------------------

def _unpack(result: IndexResult):
    """Tests historically asserted on ``(waypoint_count, event_count)``;
    keep that shape locally so individual tests stay readable while the
    public API returns the structured :class:`IndexResult`."""
    return result.waypoints, result.events


class TestIndexVideo:
    def test_index_synthetic_video(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        # Create synthetic video with GPS data
        payloads = [
            _make_sei_protobuf(lat=37.7749, lon=-122.4194, speed=25.0),
            _make_sei_protobuf(lat=37.7750, lon=-122.4195, speed=26.0),
            _make_sei_protobuf(lat=37.7751, lon=-122.4196, speed=27.0),
        ]
        mp4_data = _make_synthetic_mp4(payloads)

        teslacam = tmp_path / "TeslaCam" / "RecentClips"
        teslacam.mkdir(parents=True)
        video_file = teslacam / "2025-11-08_08-15-44-front.mp4"
        video_file.write_bytes(mp4_data)

        wc, ec = _unpack(_index_video(
            conn, str(video_file), str(tmp_path / "TeslaCam"),
            sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

        ))

        assert wc == 3
        trips = conn.execute("SELECT * FROM trips").fetchall()
        assert len(trips) == 1

        waypoints = conn.execute("SELECT * FROM waypoints").fetchall()
        assert len(waypoints) == 3
        conn.close()

    def test_index_with_events(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        payloads = [
            _make_sei_protobuf(lat=37.7749, lon=-122.4194, speed=25.0, accel_x=-6.0),
            _make_sei_protobuf(lat=37.7750, lon=-122.4195, speed=26.0, accel_x=0.0),
        ]
        mp4_data = _make_synthetic_mp4(payloads)

        teslacam = tmp_path / "TeslaCam" / "RecentClips"
        teslacam.mkdir(parents=True)
        video_file = teslacam / "2025-11-08_08-15-44-front.mp4"
        video_file.write_bytes(mp4_data)

        wc, ec = _unpack(_index_video(
            conn, str(video_file), str(tmp_path / "TeslaCam"),
            sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

        ))

        assert wc == 2
        assert ec >= 1  # Should detect harsh braking

        events = conn.execute("SELECT * FROM detected_events").fetchall()
        assert len(events) >= 1
        assert any(e['event_type'] == 'harsh_brake' for e in events)
        conn.close()

    def test_skip_no_gps_video(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        # Video with lat=0, lon=0 (no GPS)
        payloads = [_make_sei_protobuf(lat=0.0, lon=0.0)]
        mp4_data = _make_synthetic_mp4(payloads)

        teslacam = tmp_path / "TeslaCam" / "RecentClips"
        teslacam.mkdir(parents=True)
        video_file = teslacam / "2025-11-08_08-15-44-front.mp4"
        video_file.write_bytes(mp4_data)

        result = _index_video(
            conn, str(video_file), str(tmp_path / "TeslaCam"),
            sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

        )

        assert result.waypoints == 0
        assert result.events == 0
        # Recent-folder no-GPS clips are recorded as NO_MOVEMENT_RECORDED so the
        # queue worker can drop the row without flapping retries.
        assert result.outcome == IndexOutcome.NO_MOVEMENT_RECORDED
        conn.close()

    def test_indexed_files_fallback_dedup_when_video_path_nulled(self, tmp_path):
        # Defense-in-depth: when ``waypoints.video_path`` was nulled by
        # ``purge_deleted_videos`` (because a sibling copy of the clip
        # was deleted), the primary canonical-key check on
        # ``waypoints.video_path IN (...)`` returns no rows. Without
        # this fallback the indexer would re-parse the clip and insert
        # a SECOND set of waypoints + detected_events, producing the
        # duplicate event pins we hit on May 10/11.
        #
        # The fallback uses ``indexed_files`` as the authoritative
        # "we processed this physical file" record and refuses to
        # re-index when a row exists with ``waypoint_count > 0``.
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        payloads = [
            _make_sei_protobuf(lat=37.7749, lon=-122.4194, speed=25.0),
            _make_sei_protobuf(lat=37.7750, lon=-122.4195, speed=26.0),
        ]
        mp4_data = _make_synthetic_mp4(payloads)
        teslacam = tmp_path / "TeslaCam" / "RecentClips"
        teslacam.mkdir(parents=True)
        video_file = teslacam / "2025-11-08_08-15-44-front.mp4"
        video_file.write_bytes(mp4_data)

        # First index — populates waypoints and indexed_files normally.
        first = _index_video(
            conn, str(video_file), str(tmp_path / "TeslaCam"),
            sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

        )
        assert first.outcome == IndexOutcome.INDEXED
        assert first.waypoints == 2
        wp_count_after_first = conn.execute(
            "SELECT COUNT(*) FROM waypoints"
        ).fetchone()[0]
        assert wp_count_after_first == 2

        # ``index_single_file`` (the public entry point) records the
        # indexed_files row after ``_index_video`` returns. Simulate
        # that here so the fallback has authoritative state to consult.
        from datetime import datetime, timezone
        st = video_file.stat()
        conn.execute(
            "INSERT OR REPLACE INTO indexed_files "
            "(file_path, file_size, file_mtime, indexed_at, "
            "waypoint_count, event_count) VALUES (?, ?, ?, ?, ?, ?)",
            (str(video_file), st.st_size, st.st_mtime,
             datetime.now(timezone.utc).isoformat(), 2, 0),
        )

        # Simulate the production data anomaly: a prior
        # purge_deleted_videos run NULLed the video_path on every
        # waypoint for this clip (e.g., because a sibling copy was
        # deleted before the surviving-copy check found this one).
        # ``indexed_files`` keeps its row — that's the asymmetry
        # the fallback exploits.
        conn.execute("UPDATE waypoints SET video_path = NULL")
        conn.execute("UPDATE detected_events SET video_path = NULL")
        conn.commit()

        # Re-index the same clip. With ONLY the
        # ``waypoints.video_path IN (...)`` dedup, this would
        # fall through to SEI extraction and double the row count.
        # The new fallback should return ALREADY_INDEXED.
        second = _index_video(
            conn, str(video_file), str(tmp_path / "TeslaCam"),
            sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

        )
        assert second.outcome == IndexOutcome.ALREADY_INDEXED
        wp_count_after_second = conn.execute(
            "SELECT COUNT(*) FROM waypoints"
        ).fetchone()[0]
        # Critical: NO new waypoint inserts. The pre-existing
        # 2 rows (with NULL video_path) are intact, no duplicates
        # added.
        assert wp_count_after_second == 2
        conn.close()

    def test_indexed_files_fallback_does_not_overmatch_underscore(self, tmp_path):
        # Tesla filenames contain ``_`` separators (the SQLite LIKE
        # single-character wildcard). Without an ``ESCAPE`` clause, the
        # fallback's ``LIKE '%basename'`` could match a different clip
        # whose basename happens to align character-for-character with
        # ``_`` standing in for any character. This test seeds an
        # ``indexed_files`` row whose basename differs from the clip
        # only at the ``_`` positions and confirms the fallback does
        # NOT short-circuit (the indexer still runs and produces real
        # waypoints).
        from datetime import datetime, timezone
        db_path = str(tmp_path / "test.db")
        conn = _init_db(db_path)

        payloads = [
            _make_sei_protobuf(lat=37.7749, lon=-122.4194, speed=25.0),
        ]
        mp4_data = _make_synthetic_mp4(payloads)
        teslacam = tmp_path / "TeslaCam" / "RecentClips"
        teslacam.mkdir(parents=True)
        # The clip we're about to index:
        video_file = teslacam / "2025-11-08_08-15-44-front.mp4"
        video_file.write_bytes(mp4_data)

        # Seed an indexed_files row for a DIFFERENT clip whose basename
        # matches the target clip's basename only if ``_`` is treated
        # as a wildcard (every ``_`` replaced with another character).
        # Without escaping, the naive ``LIKE '%2025-11-08_08-15-44-...'``
        # query would mistakenly match this row.
        impostor_basename = "2025-11-08X08-15-44-front.mp4"
        impostor_abs = "/some/other/path/" + impostor_basename
        conn.execute(
            "INSERT INTO indexed_files "
            "(file_path, file_size, file_mtime, indexed_at, "
            "waypoint_count, event_count) VALUES (?, ?, ?, ?, ?, ?)",
            (impostor_abs, 9999, 1.0,
             datetime.now(timezone.utc).isoformat(), 5, 0),
        )
        conn.commit()

        # The indexer must NOT treat the impostor row as evidence
        # that THIS clip was already indexed. It should index normally.
        result = _index_video(
            conn, str(video_file), str(tmp_path / "TeslaCam"),
            sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

        )
        assert result.outcome == IndexOutcome.INDEXED
        assert result.waypoints == 1
        conn.close()


# ---------------------------------------------------------------------------
# Trip Fragmentation Defense Tests
# ---------------------------------------------------------------------------
# These tests guard against the May 2026 phantom-duplicate-trips incident
# where one round-trip drive was split into 6 fragments because:
#   1. The indexer paused mid-drive due to archive-lock starvation
#   2. New files queued during the pause got processed AFTER the pause
#   3. So files arrived out-of-order: t=0..t=5min, [pause], t=10..t=12min,
#      then t=6..t=9min
#   4. The matching SQL's old "ORDER BY ABS(new_start - existing.start)"
#      tie-breaker mis-assigned the t=6..9 fillers
#   5. Once split, no code re-merged adjacent trips at runtime (only the
#      one-shot v2→v3 migration did)
# Both the matching-order fix AND the post-insert merge are exercised here.

def _index_synthetic_at(conn, tmp_path, filename: str, lat: float = 37.7749,
                        lon: float = -122.4194):
    """Index one synthetic single-waypoint clip into ``conn``.

    The waypoint timestamp comes from the filename — see
    ``_timestamp_from_filename`` — so callers control trip placement
    purely through the filename.
    """
    payloads = [_make_sei_protobuf(lat=lat, lon=lon, speed=20.0)]
    mp4_data = _make_synthetic_mp4(payloads)
    teslacam = tmp_path / "TeslaCam" / "RecentClips"
    teslacam.mkdir(parents=True, exist_ok=True)
    video_file = teslacam / filename
    video_file.write_bytes(mp4_data)
    return _index_video(
        conn, str(video_file), str(tmp_path / "TeslaCam"),
        sample_rate=1, thresholds=DEFAULT_THRESHOLDS,

    )


class TestTripFragmentationDefense:
        )
        conn.close()


# ---------------------------------------------------------------------------
# Driving Stats & Event Chart Data Tests
# ---------------------------------------------------------------------------

class TestDrivingStats:
    @pytest.fixture
    def db_with_driving_data(self, tmp_path):
        db_path = str(tmp_path / "stats.db")
        conn = _init_db(db_path)

        conn.execute(
            """INSERT INTO trips (id, start_time, end_time, start_lat, start_lon,
               end_lat, end_lon, distance_km, duration_seconds, source_folder)
               VALUES (1, '2025-11-08T08:15:44', '2025-11-08T08:25:44',
               37.7749, -122.4194, 37.7850, -122.4100, 15.5, 600, 'RecentClips')"""
        )
        conn.execute(
            """INSERT INTO trips (id, start_time, end_time, distance_km, duration_seconds, source_folder)
               VALUES (2, '2025-11-09T10:00:00', '2025-11-09T10:30:00', 25.0, 1800, 'RecentClips')"""
        )

        # Waypoints with mixed autopilot states
        for i in range(10):
            ap = 'AUTOSTEER' if i < 4 else 'NONE'
            conn.execute(
                """INSERT INTO waypoints (trip_id, timestamp, lat, lon, speed_mps,
                   autopilot_state, video_path, frame_offset)
                   VALUES (1, ?, ?, ?, ?, ?, 'test.mp4', ?)""",
                (f'2025-11-08T08:1{5+i}:44', 37.77 + i*0.001, -122.41 + i*0.001,
                 20.0 + i, ap, i*30)
            )

        # Events
        conn.execute(
            """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
               event_type, severity, description)
               VALUES (1, '2025-11-08T08:17:44', 37.77, -122.41,
               'harsh_brake', 'warning', 'test')"""
        )
        conn.execute(
            """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
               event_type, severity, description)
               VALUES (1, '2025-11-08T08:18:44', 37.77, -122.41,
               'speeding', 'info', 'test')"""
        )
        conn.execute(
            """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
               event_type, severity, description)
               VALUES (1, '2025-11-08T08:19:44', 37.77, -122.41,
               'emergency_brake', 'critical', 'test')"""
        )

        conn.commit()
        conn.close()
        return db_path

    def test_has_data(self, db_with_driving_data, tmp_path):
        stats = get_driving_stats(db_with_driving_data)
        assert stats['has_data'] is True

        empty_db = str(tmp_path / "empty.db")
        stats = get_driving_stats(empty_db)
        assert stats['has_data'] is False


class TestEventChartData:
    @pytest.fixture
    def db_with_events(self, tmp_path):
        db_path = str(tmp_path / "charts.db")
        conn = _init_db(db_path)

        # Need a trip for FK constraints
        conn.execute(
            """INSERT INTO trips (id, start_time, distance_km, duration_seconds, source_folder)
               VALUES (1, '2025-11-08T08:10:00', 10.0, 600, 'RecentClips')"""
        )

        # Insert waypoints with FSD data
        for i in range(5):
            ap = 'SELF_DRIVING' if i < 2 else 'NONE'
            conn.execute(
                """INSERT INTO waypoints (trip_id, timestamp, lat, lon, speed_mps,
                   autopilot_state) VALUES (1, ?, 37.0, -122.0, 25.0, ?)""",
                (f'2025-11-08T08:1{i}:00', ap)
            )

        events = [
            ('harsh_brake', 'warning'), ('harsh_brake', 'warning'),
            ('speeding', 'info'), ('emergency_brake', 'critical'),
            ('fsd_disengage', 'warning'),
        ]
        for i, (etype, sev) in enumerate(events):
            conn.execute(
                """INSERT INTO detected_events (trip_id, timestamp, lat, lon,
                   event_type, severity, description)
                   VALUES (1, ?, 37.0, -122.0, ?, ?, 'test')""",
                (f'2025-11-08T08:1{i}:00', etype, sev)
            )

        conn.commit()
        conn.close()
        return db_path

    def test_by_type(self, db_with_events):
        data = get_event_chart_data(db_with_events)
        assert len(data['by_type']['labels']) > 0
        assert sum(data['by_type']['values']) == 5

    def test_by_severity(self, db_with_events):
        data = get_event_chart_data(db_with_events)
        assert len(data['by_severity']['labels']) == 3  # critical, warning, info
        assert len(data['by_severity']['colors']) == 3

    def test_over_time(self, db_with_events):
        data = get_event_chart_data(db_with_events)
        assert 'labels' in data['over_time']
        assert 'values' in data['over_time']

    def test_fsd_timeline(self, db_with_events):
        data = get_event_chart_data(db_with_events)
        assert 'labels' in data['fsd_timeline']
        assert 'fsd' in data['fsd_timeline']
        assert 'manual' in data['fsd_timeline']

    def test_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        _init_db(db_path)
        data = get_event_chart_data(db_path)
        assert data['by_type']['labels'] == []
        assert data['by_type']['values'] == []




# ---------------------------------------------------------------------------
# IndexResult Outcome Dispatch Tests
# ---------------------------------------------------------------------------

class TestIndexResultOutcomes:
    """Each non-INDEXED outcome maps to a specific queue dispatch decision.
    These tests pin the contract so the worker can rely on it."""

    def test_terminal_outcomes(self):
        # All of these allow the queue worker to delete the row.
        for outcome in (
            IndexOutcome.INDEXED,
            IndexOutcome.ALREADY_INDEXED,
            IndexOutcome.DUPLICATE_UPGRADED,
            IndexOutcome.NO_MOVEMENT_RECORDED,
            IndexOutcome.NOT_FRONT_CAMERA,
            IndexOutcome.FILE_MISSING,
        ):
            assert IndexResult(outcome).terminal, outcome

    def test_non_terminal_outcomes_require_retry(self):
        # The queue must NOT delete these — worker either reschedules
        # (TOO_NEW), backs off (PARSE_ERROR), or releases the claim
        # (DB_BUSY).
        for outcome in (
            IndexOutcome.TOO_NEW,
            IndexOutcome.PARSE_ERROR,
            IndexOutcome.DB_BUSY,
        ):
            assert not IndexResult(outcome).terminal, outcome


class TestIndexSingleFileOutcomes:
    def test_not_front_camera(self, tmp_path):
        # Right basename for a Tesla clip but wrong camera.
        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-back.mp4"
        clip.write_bytes(b'')

        result = index_single_file(str(clip), db, str(tmp_path))
        assert result.outcome == IndexOutcome.NOT_FRONT_CAMERA
        assert result.terminal

    def test_file_missing(self, tmp_path):
        db = str(tmp_path / "geo.db")
        _init_db(db)
        result = index_single_file(
            str(tmp_path / "does-not-exist-front.mp4"),
            db,
            str(tmp_path),
        )
        assert result.outcome == IndexOutcome.FILE_MISSING
        assert result.terminal

    def test_too_new(self, tmp_path):
        # File exists but mtime is now() — Tesla may still be writing.
        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'')
        result = index_single_file(str(clip), db, str(tmp_path))
        assert result.outcome == IndexOutcome.TOO_NEW
        assert not result.terminal  # worker should retry once mtime ages

    def test_parse_error_caught(self, tmp_path):
        # Old-enough file (>120s) with no MP4 atoms at all → parser raises.
        # Result is reported as PARSE_ERROR so the queue worker can apply
        # exponential backoff instead of looping forever.
        import os as _os
        import time as _time
        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'not an mp4 file at all')
        # Backdate so the TOO_NEW guard doesn't intercept us.
        old = _time.time() - 600
        _os.utime(str(clip), (old, old))

        result = index_single_file(str(clip), db, str(tmp_path))
        # Could legitimately come back as either NO_MOVEMENT_RECORDED (parser
        # found 0 SEI frames) or PARSE_ERROR (parser raised). Both are
        # acceptable terminal-or-retry classifications — the assertion
        # we care about is that we never get an INDEXED with 0 waypoints.
        assert result.outcome in (
            IndexOutcome.NO_MOVEMENT_RECORDED,
            IndexOutcome.PARSE_ERROR,
        )
        if result.outcome == IndexOutcome.PARSE_ERROR:
            assert result.error is not None
            assert not result.terminal


class TestIndexSingleFileSidecarConsumption:
    """Issue #197: ``_index_video`` (via ``index_single_file``) must
    prefer a sidecar JSON over a fresh mmap walk when one exists.
    """

    def _make_sidecar_with_messages(
        self, video_path, sample_rate=30, messages=None, mvhd=None,
    ):
        """Hand-build a sidecar JSON the indexer should consume
        without calling the real SEI parser. Lets us isolate the
        indexer's sidecar branch from the rest of the parser stack."""
        import json as _json
        import os as _os
        from services import sei_parser

        if messages is None:
            messages = [
                {
                    'frame_index': 0,
                    'timestamp_ms': 0.0,
                    'latitude_deg': 37.7749,
                    'longitude_deg': -122.4194,
                    'heading_deg': 90.0,
                    'vehicle_speed_mps': 25.0,
                    'linear_acceleration_x': 0.1,
                    'linear_acceleration_y': 0.0,
                    'linear_acceleration_z': -0.1,
                    'steering_wheel_angle': 0.5,
                    'accelerator_pedal_position': 0.2,
                    'brake_applied': False,
                    'gear_state': 'DRIVE',
                    'autopilot_state': 'NONE',
                    'blinker_on_left': False,
                    'blinker_on_right': False,
                    'frame_seq_no': 0,
                },
            ]
        st = _os.stat(video_path)
        payload = {
            'schema_version': sei_parser.SIDECAR_SCHEMA_VERSION,
            'sample_rate': sample_rate,
            'sei_count': len(messages),
            'no_movement_count': 0,
            'mvhd_creation_time_utc': mvhd,
            'video_size_bytes': st.st_size,
            'video_mtime_unix': st.st_mtime,
            'messages': messages,
        }
        with open(sei_parser.sidecar_path_for(video_path), 'w',
                  encoding='utf-8') as f:
            _json.dump(payload, f)

    def test_index_consumes_sidecar_without_mmap_walk(
        self, tmp_path, monkeypatch,
    ):
        """When a valid sidecar exists, ``_index_video`` must NOT
        call ``parser.extract_sei_messages`` — proves the sidecar
        path is short-circuiting the mmap walk."""
        import os as _os
        import time as _time
        from services import sei_parser

        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'\x00' * 64)
        old = _time.time() - 600
        _os.utime(str(clip), (old, old))

        self._make_sidecar_with_messages(str(clip))

        # Sentinel: explode if extract_sei_messages is touched.
        called: list = []

        def _exploder(*a, **kw):
            called.append((a, kw))
            raise AssertionError(
                "extract_sei_messages was called even though a "
                "valid sidecar exists — sidecar fast-path is broken."
            )

        monkeypatch.setattr(
            sei_parser, 'extract_sei_messages', _exploder,
        )

        result = index_single_file(
            str(clip), db, str(tmp_path), sample_rate=30,
        )
        assert result.outcome == IndexOutcome.INDEXED
        assert result.waypoints == 1
        assert called == []

    def test_index_falls_back_to_mmap_when_sidecar_missing(
        self, tmp_path, monkeypatch,
    ):
        """Without a sidecar, the indexer must transparently fall
        back to ``extract_sei_messages``. Pre-issue-#197 baseline
        path — must continue to work for clips that pre-date the
        sidecar feature or for clips whose sidecar was lost."""
        import os as _os
        import time as _time
        from services import sei_parser

        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'\x00' * 64)
        old = _time.time() - 600
        _os.utime(str(clip), (old, old))

        # No sidecar created. Stub extract_sei_messages with a
        # synthetic generator so the test doesn't need a real MP4.
        called: list = []

        def _gen(video_path, sample_rate):
            called.append((video_path, sample_rate))
            yield sei_parser.SeiMessage(
                frame_index=0, timestamp_ms=0.0,
                latitude_deg=37.7749, longitude_deg=-122.4194,
                heading_deg=90.0, vehicle_speed_mps=25.0,
                linear_acceleration_x=0.0, linear_acceleration_y=0.0,
                linear_acceleration_z=0.0,
                steering_wheel_angle=0.0, accelerator_pedal_position=0.0,
                brake_applied=False,
                gear_state='DRIVE', autopilot_state='NONE',
                blinker_on_left=False, blinker_on_right=False,
                frame_seq_no=0, video_path=video_path,
            )

        monkeypatch.setattr(sei_parser, 'extract_sei_messages', _gen)

        result = index_single_file(
            str(clip), db, str(tmp_path), sample_rate=30,
        )
        assert result.outcome == IndexOutcome.INDEXED
        assert called and called[0][1] == 30, (
            "extract_sei_messages was not called on the fallback "
            "path — indexer would have produced no waypoints."
        )

    def test_index_falls_back_to_mmap_on_sidecar_size_drift(
        self, tmp_path, monkeypatch,
    ):
        """Drift detection: sidecar's recorded size differs from
        the live file's size → ``read_sei_sidecar`` returns None →
        indexer mmap-parses."""
        import os as _os
        import time as _time
        from services import sei_parser

        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'\x00' * 64)
        old = _time.time() - 600
        _os.utime(str(clip), (old, old))

        # Sidecar describes the file as it is now …
        self._make_sidecar_with_messages(str(clip))
        # … then we overwrite with a different size (post-sidecar
        # write). Drift → invalidation → fallback.
        with open(str(clip), 'ab') as f:
            f.write(b'\x00' * 1024)
        _os.utime(str(clip), (old, old))

        called: list = []

        def _gen(video_path, sample_rate):
            called.append(True)
            yield sei_parser.SeiMessage(
                frame_index=0, timestamp_ms=0.0,
                latitude_deg=37.0, longitude_deg=-122.0,
                heading_deg=0.0, vehicle_speed_mps=10.0,
                linear_acceleration_x=0.0, linear_acceleration_y=0.0,
                linear_acceleration_z=0.0,
                steering_wheel_angle=0.0, accelerator_pedal_position=0.0,
                brake_applied=False, gear_state='DRIVE',
                autopilot_state='NONE',
                blinker_on_left=False, blinker_on_right=False,
                frame_seq_no=0, video_path=video_path,
            )

        monkeypatch.setattr(sei_parser, 'extract_sei_messages', _gen)

        result = index_single_file(
            str(clip), db, str(tmp_path), sample_rate=30,
        )
        assert result.outcome == IndexOutcome.INDEXED
        assert called == [True], (
            "Indexer did not fall back to mmap parse despite "
            "sidecar size-drift invalidation — would silently "
            "use stale data."
        )

    def test_index_falls_back_when_sample_rate_mismatches(
        self, tmp_path, monkeypatch,
    ):
        """If the cached sidecar was written at a different
        sample_rate than the indexer is requesting, the
        ``required_sample_rate`` guard invalidates the sidecar."""
        import os as _os
        import time as _time
        from services import sei_parser

        db = str(tmp_path / "geo.db")
        _init_db(db)
        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'\x00' * 64)
        old = _time.time() - 600
        _os.utime(str(clip), (old, old))

        # Sidecar at sample_rate=1; indexer asks for 30 → mismatch.
        self._make_sidecar_with_messages(str(clip), sample_rate=1)

        called: list = []

        def _gen(video_path, sample_rate):
            called.append(sample_rate)
            yield sei_parser.SeiMessage(
                frame_index=0, timestamp_ms=0.0,
                latitude_deg=37.0, longitude_deg=-122.0,
                heading_deg=0.0, vehicle_speed_mps=10.0,
                linear_acceleration_x=0.0, linear_acceleration_y=0.0,
                linear_acceleration_z=0.0,
                steering_wheel_angle=0.0, accelerator_pedal_position=0.0,
                brake_applied=False, gear_state='DRIVE',
                autopilot_state='NONE',
                blinker_on_left=False, blinker_on_right=False,
                frame_seq_no=0, video_path=video_path,
            )

        monkeypatch.setattr(sei_parser, 'extract_sei_messages', _gen)

        result = index_single_file(
            str(clip), db, str(tmp_path), sample_rate=30,
        )
        assert result.outcome == IndexOutcome.INDEXED
        assert called == [30]


class TestPurgeDeletedVideosSidecar:
    """Issue #197: ``purge_deleted_videos`` must delete the SEI
    sidecar JSON alongside the indexed_files row, so a deleted
    .mp4 doesn't leave dead sidecar weight in the directory."""

    def test_purge_deletes_sidecar(self, tmp_path):
        from services import sei_parser
        from services.mapping_service import (
            _init_db, purge_deleted_videos,
        )

        db = str(tmp_path / "geo.db")
        _init_db(db).close()

        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'\x00' * 64)
        sidecar_path = sei_parser.sidecar_path_for(str(clip))
        # Hand-create a fake sidecar — content doesn't matter; we
        # only assert it's gone after purge.
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            f.write('{}')
        assert os.path.isfile(sidecar_path)

        # Pretend the .mp4 is gone (the watcher's normal fire path).
        clip.unlink()

        result = purge_deleted_videos(
            db, deleted_paths=[str(clip)],
        )
        assert result['purged_files'] == 0  # no indexed_files row
        assert not os.path.isfile(sidecar_path), (
            "Sidecar was not deleted alongside the .mp4 — "
            "would accumulate as dead weight in the directory."
        )

    def test_purge_handles_missing_sidecar(self, tmp_path):
        """A clip whose sidecar never existed (pre-#197 file, or
        sidecar write failed) must not break purge."""
        from services.mapping_service import (
            _init_db, purge_deleted_videos,
        )

        db = str(tmp_path / "geo.db")
        _init_db(db).close()

        clip = tmp_path / "2025-11-08_08-15-44-front.mp4"
        clip.write_bytes(b'\x00')
        clip.unlink()

        result = purge_deleted_videos(
            db, deleted_paths=[str(clip)],
        )
        assert result['purged_files'] == 0


# ---------------------------------------------------------------------------
# Phase 2: Indexing queue
# ---------------------------------------------------------------------------


class TestPriorityForPath:
    def test_sentry_clip_is_highest_priority(self):
        path = '/mnt/teslacam/SentryClips/2025-01-01_event/clip-front.mp4'
        assert priority_for_path(path) == _PRIORITY_SENTRY_SAVED

    def test_saved_clip_is_highest_priority(self):
        path = '/mnt/teslacam/SavedClips/event/clip-front.mp4'
        assert priority_for_path(path) == _PRIORITY_SENTRY_SAVED

    def test_archived_clip_lower_than_event(self):
        path = '/mnt/sd/ArchivedClips/2025-01-01/clip-front.mp4'
        assert priority_for_path(path) == _PRIORITY_ARCHIVE
        assert _PRIORITY_ARCHIVE > _PRIORITY_SENTRY_SAVED

    def test_recent_clip_lowest_among_known_folders(self):
        path = '/mnt/teslacam/RecentClips/clip-front.mp4'
        assert priority_for_path(path) == _PRIORITY_RECENT
        assert _PRIORITY_RECENT > _PRIORITY_ARCHIVE

    def test_windows_path_separator_is_normalized(self):
        path = r'D:\TeslaCam\SentryClips\event\clip-front.mp4'
        assert priority_for_path(path) == _PRIORITY_SENTRY_SAVED

    def test_unknown_folder_gets_default(self):
        path = '/some/random/place/clip.mp4'
        assert priority_for_path(path) == 50

    def test_empty_path_gets_default(self):
        assert priority_for_path('') == 50


class TestComputeBackoff:
    def test_first_failure_uses_base_backoff(self):
        # attempts=0 means "no failures yet, computing wait for the first
        # retry". delay = base * 2^0 = base.
        assert compute_backoff(0) == 60.0

    def test_backoff_doubles_each_attempt(self):
        assert compute_backoff(1) == 120.0
        assert compute_backoff(2) == 240.0
        assert compute_backoff(3) == 480.0

    def test_backoff_is_capped(self):
        # 60 * 2^10 = 61440, well past the 3600 cap.
        assert compute_backoff(10) == 3600.0

    def test_negative_attempts_treated_as_zero(self):
        assert compute_backoff(-5) == 60.0


class TestEnqueue:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "queue.db")
        _init_db(db_path)
        return db_path

    def test_enqueue_writes_one_row(self, db):
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/2025-01-01_clip-front.mp4',
            source='watcher',
        )
        with sqlite3.connect(db) as c:
            rows = c.execute("SELECT * FROM indexing_queue").fetchall()
        assert len(rows) == 1

    def test_enqueue_uses_canonical_key(self, db):
        # Same canonical_key for both Recent and Archived versions.
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/2025-01-01_clip-front.mp4',
        )
        assert enqueue_for_indexing(
            db, '/mnt/sd/ArchivedClips/2025-01-01_clip-front.mp4',
        )
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        # One row, deduplicated by canonical_key.
        assert count == 1

    def test_enqueue_lowers_priority_when_more_urgent(self, db):
        # First enqueue at default (50), then upgrade to sentry priority.
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',  # canonical_key = "clip.mp4"
            priority=50,
        )
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
            priority=10,
        )
        with sqlite3.connect(db) as c:
            prio = c.execute(
                "SELECT priority FROM indexing_queue"
            ).fetchone()[0]
        assert prio == 10

    def test_enqueue_does_not_raise_priority(self, db):
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
            priority=10,
        )
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
            priority=50,
        )
        with sqlite3.connect(db) as c:
            prio = c.execute(
                "SELECT priority FROM indexing_queue"
            ).fetchone()[0]
        # MIN(50, 10) = 10 — re-enqueue at lower priority is a no-op.
        assert prio == 10

    def test_enqueue_empty_path_returns_false(self, db):
        assert enqueue_for_indexing(db, '') is False
        assert enqueue_for_indexing(db, None) is False  # type: ignore

    def test_enqueue_does_not_overwrite_claimed_row(self, db):
        # Simulate a worker holding a claim. A new enqueue for the same
        # canonical_key must NOT change file_path or source — that would
        # rip the file out from under the worker.
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
            source='watcher',
        )
        with sqlite3.connect(db) as c:
            c.execute(
                """UPDATE indexing_queue SET claimed_by='w1', claimed_at=?
                   WHERE canonical_key='clip.mp4'""",
                (time.time(),),
            )
        # Try to "upgrade" the path/source while it's claimed.
        enqueue_for_indexing(
            db, '/mnt/sd/ArchivedClips/clip.mp4',
            source='archive',
        )
        with sqlite3.connect(db) as c:
            row = c.execute(
                """SELECT file_path, source FROM indexing_queue
                   WHERE canonical_key='clip.mp4'"""
            ).fetchone()
        assert row[0] == '/mnt/teslacam/RecentClips/clip.mp4'
        assert row[1] == 'watcher'

    def test_enqueue_with_next_attempt_at_defers_first_claim(self, db):
        # Producers (the archive flow in particular) need to defer the
        # first attempt atomically with the INSERT to avoid racing the
        # worker. Verify the deferral lands on a fresh row.
        future = time.time() + 120
        assert enqueue_for_indexing(
            db, '/mnt/sd/ArchivedClips/clip-front.mp4',
            source='archive',
            next_attempt_at=future,
        ) is True
        with sqlite3.connect(db) as c:
            row = c.execute(
                """SELECT next_attempt_at FROM indexing_queue
                   WHERE canonical_key='clip-front.mp4'"""
            ).fetchone()
        assert abs(row[0] - future) < 0.01

    def test_enqueue_without_next_attempt_at_is_immediate(self, db):
        # The default is "available right now" so the watcher path
        # doesn't need to know about the deferral feature.
        assert enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip-front.mp4',
        ) is True
        with sqlite3.connect(db) as c:
            row = c.execute(
                """SELECT next_attempt_at FROM indexing_queue
                   WHERE canonical_key='clip-front.mp4'"""
            ).fetchone()
        assert row[0] == 0.0


class TestEnqueueMany:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "queue.db")
        _init_db(db_path)
        return db_path

    def test_batch_inserts_all(self, db):
        items = [
            ('/mnt/teslacam/RecentClips/a-front.mp4', None),
            ('/mnt/teslacam/RecentClips/b-front.mp4', None),
            ('/mnt/teslacam/SentryClips/event/c-front.mp4', None),
        ]
        n = enqueue_many_for_indexing(db, items, source='catchup')
        assert n == 3
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        assert count == 3

    def test_batch_dedups_by_canonical_key(self, db):
        # The Recent and Archived versions share canonical_key — second
        # one collapses into the first.
        items = [
            ('/mnt/teslacam/RecentClips/clip-front.mp4', None),
            ('/mnt/sd/ArchivedClips/clip-front.mp4', None),
        ]
        enqueue_many_for_indexing(db, items)
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        assert count == 1

    def test_batch_skips_empty(self, db):
        items = [
            ('', None),
            ('/mnt/teslacam/RecentClips/a-front.mp4', None),
        ]
        n = enqueue_many_for_indexing(db, items)
        assert n == 1


class TestClaimQueueItem:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "queue.db")
        _init_db(db_path)
        return db_path

    def test_returns_none_when_empty(self, db):
        assert claim_next_queue_item(db, 'worker-1') is None

    def test_claim_returns_highest_priority_first(self, db):
        # Insert in random order.
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/recent.mp4',
            priority=_PRIORITY_RECENT,
        )
        enqueue_for_indexing(
            db, '/mnt/teslacam/SentryClips/event/sentry-front.mp4',
            priority=_PRIORITY_SENTRY_SAVED,
        )
        enqueue_for_indexing(
            db, '/mnt/sd/ArchivedClips/archive.mp4',
            priority=_PRIORITY_ARCHIVE,
        )
        row = claim_next_queue_item(db, 'worker-1')
        assert row is not None
        assert row['canonical_key'] == 'SentryClips/event/sentry-front.mp4'
        assert row['priority'] == _PRIORITY_SENTRY_SAVED

    def test_claim_marks_row_claimed(self, db):
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        claim_next_queue_item(db, 'worker-X')
        with sqlite3.connect(db) as c:
            row = c.execute(
                "SELECT claimed_by, claimed_at FROM indexing_queue"
            ).fetchone()
        assert row[0] == 'worker-X'
        assert row[1] is not None

    def test_two_concurrent_claims_dont_double_book(self, db, tmp_path):
        # The atomic-claim contract: even with two threads racing, a
        # given canonical_key can only be picked once per release cycle.
        # Enqueue 5 items, spawn 2 worker threads, each claiming as fast
        # as possible. No canonical_key should appear in both workers'
        # results.
        import threading
        for i in range(5):
            enqueue_for_indexing(
                db, f'/mnt/teslacam/RecentClips/clip{i}.mp4',
            )
        results = {'a': [], 'b': []}

        def claim_loop(label):
            for _ in range(10):
                row = claim_next_queue_item(db, label)
                if row is None:
                    break
                results[label].append(row['canonical_key'])

        ta = threading.Thread(target=claim_loop, args=('a',))
        tb = threading.Thread(target=claim_loop, args=('b',))
        ta.start(); tb.start()
        ta.join(timeout=10); tb.join(timeout=10)

        all_claimed = results['a'] + results['b']
        assert len(all_claimed) == 5
        assert len(set(all_claimed)) == 5  # No duplicates.

    def test_claim_skips_future_attempts(self, db):
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        # Defer to 100s in the future.
        defer_queue_item(
            db, 'clip.mp4', time.time() + 100,
        )
        assert claim_next_queue_item(db, 'worker-1') is None

    def test_claim_skips_dead_letter_rows(self, db):
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        # Drive attempts past the cap.
        with sqlite3.connect(db) as c:
            c.execute(
                "UPDATE indexing_queue SET attempts = ?",
                (_PARSE_ERROR_MAX_ATTEMPTS,),
            )
        assert claim_next_queue_item(db, 'worker-1') is None


class TestCompleteAndRelease:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "queue.db")
        _init_db(db_path)
        enqueue_for_indexing(
            db_path, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        claim_next_queue_item(db_path, 'worker-1')
        return db_path

    def test_complete_deletes_row(self, db):
        assert complete_queue_item(db, 'clip.mp4') is True
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        assert count == 0

    def test_complete_no_row_returns_false(self, db):
        assert complete_queue_item(db, 'nonexistent.mp4') is False

    def test_release_clears_claim_but_keeps_row(self, db):
        assert release_claim(db, 'clip.mp4') is True
        with sqlite3.connect(db) as c:
            row = c.execute(
                "SELECT claimed_by, claimed_at, attempts FROM indexing_queue"
            ).fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] == 0  # release does NOT bump attempts

    def test_after_release_can_be_reclaimed(self, db):
        release_claim(db, 'clip.mp4')
        row = claim_next_queue_item(db, 'worker-2')
        assert row is not None
        assert row['canonical_key'] == 'clip.mp4'


class TestDeferQueueItem:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "queue.db")
        _init_db(db_path)
        enqueue_for_indexing(
            db_path, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        claim_next_queue_item(db_path, 'worker-1')
        return db_path

    def test_defer_without_bump_does_not_increment_attempts(self, db):
        future = time.time() + 200
        assert defer_queue_item(
            db, 'clip.mp4', future, bump_attempts=False,
        )
        with sqlite3.connect(db) as c:
            row = c.execute(
                """SELECT attempts, next_attempt_at, claimed_by
                   FROM indexing_queue"""
            ).fetchone()
        assert row[0] == 0
        assert abs(row[1] - future) < 1e-3
        assert row[2] is None

    def test_defer_with_bump_increments_attempts(self, db):
        defer_queue_item(
            db, 'clip.mp4', time.time() + 60,
            bump_attempts=True, last_error='boom',
        )
        with sqlite3.connect(db) as c:
            row = c.execute(
                "SELECT attempts, last_error FROM indexing_queue"
            ).fetchone()
        assert row[0] == 1
        assert row[1] == 'boom'


class TestRecoverStaleClaims:
    def test_releases_old_claim(self, tmp_path):
        db = str(tmp_path / "stale.db")
        _init_db(db)
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        # Manually plant an ancient claim.
        with sqlite3.connect(db) as c:
            c.execute(
                """UPDATE indexing_queue
                   SET claimed_by='dead-worker', claimed_at=?""",
                (time.time() - 7200,),  # 2 hours ago
            )
        n = recover_stale_claims(db, max_age_seconds=1800)
        assert n == 1
        with sqlite3.connect(db) as c:
            row = c.execute(
                "SELECT claimed_by FROM indexing_queue"
            ).fetchone()
        assert row[0] is None

    def test_keeps_recent_claim(self, tmp_path):
        db = str(tmp_path / "stale.db")
        _init_db(db)
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/clip.mp4',
        )
        with sqlite3.connect(db) as c:
            c.execute(
                """UPDATE indexing_queue
                   SET claimed_by='active-worker', claimed_at=?""",
                (time.time() - 60,),  # 1 minute ago
            )
        assert recover_stale_claims(db, max_age_seconds=1800) == 0


class TestQueueStatus:
    def test_status_on_empty_queue(self, tmp_path):
        db = str(tmp_path / "q.db")
        _init_db(db)
        st = get_queue_status(db)
        assert st['queue_depth'] == 0
        assert st['claimed_count'] == 0
        assert st['dead_letter_count'] == 0
        assert st['next_ready_at'] is None

    def test_status_reflects_state(self, tmp_path):
        db = str(tmp_path / "q.db")
        _init_db(db)
        # Three pending, one claimed, one dead-lettered.
        for i in range(3):
            enqueue_for_indexing(
                db, f'/mnt/teslacam/RecentClips/p{i}.mp4',
            )
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/claimed.mp4',
        )
        with sqlite3.connect(db) as c:
            c.execute(
                """UPDATE indexing_queue
                   SET claimed_by='w', claimed_at=?
                   WHERE canonical_key='claimed.mp4'""",
                (time.time(),),
            )
        enqueue_for_indexing(
            db, '/mnt/teslacam/RecentClips/dead.mp4',
        )
        with sqlite3.connect(db) as c:
            c.execute(
                """UPDATE indexing_queue SET attempts=?
                   WHERE canonical_key='dead.mp4'""",
                (_PARSE_ERROR_MAX_ATTEMPTS,),
            )
        st = get_queue_status(db)
        assert st['queue_depth'] == 3
        assert st['claimed_count'] == 1
        assert st['dead_letter_count'] == 1
        assert st['next_ready_at'] is not None


class TestClearQueue:
    def test_removes_everything(self, tmp_path):
        db = str(tmp_path / "q.db")
        _init_db(db)
        for i in range(5):
            enqueue_for_indexing(
                db, f'/mnt/teslacam/RecentClips/c{i}.mp4',
            )
        n = clear_queue(db)
        assert n == 5
        with sqlite3.connect(db) as c:
            assert c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0] == 0

    def test_clear_pending_preserves_claimed_rows(self, tmp_path):
        # /api/index/cancel must keep the in-flight file's claim row
        # intact so the worker can finish the file without its
        # owner-guarded complete failing on a vanished row.
        db = str(tmp_path / "q.db")
        _init_db(db)
        for i in range(3):
            enqueue_for_indexing(
                db, f'/mnt/teslacam/RecentClips/c{i}-front.mp4',
            )
        # Claim one — simulates the worker mid-file.
        claimed = claim_next_queue_item(db, worker_id='wk-1')
        assert claimed is not None

        n = clear_pending_queue(db)
        # Two pending unclaimed rows removed; the claimed row stays.
        assert n == 2

        with sqlite3.connect(db) as c:
            rows = c.execute(
                "SELECT canonical_key, claimed_by FROM indexing_queue"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == claimed['canonical_key']
        assert rows[0][1] == 'wk-1'

    def test_clear_all_removes_claimed_rows_too(self, tmp_path):
        # The advanced-rebuild path uses clear_all_queue (after pausing
        # the worker) to wipe everything.
        db = str(tmp_path / "q.db")
        _init_db(db)
        for i in range(3):
            enqueue_for_indexing(
                db, f'/mnt/teslacam/RecentClips/c{i}-front.mp4',
            )
        claim_next_queue_item(db, worker_id='wk-1')

        n = clear_all_queue(db)
        assert n == 3
        with sqlite3.connect(db) as c:
            assert c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0] == 0


class TestPurgeDeletedVideos:
    """Targeted-purge regression tests.

    The targeted purge flow runs from the file-watcher's delete
    callback whenever Tesla rotates a clip out of RecentClips. Older
    versions used ``LIKE '%basename%'`` to delete waypoints, which
    erased ArchivedClips geodata for clips that had a same-basename
    rotated copy in RecentClips. These tests pin the safe behavior:

      - skip purge entirely when a surviving on-disk copy exists
      - exact-match candidate relative paths instead of basename LIKE
    """

    def _seed(self, db, *, waypoint_video_path, indexed_abs_path,
              file_size=1024, file_mtime=100.0):
        from services.mapping_service import purge_deleted_videos  # noqa: F401
        with sqlite3.connect(db) as c:
            c.execute(
                "INSERT INTO trips (start_time, indexed_at) "
                "VALUES ('2025-01-01T00:00:00', '2025-01-01T00:00:00')"
            )
            trip_id = c.execute(
                "SELECT id FROM trips ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            c.execute(
                "INSERT INTO waypoints "
                "(trip_id, timestamp, lat, lon, video_path) "
                "VALUES (?, '2025-01-01T00:00:01', 37.0, -122.0, ?)",
                (trip_id, waypoint_video_path),
            )
            c.execute(
                "INSERT INTO indexed_files "
                "(file_path, file_size, file_mtime, indexed_at, "
                " waypoint_count, event_count) "
                "VALUES (?, ?, ?, '2025-01-01T00:00:00', 1, 0)",
                (indexed_abs_path, file_size, file_mtime),
            )
            c.commit()
        return trip_id

    def test_purge_skips_when_archive_copy_exists(self, tmp_path):
        # Reproduces BLOCKING bug: Tesla rotates RecentClips/foo-front
        # while ArchivedClips/foo-front (same basename) still exists.
        # The waypoint MUST survive — it's tied to the archived copy.
        from services.mapping_service import purge_deleted_videos
        db = str(tmp_path / "geo.db")
        _init_db(db)

        recent_path = str(tmp_path / "TeslaCam" / "RecentClips" /
                           "2025-01-01_00-foo-front.mp4")
        archive_path = str(tmp_path / "ArchivedClips" /
                            "2025-01-01_00-foo-front.mp4")

        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        # Surviving archive copy
        with open(archive_path, "wb") as f:
            f.write(b"hello")
        # No file at recent_path — Tesla deleted it. The watcher fires
        # purge_deleted_videos with the recent_path.

        # The waypoint reflects the archived copy (post-archive rewrite)
        self._seed(
            db,
            waypoint_video_path='ArchivedClips/'
                                  '2025-01-01_00-foo-front.mp4',
            indexed_abs_path=archive_path,
        )

        # Patch ARCHIVE_DIR so the surviving-copy check finds it.
        import config as _cfg
        old_dir = getattr(_cfg, 'ARCHIVE_DIR', None)
        old_en = getattr(_cfg, 'ARCHIVE_ENABLED', None)
        _cfg.ARCHIVE_DIR = str(tmp_path / "ArchivedClips")
        _cfg.ARCHIVE_ENABLED = True
        try:
            result = purge_deleted_videos(
                db, deleted_paths=[recent_path],
            )
        finally:
            if old_dir is not None:
                _cfg.ARCHIVE_DIR = old_dir
            if old_en is not None:
                _cfg.ARCHIVE_ENABLED = old_en

        # Nothing purged — surviving copy detected.
        assert result['purged_waypoints'] == 0
        assert result['purged_files'] == 0

        with sqlite3.connect(db) as c:
            wp_count = c.execute(
                "SELECT COUNT(*) FROM waypoints"
            ).fetchone()[0]
            file_count = c.execute(
                "SELECT COUNT(*) FROM indexed_files"
            ).fetchone()[0]
        assert wp_count == 1
        assert file_count == 1

    def test_purge_exact_matches_when_no_surviving_copy(self, tmp_path):
        # Counterpart: with no surviving copy on disk, the targeted
        # purge SHOULD remove the matching ``indexed_files`` row and
        # NULL out the waypoint's ``video_path`` (so the playback
        # link is severed) — but the waypoint row itself MUST survive.
        # The user's GPS history is independent of whether the dashcam
        # clip is still on disk.
        from services.mapping_service import purge_deleted_videos
        db = str(tmp_path / "geo.db")
        _init_db(db)

        recent_path = str(tmp_path / "TeslaCam" / "RecentClips" /
                           "2025-01-01_00-bar-front.mp4")
        # No file written anywhere — both Recent and Archived missing.

        trip_id = self._seed(
            db,
            waypoint_video_path='RecentClips/'
                                  '2025-01-01_00-bar-front.mp4',
            indexed_abs_path=recent_path,
        )

        import config as _cfg
        old_dir = getattr(_cfg, 'ARCHIVE_DIR', None)
        old_en = getattr(_cfg, 'ARCHIVE_ENABLED', None)
        _cfg.ARCHIVE_DIR = str(tmp_path / "ArchivedClips")  # nonexistent
        _cfg.ARCHIVE_ENABLED = True
        try:
            result = purge_deleted_videos(
                db, deleted_paths=[recent_path],
            )
        finally:
            if old_dir is not None:
                _cfg.ARCHIVE_DIR = old_dir
            if old_en is not None:
                _cfg.ARCHIVE_ENABLED = old_en

        assert result['purged_files'] == 1
        # Waypoint count reflects rows whose video_path was nulled.
        assert result['purged_waypoints'] == 1
        # Trips are NEVER deleted by reconciliation.

        with sqlite3.connect(db) as c:
            # indexed_files row gone (file truly missing).
            assert c.execute(
                "SELECT COUNT(*) FROM indexed_files"
            ).fetchone()[0] == 0
            # Waypoint preserved — GPS history outlives the video.
            assert c.execute(
                "SELECT COUNT(*) FROM waypoints"
            ).fetchone()[0] == 1
            # video_path nulled so the UI knows playback is unavailable.
            assert c.execute(
                "SELECT video_path FROM waypoints"
            ).fetchone()[0] is None
            # Trip survives — the user still drove that route.
            assert c.execute(
                "SELECT COUNT(*) FROM trips WHERE id = ?", (trip_id,),
            ).fetchone()[0] == 1

    def test_purge_does_not_substring_match_unrelated_basename(
        self, tmp_path,
    ):
        # A clip named "front.mp4" must not erase waypoints for
        # unrelated clips like "front-cam-extra.mp4". Older basename-
        # LIKE matching would have done so — the new candidate-path
        # exact match prevents it.
        from services.mapping_service import purge_deleted_videos
        db = str(tmp_path / "geo.db")
        _init_db(db)

        # Waypoint for an UNRELATED clip — substring of victim basename
        unrelated_path = str(tmp_path / "TeslaCam" / "RecentClips" /
                              "2025-01-01_00-extra-front.mp4")
        self._seed(
            db,
            waypoint_video_path='RecentClips/'
                                  '2025-01-01_00-extra-front.mp4',
            indexed_abs_path=unrelated_path,
        )

        # Purge a file with a DIFFERENT basename. The unrelated row
        # must not be touched.
        victim_path = str(tmp_path / "TeslaCam" / "RecentClips" /
                           "2025-01-01_00-front.mp4")
        result = purge_deleted_videos(
            db, deleted_paths=[victim_path],
        )

        assert result['purged_waypoints'] == 0
        with sqlite3.connect(db) as c:
            assert c.execute(
                "SELECT COUNT(*) FROM waypoints"
            ).fetchone()[0] == 1

    def test_purge_preserves_trip_when_all_videos_gone(self, tmp_path):
        """Regression test for the May 7 trip-loss incident.

        BUG: when stale-scan caught up to RecentClips files Tesla had
        rotated out before the archive subsystem copied them to SD, the
        cascade-delete logic removed the corresponding waypoints, then
        the trip itself when its waypoint count hit zero. Result: the
        user's drive history vanished from the map even though the GPS
        evidence was real.

        FIX: ``purge_deleted_videos`` now deletes only the orphan
        ``indexed_files`` row and NULLs ``waypoints.video_path``.
        Trips and their waypoints survive even when every video file
        for the trip is gone.
        """
        from services.mapping_service import purge_deleted_videos
        db = str(tmp_path / "geo.db")
        _init_db(db)

        # Three RecentClips videos all belonging to the same trip
        # (think: 1-min front-camera segments from a 3-min drive).
        recent_paths = [
            str(tmp_path / "TeslaCam" / "RecentClips" /
                f"2025-05-07_12-{m:02d}-front.mp4")
            for m in (57, 58, 59)
        ]
        with sqlite3.connect(db) as c:
            c.execute(
                "INSERT INTO trips "
                "(start_time, end_time, indexed_at, distance_km) "
                "VALUES ('2025-05-07T12:57:00', '2025-05-07T13:00:00', "
                "        '2025-05-07T13:00:00', 8.2)"
            )
            trip_id = c.execute(
                "SELECT id FROM trips ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            for i, rp in enumerate(recent_paths):
                rel = 'RecentClips/' + os.path.basename(rp)
                c.execute(
                    "INSERT INTO waypoints "
                    "(trip_id, timestamp, lat, lon, video_path) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (trip_id, f'2025-05-07T12:5{7+i}:30',
                     37.0 + i * 0.001, -122.0, rel),
                )
                c.execute(
                    "INSERT INTO indexed_files "
                    "(file_path, file_size, file_mtime, indexed_at, "
                    " waypoint_count, event_count) "
                    "VALUES (?, 1024, 100.0, "
                    "        '2025-05-07T13:00:00', 1, 0)",
                    (rp,),
                )
            c.commit()

        # All RecentClips files vanished (Tesla rotated them out before
        # archive copied them to SD). No surviving copy on disk.
        import config as _cfg
        old_dir = getattr(_cfg, 'ARCHIVE_DIR', None)
        old_en = getattr(_cfg, 'ARCHIVE_ENABLED', None)
        _cfg.ARCHIVE_DIR = str(tmp_path / "ArchivedClips")  # nonexistent
        _cfg.ARCHIVE_ENABLED = True
        try:
            result = purge_deleted_videos(db, deleted_paths=recent_paths)
        finally:
            if old_dir is not None:
                _cfg.ARCHIVE_DIR = old_dir
            if old_en is not None:
                _cfg.ARCHIVE_ENABLED = old_en

        # All three indexed_files rows purged.
        assert result['purged_files'] == 3
        # All three waypoints' video_path nulled.
        assert result['purged_waypoints'] == 3

        with sqlite3.connect(db) as c:
            # Trip survives intact.
            row = c.execute(
                "SELECT id, distance_km FROM trips WHERE id = ?",
                (trip_id,),
            ).fetchone()
            assert row is not None
            assert row[1] == 8.2
            # All three waypoints survive.
            wps = c.execute(
                "SELECT COUNT(*) FROM waypoints WHERE trip_id = ?",
                (trip_id,),
            ).fetchone()[0]
            assert wps == 3
            # video_path nulled on every one — UI knows playback is gone.
            null_count = c.execute(
                "SELECT COUNT(*) FROM waypoints "
                "WHERE trip_id = ? AND video_path IS NULL",
                (trip_id,),
            ).fetchone()[0]
            assert null_count == 3
            # No indexed_files rows left for the missing clips.
            assert c.execute(
                "SELECT COUNT(*) FROM indexed_files"
            ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Phase 3: Boot catch-up scan
# ---------------------------------------------------------------------------


class TestBootCatchupScan:
    """Phase 2b (issue #76): boot_catchup_scan now walks ONLY
    ``ARCHIVE_DIR`` (the SD-card ArchivedClips), never the RO USB
    mount. The USB-side catch-up is handled by the
    ``archive_producer`` thread, which enqueues into ``archive_queue``;
    the worker then copies into ArchivedClips, where THIS catch-up
    finds them on the next gadget_web start.

    The legacy test signature ``boot_catchup_scan(db, tc)`` still
    accepts the ``tc`` argument for back-compat, but it's now ignored.
    All these tests populate ARCHIVE_DIR via monkeypatch instead.
    """
    def _make_archive(self, root, files):
        """Create a fake ArchivedClips tree with the given relative paths."""
        for rel in files:
            full = root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(b'')
        return str(root)

    def _make_teslacam(self, root, files):
        """Compatibility helper kept so the dedup test below still works
        (the test populates BOTH ArchivedClips and a legacy TeslaCam
        tree, then verifies that only the ArchivedClips side gets
        enqueued)."""
        for rel in files:
            full = root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(b'')
        return str(root)

    @pytest.fixture(autouse=True)
    def _patch_archive_dir(self, tmp_path, monkeypatch):
        """Point ARCHIVE_DIR at a per-test tmpdir so the scanner sees
        a clean slate. This must run BEFORE each test populates files."""
        archive_root = tmp_path / "ArchivedClips"
        archive_root.mkdir()
        import config as _cfg
        monkeypatch.setattr(_cfg, 'ARCHIVE_DIR', str(archive_root))
        monkeypatch.setattr(_cfg, 'ARCHIVE_ENABLED', True)
        self._archive_root = archive_root

    def test_no_files_returns_zero_counts(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        result = boot_catchup_scan(db, '')
        assert result == {
            'scanned': 0, 'already_indexed': 0, 'enqueued': 0,
            'skipped_by_watermark': 0,
        }

    def test_enqueues_orphan_clips(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        # Populate the ArchivedClips tree with three distinct
        # canonical_keys: a flat RecentClips file, a SavedClips event
        # folder file, and a SentryClips event folder file.
        self._make_archive(self._archive_root, [
            'RecentClips/2025-11-08_08-15-44-front.mp4',
            'SavedClips/2025-11-08_evt/2025-11-08_08-15-44-front.mp4',
            'SentryClips/2025-11-08_evt2/2025-11-08_08-20-00-front.mp4',
        ])
        result = boot_catchup_scan(db, '')
        assert result['scanned'] >= 3
        assert result['enqueued'] == 3
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        assert count == 3

    def test_skips_already_indexed_clips(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        self._make_archive(self._archive_root, [
            'RecentClips/2025-11-08_08-15-44-front.mp4',
        ])
        # Pre-populate indexed_files with the canonical_key matching
        # the ArchivedClips path. canonical_key for a RecentClips file
        # is just the basename (so any pre-existing row with the same
        # basename counts as "already indexed").
        full_path = os.path.join(
            str(self._archive_root),
            'RecentClips', '2025-11-08_08-15-44-front.mp4',
        )
        with sqlite3.connect(db) as c:
            c.execute(
                """INSERT INTO indexed_files
                   (file_path, file_size, file_mtime, indexed_at,
                    waypoint_count, event_count)
                   VALUES (?, 0, 0, '2025-01-01', 5, 0)""",
                (full_path,),
            )
        result = boot_catchup_scan(db, '')
        assert result['scanned'] >= 1
        assert result['already_indexed'] >= 1
        assert result['enqueued'] == 0

    def test_skips_already_queued_clips(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        self._make_archive(self._archive_root, [
            'RecentClips/2025-11-08_08-15-44-front.mp4',
        ])
        full_path = os.path.join(
            str(self._archive_root),
            'RecentClips', '2025-11-08_08-15-44-front.mp4',
        )
        # Pre-queue (e.g. from a watcher event during the scan).
        enqueue_for_indexing(db, full_path)
        # Catch-up must not double-enqueue.
        result = boot_catchup_scan(db, '')
        assert result['enqueued'] == 0
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        assert count == 1

    def test_recent_and_archived_dedup_by_canonical_key(self, tmp_path):
        # Two files with the same basename but DIFFERENT canonical_key
        # (one flat under RecentClips, one nested under SavedClips/evt)
        # should both enqueue — they're distinct canonical keys.
        db = str(tmp_path / "g.db")
        _init_db(db)
        self._make_archive(self._archive_root, [
            'RecentClips/dup-front.mp4',
            'SavedClips/evt/dup-front.mp4',
        ])
        result = boot_catchup_scan(db, '')
        # Two distinct canonical keys: bare 'dup-front.mp4' (Recent)
        # and 'SavedClips/evt/dup-front.mp4' (Saved).
        assert result['enqueued'] == 2

    def test_legacy_teslacam_argument_is_ignored(self, tmp_path):
        """Phase 2b: even if the caller passes a TeslaCam path with
        clips on it, the scanner walks ArchivedClips ONLY. This is the
        whole point of the redesign — the indexer must never touch
        the RO USB mount."""
        db = str(tmp_path / "g.db")
        _init_db(db)
        # Populate a legacy TeslaCam tree with clips that should NOT
        # be enqueued.
        legacy_tc = self._make_teslacam(tmp_path / "TeslaCam", [
            'RecentClips/should-not-enqueue-front.mp4',
            'SavedClips/evt/should-not-enqueue-front.mp4',
        ])
        # ArchivedClips is empty — scanner must report zero.
        result = boot_catchup_scan(db, legacy_tc)
        assert result == {
            'scanned': 0, 'already_indexed': 0, 'enqueued': 0,
            'skipped_by_watermark': 0,
        }
        with sqlite3.connect(db) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM indexing_queue"
            ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Issue #184 Wave 2 — Phase E: boot catch-up watermark
# ---------------------------------------------------------------------------


class TestBootCatchupWatermark:
    """The boot catch-up scan persists a high-water mark of the highest
    file mtime it has ever seen and uses it on subsequent boots to
    skip files older than the watermark — turning the steady-state
    boot scan from O(N) into O(new files)."""

    def _make_archive(self, root, files):
        for rel in files:
            full = root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(b'')
        return str(root)

    @pytest.fixture(autouse=True)
    def _patch_archive_dir(self, tmp_path, monkeypatch):
        archive_root = tmp_path / "ArchivedClips"
        archive_root.mkdir()
        import config as _cfg
        monkeypatch.setattr(_cfg, 'ARCHIVE_DIR', str(archive_root))
        monkeypatch.setattr(_cfg, 'ARCHIVE_ENABLED', True)
        self._archive_root = archive_root

    def test_first_run_writes_watermark(self, tmp_path):
        from services.mapping_service import (
            _kv_get, _BOOT_CATCHUP_WATERMARK_KEY,
        )
        db = str(tmp_path / "g.db")
        _init_db(db)
        self._make_archive(self._archive_root, [
            "RecentClips/2026-05-11_09-00-00-front.mp4",
        ])
        result = boot_catchup_scan(db, '')
        assert result['scanned'] == 1
        assert result['enqueued'] == 1
        assert result['skipped_by_watermark'] == 0
        # Watermark must be set to the file's mtime.
        with sqlite3.connect(db) as conn:
            stored = _kv_get(conn, _BOOT_CATCHUP_WATERMARK_KEY)
        assert stored is not None
        assert float(stored) > 0.0

    def test_second_run_skips_unchanged_files(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        self._make_archive(self._archive_root, [
            "RecentClips/2026-05-11_09-00-00-front.mp4",
            "RecentClips/2026-05-11_09-01-00-front.mp4",
        ])
        # First run — full scan.
        first = boot_catchup_scan(db, '')
        assert first['scanned'] == 2
        # Second run — watermark covers both files.
        second = boot_catchup_scan(db, '')
        assert second['scanned'] == 2
        assert second['skipped_by_watermark'] == 2
        assert second['enqueued'] == 0
        assert second['already_indexed'] == 0

    def test_new_file_after_watermark_is_processed(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        self._make_archive(self._archive_root, [
            "RecentClips/2026-05-11_09-00-00-front.mp4",
        ])
        boot_catchup_scan(db, '')
        # Add a new file with a strictly newer mtime.
        new_path = (
            self._archive_root / "RecentClips" /
            "2026-05-11_09-05-00-front.mp4"
        )
        new_path.write_bytes(b'')
        # Bump its mtime explicitly so the test isn't sensitive to
        # filesystem timestamp granularity (FAT32 has 2-s resolution).
        future = time.time() + 60
        os.utime(str(new_path), (future, future))
        result = boot_catchup_scan(db, '')
        assert result['scanned'] == 2
        assert result['skipped_by_watermark'] == 1
        assert result['enqueued'] == 1


# ---------------------------------------------------------------------------
# Phase 5: Daily stale scan
# ---------------------------------------------------------------------------


class TestDailyStaleScan:
    def test_start_returns_true_first_time_false_second(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        try:
            assert start_daily_stale_scan(db, lambda: None) is True
            # Idempotent — second call should not start another thread.
            assert start_daily_stale_scan(db, lambda: None) is False
        finally:
            stop_daily_stale_scan(timeout=2.0)

    def test_stop_terminates_thread(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        start_daily_stale_scan(db, lambda: None)
        # Cleanly stop within a reasonable time.
        assert stop_daily_stale_scan(timeout=5.0) is True

    def test_stop_when_not_running_is_safe(self, tmp_path):
        # Should be idempotent even when nothing is running.
        assert stop_daily_stale_scan(timeout=1.0) is True
        assert stop_daily_stale_scan(timeout=1.0) is True

    def test_initial_delay_within_5_to_10_minutes(self):
        # Issue #75: stale scan must fire within ~10 minutes of boot
        # so orphans left behind by the previous boot get cleaned up
        # before the user opens the map page.
        for _ in range(50):
            d = _initial_stale_scan_delay()
            assert 5 * 60 <= d <= 10 * 60, (
                f"Expected delay in [300, 600], got {d}"
            )


# ---------------------------------------------------------------------------
# Phase 5b: Out-of-cycle stale-scan trigger (issue #75)
# ---------------------------------------------------------------------------


class TestStaleScanTrigger:
    """trigger_stale_scan_now() lets services nudge the stale scan
    after high-signal events (archive cycle, map page load) without
    waiting for the daily safety net. Debounced so concurrent
    triggers from different services collapse into one scan.
    """

    def setup_method(self):
        _reset_stale_scan_state_for_tests()

    def teardown_method(self):
        _reset_stale_scan_state_for_tests()

    def test_trigger_fires_when_no_recent_run(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()
        result = trigger_stale_scan_now(db, str(tc), source='test')
        assert result['status'] == 'fired'

    def test_trigger_debounces_within_window(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()
        first = trigger_stale_scan_now(
            db, str(tc), source='archive', debounce_seconds=60.0,
        )
        assert first['status'] == 'fired'
        # Wait for the spawned thread so the timestamp is settled.
        # The scan against an empty TeslaCam is essentially instant.
        time.sleep(0.2)
        second = trigger_stale_scan_now(
            db, str(tc), source='map_load', debounce_seconds=60.0,
        )
        assert second['status'] == 'debounced'
        assert 'last_run_age_seconds' in second
        assert second['last_run_age_seconds'] >= 0.0

    def test_trigger_fires_after_debounce_expires(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()
        first = trigger_stale_scan_now(
            db, str(tc), source='archive', debounce_seconds=60.0,
        )
        assert first['status'] == 'fired'
        time.sleep(0.2)
        # Use a tiny debounce window — should fire again.
        third = trigger_stale_scan_now(
            db, str(tc), source='map_load', debounce_seconds=0.0,
        )
        assert third['status'] == 'fired'

    def test_trigger_accepts_callable_provider(self, tmp_path):
        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()
        calls = []

        def _provider():
            calls.append(1)
            return str(tc)

        result = trigger_stale_scan_now(db, _provider, source='test')
        assert result['status'] == 'fired'
        # Wait for the spawned scan thread to consume the provider.
        time.sleep(0.3)
        assert len(calls) == 1

    def test_trigger_with_missing_teslacam_returns_fired(self, tmp_path):
        # Provider returns None — scan is fired but exits early
        # without raising. Status is still 'fired' (the trigger
        # contract is "we attempted a scan", not "the scan found a
        # path").
        db = str(tmp_path / "g.db")
        _init_db(db)
        result = trigger_stale_scan_now(
            db, lambda: None, source='test',
        )
        assert result['status'] == 'fired'

    def test_blocking_helper_purges_orphan_indexed_files_row(
        self, tmp_path,
    ):
        # Synthetic regression test: insert an indexed_files row
        # pointing to a path that doesn't exist, run the blocking
        # helper, verify the row is gone. This is the exact scenario
        # the McDonald's-trip incident (issue #75) created live.
        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()
        recent_clips = tc / "RecentClips"
        recent_clips.mkdir()
        ghost_path = str(
            recent_clips / "2026-05-07_11-36-00-front.mp4"
        )
        with sqlite3.connect(db) as c:
            c.execute(
                """INSERT INTO indexed_files
                   (file_path, file_size, file_mtime, indexed_at,
                    waypoint_count, event_count)
                   VALUES (?, 12345, 1700000000, '2026-05-07', 22, 4)""",
                (ghost_path,),
            )
        # Pre-condition: the row exists.
        with sqlite3.connect(db) as c:
            n = c.execute(
                "SELECT COUNT(*) FROM indexed_files",
            ).fetchone()[0]
        assert n == 1

        result = _run_stale_scan_blocking(db, str(tc), source='test')
        assert result is not None
        assert result.get('purged_files', 0) >= 1

        with sqlite3.connect(db) as c:
            n = c.execute(
                "SELECT COUNT(*) FROM indexed_files",
            ).fetchone()[0]
        assert n == 0

    def test_blocking_helper_updates_debounce_timestamp(self, tmp_path):
        # Both the scheduled loop and out-of-cycle triggers go
        # through _run_stale_scan_blocking, so a scheduled fire
        # must also debounce subsequent triggers (otherwise a
        # trigger that arrives moments after the loop wakes would
        # double the work).
        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()
        _run_stale_scan_blocking(db, str(tc), source='scheduled')
        result = trigger_stale_scan_now(
            db, str(tc), source='archive', debounce_seconds=60.0,
        )
        assert result['status'] == 'debounced'

    def test_blocking_helper_handles_missing_teslacam_gracefully(
        self, tmp_path,
    ):
        db = str(tmp_path / "g.db")
        _init_db(db)
        # Path doesn't exist — helper should return None, not raise.
        result = _run_stale_scan_blocking(
            db, '/nonexistent/path/abc123', source='test',
        )
        assert result is None

    def test_blocking_helper_purges_orphaned_dead_letters(self, tmp_path):
        """Issue #110 — _run_stale_scan_blocking also removes
        ``indexing_queue`` dead-letter rows whose source file no
        longer exists (typically because retention deleted a
        truncated archive copy)."""
        from services.indexing_queue_service import (
            _PARSE_ERROR_MAX_ATTEMPTS,
            enqueue_for_indexing,
            get_queue_status,
        )

        db = str(tmp_path / "g.db")
        _init_db(db)
        tc = tmp_path / "TeslaCam"
        tc.mkdir()

        # Create a "front" clip, enqueue it for indexing, force it
        # to dead-letter, then delete the file (simulating retention).
        clip = tmp_path / "2026-05-11_08-41-58-front.mp4"
        clip.write_bytes(b"fake")
        assert enqueue_for_indexing(db, str(clip)) is True
        with sqlite3.connect(db) as c:
            c.execute(
                "UPDATE indexing_queue SET attempts = ?, "
                "last_error = 'No mdat box found' "
                "WHERE file_path = ?",
                (_PARSE_ERROR_MAX_ATTEMPTS, str(clip)),
            )
            c.commit()
        assert get_queue_status(db)['dead_letter_count'] == 1
        clip.unlink()

        result = _run_stale_scan_blocking(db, str(tc), source='test')
        assert result is not None
        assert result.get('purged_dead_letters') == 1

        # Row should be gone after the sweep.
        assert get_queue_status(db)['dead_letter_count'] == 0



class TestGapDetectionHelpers:
    """Pure-function tests for the gap-detection helpers used by the
    map polyline renderer to avoid drawing diagonal straight lines
    across actual GPS dropouts (parking breaks, missing clips, SEI
    clock skew).
    """

