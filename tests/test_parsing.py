import duckdb
import pandas as pd
import pytest

from logad.adapters.hdfs import ingest
from logad.parsing import parse_dataset, sql_mask_expression
from logad.parsing.accuracy import grouping_accuracy
from logad.parsing.drain import LogParser, Mask

# Same masks as configs/hdfs.yaml.
MASKS = [
    Mask("BLK", r"blk_-?\d+( +blk_-?\d+)*"),
    Mask("IP", r"(\d+\.){3}\d+(:\d+)?"),
    Mask("NUM", r"\d+"),
]
MESSAGES = [
    "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "BLOCK* ask 10.250.17.177:50010 to delete  blk_-857 blk_912 blk_-43",
    "PacketResponder 1 for block blk_38865049064139660 terminating",
    "Verification succeeded for blk_7",
    "no numbers here",
]


def make_parser():
    return LogParser(sim_threshold=0.4, depth=4, masks=MASKS)


def test_mask_replaces_known_variables_in_order():
    parser = make_parser()
    assert parser.mask(MESSAGES[0]) == "Receiving block <BLK> src: /<IP> dest: /<IP>"
    assert parser.mask(MESSAGES[1]) == "BLOCK* ask <IP> to delete  <BLK>"  # list → one <BLK>
    assert parser.mask(MESSAGES[2]) == "PacketResponder <NUM> for block <BLK> terminating"


def test_duckdb_masking_matches_python_masking():
    # DuckDB masks the bulk data, Python masks single new messages: they must agree.
    parser = make_parser()
    df = pd.DataFrame({"message": MESSAGES})  # noqa: F841  (DuckDB reads it by name)
    sql = f"SELECT {sql_mask_expression('message', MASKS)} AS masked FROM df"
    assert [row[0] for row in duckdb.sql(sql).fetchall()] == [parser.mask(m) for m in MESSAGES]


def test_sql_mask_expression_escapes_quotes():
    expression = sql_mask_expression("message", [Mask("Q", "it's")])
    assert "'it''s'" in expression


def test_fit_groups_similar_messages_and_separates_different_ones():
    parser = make_parser()
    masked = [
        parser.mask(m)
        for m in [
            "Deleting block blk_1 file /mnt/disk_a/blk_1",
            "Deleting block blk_2 file /mnt/disk_b/blk_2",
            "Verification succeeded for blk_3",
        ]
    ]
    ids = parser.fit(masked)
    assert ids[0] == ids[1] != ids[2]
    assert parser.templates[ids[0]] == "Deleting block <BLK> file <*>"


def test_match_known_and_unknown_messages():
    parser = make_parser()
    parser.fit([parser.mask("Verification succeeded for blk_3")])
    assert parser.match("Verification succeeded for blk_-999") == "E1"
    assert parser.match("something completely different") is None


def test_save_and_load_roundtrip(tmp_path):
    parser = make_parser()
    parser.fit([parser.mask(m) for m in MESSAGES])
    path = tmp_path / "parser.json.zlib"
    parser.save(path)

    loaded = LogParser.load(path)
    assert loaded.templates == parser.templates
    assert loaded.masks == parser.masks
    assert loaded.match(MESSAGES[3]) == parser.match(MESSAGES[3])


def test_from_config_reads_parsing_section():
    config = {
        "parsing": {"sim_threshold": 0.5, "depth": 5, "masks": [{"name": "N", "pattern": r"\d+"}]}
    }
    parser = LogParser.from_config(config)
    assert (parser.sim_threshold, parser.depth) == (0.5, 5)
    assert parser.mask("a 12 b") == "a <N> b"


@pytest.mark.parametrize(
    ("true", "pred", "expected"),
    [
        (["A", "A", "B"], ["x", "x", "y"], 1.0),  # names differ, grouping identical
        (["A", "A", "B"], ["x", "y", "z"], 1 / 3),  # A split in two → both A lines wrong
        (["A", "A", "B"], ["x", "x", "x"], 0.0),  # A and B merged → everything wrong
        (["A", "B", "B", "C"], ["x", "y", "y", "y"], 0.25),  # B merged with C → only A right
    ],
)
def test_grouping_accuracy(true, pred, expected):
    assert grouping_accuracy(true, pred) == pytest.approx(expected)


def test_grouping_accuracy_rejects_bad_input():
    with pytest.raises(ValueError):
        grouping_accuracy(["A"], ["x", "y"])
    with pytest.raises(ValueError):
        grouping_accuracy([], [])


HDFS_LINES = [
    "081109 203615 148 INFO dfs.DataNode: PacketResponder 1 for block blk_1 terminating",
    "081109 203615 149 INFO dfs.DataNode: PacketResponder 2 for block blk_2 terminating",
    "081109 203616 19 INFO dfs.FSNamesystem: BLOCK* ask 10.0.0.1:50010 to delete  blk_1 blk_2",
    "081109 203617 20 INFO dfs.DataBlockScanner: Verification succeeded for blk_2",
]


def make_dataset(tmp_path, truth_ids=None):
    raw, artifacts = tmp_path / "raw", tmp_path / "artifacts"
    raw.mkdir()
    (raw / "test.log").write_text("\n".join(HDFS_LINES) + "\n", encoding="utf-8")
    ingest(raw / "test.log", artifacts / "logs.parquet")
    config = {
        "dataset": "hdfs",
        "paths": {
            "raw_dir": str(raw),
            "artifacts_dir": str(artifacts),
            "reports_dir": str(tmp_path / "reports"),
        },
        "data": {},
        "parsing": {"masks": [{"name": m.name, "pattern": m.pattern} for m in MASKS]},
    }
    if truth_ids:
        pd.DataFrame({"LineId": range(1, len(truth_ids) + 1), "EventId": truth_ids}).to_csv(
            raw / "truth.csv", index=False
        )
        config["data"]["truth_file"] = "truth.csv"
    return config, artifacts


def test_parse_dataset_end_to_end(tmp_path):
    config, artifacts = make_dataset(tmp_path, truth_ids=["T1", "T1", "T2", "T3"])
    stats = parse_dataset(config)

    assert stats.lines == 4
    assert stats.templates == 3
    assert stats.grouping_accuracy == 1.0
    assert stats.true_templates == 3

    events = pd.read_parquet(artifacts / "events.parquet")
    assert list(events.columns) == ["line_id", "timestamp", "group_id", "event_id", "label"]
    assert len(events) == 5  # the "delete blk_1 blk_2" line belongs to two blocks
    assert (
        events.loc[events["line_id"] == 0, "event_id"].item()
        == events.loc[events["line_id"] == 1, "event_id"].item()
    )  # both PacketResponder lines share one template

    templates = pd.read_csv(artifacts / "templates.csv")
    assert templates["lines"].sum() == 4
    assert (artifacts / "drain_parser.json.zlib").is_file()
    assert "PacketResponder <NUM> for block <BLK> terminating" in (
        tmp_path / "reports" / "templates.md"
    ).read_text(encoding="utf-8")


def test_parse_dataset_without_ingest_explains_what_to_run(tmp_path):
    config = {"dataset": "hdfs", "paths": {"artifacts_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError, match="logad ingest"):
        parse_dataset(config)


def test_template_report_text_is_safe_to_publish():
    from logad.parsing import for_display

    # the NUL bytes become the visible text \x00 (a backslash, then x00)
    assert for_display("Part Number: <*>\x00\x00 end") == r"Part Number: <*>\x00\x00 end"
    assert for_display("chdir(/home/alice/run) failed") == "chdir(/home/<user>/run) failed"
    assert for_display("normal text") == "normal text"
