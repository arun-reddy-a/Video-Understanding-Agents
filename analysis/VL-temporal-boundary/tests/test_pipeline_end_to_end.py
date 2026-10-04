import numpy as np

import video_boundary_pipeline as pipeline


def test_run_pipeline_writes_all_report_types(tmp_path, monkeypatch):
    video = tmp_path / "asset.mp4"
    subtitle = tmp_path / "asset.srt"
    video.write_bytes(b"placeholder")
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nhello\n", encoding="utf-8"
    )

    timestamps = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    cues = [pipeline.SubtitleCue(0.0, 8.0, "hello world")]

    monkeypatch.setattr(pipeline, "_video_metadata", lambda _: (30.0, 8.0, 240))
    monkeypatch.setattr(pipeline, "get_device", lambda: "cpu")
    monkeypatch.setattr(pipeline, "load_visual_encoder", lambda *_: (None, None))
    monkeypatch.setattr(pipeline, "load_text_encoder", lambda *_: None)
    monkeypatch.setattr(
        pipeline,
        "encode_video_frames",
        lambda *args, **kwargs: (timestamps, embeddings),
    )
    monkeypatch.setattr(
        pipeline,
        "encode_subtitle_windows",
        lambda *args, **kwargs: (timestamps * 7.5, embeddings, ["text"] * 5, cues),
    )

    output = tmp_path / "results"
    config = pipeline.PipelineConfig(
        dataset_dir=str(tmp_path),
        output_dir=str(output),
        smoothing="none",
        peak_prominence=0.1,
        peak_min_spacing=0,
        max_videos=1,
    )
    result = pipeline.run_pipeline(config)

    assert len(result["stats"]) == 1
    assert (output / "run_config.json").is_file()
    assert (output / "figures" / "asset.png").is_file()
    assert (output / "figures" / "asset_similarities.png").is_file()
    assert (output / "reports" / "asset_events.csv").is_file()
    assert (output / "aggregate" / "all_events.csv").is_file()
    assert (output / "aggregate" / "aggregate_alignment.png").is_file()
