# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Describe the observation and action spaces with unienv space metadata (requires `unienv>=0.0.1b16`): per-term units / names / frames for proprioceptive, command and sensor terms, `terms` breakdowns and per-element `names` for concatenated groups, image / depth metadata for camera terms (`CHW`, normalized, source, fps) and `control` / `mode` / joint `names` / `scale` / `offset` for action terms.

## 0.0.1a1 (2026-04-11)

- Initial release