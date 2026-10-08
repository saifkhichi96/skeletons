Playground Helpers
==================

Package helpers used by the optional Qt playground. Public symbols from this
module are re-exported and autodocumented in the top-level package API.

The package-backed seams currently cover procedural animation presets,
range-of-motion payload parsing, synthetic fitting-lab data generation, and
stable fitted-sequence export. Qt-specific worker orchestration lives in
``skeletons.playground.qt_workers`` and is imported only by the
optional launcher.
