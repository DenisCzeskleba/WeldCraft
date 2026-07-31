"""Special batch preset that renders every ordinary still-diagram preset."""

PRESET_NAME = "Render All Diagram Presets"
BATCH_RENDER_ALL_PRESETS = True

# Canonical outward-facing source for regenerating the published examples.
# It is deliberately independent of the single-diagram input selected in c3.
BATCH_INPUT_H5_FILENAME = "Examples/published_examples_source.h5"
BATCH_SNAPSHOT_INDEX = -1
BATCH_OUTPUT_FOLDER = "Examples"
BATCH_SAVE_DPI = 300
# Internal bookkeeping stored beside this preset, not in the public Examples folder.
BATCH_MANIFEST_FILENAME = "published_examples_manifest.json"

# This is the stable public numbering order documented in
# c3_Brown_Make_Diagram.py. Add every new preset here at the end. Do not reorder
# existing entries unless the published filenames are intentionally being changed.
# Unlisted preset files are still discovered and appended alphabetically as a
# safety net, but they should be added here before publication.
BATCH_PRESET_ORDER = (
    "default",
    "two_regions_w_solubility",
    "simple_1_region_source_sink",
    "simple_concentration_profile",
    "depletion_heatmap",
    "printer_friendly",
    "area_summary",
    "chapter_2_3_brown_overview",
    "area_summary_transient",
)
