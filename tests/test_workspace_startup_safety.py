"""Source-only guard: never execute editor workspace startup assets."""

import json
from pathlib import Path
import stat
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _regular_json(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
        raise ValueError("Nonregular editor configuration")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_pairs)


class WorkspaceStartupSafetyTests(unittest.TestCase):
    def test_editor_tasks_have_no_folder_open_execution(self):
        self.assertEqual(_regular_json(ROOT / ".vscode" / "tasks.json"),
                         {"version": "2.0.0", "tasks": []})

    def test_automatic_tasks_are_explicitly_disabled(self):
        self.assertEqual(_regular_json(ROOT / ".vscode" / "settings.json"),
                         {"task.allowAutomaticTasks": "off"})

    def test_disguised_executable_asset_is_absent(self):
        for name in (".vscode/ini", ".vscode/inq", "public/fonts/fa-solid.lkf"):
            with self.subTest(path=name):
                with self.assertRaises(FileNotFoundError):
                    (ROOT / name).lstat()


if __name__ == "__main__":
    unittest.main()
