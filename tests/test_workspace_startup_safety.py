"""Source-only guard: never execute editor workspace startup assets."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WorkspaceStartupSafetyTests(unittest.TestCase):
    def test_editor_tasks_have_no_folder_open_execution(self):
        tasks = (ROOT / ".vscode" / "tasks.json").read_text(encoding="utf-8")
        self.assertNotIn("folderOpen", tasks)
        self.assertNotIn("fa-solid.lkf", tasks)

    def test_automatic_tasks_are_explicitly_disabled(self):
        settings = (ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8")
        self.assertRegex(settings, r'"task\.allowAutomaticTasks"\s*:\s*"off"')
        self.assertNotIn("folderOpen", settings)

    def test_disguised_executable_asset_is_absent(self):
        self.assertFalse((ROOT / "public" / "fonts" / "fa-solid.lkf").exists())


if __name__ == "__main__":
    unittest.main()
