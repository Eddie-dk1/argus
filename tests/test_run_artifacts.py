import json
from pathlib import Path
import tempfile
import unittest
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from experiment_tracking import finalize_run_artifacts, file_sha256


class RunArtifactTests(unittest.TestCase):
    def test_independent_bundles_reload_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state.json'
            state.write_text('{}')
            x = [[0], [1], [2], [3]]
            model = RandomForestClassifier(n_estimators=2, random_state=42).fit(x, [0, 0, 1, 1])
            for name in ('a', 'b'):
                run = root / name
                run.mkdir()
                finalize_run_artifacts(run, predictions=pd.DataFrame({'score': model.predict_proba(x)[:, 1]}),
                    model_writer=lambda p: joblib.dump(model, p), model_filename='model.joblib',
                    inference_contract={'threshold': .5}, feature_state_path=state)
            a = root / 'a'
            manifest = json.loads((a / 'complete.json').read_text())
            for path, metadata in manifest['artifacts'].items():
                self.assertEqual(file_sha256(a / path), metadata['sha256'])
            restored = joblib.load(a / 'inference/model.joblib')
            self.assertEqual(restored.predict(x).tolist(), model.predict(x).tolist())
            with self.assertRaises(FileExistsError):
                finalize_run_artifacts(a, predictions=pd.DataFrame(), model_writer=lambda p: None,
                    model_filename='model.joblib', inference_contract={}, feature_state_path=state)

    def test_failed_serialization_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state.json'
            state.write_text('{}')
            def fail(path):
                raise RuntimeError('serialization failed')
            with self.assertRaises(RuntimeError):
                finalize_run_artifacts(root, predictions=pd.DataFrame(), model_writer=fail,
                    model_filename='model.joblib', inference_contract={}, feature_state_path=state)
            self.assertFalse((root / 'complete.json').exists())
