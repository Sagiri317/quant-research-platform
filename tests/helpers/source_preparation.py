"""Exercise actual source preparation and inspect its persisted source evidence."""

from pathlib import Path
from tempfile import TemporaryDirectory

from quant_research_platform.storage.datasets.reading import open_dataset_session


def prepared_source_evidence(owner, raw_root, data_id):
    with TemporaryDirectory(prefix="qrp-source-test-") as temporary:
        output = Path(temporary) / "prepared"
        result = owner.materialize_interim_data(
            raw_root=raw_root,
            output_roots={data_id: output},
            data_ids=(data_id,),
        )
        if data_id in result.skipped:
            assert not output.exists()
            return None
        return open_dataset_session(output).manifest["source_evidence"]["content_hash"]
