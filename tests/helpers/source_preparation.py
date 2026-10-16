"""Exercise actual source preparation and inspect its persisted source evidence."""

from pathlib import Path
from tempfile import TemporaryDirectory

from quant_research_platform.storage.manifests import read_dataset_manifest_evidence


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
        return read_dataset_manifest_evidence(output)["source_evidence"]["content_hash"]
