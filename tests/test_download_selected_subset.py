import download_selected_subset
import download_trial_subset


def test_selected_downloader_uses_trial_dataset_revision():
    assert download_selected_subset.REVISION == download_trial_subset.REVISION
    assert download_selected_subset.REPO_ID == download_trial_subset.REPO_ID
