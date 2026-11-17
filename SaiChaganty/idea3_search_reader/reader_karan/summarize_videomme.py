"""Accuracy per condition from read_videomme.py's results, with differences against uniform

Every number is over the questions answered in all conditions, so a partial run compares like with
like. Intervals are 95% bootstrap intervals that resample videos, since a video's 3 questions are
not independent.

Usage: python summarize_videomme.py --results-path FILE
"""
import json
import tyro
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PAPER_LONG_ACCURACY = {
    'uniform':    50.6,
    'evidential': 55.0,
}


@dataclass
class Config:
    results_path:  Path
    baseline:      str = 'uniform'
    num_bootstrap: int = 10_000
    seed:          int = 0


def video_bootstrap(values_by_video, num_bootstrap, seed):
    """95% interval of the per-question mean when whole videos are resampled"""
    random = np.random.default_rng(seed)

    sums = np.array([values.sum() for values in values_by_video])
    counts = np.array([len(values) for values in values_by_video])

    picks = random.integers(0, len(sums), size=(num_bootstrap, len(sums)))
    resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)

    return np.percentile(resampled, [2.5, 97.5])


def main(config: Config):
    correct = defaultdict(dict)  # condition -> question id -> 0 / 1
    video_of = {}
    unanswered = defaultdict(set)  # condition -> question ids with no letter in the reply

    for line in open(config.results_path):
        row = json.loads(line)

        correct[row['condition']][row['qid']] = float(row['correct'])
        video_of[row['qid']] = row['video']
        if row['pred'] == '':
            unanswered[row['condition']].add(row['qid'])

    if config.baseline not in correct:
        raise ValueError(f'no rows for the baseline condition {config.baseline!r}; conditions read: {sorted(correct)}')

    shared = sorted(set.intersection(*(set(by_question) for by_question in correct.values())))
    videos = sorted({video_of[question_id] for question_id in shared})

    def by_video(values_of):
        return [
            np.array([values_of(question_id) for question_id in shared if video_of[question_id] == video_id])
            for video_id
            in videos
        ]

    print(f'{len(shared)} questions on {len(videos)} videos answered in every condition\n')
    print(f"{'condition':16} {'accuracy':>8} {'95% CI':>14} {'paper':>6} {'vs ' + config.baseline:>12} {'95% CI':>16} {'no letter':>9}")

    baseline = correct[config.baseline]
    for condition, by_question in sorted(correct.items()):
        accuracy = 100 * np.mean([by_question[question_id] for question_id in shared])
        low, high = 100 * video_bootstrap(by_video(lambda question_id: by_question[question_id]), config.num_bootstrap, config.seed)

        paper = PAPER_LONG_ACCURACY.get(condition)
        paper_text = f'{paper:.1f}' if paper is not None else ''

        comparison = ''
        if condition != config.baseline:
            def difference(question_id):
                return by_question[question_id] - baseline[question_id]

            mean_difference = 100 * np.mean([difference(question_id) for question_id in shared])
            difference_low, difference_high = 100 * video_bootstrap(by_video(difference), config.num_bootstrap, config.seed)
            comparison = f'{mean_difference:+12.1f} [{difference_low:+5.1f}, {difference_high:+5.1f}]'

        print(
            f'{condition:16} {accuracy:8.1f} [{low:5.1f}, {high:5.1f}] {paper_text:>6} '
            f'{comparison:>29} {len(unanswered[condition] & set(shared)):9d}'
        )


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
