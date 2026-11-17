"""JSONL files that runs append to and resume from; standard library only, so both venvs import it"""
import json


def read_resumable(path):
    """Rows of an append-only JSONL file. A last line cut short by a killed run is dropped from the file,
    so the run can resume and append after it"""
    if not path.exists():
        return []

    lines = path.read_text().splitlines()

    rows = []
    for line_index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if line_index != len(lines) - 1:
                raise

            print(f'{path}: dropping a truncated last line')
            path.write_text(''.join(complete + '\n' for complete in lines[:-1]))

    return rows
