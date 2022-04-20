import click
from pathlib import Path
from judge import score
import pandas as pd
import json


@click.command()
@click.option('-gt',
              '--ground_truth_path',
              default='',
              help='Ground truth file path')
@click.option('-r', '--result_path', default='', help='Your answer file path')
def cal_score(ground_truth_path, result_path):
    click.echo(f"ground_truth_path: {ground_truth_path}")
    click.echo(f"result_path: {result_path}")
    gt_path = Path(ground_truth_path)
    res_path = Path(result_path)
    if not gt_path.exists():
        raise FileNotFoundError("ground_truth file do not exists")
    if not res_path.exists():
        raise FileNotFoundError("result file do not exists")
    ground_truth = pd.read_csv(str(gt_path))
    answers = []
    with res_path.open('r') as f:
        answers = json.load(f)
    res = score(
        {
            "timestamp": ground_truth["timestamp"].values.tolist(),
            "cmdb_id": ground_truth["cmdb_id"].values.tolist(),
            "failure_type": ground_truth["failure_type"].values.tolist()
        },
        answers,
        0,
        method="last",
    )
    click.echo(res)


if __name__ == '__main__':
    cal_score()
