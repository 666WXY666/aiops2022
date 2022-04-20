from judge import *
import json
import unittest
import pandas as pd


def cal_one_ans_score(team_ans, ground_truth, param):
    context = [team_ans["createTime"]] + json.loads(team_ans["content"])
    one_team_all_answers = pd.DataFrame(
        np.array(context).reshape(-1, 3),
        columns=["timestamp", "cmdb_id", "failure_type"])
    one_team_all_answers["timestamp"] = one_team_all_answers[
        "timestamp"].astype(np.int64).apply(ensure_timestamp)
    final_score = cal_score_by_teams(ground_truth, one_team_all_answers, param)
    return sum(final_score)


class TestJudge(unittest.TestCase):

    def test_cal_score(self):
        ground_truth = pd.read_csv("result_sample.csv")
        answers = []
        with open("result.json") as f:
            answers = json.load(f)
        param = Param()
        for team_ans in answers:
            final_score = cal_one_ans_score(team_ans, ground_truth, param)
            self.assertEqual(final_score, team_ans["score"])

    def test_merge_team_answer(self):
        answers = []
        with open("result.json") as f:
            answers = json.load(f)
        team_res = merge_team_answer(answers)
        self.assertEqual(len(team_res), 4)
        self.assertEqual(list(team_res.keys()), ['1', '2', '3', '4'])
        self.assertEqual(len(team_res['1']), 8 * 3)
        self.assertEqual(len(team_res['2']), 7 * 3)
        self.assertEqual(len(team_res['3']), 7 * 3)
        self.assertEqual(len(team_res['4']), 7 * 3)

    def test_score(self):
        ground_truth = pd.read_csv("result_sample.csv")
        answers = []
        with open("result.json") as f:
            answers = json.load(f)
        res = score(
            {
                "timestamp": ground_truth["timestamp"].values.tolist(),
                "cmdb_id": ground_truth["cmdb_id"].values.tolist(),
                "failure_type": ground_truth["failure_type"].values.tolist()
            },
            answers,
            0,
        )
        res = sorted(res, key=lambda r: int(r["teamId"]))
        self.assertDictEqual(res[0], {
            "teamId": '1',
            "score": 55.0,
            'submitNum': 8
        })
        self.assertDictEqual(res[1], {
            "teamId": '2',
            "score": 34.0,
            'submitNum': 7
        })
        self.assertDictEqual(res[2], {
            "teamId": '3',
            "score": 36.0,
            'submitNum': 7
        })
        self.assertDictEqual(res[3], {
            "teamId": '4',
            "score": 31.5,
            'submitNum': 7
        })

    def test_cal_delay_score(self):
        t = 60
        param = Param()
        self.assertEqual(cal_delay_score(t, 61, param), 1)
        self.assertEqual(cal_delay_score(t, 59, param), 1)
        self.assertEqual(cal_delay_score(t, 120, param), 0.95)
        self.assertEqual(cal_delay_score(t, 659, param), 0.55)
        self.assertEqual(cal_delay_score(t, 660, param), 0.5)
        self.assertEqual(cal_delay_score(t, 661, param), 0)


if __name__ == '__main__':
    unittest.main()
