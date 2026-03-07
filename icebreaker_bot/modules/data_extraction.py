import json


def extract_linkedin_profile(mock=True):

    if mock:
        with open("data/mock_profile.json") as f:
            data = json.load(f)

        return data

    return {}
