"""Read-only locate the original source passages behind pilot claims."""
import json

NEEDLES = [
    "获得了大家的信任与拥戴",
    "大荒城稻米香甜",
    "羽蛇家族四百五十一年",
    "会获得永恒的宁静",
    "那里存放着一些独属于我的东西",
    "你知道吗，我从小",
]
with open("${REMOTE_HOME}/ASA-ArknightStoryAgent/indexes/arknights_story/documents.jsonl") as stream:
    for line in stream:
        row = json.loads(line)
        text = row["clean_text"]
        if any(needle in text for needle in NEEDLES):
            print(json.dumps({"id": row["id"], "clean_text": text}, ensure_ascii=False))
