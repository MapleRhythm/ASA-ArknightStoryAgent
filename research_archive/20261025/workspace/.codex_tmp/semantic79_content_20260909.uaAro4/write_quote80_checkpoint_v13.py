"""Record the rejected quote-ceiling ablation and observed training-mirror gap."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
DEST = ROOT / "docs/CANONICAL_QUOTE_CEILING_AND_TRAINING_MIRROR_AUDIT_20260908_irPugk_v13.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert not DEST.exists()
    replay = load(HERE / "quote80_ablation_comparison_v1.json")
    assert len(replay["newly_answering_states"]) == 5
    assert sum(g["summary"]["baseline"]["states"] for g in replay["groups"]) == 199
    source = ROOT / "release/ASA-ArknightStoryAgent/src"
    assert {str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")} == replay["source_hashes"]
    assert all(sha(Path(p)) == digest for p, digest in replay["source_files"].items())
    now = datetime.now(timezone.utc)
    notes = []
    for state in replay["newly_answering_states"]:
        rid = state["id"]
        if rid.startswith("relabel-4835313d"):
            verdict = "unsupported_causal_link_and_question_mismatch"
            reason = (
                "完整E1至E6已读。E1说莱欧斯其实更着急、齐尔查克缩短路上时间，"
                "玛露西尔仍希望为重逢作美好打算。答案却以更着急推出因此不急着赶路，"
                "这个因果连接没有被源文建立，也未回答提问中她为何作出不急着赶路判断的观察理由。"
            )
        elif rid.startswith("relabel-28d915"):
            verdict = "supported_core_with_temporal_overstatement_concern"
            reason = (
                "完整E1至E6已读。E3支持固执、自大、透支钱/生活/儿子选择的自责，也支持大儿子战死、"
                "二儿子离开。原文达维镇是马上就要完蛋，答案压成导致达维镇完蛋，"
                "可能把临近困境写成既成结果。核心有据，但此扩展需复核，不把它整条判为虚构，"
                "也不直接认证为完整正确。"
            )
        elif rid.startswith("relabel-c637f5"):
            verdict = "unsupported_subject_attribution"
            reason = (
                "完整E1至E6已读。第一事实的核心自责有E3支持，另有与上一状态相同的时态压缩顾虑。"
                "第二事实把E1明确围绕莱昂图索写的被矫饰的暴力归给里昂，"
                "材料没有提供二者同一人的根据，不能仅凭最大错误的词面相似转移人物归属。"
            )
        else:
            assert rid.startswith("896419")
            verdict = "supported_complete_visible_context"
            reason = (
                "两次新放行的展示答案完全相同，是同一个美食节问题，不是两个独立答对问题。"
                "已读相关完整E块：E1/E4支持美食节及投入准备，E6支持凛冬解谜失败后吃罐头刨冰；"
                "内容按全可见证据支持。该复核不等于每个fact的短quote均单独覆盖全部语义。"
            )
        notes.append({**state, "source_review_verdict": verdict, "reason": reason})
    notes_path = HERE / "quote80_new_answers_source_review_v1.json"
    save(notes_path, {
        "created_at_utc": now.isoformat(), "reviewer_type": "assistant_source_review_not_human_gold",
        "evaluation_only": True, "training_ready": False, "production_promotion": "not_promoted",
        "source_replay_sha256": sha(HERE / "quote80_ablation_comparison_v1.json"), "notes": notes,
    })
    observed_builder = "fc426928db98380eaad6708ce6d05df23b441d86114d0e1f180ad78c96d85e4f"
    observed_cleaner = "ba6ee2a97839e644c9cc0397cc1d7f0f7488066ecf0977320cef38610686dfb4"
    builder_backup = RUN / "verifier_contract_patch_v1.nKJjAQ/original.py"
    cleaner_backup = RUN / "cleaner_integrity_patch_v1.LTD6na/cleaner_original.py"
    assert sha(builder_backup) == observed_builder and sha(cleaner_backup) == observed_cleaner
    mirror_path = HERE / "training_mirror_readonly_observations_v1.json"
    save(mirror_path, {
        "recorded_at_utc": now.isoformat(), "source": "read_only_ssh_checks_in_this_turn",
        "host": "${REMOTE_HOST}", "path": "${REMOTE_HOME}/ASA-ArknightStoryAgent",
        "git_rev_parse_result": "fatal: not a git repository (or any of the parent directories): .git",
        "git_search_scope": "${REMOTE_HOME}, directories at maximum depth 3, no .git found",
        "release_directory_present": False,
        "observed_sha256": {
            "scripts/build_soda_api_verifier_dataset.py": observed_builder,
            "scripts/clean_soda_api_verifier_dataset.py": observed_cleaner,
        },
        "missing_paths": [
            "scripts/soda_dataset_integrity.py",
            "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/planning/retrieval_term_cleaning.py",
        ],
        "matches_241_pre_fix_backups": {
            str(builder_backup): sha(builder_backup), str(cleaner_backup): sha(cleaner_backup),
        },
        "gpu_compute_process": {
            "gpu_uuid": "GPU-17dfd5ca-b6d1-7cd5-fdc7-b1e95d13d960",
            "pid": 450914, "process_name": ".venv/bin/python", "memory_mib": 1892,
        },
        "remote_writes": False, "processes_stopped": False,
        "limitations": [
            "No claim about every possible repository path on 248; only the stated path and bounded git search were checked.",
            "Current files matching unsafe old implementations does not prove a running job is using them.",
            "Copying only the repaired builder would leave required modules missing; synchronization needs its dependency set and training-side tests.",
        ],
    })
    table = "\n".join(
        f"| {g['name']} | {g['summary']['baseline']['states']} | "
        f"{g['summary']['baseline']['actions'].get('answer_directly', 0)} | "
        f"{g['summary']['quote160']['actions'].get('answer_directly', 0)} |"
        for g in replay["groups"]
    )
    text = f"""# ASA：单引用长度上限与248训练镜像审计（v13）

UTC：{now.isoformat()}；北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat()}。
文件名延续UTC 20260908系列。241为规范源。

## 结论

**不直接放宽线上quote长度规则。** 离线去掉单条quote超过80字的失败项后，只有5个状态新增直接回答，
其中出现无依据因果和人物归属错误。长度上限不是语义审核；这些错误之前只是被长度条件偶然拦住。
不能把更多直接回答当成更高正确率。本试验没有修改规范源码、提示、数据或模型。

248的指定目录是裁剪训练镜像，不是当前可确认的完整Git检出。两个构建/清洗脚本与241修复前备份哈希完全相同，
并缺少新修复依赖。没有冒然覆盖文件或停止其A100进程；后续训练前需受控同步完整依赖并验证。

## 1. 长引用离线反事实

复用199个已有生成状态，按7个实验分开报告；包含不同适配器和重复问题，绝不能把199作为独立准确率分母。
唯一改变是进程内忽略精确的supported/inferred_fact_N_ref_N_quote_over_80失败项。
保留每fact总引文160字、全答案400字、可见证据匹配、证据ID、事实内容检查和原解析器。
不截断或重写引文/事实，不补答案，不重新生成/检索，不触碰生产文件。
独立分类器11项测试通过，不并入276项规范回归总数。

| 原实验 | 状态数 | 原直接回答 | 离线取消单条80限制 |
|---|---:|---:|---:|
{table}

fixed79仍是58个去重问题的79个历史证据状态；这里只由39变40，不是实测正确率上升。
新增5个状态只涉及3个问题（里昂、美食节均有重复状态/生成）。
其余部分状态仅内部诊断字段变化，不算新增回答。

## 2. 对新增答案逐源复核

- 玛露西尔：全E1至E6已读。原文说莱欧斯其实更着急、同伴在缩短路上时间；答案写成“更着急，因此不急着赶路”。
  源文未建立这个因果，也未给出提问所要的观察理由，不应放行成正确答案。
- 里昂第一个状态：固执、自大、透支选择的核心自责有E3支持。把“马上就要完蛋”的达维镇压成
  “导致达维镇完蛋”可能抹去了将要/既成的区别；保留时态过强疑虑，不把核心也一并判成虚构。
- 里昂第二个状态：新增断言把E1围绕莱昂图索的“被矫饰的暴力”归给里昂，材料没有同一人物的根据。
  同题长引文通过并不能解决跨人物错误。两条里昂状态的完整E1至E6均已核对。
- 美食节两个状态：实际展示答案相同，相关完整证据支持其节日、准备和惩罚经历；
  它们不是两个独立问题，也不表示每一条短quote都单独支持其fact全部内容。

这是assistant_source_review_not_human_gold，不覆盖旧自动标签、不生成训练正负标签、不报事后挑选的正确率。
只取消80限制会同时放行有据回答和错误回答，必须先解决事实归属/因果以及引用生成质量，不能直接上线。

## 3. 248镜像实际状态

检查位置：`${REMOTE_HOME}/ASA-ArknightStoryAgent`。

- git rev-parse失败；${REMOTE_HOME}深度3内未发现.git目录。没有据此声称整台机器都无仓库。
- 此位置没有release目录。
- builder SHA256：`{observed_builder}`，等于241 v7修复前备份。
- cleaner SHA256：`{observed_cleaner}`，等于241 v10修复前备份。
- 缺scripts/soda_dataset_integrity.py及release运行时诊断清理模块。
- A100仍有PID450914，约1892MiB，即使低利用率也不占用/停止。

上述能证明镜像文件未包含这些修复，不能证明正在运行的那个进程正在使用它们。
不能只复制最新builder：它的新依赖在此目录不存在。必须先核对训练镜像布局、保留其改动，
准备含依赖的受控同步，并运行训练侧回归；未审定数据仍不训练。
本次248只有只读检查，没有远端写入。

## 4. 后续

继续把读题/事实归属能力与格式失败分开诊断。已有不带LoRA的strict9实验，需要先读其条件和结果，
避免重复试验；若做新的适配器消融，应固定同一输入、tokenizer、seed和证据，并保留逐请求适配器记录。
同时处理训练镜像依赖同步问题，而不是让旧构建器重新产生已修复的污染目标。
目标仍进行中。

实验目录：`{HERE}`。
"""
    with DEST.open("x", encoding="utf-8") as stream:
        stream.write(text)
    save(HERE / "checkpoint_v13_manifest.json", {
        "created_at_utc": now.isoformat(), "report_path": str(DEST), "report_sha256": sha(DEST),
        "training_ready": False, "goal_complete": False,
        "artifacts_sha256": {
            str(p): sha(p) for p in (
                HERE / "quote80_ablation_comparison_v1.json", notes_path, mirror_path,
                HERE / "ablation_classifier_tests_v1.xml",
            )
        },
    })
    print(json.dumps({"report": str(DEST), "sha256": sha(DEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
