from __future__ import annotations

import re


QUESTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
CHINESE_TOKEN_SPLIT_RE = re.compile(
    r"(?:[的是和与及或为在把被让给从向对将要]|为什么|为何|怎么|如何|具体|真正|目的|原因|动机|"
    r"发生了什么|发生了|发生|什么|启动|开启|启用|动用|使用|发动|打开|关闭|解除|建造|修建|建设|制造|改造|布局|设下|安排)"
)
QUOTED_TERM_RE = re.compile(r"[“\"'「『]([^”\"'」』]{2,16})[”\"'」』]")
ACTION_TARGET_RE = re.compile(
    r"(?:启动|开启|启用|动用|使用|发动|打开|关闭|解除|建造|修建|建设|制造|改造|布局|设下|安排)"
    r"(?:[“\"'「『])?([^\s，。！？；、”\"'」』?？]{2,18})(?:[”\"'」』])?"
)
ACTION_TARGET_BOUNDARY_RE = re.compile(r"(?:的|是|和|与|及|为|为了|为什么|为何|怎么|如何|关系|原因|目的|区别|吗|么)")
CHAPTER_TOKEN_RE = re.compile(r"(?:第[一二三四五六七八九十百零〇两0-9]+章|[0-9]{1,2}章)")
MAIN_CHAPTER_REF_RE = re.compile(
    r"(?:第\s*([一二三四五六七八九十百零〇两0-9]{1,4})\s*章|([0-9]{1,2})\s*章|level_main[_-]([0-9]{1,2})|main[_-]([0-9]{1,2}))",
    re.IGNORECASE,
)
LINE_SPLIT_RE = re.compile(r"[\n\r。！？；]+")
INTERNAL_EVIDENCE_META_RE = re.compile(
    r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]+\]\s*|\[E\d+\]\s*"
)
INHERITANCE_RE = re.compile(r"([\u4e00-\u9fff]{2,8})的(后人|女儿|儿子|传人)")
KINSHIP_RE = re.compile(r"(亲生父亲|父亲|母亲|家人|老师|师父|弟子|学生)")
REAL_NAME_RE = re.compile(r"(?:原名|本名|真名)[为叫是：:\s]*([\u4e00-\u9fff]{2,8}(?:·[\u4e00-\u9fff]{1,8})?)")
CONSPIRACY_ANCHOR_RE = re.compile(r"(?:撞破|发现|曝光|阻止)?([\u4e00-\u9fff]{2,4})城议员的阴谋")
DIALOGUE_ROLE_PREFIX_RE = re.compile(r"^(user|assistant)\s*:\s*(.*)$", re.IGNORECASE)
