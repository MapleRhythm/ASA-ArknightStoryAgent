"""Native chat adapter; neither input truncation nor answer validation."""
import json
import re
import urllib.request


def messages_from_prompt(prompt):
    match = re.fullmatch(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>assistant\n(.*)",
        prompt, flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Unexpected prompt envelope")
    system, user, prefix = match.groups()
    if prefix.strip():
        raise ValueError("Nonempty assistant prefix requires explicit handling")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class BonsaiClient:
    def __init__(self, url="http://127.0.0.1:18127", context=32768):
        self.url = url.rstrip("/")
        self.context = context
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, body=None):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(req, timeout=600) as response:
            return json.load(response)

    def generate(self, prompt, *, thinking=False, max_tokens=1536,
                 temperature=0.2, top_p=0.9, repeat_penalty=1.05):
        messages = messages_from_prompt(prompt)
        effort = "medium" if thinking else "none"
        template_kw = {"enable_thinking": thinking, "reasoning_effort": effort}
        rendered = self.request("/apply-template", {
            "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": template_kw,
        })["prompt"]
        # Check actual native chat envelope retained both content strings verbatim.
        if not all(m["content"] in rendered for m in messages):
            raise ValueError("Native template changed/truncated message content")
        tokenized = self.request("/tokenize", {"content": rendered, "add_special": True})
        input_tokens = len(tokenized["tokens"])
        if input_tokens + max_tokens > self.context:
            raise ValueError(f"Full prompt plus output exceeds context: {input_tokens}+{max_tokens}")
        body = {
            "model": "bonsai2-27b", "messages": messages, "stream": False,
            "reasoning_effort": effort, "chat_template_kwargs": template_kw,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "repeat_penalty": repeat_penalty, "seed": 0,
            "cache_prompt": False,
        }
        if thinking:
            body.update(top_k=20, min_p=0.05, presence_penalty=0.0)
        else:
            # 9B did not use a top-k/min-p filter or presence penalty.
            body.update(top_k=0, min_p=0.0, presence_penalty=0.0)
        response = self.request("/v1/chat/completions", body)
        choice = response["choices"][0]
        actual_tokens = response.get("usage", {}).get("prompt_tokens")
        if actual_tokens != input_tokens:
            raise ValueError(f"Template preflight/API token mismatch: {input_tokens} != {actual_tokens}")
        return {
            "body": body, "native_prompt": rendered, "input_tokens": input_tokens,
            "api_response": response, "text": choice["message"].get("content") or "",
            "finish_reason": choice["finish_reason"],
        }
