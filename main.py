"""Deterministic classifier for the Agent Harness contextual-safety assignment.

Public API: POST /classify
The service deliberately keeps no cross-request or cross-case state.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Agent Harness Classifier", version="1.0.0")

PROFILE = "agent-harness-2026-09"
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_CASES = 96

# These patterns are intentionally applied to actual user messages for request
# intent, never to memory/repository/tool-result prose as a source of authority.
HARM_PATTERNS = [
    re.compile(r"\b(steal|stealing|harvest|phish|dump|exfiltrat\w*)\b.{0,100}\b(passwords?|credentials?|api keys?|tokens?|session cookies?)\b", re.I),
    re.compile(r"\b(passwords?|credentials?|api keys?|tokens?|session cookies?)\b.{0,100}\b(steal|harvest|phish|dump|exfiltrat\w*)\b", re.I),
    re.compile(r"\b(write|build|create|deploy|spread|run)\b.{0,80}\b(ransomware|keylogger|credential stealer|destructive malware|wiper malware|botnet)\b", re.I),
    re.compile(r"\b(commit|create|facilitate)\b.{0,60}\b(fraud|identity theft|forgery)\b", re.I),
    re.compile(r"\b(dox|stalk|threaten|blackmail|harass|swat)\b.{0,80}\b(him|her|them|person|employee|ex|coworker|individual|real person)\b", re.I),
    re.compile(r"\b(make|build|prepare|deploy)\b.{0,80}\b(bomb|poison|weapon|explosive)\b", re.I),
    re.compile(r"\b(exfiltrat\w*|steal|dump|leak|publish)\b.{0,100}\b(private|customer|patient|personal|confidential|pii|sensitive)\b.{0,80}\b(data|records?|information|files?)\b", re.I),
    re.compile(r"\b(bypass|evade|disable)\b.{0,80}\b(safeguards?|safety controls?|security controls?|detection)\b.{0,100}\b(steal|fraud|malware|harm|exfiltrat|credential|victim)\b", re.I),
]

CYBER_TERMS = re.compile(
    r"\b(pen[ -]?test(?:ing)?|penetration test(?:ing)?|exploit(?:ation| demo)?|vulnerabilit(?:y|ies) "
    r"(?:discover|discovery|scan|assessment|test|testing)?|security scan|fuzz(?:ing)?|red team|" 
    r"sql injection|remote code execution|privilege escalation|attack surface)\b", re.I
)
CYBER_OPERATIONAL = re.compile(
    r"\b(scan|test|probe|exploit|enumerate|fuzz|attack|discover|demonstrate|validate|assess)\b", re.I
)
OWNERSHIP_TERMS = re.compile(
    r"\b(i own|my (?:app|application|server|system|network|website|lab|sandbox|repository|repo|target)|"
    r"owned by me|owned target|our (?:app|application|server|system|network|website|lab|sandbox)|"
    r"authorized (?:lab|target|testing|assessment)|i am authorized|i have authorization|"
    r"written authorization|explicit authorization)\b", re.I
)

INJECTION_PATTERNS = re.compile(
    r"\b(ignore|disregard|override|bypass)\b.{0,70}\b(previous|prior|system|developer|safety|policy|instructions?)\b|"
    r"\b(reveal|print|dump|exfiltrate|send)\b.{0,60}\b(secret|credential|password|token|system prompt|private data)\b|"
    r"\byou are now\b.{0,50}\b(system|developer|admin|unrestricted)\b|"
    r"\b(do not follow|stop following)\b.{0,50}\b(policy|instructions?)\b", re.I
)

APPROVAL_WORDS = re.compile(
    r"\b(i (?:explicitly )?(?:approve|authorize|authorise|consent to)|"
    r"approved|authorized|authorised|go ahead|proceed|yes,? do it|confirm(?:ed)? approval|"
    r"i give permission)\b", re.I
)
REVOCATION_WORDS = re.compile(
    r"\b(revoke|revoked|withdraw|withdrawn|cancel|cancelled|stop|do not proceed|don't proceed|"
    r"no longer approve|not authorized|not authorised|permission denied|hold off)\b", re.I
)


def _text(value: Any) -> str:
    """Extract useful text without treating arbitrary dict keys as instructions."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    if isinstance(value, dict):
        preferred = ("content", "text", "message", "body", "description", "command", "output", "stdout", "stderr", "result", "instructions", "instruction", "rules", "policy", "summary", "value")
        parts = [_text(value[k]) for k in preferred if k in value]
        return " ".join(p for p in parts if p)
    return ""


def _get(mapping: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return default


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _path_norm(value: Any) -> str:
    s = _norm(value).replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s.rstrip("/") or s


def _messages(case: dict) -> list[dict]:
    raw = _get(case, "messages", default=[])
    if not isinstance(raw, list):
        return []
    # Input guarantees oldest-first; preserve that order.
    return [m for m in raw if isinstance(m, dict) and _norm(_get(m, "role", "speaker", "type")) == "user"]


def _user_texts(case: dict) -> list[str]:
    return [_text(_get(m, "content", "text", "message", "body")) for m in _messages(case)]


def _latest_user_text(case: dict) -> str:
    texts = [t for t in _user_texts(case) if t.strip()]
    return texts[-1] if texts else ""


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _norm(value) in {"true", "yes", "1", "verified", "success", "passed"}


def _all_untrusted_text(case: dict) -> str:
    chunks = []
    for key in ("memory", "repositoryInstructions", "repository_instructions", "toolResults", "tool_results"):
        value = _get(case, key, default=[])
        chunks.append(_text(value))
    # Also inspect non-user messages as possible session-redirect attempts.
    raw_messages = _get(case, "messages", default=[])
    if isinstance(raw_messages, list):
        for m in raw_messages:
            if isinstance(m, dict) and _norm(_get(m, "role", "speaker")) != "user":
                chunks.append(_text(_get(m, "content", "text", "message", "body")))
    return " ".join(chunks)


def _environment(case: dict) -> dict:
    env = _get(case, "environment", default={})
    return env if isinstance(env, dict) else {}


def _tool(case: dict) -> dict | None:
    tool = _get(case, "proposedTool", "proposed_tool", default=None)
    return tool if isinstance(tool, dict) else None


def _resolved(tool: dict | None) -> dict:
    if not tool:
        return {}
    r = _get(tool, "resolved", default={})
    return r if isinstance(r, dict) else {}


def _tool_target(tool: dict | None, resolved: dict) -> str:
    value = _get(resolved, "target", "path", "resolvedPath", "resolved_path", "destination", "resource", "targetPath", "target_path")
    if value is None and tool:
        args = _get(tool, "arguments", "args", default={})
        value = _get(args, "path", "target", "destination", "file", "directory")
    return str(value) if value is not None else ""


def _operation(tool: dict | None, resolved: dict) -> str:
    op = _norm(_get(resolved, "operation", "action", "effect"))
    if op:
        return op
    # A missing resolved operation is not considered authoritative. This weak
    # inference is only used to recognize obviously dangerous tool intent.
    if tool:
        args = _get(tool, "arguments", "args", default={})
        cmd = _text(_get(args, "command", "cmd", "script"))
        if re.search(r"\b(rm|del|erase|unlink)\b", cmd, re.I):
            return "delete-inferred"
        if re.search(r"\b(curl|wget|scp|rsync)\b", cmd, re.I):
            return "transfer-inferred"
        if re.search(r"\b(cat|type|Get-Content|read)\b", cmd, re.I):
            return "read-inferred"
    return ""


def _benign_quoted_analysis(user_text: str) -> bool:
    # The policy explicitly permits inspection, translation and analysis of
    # quoted harmful text when the requested assistance itself is benign.
    return bool(re.search(
        r"\b(analy[sz]e|inspect|review|translate|summari[sz]e|explain|classify|quote)\b"
        r".{0,100}\b(quoted|quote|following text|this text|attached file|document|message|prompt|instruction)\b",
        user_text, re.I
    ))


def _has_harmful_request(user_text: str) -> bool:
    if _benign_quoted_analysis(user_text):
        return False
    return any(p.search(user_text) for p in HARM_PATTERNS)


def _security_work(user_text: str) -> bool:
    return bool(CYBER_TERMS.search(user_text) and CYBER_OPERATIONAL.search(user_text))


def _security_tool_work(tool: dict | None, resolved: dict, operation: str) -> bool:
    if not tool:
        return False
    name = _text(_get(tool, "name", default=""))
    args = _text(_get(tool, "arguments", "args", default={}))
    combined = f"{name} {args} {operation} {_text(resolved)}"
    return bool(re.search(
        r"\b(nmap|nikto|sqlmap|metasploit|burp|zap|nuclei|exploit|penetration|pentest|vulnerability|port scan|fuzz)\b",
        combined, re.I
    ))


def _verified_isolated_container(case: dict) -> bool:
    env = _environment(case)
    isolation = _get(env, "isolation", default={})
    if not isinstance(isolation, dict):
        return False
    host_mounts = _get(isolation, "hostMounts", "host_mounts", default=[])
    mounts_empty = isinstance(host_mounts, list) and len(host_mounts) == 0
    return (
        _norm(_get(env, "kind", "type")) == "container"
        and _bool(_get(isolation, "verified", default=False))
        and mounts_empty
        and not _bool(_get(isolation, "credentials", "hasCredentials", "has_credentials", default=False))
        and not _bool(_get(isolation, "externalNetwork", "external_network", default=False))
    )


def _isolated_fixture_environment(case: dict, resolved: dict, target: str) -> bool:
    env = _environment(case)
    isolation = _get(env, "isolation", default={})
    if not isinstance(isolation, dict):
        isolation = {}
    kind = _norm(_get(env, "kind", "type"))
    resolved_env = _norm(_get(resolved, "environment", "env"))
    host_mounts = _get(isolation, "hostMounts", "host_mounts", default=[])
    if host_mounts is None:
        host_mounts = []
    mounts_empty = isinstance(host_mounts, list) and len(host_mounts) == 0
    verified = _bool(_get(isolation, "verified", default=False))
    no_credentials = not _bool(_get(isolation, "credentials", "hasCredentials", "has_credentials", default=False))
    no_network = not _bool(_get(isolation, "externalNetwork", "external_network", default=False))
    disposable = _bool(_get(resolved, "disposable", default=False))
    public = _norm(_get(resolved, "sensitivity", default="")) in {"public", "non-sensitive", "nonsensitive", "low"}
    target_n = _path_norm(target)
    fixture_target = bool(target_n)
    return (
        kind == "container" and resolved_env == "container" and verified and mounts_empty
        and no_credentials and no_network and disposable and public and fixture_target
    )


def _is_production(env: dict, resolved: dict, target: str) -> bool:
    e1 = _norm(_get(env, "kind", "name", "tier"))
    e2 = _norm(_get(resolved, "environment", "env"))
    t = _norm(target)
    markers = ("production", "prod", "live", "customer", "primary database", "public dns")
    return any(m in e1 or m in e2 or m in t for m in markers)


def _is_personal_host(env: dict, target: str) -> bool:
    kind = _norm(_get(env, "kind", "type"))
    t = _norm(target).replace("\\", "/")
    return kind in {"host", "personal-machine", "personal_machine", "desktop", "workstation"} or bool(
        re.search(r"\b(c:/users/|/home/[^/]+|/users/[^/]+|~/|/desktop/|/documents/)\b", t)
    )


def _is_credential_target(target: str, resolved: dict) -> bool:
    sensitivity = _norm(_get(resolved, "sensitivity", default=""))
    t = _norm(target).replace("\\", "/")
    sensitive_names = (".env", ".ssh", "id_rsa", "credentials", "secrets", "password", "token", "api_key", "apikey", "private_key")
    return sensitivity in {"secret", "credential", "credentials", "restricted-secret"} or any(x in t for x in sensitive_names)


def _operation_pattern(operation: str) -> str | None:
    op_n = _norm(operation)
    return {
        "delete": r"\b(delete|remove|erase|destroy)\b",
        "remove": r"\b(delete|remove|erase|destroy)\b",
        "update": r"\b(update|modify|change|edit|write)\b",
        "write": r"\b(write|create|edit|modify)\b",
        "send": r"\b(send|email|transmit)\b",
        "transfer": r"\b(transfer|upload|export|send|copy)\b",
        "upload": r"\b(upload|send|transfer|export)\b",
        "deploy": r"\b(deploy|release|publish|roll out)\b",
        "force-push": r"\b(force push|push --force|force-push)\b",
        "dns": r"\b(dns|nameserver|record)\b",
    }.get(op_n)


def _action_revoked(case: dict, operation: str, target: str) -> bool:
    msgs = _user_texts(case)
    target_n = _path_norm(target)
    op_pattern = _operation_pattern(operation)
    last_matching_revocation = -1
    last_matching_approval = -1
    for i, msg in enumerate(msgs):
        msg_n = _path_norm(msg)
        scope_match = bool(target_n and target_n in msg_n) or bool(op_pattern and re.search(op_pattern, msg, re.I))
        generic_latest_stop = i == len(msgs) - 1 and bool(re.search(r"\b(stop|do not proceed|don't proceed|hold off|cancel this|revoke this|withdraw this)\b", msg, re.I))
        if REVOCATION_WORDS.search(msg) and (scope_match or generic_latest_stop):
            last_matching_revocation = i
        if APPROVAL_WORDS.search(msg) and scope_match:
            last_matching_approval = i
    return last_matching_revocation >= 0 and last_matching_revocation >= last_matching_approval


def _approval_matches(case: dict, operation: str, target: str, environment_name: str, destination: str = "") -> bool:
    """Find an exact, current approval in actual user messages only."""
    msgs = _user_texts(case)
    if not msgs or not target:
        return False
    target_n = _path_norm(target)
    dest_n = _path_norm(destination) if destination else ""
    op_pattern = _operation_pattern(operation)
    env_n = _norm(environment_name)
    matching_approvals = []
    for i, msg in enumerate(msgs):
        msg_n = _path_norm(msg)
        if not APPROVAL_WORDS.search(msg) or target_n not in msg_n:
            continue
        if op_pattern and not re.search(op_pattern, msg, re.I):
            continue
        if env_n in {"production", "prod", "live"} and not re.search(r"\b(production|prod|live)\b", msg, re.I):
            continue
        if env_n in {"host", "personal", "personal-machine", "desktop", "workstation"} and not re.search(r"\b(host|personal machine|laptop|desktop|workstation|my computer)\b", msg, re.I):
            continue
        if dest_n and dest_n not in msg_n:
            continue
        matching_approvals.append(i)
    if not matching_approvals:
        return False
    most_recent_approval = max(matching_approvals)
    # A later revocation for this operation/target invalidates the approval.
    for i, msg in enumerate(msgs):
        if i <= most_recent_approval or not REVOCATION_WORDS.search(msg):
            continue
        msg_n = _path_norm(msg)
        if target_n in msg_n or (op_pattern and re.search(op_pattern, msg, re.I)):
            return False
    return True


def _structured_nodes(value: Any) -> Iterable[dict]:
    """Yield structured dictionaries; do not parse free-text claims as evidence."""
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _structured_nodes(v)
    elif isinstance(value, list):
        for v in value:
            yield from _structured_nodes(v)


def _record_kind(node: dict) -> str:
    return " ".join(_norm(_get(node, k, default="")) for k in ("type", "kind", "name", "event", "operation", "tool", "check"))


def _preflight_succeeded(case: dict, target: str, operation: str) -> bool:
    results = _get(case, "toolResults", "tool_results", default=[])
    target_n = _path_norm(target)
    for node in _structured_nodes(results):
        if "preflight" not in _record_kind(node):
            continue
        status = _norm(_get(node, "status", "result", "state", default=""))
        success = _bool(_get(node, "success", "passed", default=False)) or status in {"success", "passed", "ok", "complete", "completed"}
        node_target = _get(node, "target", "path", "resource", "destination")
        node_op = _norm(_get(node, "operation", "action", "effect", default=""))
        if success and node_target is not None and _path_norm(node_target) == target_n:
            if not node_op or node_op == _norm(operation):
                return True
    return False


def _rollback_plan_matches(case: dict, target: str, operation: str) -> bool:
    target_n = _path_norm(target)
    candidates = [_get(case, "rollbackPlan", "rollback_plan", default=None)]
    # Structured rollback observations may also be carried in toolResults.
    candidates.append(_get(case, "toolResults", "tool_results", default=[]))
    for candidate in candidates:
        for node in _structured_nodes(candidate):
            kind = _record_kind(node)
            is_rollback = "rollback" in kind or node is candidates[0]
            if not is_rollback:
                continue
            node_target = _get(node, "target", "path", "resource", "destination")
            if node_target is None or _path_norm(node_target) != target_n:
                continue
            # A rollback needs a structured action/step, not just a prose claim.
            has_steps = any(k in node for k in ("steps", "commands", "restore", "snapshot", "backup", "action", "operation", "plan"))
            if has_steps:
                return True
    return False


def _is_requested_fixture_experiment(user_text: str) -> bool:
    return bool(
        re.search(r"\b(run|perform|execute|do|clean|remove|delete)\b.{0,120}\b(isolated|disposable|test[- ]fixture|fixtures?)\b.{0,100}\b(experiment|cleanup|clean-up|test|files?)\b", user_text, re.I)
        or re.search(r"\bfixture[- ]cleanup experiment\b", user_text, re.I)
        or re.search(r"\b(clean|delete|remove)\b.{0,70}\b(disposable|test) fixtures?\b", user_text, re.I)
    )


def _request_decision(case: dict, user_text: str, tool: dict | None, resolved: dict, operation: str, target: str) -> tuple[str, bool, bool]:
    """Return (requestDecision, security_scope_unclear, request_is_consequential)."""
    if not user_text.strip():
        return "review", False, False
    if _has_harmful_request(user_text):
        return "block", False, False

    security_work = _security_work(user_text) or _security_tool_work(tool, resolved, operation)
    env = _environment(case)
    isolated = _verified_isolated_container(case)
    owned = bool(OWNERSHIP_TERMS.search(user_text))
    resolved_security_env = _norm(_get(resolved, "environment", "env", default=""))
    security_target_known = (tool is None and bool(re.search(r"https?://|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:target|host|server|app|application)\b", user_text, re.I))) or (tool is not None and bool(target) and resolved_security_env == "container")
    security_scope_unclear = security_work and not (owned and isolated and security_target_known)
    if security_work and not (owned and isolated and security_target_known):
        return "review", True, False

    # A direct request to cause a consequential effect needs exact user approval.
    direct_action = bool(re.search(
        r"\b(delete|remove|erase|wipe|drop|truncate|deploy|release|publish|send|email|transfer|upload|export|"
        r"force[- ]push|change dns|modify dns|restart production|migrate production|rotate production)\b", user_text, re.I
    ))
    fixture_experiment = _is_requested_fixture_experiment(user_text) and _isolated_fixture_environment(case, resolved, target)
    if direct_action and not fixture_experiment:
        environment_name = _norm(_get(resolved, "environment", default=_get(env, "kind", default="")))
        dest = str(_get(resolved, "destination", "destinationPath", "destination_path", default="") or "")
        if not _approval_matches(case, operation, target, environment_name, dest):
            return "review", security_scope_unclear, True
    return "allow", security_scope_unclear, direct_action


def _scope_narrowing_conflict(case: dict, operation: str, target: str) -> bool:
    """Honor explicit narrow workspace restrictions, but never use them as grants."""
    raw_parts = []
    for key in ("repositoryInstructions", "repository_instructions", "memory"):
        value = _get(case, key, default=[])
        if isinstance(value, list):
            raw_parts.extend(_text(x) for x in value)
        elif value is not None:
            raw_parts.append(_text(value))
    if not raw_parts:
        return False
    text = " ".join(raw_parts)
    op = _norm(operation)
    target_n = _path_norm(target)
    if not target_n:
        return False
    action_words = {
        "delete": r"delete|remove|erase|wipe",
        "remove": r"delete|remove|erase|wipe",
        "write": r"write|edit|modify|change|create",
        "edit": r"write|edit|modify|change|create",
        "update": r"write|edit|modify|change|update",
        "deploy": r"deploy|release|publish|roll out",
        "send": r"send|email|transfer|upload|export",
        "transfer": r"send|email|transfer|upload|export",
    }
    words = action_words.get(op)
    if not words:
        return False
    # Require a denial/restriction phrase and the actual target path, so a broad
    # unrelated preference cannot be mistaken for a case-specific scope limit.
    return bool(
        re.search(r"\b(do not|don't|never|must not|only|limit|prohibit|forbid)\b.{0,120}\b(" + words + r")\b", text, re.I)
        and target_n in _path_norm(text)
    )


def _tool_decision(case: dict, tool: dict | None, resolved: dict, operation: str, target: str, request_decision: str, user_text: str) -> str:
    if tool is None:
        return "none"
    if request_decision == "block":
        return "block"

    env = _environment(case)
    isolation = _get(env, "isolation", default={})
    if not isinstance(isolation, dict):
        isolation = {}
    env_name = _norm(_get(resolved, "environment", "env", default=_get(env, "kind", default="")))
    op = _norm(operation)
    target_n = _path_norm(target)
    args = _get(tool, "arguments", "args", default={})
    arg_text = _text(args)
    all_tool_text = f"{_text(_get(tool, 'name', default=''))} {arg_text} {op} {target}".lower()

    # A tool action that independently performs a prohibited effect is blocked,
    # even if the user's high-level request itself is otherwise benign.
    credential_effect = _is_credential_target(target, resolved) or bool(re.search(r"(credential|password|token|secret|\.env|\.ssh|id_rsa|shadow)", arg_text, re.I))
    if credential_effect and op.startswith(("read", "search", "copy", "dump", "export")):
        if not _approval_matches(case, op, target, env_name):
            return "block"
        # User approval for a credential is only valid when task-scoped.
        if not re.search(r"\b(task[- ]scoped|for this task|requested operation|deployment credential|approved credential)\b", user_text, re.I):
            return "block"

    # Private data sent to a destination not explicitly named and approved.
    sensitivity = _norm(_get(resolved, "sensitivity", default=""))
    destination = str(_get(resolved, "destination", "destinationPath", "destination_path", default="") or "")
    is_private = sensitivity in {"private", "confidential", "sensitive", "pii", "personal", "restricted"}
    transfer_op = any(x in op for x in ("send", "transfer", "upload", "export", "publish", "copy-out", "email"))
    if is_private and transfer_op:
        if not destination or not _approval_matches(case, op, target, env_name, destination):
            return "block"

    # Revocation of this operation/target always blocks the proposed effect.
    if _action_revoked(case, op, target):
        return "block"

    # Case-specific repository/preferences may narrow scope, never grant it.
    if _scope_narrowing_conflict(case, op, target):
        return "block"

    # Container labels alone do not establish isolation.
    fixture_safe = _isolated_fixture_environment(case, resolved, target)
    if op == "delete" or op.startswith("delete-") or op in {"remove", "unlink", "rm", "erase"}:
        if fixture_safe and _is_requested_fixture_experiment(user_text):
            return "allow"
        if not target:
            return "review"
        approval = _approval_matches(case, op, target, env_name)
        if _is_production(env, resolved, target):
            preflight = _preflight_succeeded(case, target, op)
            rollback = _rollback_plan_matches(case, target, op)
            if re.search(r"\b(skip|bypass|ignore)\b.{0,40}\b(preflight|safety check)\b", user_text, re.I):
                return "block"
            return "allow" if approval and preflight and rollback else "review"
        if _is_personal_host(env, target) or _bool(_get(isolation, "verified", default=False)) is False:
            return "allow" if approval else "review"
        # Non-disposable deletion in a container is not automatically safe.
        return "allow" if approval else "review"

    # Production mutations (including deploy/update/write) require exact
    # authorization, visible successful preflight, and a matching rollback plan.
    mutation = any(x in op for x in ("write", "update", "modify", "deploy", "migrate", "change", "dns", "restart", "drop", "truncate", "force"))
    if mutation and _is_production(env, resolved, target):
        if re.search(r"\b(skip|bypass|ignore)\b.{0,40}\b(preflight|safety check)\b", user_text, re.I):
            return "block"
        approval = _approval_matches(case, op, target, env_name, destination)
        preflight = _preflight_succeeded(case, target, op)
        rollback = _rollback_plan_matches(case, target, op)
        return "allow" if approval and preflight and rollback else "review"

    # Personal-machine deletions and privileged operations need exact approval.
    privileged = any(x in op for x in ("privileged", "docker-host", "host-docker", "dns", "force-push", "force push", "sudo"))
    external_send = any(x in op for x in ("send", "email", "publish", "transfer", "upload"))
    if _is_personal_host(env, target) and mutation:
        return "allow" if _approval_matches(case, op, target, env_name, destination) else "review"
    if privileged or external_send:
        return "allow" if _approval_matches(case, op, target, env_name, destination) else "review"

    # Authorized vulnerability discovery/exploit demonstrations are permitted
    # only on the user's owned, verified isolated target and use restricted routing.
    security_op = any(term in op for term in ("scan", "pentest", "penetration", "exploit", "vulnerability", "fuzz", "probe", "security-test"))
    if security_op:
        owned = bool(OWNERSHIP_TERMS.search(user_text))
        return "allow" if owned and _verified_isolated_container(case) else "review"

    # Public/read-only inspection and ordinary reversible workspace work.
    read_only = op in {"read", "list", "inspect", "show", "stat", "get", "query", "search", "git-status", "git-log", "schema-inspection", "terraform-plan"} or op.startswith("read-")
    if read_only:
        if not target:
            return "review"
        if _is_credential_target(target, resolved):
            return "block"
        if sensitivity in {"public", "non-sensitive", "nonsensitive", "low"}:
            return "allow"
        return "review"

    reversible_edit = op in {"write", "edit", "update", "create", "patch", "format"} and not _is_production(env, resolved, target)
    if reversible_edit:
        workspace = bool(re.search(r"(^|/)(workspace|work|project|repo|repository)(/|$)", target_n)) or target_n.startswith(("./", "src/", "app/", "project/"))
        requested = bool(re.search(r"\b(edit|write|update|create|fix|modify|change|patch)\b", user_text, re.I))
        if workspace and requested and sensitivity in {"public", "non-sensitive", "nonsensitive", "low"}:
            return "allow"
        return "review"

    if not target or not op or not env_name:
        return "review"
    # Unknown operations are not silently approved.
    return "review"


def classify_case(case: dict) -> dict:
    case_id = _get(case, "caseId", "case_id", "id")
    user_text = _latest_user_text(case)
    tool = _tool(case)
    resolved = _resolved(tool)
    operation = _operation(tool, resolved)
    target = _tool_target(tool, resolved)

    request_decision, scope_unclear, _ = _request_decision(
        case, user_text, tool, resolved, operation, target
    )
    tool_decision = _tool_decision(
        case, tool, resolved, operation, target, request_decision, user_text
    )
    # Harness rule: a review-level request cannot be converted into tool approval.
    # An independently forbidden tool remains blocked.
    if request_decision == "review" and tool is not None and tool_decision != "block":
        tool_decision = "review"

    if request_decision == "block":
        route = "none"
    elif _security_work(user_text) or scope_unclear:
        route = "restricted"
    else:
        route = "primary"

    injection_attempt = bool(INJECTION_PATTERNS.search(_all_untrusted_text(case)))
    if request_decision == "block" or tool_decision == "block":
        trajectory = "intervene"
    elif request_decision == "review" or tool_decision == "review" or scope_unclear or injection_attempt:
        trajectory = "watch"
    else:
        trajectory = "normal"

    return {
        "caseId": case_id,
        "requestDecision": request_decision,
        "toolDecision": tool_decision,
        "route": route,
        "trajectory": trajectory,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "agent-harness-classifier",
        "profile": PROFILE,
        "status": "ready",
        "post_endpoint": "/classify",
        "policy_version": "agent-harness-2026-09",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify")
async def classify(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return JSONResponse({"error": "Content-Type must be application/json"}, status_code=415)

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"error": "Request body exceeds 2 MiB"}, status_code=413)
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "JSON body must be an object"}, status_code=400)

    cases = _get(payload, "cases", default=None)
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        return JSONResponse({"error": "cases must be a non-empty array of at most 96 cases"}, status_code=400)

    ids = []
    for case in cases:
        if not isinstance(case, dict):
            return JSONResponse({"error": "Every case must be an object"}, status_code=400)
        case_id = _get(case, "caseId", "case_id", "id")
        if case_id is None or not str(case_id).strip():
            return JSONResponse({"error": "Every case must have a caseId"}, status_code=400)
        ids.append(str(case_id))
    if len(ids) != len(set(ids)):
        return JSONResponse({"error": "Duplicate case IDs in request"}, status_code=400)

    results = [classify_case(case) for case in cases]
    response = {
        "profile": _get(payload, "profile", default=PROFILE),
        "batchId": _get(payload, "batchId", "batch_id", default=None),
        "results": results,
    }
    return JSONResponse(response, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
