#!/usr/bin/env python3
"""OpenAI-compatible bridge from OmniRoute to a Claude Pro/Max subscription.

Wraps the official Nous plugin `hermes-plugin-claude-subscription-directsdk` (vendored in ./plugin,
pinned commit) so every request runs through the unmodified official `claude` CLI, billed against
the account's Agent SDK allowance. OmniRoute talks to this bridge as a plain OpenAI-compatible
provider on the synthpulse docker bridge.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream and non-stream).
Auth: Bearer CLAUDESUB_BRIDGE_KEY. No prompt content is logged.
"""
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'plugin'))
sys.path.insert(0, os.environ.get('HERMES_AGENT_DIR', str(Path.home() / '.hermes/hermes-agent')))

from account_pool import AccountPool, AccountsUnavailable

import directsdk  # noqa: E402  (vendored plugin, flat import path)

HOST = os.environ.get('CLAUDESUB_BRIDGE_HOST', '127.0.0.1')
PORT = int(os.environ.get('CLAUDESUB_BRIDGE_PORT', '20149'))
KEY = os.environ.get('CLAUDESUB_BRIDGE_KEY', '')
MAX_CONCURRENT = int(os.environ.get('CLAUDESUB_BRIDGE_MAX_CONCURRENT', '4'))
QUEUE_WAIT = float(os.environ.get('CLAUDESUB_BRIDGE_QUEUE_WAIT', '20'))
READ_TIMEOUT = float(os.environ.get('CLAUDESUB_BRIDGE_READ_TIMEOUT', '180'))
MISSING_CLI_WAIT = float(os.environ.get('CLAUDESUB_BRIDGE_MISSING_CLI_WAIT', '15'))

MODELS = {
    'claude-haiku-5-5': 1_000_000,
    'claude-sonnet-5-5': 1_000_000,
    'claude-sonnet-5': 1_000_000,
    'claude-opus-5-5': 1_000_000,
    'claude-opus-5': 1_000_000,
    'claude-opus-4-8': 1_000_000,
    'claude-fable-5-1': 1_000_000,
    'claude-haiku-4-5': 200_000,
}

log = logging.getLogger('claudesub-bridge')
slots = threading.BoundedSemaphore(MAX_CONCURRENT)

# Never let an inherited API key or custom endpoint turn this into a pay-per-token route.
CHILD_ENV = {k: v for k, v in os.environ.items() if k not in (
    'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_FOUNDRY_API_KEY',
    'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', 'CLAUDESUB_BRIDGE_KEY')}
client = directsdk.Client(env=CHILD_ENV, timeout=READ_TIMEOUT)
POOL_CONFIG = os.environ.get('CLAUDESUB_BRIDGE_ACCOUNTS_FILE')
pool = AccountPool(json.loads(Path(POOL_CONFIG).read_text()),
                   os.environ.get('CLAUDESUB_BRIDGE_POOL_STATE', str(HERE / 'account-cooldowns.json')),
                   directsdk.Client, CHILD_ENV, READ_TIMEOUT) if POOL_CONFIG else None


ALWAYS_THINKING = ('claude-opus-5-5', 'claude-fable-5-1')  # thinking cannot be disabled (400)
_request_body = directsdk.request_body


def request_body_with_visible_thinking(kwargs):
    """Opus 5.5 / Fable / Opus 5 / Sonnet 5 default to thinking display "omitted": thinking runs and
    is billed, but streams back empty, so the WebUI shows no reasoning. Ask for summaries instead."""
    encoded, manifest, names = _request_body(kwargs)
    body = json.loads(encoded)
    model = str(kwargs.get('model') or '').removesuffix('[1m]')
    thinking = body.get('thinking') or {}
    if thinking.get('type') == 'disabled' and model in ALWAYS_THINKING:
        body.pop('thinking', None)
        body.pop('context_management', None)
        thinking = {}
    if thinking.get('type') != 'disabled' and model != 'claude-haiku-4-5':
        body['thinking'] = {'type': 'adaptive', 'display': 'summarized'}
    return json.dumps(body, separators=(',', ':'), allow_nan=False), manifest, names


directsdk.request_body = request_body_with_visible_thinking

# Optional cache diagnostics: per upstream request, dump only hashes and cache markers (no content).
DEBUG_DIR = os.environ.get('CLAUDESUB_BRIDGE_DEBUG_DIR')
if DEBUG_DIR:
    import http.client
    _orig_request = http.client.HTTPConnection.request

    def _dumping_request(self, method, url, body=None, headers={}, **kw):
        if isinstance(body, (bytes, str)) and '/v1/messages' in url:
            try:
                d = json.loads(body)
                h = lambda o: hashlib.sha1(json.dumps(o, sort_keys=True).encode()).hexdigest()[:10]
                blocks = lambda c: [(b.get('type'), h({k: v for k, v in b.items() if k != 'cache_control'}), bool(b.get('cache_control')))
                                    for b in (c if isinstance(c, list) else [{'type': 'text', 'text': c}])]
                dump = {'system': blocks(d.get('system') or []), 'tools': h(d.get('tools')),
                        'messages': [(m['role'], blocks(m['content'])) for m in d.get('messages', [])]}
                if os.environ.get('CLAUDESUB_BRIDGE_DEBUG_FULL') == '1':  # test instances only: full content
                    dump['full'] = d.get('messages', [])
                Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
                (Path(DEBUG_DIR) / f'{time.time():.3f}.json').write_text(json.dumps(dump))
            except Exception as error:  # noqa: BLE001
                log.warning('debug dump failed: %s', error)
        return _orig_request(self, method, url, body, headers, **kw)

    http.client.HTTPConnection.request = _dumping_request


def cli_ready():
    """`claude` resolves to a complete executable; a self-update can briefly leave it missing
    or half written ("Exec format error")."""
    resolved = directsdk.resolve_claude(client.command, CHILD_ENV)
    if resolved is None:
        return False
    try:
        path = os.path.realpath(resolved[0])
        with open(path, 'rb') as handle:
            magic = handle.read(4)
        return magic == b'\x7fELF' and os.path.getsize(path) > 50_000_000 or magic[:2] == b'#!'
    except OSError:
        return False


class BadRequest(ValueError):
    pass


TOOL_NAME_OK = re.compile(r'[A-Za-z0-9_-]{1,50}')


def alias_tool(name):
    """Plugin requires ASCII names of at most 50 chars; long MCP tool names get a stable alias."""
    if TOOL_NAME_OK.fullmatch(name):
        return name
    clean = re.sub(r'[^A-Za-z0-9_-]', '_', name)
    return clean[:41] + '_' + hashlib.sha1(name.encode()).hexdigest()[:8]


def safe_id(value):
    value = str(value or '')
    return value if re.fullmatch(r'[A-Za-z0-9_-]+', value) else 'id_' + hashlib.sha1(value.encode()).hexdigest()[:24]


def text_of(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') in ('text', 'input_text', 'output_text'))


def norm_content(content):
    """OpenAI content parts to what the plugin accepts (text, base64 image_url)."""
    if content is None or isinstance(content, str):
        return content
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get('type')
        if kind in ('text', 'input_text', 'output_text'):
            if part.get('text'):
                out.append({'type': 'text', 'text': part['text']})
        elif kind == 'image_url':
            url = (part.get('image_url') or {}).get('url', '') if isinstance(part.get('image_url'), dict) else part.get('image_url', '')
            if url.startswith('data:') and ';base64,' in url:
                out.append({'type': 'image_url', 'image_url': {'url': url}})
            else:
                out.append({'type': 'text', 'text': f'[image not forwarded: {url[:200]}]'})
        elif kind in ('image', 'document'):
            out.append(part)
    return out


def translate(body):
    """Map an arbitrary OpenAI chat request onto the plugin's strict create() kwargs."""
    model = str(body.get('model') or '')
    model = model.split('/', 1)[1] if '/' in model else model
    if model not in MODELS:
        raise BadRequest(f'unknown model {model!r}; available: {", ".join(MODELS)}')
    if body.get('n', 1) not in (1, None):
        raise BadRequest('only n=1 is supported')

    tool_choice = body.get('tool_choice')
    tools_in = [] if tool_choice == 'none' else (body.get('tools') or [])
    name_map, tools = {}, []
    for tool in tools_in:
        if tool.get('type') != 'function':
            continue
        f = dict(tool['function'])
        alias = alias_tool(f['name'])
        if alias in name_map:
            continue
        name_map[alias] = f['name']
        f['name'] = alias
        f.pop('strict', None)
        tools.append({'type': 'function', 'function': f})
    reverse = {v: k for k, v in name_map.items()}

    system, messages = [], []
    for m in body.get('messages') or []:
        role = m.get('role')
        if role in ('system', 'developer'):
            if not messages:
                system.append(text_of(m.get('content')))
            else:  # plugin requires system first; later system notes become user text
                messages.append({'role': 'user', 'content': '[system note]\n' + text_of(m.get('content'))})
        elif role == 'assistant':
            calls = []
            for tc in m.get('tool_calls') or []:
                fn = tc.get('function') or {}
                calls.append({'id': safe_id(tc.get('id')), 'type': 'function',
                              'function': {'name': reverse.get(fn.get('name'), alias_tool(fn.get('name', 'tool'))),
                                           'arguments': fn.get('arguments') or '{}'}})
            content = norm_content(m.get('content'))
            if not calls and not text_of(content).strip() and not (isinstance(content, list) and content):
                continue  # empty assistant turns have nothing to replay
            messages.append({'role': 'assistant', 'content': content, 'tool_calls': calls or None})
        elif role == 'tool':
            content = m.get('content')
            messages.append({'role': 'tool', 'tool_call_id': safe_id(m.get('tool_call_id')),
                             'content': content if isinstance(content, str) else text_of(content)})
        elif role in ('user', 'function'):
            messages.append({'role': 'user', 'content': norm_content(m.get('content')) or ''})
    if system:
        messages.insert(0, {'role': 'system', 'content': '\n\n'.join(s for s in system if s)})
    if not messages or messages[-1]['role'] not in ('user', 'tool'):
        raise BadRequest('history must end with a user or tool message (assistant prefill is unsupported)')

    kwargs = {'model': model + ('[1m]' if MODELS[model] == 1_000_000 else ''), 'messages': messages, 'stream': True}
    if tools:
        kwargs['tools'] = tools
    max_tokens = body.get('max_completion_tokens') or body.get('max_tokens')
    if isinstance(max_tokens, int) and max_tokens > 0:
        kwargs['max_tokens'] = max_tokens
    if body.get('stop'):
        kwargs['stop'] = body['stop']
    reasoning = body.get('reasoning') if isinstance(body.get('reasoning'), dict) else {}
    effort = body.get('reasoning_effort') or reasoning.get('effort')
    if effort:
        kwargs['extra_body'] = {'reasoning': {'enabled': effort != 'none', 'effort': effort}}
    rf = body.get('response_format')
    if isinstance(rf, dict) and rf.get('type') == 'json_schema':
        kwargs['response_format'] = rf
    return model, kwargs, name_map


def status_for(error):
    if isinstance(error, AccountsUnavailable):
        return 429, 'rate_limit_error'
    text = str(error)
    if isinstance(error, (BadRequest, ValueError)) and not isinstance(error, directsdk.ClaudeCodeLoggedOut):
        return 400, 'invalid_request_error'
    if isinstance(error, (directsdk.ClaudeCodeLoggedOut, directsdk.ClaudeCodeMissing)):
        return 503, 'provider_unavailable'
    if 'status 429' in text or 'rate_limit' in text or 'usage limit' in text.lower() or 'limit reached' in text.lower():
        return 429, 'rate_limit_error'
    if isinstance(error, TimeoutError):
        return 504, 'timeout'
    return 502, 'upstream_error'


def clean_usage(usage):
    if not usage:
        return None
    u = usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage)
    return {'prompt_tokens': u.get('prompt_tokens', 0), 'completion_tokens': u.get('completion_tokens', 0),
            'total_tokens': u.get('total_tokens', 0),
            'prompt_tokens_details': {'cached_tokens': (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)},
            'completion_tokens_details': {'reasoning_tokens': (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'claudesub-bridge/1.0'

    def log_message(self, *_):
        pass

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        if getattr(self, 'pool_account', None):
            self.send_header('x-claudesub-account', self.pool_account)
        if getattr(self, 'retry_after', None):
            self.send_header('Retry-After', str(self.retry_after))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, kind, message):
        self._json(code, {'error': {'message': message, 'type': kind, 'code': code}})

    def _authorized(self):
        return bool(KEY) and self.headers.get('Authorization', '') == f'Bearer {KEY}'

    def do_GET(self):
        self.pool_account = None
        self.retry_after = None
        if self.path == '/health':
            return self._json(200, {'ok': True, 'busy': MAX_CONCURRENT - slots._value, 'max': MAX_CONCURRENT})
        if not self._authorized():
            return self._error(401, 'authentication_error', 'invalid bridge key')
        if self.path.rstrip('/') == '/v1/accounts':
            return self._json(200, {'accounts': pool.status() if pool else []})
        if self.path.rstrip('/') == '/v1/models':
            return self._json(200, {'object': 'list', 'data': [
                {'id': m, 'object': 'model', 'owned_by': 'anthropic-subscription', 'context_length': c} for m, c in MODELS.items()]})
        self._error(404, 'not_found', 'not found')

    def do_POST(self):
        self.pool_account = None
        self.retry_after = None
        if not self._authorized():
            return self._error(401, 'authentication_error', 'invalid bridge key')
        if self.path.rstrip('/') != '/v1/chat/completions':
            return self._error(404, 'not_found', 'not found')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            model, kwargs, name_map = translate(body)
        except (ValueError, json.JSONDecodeError) as error:
            return self._error(400, 'invalid_request_error', str(error))
        if not slots.acquire(timeout=QUEUE_WAIT):
            return self._error(429, 'rate_limit_error', f'bridge busy: {MAX_CONCURRENT} concurrent claude runs')
        started, outcome, usage = time.monotonic(), 'error', None
        try:
            outcome, usage = self._complete(model, kwargs, name_map, bool(body.get('stream')),
                                            (body.get('stream_options') or {}).get('include_usage', True))
        finally:
            slots.release()
            u = usage or {}
            log.info('account=%s model=%s msgs=%d tools=%d stream=%s outcome=%s in=%s cached=%s out=%s secs=%.1f',
                     self.pool_account or 'default', model, len(kwargs['messages']), len(kwargs.get('tools') or []), bool(body.get('stream')), outcome,
                     u.get('prompt_tokens'), (u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                     u.get('completion_tokens'), time.monotonic() - started)

    def _complete(self, model, kwargs, name_map, stream_out, include_usage):
        rid = 'chatcmpl-cs' + os.urandom(8).hex()
        created = int(time.time())
        headers_sent = False
        stream = None
        text, reasoning, calls, finish, usage = [], [], [], 'stop', None

        def sse(delta, finish_reason=None, usage_out=None):
            chunk = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model,
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]}
            if usage_out is not None:
                chunk['usage'] = usage_out
            self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
            self.wfile.flush()

        # Claude Code updates itself by swapping its binary; for a few seconds `claude` is missing
        # from PATH. Wait it out instead of answering 503, which makes OmniRoute cool the
        # connection down and leaves a Codex-less chain without any model.
        for _ in range(int(MISSING_CLI_WAIT)):
            if cli_ready():
                break
            time.sleep(1)
        try:
            stream = (pool.stream(kwargs, lambda ident: setattr(self, 'pool_account', ident))
                      if pool else client.chat.completions.create(**kwargs))
            for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta
                if stream_out and not headers_sent:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    if self.pool_account:
                        self.send_header('x-claudesub-account', self.pool_account)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    headers_sent = True
                    sse({'role': 'assistant', 'content': ''})
                if getattr(delta, 'content', None):
                    text.append(delta.content)
                    if stream_out:
                        sse({'content': delta.content})
                if getattr(delta, 'reasoning_content', None):
                    reasoning.append(delta.reasoning_content)
                    if stream_out:
                        sse({'reasoning_content': delta.reasoning_content})
                if choice.finish_reason:
                    finish = choice.finish_reason
                    usage = clean_usage(chunk.usage)
                    for i, tc in enumerate(getattr(delta, 'tool_calls', None) or []):
                        tc = tc.model_dump() if hasattr(tc, 'model_dump') else tc
                        name = name_map.get(tc['function']['name'], tc['function']['name'])
                        calls.append({'index': i, 'id': tc['id'], 'type': 'function',
                                      'function': {'name': name, 'arguments': tc['function']['arguments']}})
        except (BrokenPipeError, ConnectionResetError):
            return 'client_gone', usage
        except Exception as error:  # noqa: BLE001  (every failure must become an HTTP status)
            code, kind = status_for(error)
            self.retry_after = getattr(error, 'retry_after', None)
            log.warning('request failed (%s): %s', code, str(error)[:500])
            if headers_sent:
                try:
                    self.wfile.write(f'data: {json.dumps({"error": {"message": str(error)[:500], "type": kind, "code": code}})}\n\n'.encode())
                    self.close_connection = True
                except OSError:
                    pass
            else:
                self._error(code, kind, str(error)[:1000])
            return f'http_{code}', usage
        finally:
            if stream is not None:
                stream.close()

        if stream_out:
            try:
                if calls:
                    sse({'tool_calls': calls})
                sse({}, finish)
                if include_usage and usage:
                    chunk = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [], 'usage': usage}
                    self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
                self.close_connection = True
            except OSError:
                return 'client_gone', usage
            return 'ok', usage
        message = {'role': 'assistant', 'content': ''.join(text) or None}
        if reasoning:
            message['reasoning_content'] = ''.join(reasoning)
        if calls:
            message['tool_calls'] = [{k: v for k, v in c.items() if k != 'index'} for c in calls]
        self._json(200, {'id': rid, 'object': 'chat.completion', 'created': created, 'model': model,
                         'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage})
        return 'ok', usage


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not KEY:
        sys.exit('CLAUDESUB_BRIDGE_KEY is required')
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    log.info('listening on %s:%d, max %d concurrent', HOST, PORT, MAX_CONCURRENT)
    server.serve_forever()


if __name__ == '__main__':
    main()
