"""Ordered Claude CLI profiles with bounded, pre-output failover.

No OAuth tokens are read here. Each unmodified CLI owns its profile's auth.
Cooldowns contain only account IDs, reasons and timestamps, never prompts.
"""
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time


class AccountsUnavailable(RuntimeError):
    def __init__(self, retry_after):
        super().__init__('All Claude accounts are unavailable; retry after %s seconds' % retry_after)
        self.retry_after = retry_after


def failure_kind(error):
    text = str(error).lower()
    if re.search(r'\bstatus (?:code[=: ]*)?429\b', text) or any(s in text for s in (
            'rate_limit', 'usage limit', 'limit reached', "hit your weekly limit", "hit your limit")):
        return 'quota'
    if type(error).__name__ == 'ClaudeCodeLoggedOut' or any(s in text for s in (
            'status 401', 'authentication_failed', 'invalid oauth', 'token has expired')):
        return 'auth'
    if any(s in text for s in ('status 529', 'status 503', 'overloaded_error')):
        return 'overloaded'
    return None


def visible(chunk):
    for choice in chunk.choices:
        d = choice.delta
        if (getattr(d, 'content', None) or getattr(d, 'reasoning_content', None)
                or getattr(d, 'tool_calls', None) or choice.finish_reason):
            return True
    return False


class AccountPool:
    def __init__(self, config, state_path, client_factory, env, timeout, clock=time.time):
        self.clock = clock
        self.lock = threading.RLock()
        self.state_path = Path(state_path)
        self.accounts = []
        ids, paths = set(), set()
        for item in config['accounts']:
            if not item.get('enabled', True):
                continue
            ident = item['id']
            path = str(Path(item['config_dir']).expanduser().resolve())
            if not re.fullmatch(r'[a-zA-Z0-9_-]+', ident) or ident in ids or path in paths:
                raise ValueError('Account IDs and profile paths must be unique')
            ids.add(ident); paths.add(path)
            child_env = dict(env)
            child_env['CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR'] = path
            child_env['CLAUDE_CONFIG_DIR'] = path
            self.accounts.append({**item, 'path': Path(path),
                                  'client': client_factory(env=child_env, timeout=timeout)})
        if not self.accounts:
            raise ValueError('Account pool needs at least one enabled account')
        self.cooldowns = {'quota': 300, 'auth': 300, 'overloaded': 30}
        for kind in self.cooldowns:
            self.cooldowns[kind] = max(1, min(3600, int(config.get('cooldowns', {}).get(kind, self.cooldowns[kind]))))
        try:
            self.state = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            self.state = {}

    def _identity(self, account):
        path = account['path']
        try:
            # A configured email mismatch must never silently bill a different account.
            profile_path = path / '.claude.json'
            if path == Path.home() / '.claude':
                profile_path = Path.home() / '.claude.json'
            profile = json.loads(profile_path.read_text()).get('oauthAccount', {})
            credential_stat = (path / '.credentials.json').stat()
            if profile.get('emailAddress', '').lower() != account['email'].lower():
                return None, None
            identity = profile.get('accountUuid') or profile.get('emailAddress', '').lower()
            stamp = f'{credential_stat.st_mtime_ns}:{credential_stat.st_size}'
            return identity, stamp
        except (OSError, ValueError):
            return None, None

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(dir=self.state_path.parent)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(self.state, handle)
            os.replace(name, self.state_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def status(self):
        result = []
        with self.lock:
            seen = set()
            for account in self.accounts:
                identity, stamp = self._identity(account)
                old = self.state.get(account['id'], {})
                until = old.get('until', 0)
                if old.get('reason') == 'auth' and old.get('stamp') != stamp:
                    until = 0
                reason = ('login_required' if not identity else 'duplicate_account' if identity in seen
                          else old.get('reason') if until > self.clock() else 'ready')
                if identity:
                    seen.add(identity)
                result.append({'id': account['id'], 'email': account['email'], 'status': reason,
                               'retry_after': max(0, int(until - self.clock()) + 1) if until > self.clock() else 0})
        return result

    def stream(self, kwargs, selected=lambda ident: None):
        """Retry only before yielding any content, reasoning, tool calls or finish."""
        attempted = set()
        last_error = None
        for account in self.accounts:
            with self.lock:
                identity, stamp = self._identity(account)
                current = next(s for s in self.status() if s['id'] == account['id'])
                if current['status'] != 'ready' or identity in attempted:
                    continue
                attempted.add(identity)
            upstream = None
            delivered = False
            pending = []
            try:
                upstream = account['client'].chat.completions.create(**kwargs)
                for chunk in upstream:
                    if not delivered:
                        pending.append(chunk)
                        if not visible(chunk):
                            continue
                        selected(account['id'])
                        delivered = True
                        for first in pending:
                            yield first
                        pending.clear()
                    else:
                        yield chunk
                if not delivered:
                    raise RuntimeError('Claude account returned no completed response')
                return
            except Exception as error:
                kind = failure_kind(error)
                if kind:
                    with self.lock:
                        self.state[account['id']] = {'until': self.clock() + self.cooldowns[kind],
                                                    'reason': kind, 'stamp': stamp}
                        self._save()
                if delivered or not kind:
                    raise
                last_error = error
            finally:
                if upstream is not None:
                    upstream.close()
        waits = [s['retry_after'] for s in self.status() if s['retry_after']]
        raise AccountsUnavailable(min(waits) if waits else 60) from last_error
