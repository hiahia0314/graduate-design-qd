"""DeepSeek official Chat Completions adapter; no SDK, secret logging or redirects."""
import json
import math
import os
from pathlib import Path
import shlex
import socket
import urllib.error
import urllib.request


class DeepSeekError(RuntimeError):
    def __init__(self, code, *, retryable=False):
        super().__init__(code)
        self.code, self.retryable = code, retryable


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def configuration(env_file='.env'):
    """Literal .env assignments only; process environment takes precedence."""
    known = {'DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL', 'DEEPSEEK_TIMEOUT', 'DEEPSEEK_MAX_TOKENS', 'DEEPSEEK_MAX_CALLS'}
    values = {}
    path = Path(env_file)
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.removeprefix('export ').partition('=')
            if key.strip() not in known:
                continue
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                raise DeepSeekError('invalid_env_file') from None
            if not separator or len(parts) > 1:
                raise DeepSeekError('invalid_env_file')
            values[key.strip()] = parts[0] if parts else ''
    values.update({key: os.environ[key] for key in known if key in os.environ})
    return values


class DeepSeekProvider:
    endpoint = 'https://api.deepseek.com/chat/completions'

    def __init__(self, api_key, *, model='deepseek-flash', timeout=60, max_tokens=4096, max_calls=60, transport=None):
        if not isinstance(api_key, str) or not api_key.strip() or api_key.strip() == 'replace-with-your-key':
            raise DeepSeekError('missing_deepseek_api_key')
        if any(c.isspace() for c in api_key):
            raise DeepSeekError('invalid_deepseek_api_key')
        if not isinstance(model, str) or not model.strip():
            raise DeepSeekError('missing_model')
        if not math.isfinite(timeout) or not 1 <= timeout <= 120:
            raise DeepSeekError('invalid_timeout')
        if type(max_tokens) is not int or not 128 <= max_tokens <= 16384:
            raise DeepSeekError('invalid_max_tokens')
        if type(max_calls) is not int or not 1 <= max_calls <= 500:
            raise DeepSeekError('invalid_max_calls')
        self._api_key = api_key
        self.model, self.timeout, self.max_tokens, self.max_calls = model, timeout, max_tokens, max_calls
        self.calls = []
        self._transport = transport or urllib.request.build_opener(NoRedirect()).open

    @property
    def audit_log(self):
        return self.calls

    @classmethod
    def from_env(cls, env_file='.env'):
        values = configuration(env_file)
        try:
            return cls(values.get('DEEPSEEK_API_KEY'), model=values.get('DEEPSEEK_MODEL', 'deepseek-flash'),
                       timeout=float(values.get('DEEPSEEK_TIMEOUT', 60)),
                       max_tokens=int(values.get('DEEPSEEK_MAX_TOKENS', 4096)),
                       max_calls=int(values.get('DEEPSEEK_MAX_CALLS', 60)))
        except (ValueError, TypeError):
            raise DeepSeekError('invalid_numeric_api_configuration') from None

    def __call__(self, stage, prompt):
        if len(self.calls) >= self.max_calls:
            raise DeepSeekError('api_call_budget_exhausted')
        if stage not in {'planning', 'hypothesis', 'expression', 'feedback', 'check'}:
            raise DeepSeekError('unknown_stage')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 200_000:
            raise DeepSeekError('invalid_prompt_size')
        payload = {'model': self.model, 'stream': False, 'max_tokens': self.max_tokens,
                   'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'system', 'content': 'Return one JSON object only. Follow the requested schema. For expression stage obey the requested maximum factor count.'},
                                {'role': 'user', 'content': prompt}]}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._api_key}, method='POST')
        audit = {'stage': stage, 'requested_model': self.model, 'status': 'started'}
        self.calls.append(audit)
        try:
            with self._transport(request, timeout=self.timeout) as response:
                content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise DeepSeekError('response_too_large')
            try:
                result = json.loads(content)
                choice = result['choices'][0]
                answer = choice['message']['content']
            except (ValueError, KeyError, IndexError, TypeError):
                raise DeepSeekError('invalid_api_response', retryable=True) from None
            if choice.get('finish_reason') != 'stop':
                raise DeepSeekError('incomplete_completion', retryable=True)
            if not isinstance(answer, str) or not answer.strip():
                raise DeepSeekError('empty_completion', retryable=True)
            # Keep only known scalar metadata; do not persist arbitrary HTTP content.
            if isinstance(result.get('model'), str):
                audit['returned_model'] = result['model'][:120]
            if isinstance(result.get('usage'), dict):
                audit['usage'] = {key: value for key, value in result['usage'].items()
                                  if key in {'prompt_tokens', 'completion_tokens', 'total_tokens'} and type(value) is int}
            audit['status'] = 'completed'
            return answer
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            audit.update(status='failed', http_status=code)
            raise DeepSeekError(f'http_{code}', retryable=code in {408, 429, 500, 502, 503, 504}) from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
            audit.update(status='failed', error='network_error')
            raise DeepSeekError('network_error', retryable=True) from None
        except DeepSeekError as exc:
            audit.update(status='failed', error=exc.code)
            raise
