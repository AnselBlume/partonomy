import os
import re
import json
import time
import base64
import logging
import orjson
import random
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Callable, Any
from pprint import pprint

import orjson
import jsonargparse
from openai import OpenAI
from openai import APIError
from openai._exceptions import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError
)

from qa_generation.qa_pair import QAPair, AnswerType
from qa_generation.question_type import QuestionType
from evaluation.evaluators.mc_text import MCTextEvaluator
from evaluation.evaluators.part_text import PartTextEvaluator
import numpy as np

logger = logging.getLogger(__name__)

logging.getLogger('httpx').setLevel(logging.WARNING) # Suppress lots of info about HTTP requests

# ----------------------------
# Retry / Backoff utilities
# ----------------------------
def _sleep_with_jitter(base: float, factor: float, attempt: int, cap: float, jitter: float) -> None:
    backoff = min(cap, base * (factor ** attempt))
    # Full jitter
    sleep_for = backoff + random.uniform(0, jitter)
    time.sleep(max(0.0, sleep_for))


def retry_api_call(
    fn: Callable[..., Any],
    *,
    max_retries: int = 8,
    base: float = 0.5,
    factor: float = 2.0,
    cap: float = 20.0,
    jitter: float = 0.2,
    log_prefix: str = '',
    respect_retry_after: bool = True,
    **kwargs,
) -> Any:
    '''
    Wrap any OpenAI SDK call with resilient retries.
    Retries on: 429, 5xx, timeouts, connection errors.
    '''
    attempt = 0
    while True:
        try:
            return fn(**kwargs)
        except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError, APIError) as e:
            attempt += 1
            is_last = attempt > max_retries

            # Pull HTTP status (if available)
            status = getattr(e, 'status_code', None) or getattr(e, 'status', None)
            message = str(e)

            # If it's a non-retryable 4xx (not 429), bail out early
            if isinstance(e, APIError) and status and 400 <= status < 500 and status != 429:
                logger.error(f'{log_prefix}Non-retryable API error {status}: {message}')
                raise

            # Respect Retry-After if present
            retry_after = None
            if respect_retry_after:
                headers = getattr(e, 'response_headers', None) or getattr(e, 'headers', None) or {}
                if isinstance(headers, dict):
                    ra = headers.get('retry-after') or headers.get('Retry-After')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except Exception:
                            retry_after = None

            if is_last:
                logger.error(f'{log_prefix}Giving up after {attempt-1} retries. Last error: {message}')
                raise

            if retry_after is not None:
                wait = min(cap, retry_after + random.uniform(0, jitter))
                logger.warning(f'{log_prefix}Rate-limited (429). Respecting Retry-After={retry_after:.2f}s, sleeping {wait:.2f}s...')
                time.sleep(wait)
            else:
                _sleep_with_jitter(base, factor, attempt-1, cap, jitter)
                logger.warning(f'{log_prefix}Retrying (attempt {attempt}/{max_retries}) after error: {message}')
        except Exception as e:
            # Unknown/unexpected exception: don't retry unless you explicitly want to
            logger.error(f'{log_prefix}Non-OpenAI error: {e}')
            raise


# ----------------------------
# Core helpers
# ----------------------------
def load_qa_pairs(path: str) -> List[QAPair]:
    with open(path, 'rb') as f:
        data = orjson.loads(f.read())
    return [QAPair.from_dict(d) for d in data]


def _encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower().lstrip('.') or 'jpeg'
    mime = f'image/{"jpeg" if ext in {"jpg", "jpeg"} else ext}'
    return f'data:{mime};base64,{encoded}'


def _options_string(answer_choices: list[str]) -> str:
    '''
    Converts a list of answer choices into a string of the form:
    1) choice_1
    2) choice_2
    ...
    n) choice_n
    '''
    return '\n'.join(f'{i+1}) {choice}' for i, choice in enumerate(answer_choices))


def build_messages(
    question: str,
    answer_choices: list[str],
    image_path: str,
    *,
    include_image: bool = True,
    question1_answer_num: int = None,
    question2: str = None,
    answer_choices2: list[str] = None
) -> List[dict]:
    options = _options_string(answer_choices)
    prompt = (
        f'{question}\n\nAnswer choices:\n{options}\n'
        'Respond with <answer>N</answer> where N is the number corresponding to the best answer choice. Say nothing else.'
    )

    if include_image:
        image_content = {'type': 'image_url', 'image_url': {'url': _encode_image(image_path)}}
    else:
        # logging only
        image_content = {'type': 'image_path', 'image_url': {'url': image_path}}

    content = [image_content, {'type': 'text', 'text': prompt}]
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': content},
    ]

    if question2 is not None:
        assert answer_choices2 is not None and question1_answer_num is not None

        conversation.append({
            'role': 'assistant',
            'content': f'<answer>{question1_answer_num}</answer>'
        })

        options2 = _options_string(answer_choices2)
        conversation.append({
            'role': 'user',
            'content': f'{question2}\n\nAnswer choices:\n{options2}\nRespond with <answer>N</answer> where N is the number of the correct choice. Say nothing else.'
        })

    return conversation


def parse_answer(text: str) -> int:
    match = re.search(r'<answer>(\d+)</answer>', text or '')
    if match:
        return int(match.group(1)) - 1
    digits = re.findall(r'\d+', text or '')
    return int(digits[0]) - 1 if digits else -1


# ----------------------------
# Online requests (with retries)
# ----------------------------
def send_online_requests(
    client: OpenAI,
    pairs: List[QAPair],
    model: str,
    log_every: int = 10,
    requests_per_minute: int = 5000,
    **kwargs,
) -> List[Dict]:
    kwargs = {'max_tokens': kwargs.pop('max_tokens', 128), **kwargs}
    # Client-side pacing to reduce 429s
    sleep_s = 60.0 / requests_per_minute if requests_per_minute > 0 else 0

    results = []
    logger.info(f'Sending {len(pairs)} requests')
    for i, pair in enumerate(tqdm(pairs)):
        if pair.question_type == QuestionType.WHOLE_TO_PART:
            question = pair.object_question
            answer_choices = pair.object_answer_choices
            answer_types = pair.object_answer_types
        else:
            question = pair.question
            answer_choices = pair.answer_choices
            answer_types = pair.answer_types

        round1_conv_send = build_messages(question, answer_choices, pair.image_path, include_image=True)
        round1_conv_log = build_messages(question, answer_choices, pair.image_path, include_image=False)
        logger.debug(f'Request {i} part: {round1_conv_log}')

        # Optional idempotency key per call
        # req_opts = {'timeout': 60.0, 'idempotency_key': f'{i}_round1'}
        req_opts = {'timeout': 60.0}
        round1_resp = retry_api_call(
            client.chat.completions.create,
            log_prefix=f'[round 1 i={i}] ',
            model=model,
            messages=round1_conv_send,
            **req_opts,
            **kwargs
        )
        round1_content = (round1_resp.choices[0].message.content or '').strip()
        if sleep_s:
            time.sleep(sleep_s)

        # Round 2 for WHOLE_TO_PART and PART_TO_WHOLE questions
        round2_content = None
        round2_conv_log: Optional[List[dict]] = None

        if pair.question_type in [QuestionType.WHOLE_TO_PART, QuestionType.PART_TO_WHOLE]:
            if pair.question_type == QuestionType.WHOLE_TO_PART:
                question2 = pair.question
                answer_choices2 = pair.answer_choices
            else: # PART_TO_WHOLE
                question2 = pair.object_question
                answer_choices2 = pair.object_answer_choices

            # Round 2 question should be a follow-up question with the correct answer choice from round 1
            question1_answer_num = answer_types.index(AnswerType.CORRECT) + 1

            messages_kwargs = dict(
                question=question,
                answer_choices=answer_choices,
                image_path=pair.image_path,
                question1_answer_num=question1_answer_num,
                question2=question2,
                answer_choices2=answer_choices2
            )

            round2_conv_send = build_messages(**messages_kwargs, include_image=True)
            round2_conv_log = build_messages(**messages_kwargs, include_image=False)

            logger.debug(f'Request {i} object: {round2_conv_log}')

            # req_opts = {'timeout': 60.0, 'idempotency_key': f'{i}_round2'}
            req_opts = {'timeout': 60.0}
            round2_resp = retry_api_call(
                client.chat.completions.create,
                log_prefix=f'[round 2 i={i}] ',
                model=model,
                messages=round2_conv_send,
                **req_opts,
                **kwargs,
            )
            round2_content = (round2_resp.choices[0].message.content or '').strip()
            if sleep_s:
                time.sleep(sleep_s)

        # Log progress every n requests
        if i % log_every == 0:
            logger.info(f'Request {i} response round1: {round1_content}')
            if round2_content is not None:
                logger.info(f'Request {i} response round2: {round2_content}')

        results.append({
            'round1_request': round1_conv_log,
            'round1_response': round1_content,
            'round2_request': round2_conv_log,
            'round2_response': round2_content
        })
    return results


# ----------------------------
# Batch mode (with retries)
# ----------------------------
def create_batch_file(
    pairs: List[QAPair], model: str, batch_path: str, **kwargs
) -> Tuple[str, List[Dict[str, list]]]:
    body_defaults = {'max_tokens': kwargs.pop('max_tokens', 128)}
    request_logs: List[Dict[str, list]] = []
    with open(batch_path, 'w', encoding='utf-8') as f:
        for idx, pair in enumerate(pairs):
            msg_part_send = build_messages(pair.question, pair.answer_choices, pair.image_path, include_image=True)
            msg_part_log = build_messages(pair.question, pair.answer_choices, pair.image_path, include_image=False)
            req_part = {
                'custom_id': f'{idx}_part',
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {'model': model, 'messages': msg_part_send, **body_defaults, **kwargs},
            }
            f.write(json.dumps(req_part, ensure_ascii=False) + '\n')

            req_entry = {'part_request': msg_part_log, 'object_request': None}

            if getattr(pair, 'object_question', None):
                msg_obj_send = build_messages(pair.object_question, pair.object_answer_choices, pair.image_path, include_image=True)
                msg_obj_log = build_messages(pair.object_question, pair.object_answer_choices, pair.image_path, include_image=False)
                req_obj = {
                    'custom_id': f'{idx}_object',
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {'model': model, 'messages': msg_obj_send, **body_defaults, **kwargs},
                }
                f.write(json.dumps(req_obj, ensure_ascii=False) + '\n')
                req_entry['object_request'] = msg_obj_log

            request_logs.append(req_entry)
    return batch_path, request_logs


def run_batch_inference(client: OpenAI, pairs: List[QAPair], model: str, **kwargs) -> List[Dict]:
    raise NotImplementedError('Batch mode is not carefully tested yet')

    tmp_jsonl = 'batch_requests.jsonl'
    tmp_jsonl, request_logs = create_batch_file(pairs, model, tmp_jsonl, **kwargs)

    # Upload input file
    with open(tmp_jsonl, 'rb') as f:
        file_obj = retry_api_call(
            client.files.create,
            log_prefix='[batch upload] ',
            file=f,
            purpose='batch',
        )

    # Create batch job
    batch = retry_api_call(
        client.batches.create,
        log_prefix='[batch create] ',
        completion_window='24h',
        endpoint='/v1/chat/completions',
        input_file_id=file_obj.id,
    )
    logger.info(f'Created batch {batch.id}')

    # Poll until completion with global cap
    poll_start = time.time()
    max_poll_seconds = 60 * 60 * 6  # 6 hours cap (batch window is 24h)
    while batch.status not in {'completed', 'failed', 'expired', 'canceled'}:
        if time.time() - poll_start > max_poll_seconds:
            raise TimeoutError(f'Batch {batch.id} polling exceeded {max_poll_seconds/3600:.1f} hours.')
        time.sleep(5)
        batch = retry_api_call(
            client.batches.retrieve,
            log_prefix='[batch poll] ',
            batch_id=batch.id,
        )
        logger.debug(f'Batch status: {batch.status}')

    if batch.status != 'completed':
        reason = getattr(batch, 'error', None) or getattr(batch, 'status', None)
        raise RuntimeError(f'Batch did not complete successfully. Final status: {batch.status}. Reason: {reason}')

    # Download output JSONL
    stream = retry_api_call(
        client.files.content,
        log_prefix='[batch download] ',
        file_id=batch.output_file_id,
    )
    payload = getattr(stream, 'text', None)
    if payload is None:
        payload = stream.read().decode('utf-8')

    lines = [ln for ln in payload.splitlines() if ln.strip()]
    raw_results = [orjson.loads(line) for line in lines]
    try:
        os.remove(tmp_jsonl)
    except Exception:
        pass

    # Organize responses per pair
    results: List[Dict[str, Optional[str]]] = [
        {'part_request': log['part_request'], 'object_request': log['object_request'],
         'part_response': None, 'object_response': None}
        for log in request_logs
    ]

    for entry in raw_results:
        cid = entry.get('custom_id', '')
        if not cid:
            logger.warning(f'[batch parse] Missing custom_id in line: {entry}')
            continue

        # Success path
        resp_body = entry.get('response', {}).get('body')
        if resp_body and 'choices' in resp_body:
            content = (resp_body['choices'][0]['message']['content'] or '').strip()
        else:
            # Error path (record the error as content so downstream sees it)
            err = entry.get('error') or entry.get('response', {}).get('error') or 'Unknown batch line error'
            content = f'[ERROR] {err}'

        try:
            idx_str, qtype = cid.split('_', 1)
            idx = int(idx_str)
        except Exception:
            logger.error(f'[batch parse] Invalid custom_id format: {cid}')
            continue

        key = f'{qtype}_response'
        if 0 <= idx < len(results):
            results[idx][key] = content
        else:
            logger.error(f'[batch parse] Index {idx} out of range for key {key}')

    return results


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(pairs: List[QAPair], responses: List[Dict[str, Optional[str]]]):
    logger.info(f'Evaluating {len(pairs)} responses')

    part_mc: Dict[QuestionType, MCTextEvaluator] = {}
    part_text: Dict[QuestionType, PartTextEvaluator] = {}
    object_mc: Dict[QuestionType, MCTextEvaluator] = {}

    def get_eval(d: Dict, key, cls):
        if key not in d:
            d[key] = cls()
        return d[key]

    annotated = []

    for pair, resp in tqdm(zip(pairs, responses)):
        qtype = pair.question_type

        # Determine correct sets for each round
        if pair.question_type == QuestionType.WHOLE_TO_PART:
            round1_answer_types = pair.object_answer_types
            round2_answer_types = pair.answer_types
            round1_mc_dict = object_mc
            round2_mc_dict = part_mc
        else:
            round1_answer_types = pair.answer_types
            round2_answer_types = pair.object_answer_types
            round1_mc_dict = part_mc
            round2_mc_dict = object_mc

        # ---------- Round 1 ----------
        round1_resp = resp.get('round1_response')
        if round1_resp is None:
            logger.warning(f'Round 1 response is None for question type {qtype}')

        round1_pred_index = parse_answer(round1_resp) if round1_resp else -1
        round1_correct_index = round1_answer_types.index(AnswerType.CORRECT)

        # Evaluate multiple choice accuracy
        get_eval(round1_mc_dict, qtype, MCTextEvaluator).update(round1_pred_index, round1_correct_index)

        # ---------- Round 2 ----------
        round2_resp = None
        round2_pred_index = None
        round2_correct_index = None

        if pair.question_type in [QuestionType.WHOLE_TO_PART, QuestionType.PART_TO_WHOLE]:
            round2_resp = resp.get('round2_response')

            if round2_resp is None:
                logger.warning(f'Round 2 response is None for question type {qtype}')

            round2_pred_index = parse_answer(round2_resp) if round2_resp else -1
            round2_correct_index = round2_answer_types.index(AnswerType.CORRECT)

            # Evaluate multiple choice accuracy
            get_eval(round2_mc_dict, qtype, MCTextEvaluator).update(round2_pred_index, round2_correct_index)

        else:
            if resp.get('round2_response') is not None:
                logger.warning(f'Round 2 response is not None for question type {qtype}')

        # --------- Evaluate part text precision/recall ---------
        correct_parts_index = round2_correct_index if pair.question_type == QuestionType.WHOLE_TO_PART else round1_correct_index
        pred_parts_index = round2_pred_index if pair.question_type == QuestionType.WHOLE_TO_PART else round1_pred_index

        gt_parts = pair.answer_parts[correct_parts_index]
        pred_parts = pair.answer_parts[pred_parts_index]

        get_eval(part_text, qtype, PartTextEvaluator).update(pred_parts, gt_parts)

        annotated.append({
            'qa_pair': {k : v for k, v in pair.to_dict().items() if k not in ['segmentations']},
            'round1_request': resp.get('round1_request'),
            'round1_response': round1_resp,
            'round1_predicted_index': round1_pred_index,
            'round1_correct_index': round1_correct_index,
            'round2_request': resp.get('round2_request'),
            'round2_response': round2_resp,
            'round2_predicted_index': round2_pred_index,
            'round2_correct_index': round2_correct_index,
        })

    metrics = {}
    for qtype, evalr in part_mc.items():
        metrics[f'{qtype.value}_part_mc'] = evalr.summarize()
    for qtype, evalr in part_text.items():
        metrics[f'{qtype.value}_part_text'] = evalr.summarize()
    for qtype, evalr in object_mc.items():
        metrics[f'{qtype.value}_object_mc'] = evalr.summarize()

    return metrics, annotated

def shuffle_answer_order(pairs: list[QAPair], random_seed: int) -> list[QAPair]:
    rng = np.random.default_rng(seed=random_seed)

    for pair in pairs:
        perm1 = rng.permutation(len(pair.answer_choices))

        pair.answer_choices = [pair.answer_choices[i] for i in perm1]
        pair.answer_types = [pair.answer_types[i] for i in perm1]
        pair.answer_parts = [pair.answer_parts[i] for i in perm1]

        if pair.object_question:
            perm2 = rng.permutation(len(pair.object_answer_choices))
            pair.object_answer_choices = [pair.object_answer_choices[i] for i in perm2]
            pair.object_answer_types = [pair.object_answer_types[i] for i in perm2]


# ----------------------------
# CLI
# ----------------------------
def parse_args(cl_args: list[str] = None, config_str: str = None):
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--qa_pairs_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--responses_path', type=str)
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--mode', choices=['online', 'batch'], default='online')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--requests_per_minute', type=int, default=120)  # a bit more conservative
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--max_retries', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--limit_instances', type=int, default=None)
    parser.add_argument('--shuffle_answer_order', type=bool, default=True)

    if config_str:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args)

    return args


def main(cl_args: list[str] = None, config_str: str = None):
    import coloredlogs
    coloredlogs.install(level='INFO')

    args = parse_args(cl_args, config_str)

    pairs = load_qa_pairs(args.qa_pairs_path)
    pairs = pairs[:args.limit_instances]

    if args.shuffle_answer_order:
        shuffle_answer_order(pairs, args.random_seed)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY environment variable is not set')

    client = OpenAI(api_key=api_key)

    # Thread max_retries into the retry wrapper by partially applying where needed
    # (we pass it directly in calls above via default, but you can edit retry_api_call signature if you want dynamic max_retries)

    if args.mode == 'online':
        responses = send_online_requests(
            client,
            pairs,
            model=args.model,
            log_every=args.log_every,
            requests_per_minute=args.requests_per_minute,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    else:
        responses = run_batch_inference(
            client,
            pairs,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

    # Dump responses
    with open(args.responses_path, 'wb') as f:
        f.write(orjson.dumps(responses, option=orjson.OPT_INDENT_2))

    metrics, annotated = evaluate(pairs, responses)
    logger.info(f'Metrics: {metrics}')

    output_dict = {'metrics': metrics, 'data': annotated}
    with open(args.output_path, 'wb') as f:
        f.write(orjson.dumps(output_dict, option=orjson.OPT_INDENT_2))

    logger.info(orjson.dumps(metrics, option=orjson.OPT_INDENT_2).decode('utf-8'))


if __name__ == '__main__':
    main(config_str=f'''
        qa_pairs_path: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partonomy/partonomy_qa_pairs_val.json
        responses_path: responses.json
        output_path: gpt4o_mc_eval.json
        model: gpt-4o
        mode: online
        temperature: 0
        top_p: 1
        log_every: 10
        requests_per_minute: 120
        max_tokens: 128
        max_retries: 128

        # limit_instances: 10
        shuffle_answer_order: true
    ''')