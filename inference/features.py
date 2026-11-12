"""Fixed features, never fitted to labels. See IMPLEMENTATION_CN.md for conventions."""
import math
from collections import Counter
import numpy as np

STATIC_DIM = 264
EVENT_NAMES = ("log_raw_value", "mint", "burn", "self_transfer", "from_deployer",
               "to_deployer", "first_seen_sender", "first_seen_receiver",
               "sender_out_fraction", "receiver_in_fraction", "log_unique_addresses",
               "log_cumulative_raw_value", "relative_block", "relative_block_gap",
               "relative_event_position", "log_prefix_length")
EVENT_DIM = len(EVENT_NAMES)
ZERO = "0x" + "0" * 40


def static_features(bytecode):
    """256 opcode frequencies + 7 opcode capability flags + log byte length.
    PUSH immediate bytes are not opcodes. Metadata is not heuristically stripped.
    No chain, address, security report or label-derived inputs.
    """
    if not isinstance(bytecode, str) or not bytecode.startswith("0x"):
        raise ValueError("runtime bytecode must be 0x-prefixed hex")
    raw = bytes.fromhex(bytecode[2:])
    if not raw:
        raise ValueError("empty runtime bytecode is not a valid contract input")
    ops = []; i = 0
    while i < len(raw):
        op = raw[i]; ops.append(op)
        i += 1 + (op - 0x5f if 0x60 <= op <= 0x7f else 0)
    freq = np.bincount(ops, minlength=256).astype(np.float32) / len(ops)
    flags = np.array([float(op in ops) for op in (0x55, 0xf1, 0xf4, 0xfa, 0xff, 0xf0, 0xf5)], np.float32)
    return np.concatenate((freq, flags, [math.log1p(len(raw))])).astype(np.float32)


def cursor(event, provider_order=False):
    block = int(event["block_number"])
    if "transaction_index" in event and "log_index" in event:
        tx, log = int(event["transaction_index"]), int(event["log_index"])
    elif provider_order and event.get("event_index") is not None:
        tx, log = 0, int(event["event_index"])
    else:
        raise ValueError("canonical transaction_index/log_index missing")
    if min(block, tx, log) < 0:
        raise ValueError("negative event cursor")
    return block, tx, log


def event_features(events, deployment_block, query_block, deployer=""):
    """One vector per event; time coordinates recomputed at EACH query.
    denom = max(1, query_block - deployment_block).
    No harm block, global-history max, future value, or retrospective cutoff.
    Events must already be in validated chronological order. Same-block ties
    are validated by the data adapter, which has canonical/provider provenance.
    """
    deployment_block, query_block = int(deployment_block), int(query_block)
    if query_block < deployment_block:
        raise ValueError("query predates deployment")
    denominator = max(1, query_block - deployment_block)
    src_count, dst_count, known = Counter(), Counter(), set()
    total = 0; previous = deployment_block; rows = []
    n = len(events); deployer = deployer.lower()
    for j, e in enumerate(events, 1):
        b = int(e["block_number"])
        if not previous <= b <= query_block:
            raise ValueError("unordered event or event outside current query")
        src, dst = e["from_address"].lower(), e["to_address"].lower()
        value = int(e["token_value_raw"])
        if not src or not dst or not 0 <= value < 2**256:
            raise ValueError("invalid ERC-20 value or address")
        new_src, new_dst = src not in known, dst not in known
        known.update((src, dst)); src_count[src] += 1; dst_count[dst] += 1; total += value
        rows.append([math.log1p(value), src == ZERO, dst == ZERO, src == dst,
                     bool(deployer) and src == deployer, bool(deployer) and dst == deployer,
                     new_src, new_dst, src_count[src]/j, dst_count[dst]/j,
                     math.log1p(len(known)), math.log1p(total),
                     (b-deployment_block)/denominator, (b-previous)/denominator,
                     j/max(n, 1), math.log1p(j)])
        previous = b
    return np.asarray(rows, dtype=np.float32).reshape(n, EVENT_DIM)

