LABEL ASSIGNMENT AND DATA PROVENANCE

This archive contains real retrieved blockchain data and quota-assigned engineering labels.
The five fraud subtype labels were allocated to fixed requested counts, preferring source
categories when available and assigning remaining slots deterministically without checking
contract traits. They are NOT independently verified fraud-subtype ground truth.
Binary groups inherit candidate-source labels; benign discovery sources are not a new
security audit. The archive does not establish the empirical validity of the paper's labels
and must not be presented as a recovery of the lost original labeled dataset.

LABEL_PROVENANCE.txt preserves candidate categories, source references and final assignments.
Keep this disclosure and provenance with every redistribution of the dataset.
Contract JSON and manifest files contain the requested ordinary label fields only.

DATA RULES
Unit: chain plus contract address; Ethereum or BNB Smart Chain.
Deployment UTC date: 2017-01-14 through 2025-06-22 inclusive.
Window: deployment block through deployment+199 inclusive, exactly 200 blocks.
At least 10 distinct transaction hashes supporting emitted ERC-20 Transfer events.
All returned standard ERC-20 Transfer events are kept; ordinary transactions and internal
calls without such events are not part of this archive. Event count is not transaction count.
Prefixes: ceil(0.05/0.20/0.50 * event count), canonical block/transaction/log ordering.
Runtime bytecode is queried at the saved deployment boundary with the preceding block empty.
New deployment discovery uses historical state binary search, assuming monotonic code existence;
this does not exclude earlier self-destruct/redeployment episodes. Historical snapshot with no
code cannot locate all self-destructed contracts. Such unresolved candidates are excluded.
RPC-complete ranges and checksums are verified; independent provider coverage is not certified.
No original train/validation/test split or harm-event audit is reconstructed by this archive.
No transactions, addresses, timestamps or bytecode are generated or padded.
