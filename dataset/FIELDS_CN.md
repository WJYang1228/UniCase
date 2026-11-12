# Multi-Fraud 字段说明

每个 contracts/*.json 对应一个“链 + 合约地址”，不是一笔交易。

| 字段 | 含义 |
|---|---|
| contract_id / chain / address | 合约标识、所在链和合约地址 |
| binary_label | 0=良性，1=欺诈；沿用候选来源的二分类标签 |
| fraud_type | 分配的五类标签或 Benign，适用边界见标签来源说明 |
| deployment_block / deployment_timestamp | 保存的部署边界区块及链上 Unix 时间（秒） |
| window_start_block / window_end_block | 部署区块至部署+199，含两端，共200块 |
| runtime_bytecode / runtime_bytecode_sha256 | 保存的运行时代码及原始字节的 SHA256 |
| event_count | 此合约保存的 ERC-20 Transfer 日志数量 |
| distinct_transaction_count | 此合约日志对应的不同交易哈希数量 |
| erc20_transfer_events | 按 block_number、transaction_index、log_index 排序的日志 |
| block_number / block_hash | 日志所在区块高度及区块哈希 |
| transaction_index / log_index | 区块内交易位置及日志位置 |
| tx_hash | 交易哈希；一笔交易可以产生多条事件 |
| from_address / to_address | Transfer 事件中的转出方、转入方 |
| token_value_raw | 未按 decimals 换算的原始整数，使用字符串保存 |
| timestamp | 所在区块的 Unix 时间（秒），不是采集时间 |
| split / entity_group_id | 仅在正式 manifest.csv 中：固定划分及相同 runtime 字节码分组 |

manifest.csv 是模型使用的固定清单；source_manifest.csv 是原压缩包清单。
model_input/samples.json 是训练时的原始输入缓存，保留原字节与 SHA256，包含5%、20%、50%三种事件前缀。
事件前缀长度为 ceil(N200 * ratio / 100)，时间特征使用当次 query-deployment 的差值（最小为1）归一化。
N200 比例属于回溯评估预算，不是实时已知的全量事件计数。普通交易和内部调用若没有 Transfer 日志，不在此 H200 包内。
