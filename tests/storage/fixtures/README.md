`mixed_dictionary_row_groups.parquet` 仅包含合成字符串，使用 DuckDB 1.5.3
单线程一次生成；测试直接读取此夹具，无需 DuckDB 依赖。
第一个 2,048 行的行组使用 PLAIN，第二个使用 PLAIN_DICTIONARY。

```sql
COPY (SELECT CASE WHEN i < 2048 THEN repeat(lpad(i::VARCHAR, 8, '0'), 16)
                  ELSE repeat('x', 128) END AS value
      FROM range(4096) t(i))
TO 'mixed_dictionary_row_groups.parquet'
(FORMAT PARQUET, ROW_GROUP_SIZE 2048, COMPRESSION ZSTD);
```
