`mixed_dictionary_row_groups.parquet` 是只含合成字符串的测试文件，使用 DuckDB 1.5.3
单线程生成。测试直接读取这个已保存文件，不需要安装 DuckDB。
第一个含 2,048 行的行组使用普通编码（PLAIN），第二个使用字典编码（PLAIN_DICTIONARY）。

```sql
COPY (SELECT CASE WHEN i < 2048 THEN repeat(lpad(i::VARCHAR, 8, '0'), 16)
                  ELSE repeat('x', 128) END AS value
      FROM range(4096) t(i))
TO 'mixed_dictionary_row_groups.parquet'
(FORMAT PARQUET, ROW_GROUP_SIZE 2048, COMPRESSION ZSTD);
```
