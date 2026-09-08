`mixed_dictionary_row_groups.parquet` contains only synthetic strings. It was
generated once with DuckDB 1.5.3 (one thread); DuckDB is not a test dependency.
The first 2,048-row group uses PLAIN and the second uses PLAIN_DICTIONARY.

```sql
COPY (SELECT CASE WHEN i < 2048 THEN repeat(lpad(i::VARCHAR, 8, '0'), 16)
                  ELSE repeat('x', 128) END AS value
      FROM range(4096) t(i))
TO 'mixed_dictionary_row_groups.parquet'
(FORMAT PARQUET, ROW_GROUP_SIZE 2048, COMPRESSION ZSTD);
```
