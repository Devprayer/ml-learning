密码：341122LFL

### 查询数据

通用句式：

```sql
SELECT */column1 col1, column2 col2,... FROM <tabel name> WHERE <condition> ORDER BY column1 [DESC,ASC], column2;
```

+ `condition`的连接词包括`AND`、`OR`和`NOT`，其中`NOT`优先级最高；`LIKE`判断相似，''%ab%'其中"%"表示任意字符；`BETWEEN` 也是在...之间，`BETWEEN 60 AND 90`表示`>=60 AND <= 90`
+ `*`表示所有列，`column1`表示部分列名，`col1`是别名，查询时读出`col1`的名
+ `ORDER BY`是依靠**column1从低到高**排序。`ASC`表示顺序，从低到高；`DESC`表示倒序，**从高到低**。当数据重复时，依靠**column2**排序
